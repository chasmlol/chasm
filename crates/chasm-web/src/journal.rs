//! The journal pass — the save-driven, per-NPC inner voice of the
//! self-improving-NPC system.
//!
//! It runs right after the Gamemaster relationships pass commits (chained at
//! the tail of [`crate::gamemaster::spawn_pass`]), so relationships are already
//! written when it starts. For each NPC who spoke or was present since their
//! last journal entry, it makes ONE structured LLM call — with THAT NPC's own
//! character card injected (the key difference from the persona-less GM pass) —
//! and appends a single new journal entry in the NPC's own voice, noting any
//! PATTERNS they see and what their PERSONALITY inclines them to do about it.
//!
//! The journal is APPEND-ONLY: a pass never rewrites or removes an earlier
//! entry (see [`chasm_st_compat::JournalStore::append`]). Per-session
//! watermarks (the journal store's own, independent of the GM's) only advance
//! when a pass succeeds, so a failed LLM call is retried next save over the
//! same content.
//!
//! When the pass finishes it chains the persona-less skill-creator pass
//! ([`crate::skill_creator::spawn_pass`]), which reads these journals.
//!
//! Like the GM pass this is fire-and-forget behind a busy flag: it never delays
//! the capture response, the persona generation, or an NPC turn.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    sync::Arc,
};

use serde_json::{json, Value};

use chasm_core::AppSettings;
use chasm_st_compat::{JournalEntry, LiveChat, STJsonlChatMessage};

use crate::AppState;

/// One journal pass at a time, process-wide (mirrors the GM pass). A save
/// arriving mid-pass is skipped — its content stays past the watermark.
static RUNNING: AtomicBool = AtomicBool::new(false);

/// `max_tokens` for one journal entry: a few sentences up to a short paragraph.
const JOURNAL_MAX_TOKENS: i64 = 700;
/// Per-session cap of NEW transcript lines fed to one pass (bounds a first-ever
/// pass over a long history).
const MAX_NEW_LINES_PER_SESSION: usize = 300;
/// Upper bound on one stored entry's text (truncated at a char boundary).
const MAX_ENTRY_CHARS: usize = 1200;
/// How many of an NPC's PRIOR entries to show as read-only context (the tail;
/// they are never rewritten, only added to).
const MAX_PRIOR_ENTRIES: usize = 6;
/// How many recent ambient events to show an NPC as "what happened around you".
const MAX_AMBIENT_EVENTS: usize = 40;

/// Save-driven check: journal store path helpers (byte-copy sidecars keyed by
/// the save-sync checkpoint id, exactly like the scheduler store).
fn journal_store_path_at(content_root: &Path) -> std::path::PathBuf {
    content_root.join("headless").join("journals.json")
}

fn journal_checkpoint_path(content_root: &Path, checkpoint_id: &str) -> std::path::PathBuf {
    content_root
        .join("headless")
        .join("save-sync")
        .join("journal-checkpoints")
        .join(format!("{checkpoint_id}.json"))
}

/// Snapshot the journal store for a save checkpoint. A missing store writes an
/// EMPTY snapshot so a later restore correctly clears journals authored after
/// this checkpoint (rollback of a discarded branch).
pub fn checkpoint_journal_store(content_root: &Path, checkpoint_id: &str) {
    let dst = journal_checkpoint_path(content_root, checkpoint_id);
    if let Some(dir) = dst.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match std::fs::read(journal_store_path_at(content_root)) {
        Ok(bytes) => {
            let _ = std::fs::write(&dst, bytes);
        }
        Err(_) => {
            let _ = std::fs::write(&dst, b"{}");
        }
    }
    tracing::info!("journal: checkpointed store for {checkpoint_id}");
}

/// Restore the journal store from a checkpoint's sidecar on load. A missing
/// sidecar means the save predates any journal, so the live store is CLEARED.
pub fn restore_journal_store(content_root: &Path, checkpoint_id: &str) {
    let dst = journal_store_path_at(content_root);
    if let Some(dir) = dst.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match std::fs::read(journal_checkpoint_path(content_root, checkpoint_id)) {
        Ok(bytes) => {
            let _ = std::fs::write(&dst, bytes);
        }
        Err(_) => {
            tracing::info!("journal: cleared store (no sidecar for {checkpoint_id})");
            let _ = std::fs::write(&dst, b"{}");
        }
    }
}

/// Spawns a journal pass on a background task unless one is already running.
/// On completion (success OR failure) it chains the skill-creator pass, which
/// reads whatever journals now exist. Returns whether a task was started.
pub(crate) fn spawn_pass(state: Arc<AppState>) -> bool {
    if RUNNING.swap(true, Ordering::SeqCst) {
        return false;
    }
    tokio::spawn(async move {
        match run_pass(&state).await {
            Ok(outcome) => tracing::info!(
                target: "chasm::journal",
                npcs = outcome.npcs_considered,
                entries = outcome.entries_written,
                "journal pass complete"
            ),
            Err(error) => tracing::warn!(
                target: "chasm::journal",
                error = %error,
                "journal pass failed (content stays queued for the next save)"
            ),
        }
        RUNNING.store(false, Ordering::SeqCst);
        // Chain the skill-creator over the (now updated) journals. Its own busy
        // flag + settings gate apply; nothing to do is a fast no-op.
        crate::skill_creator::spawn_pass(state.clone());
    });
    true
}

/// True while a journal pass is in flight (for the UI).
pub(crate) fn pass_in_flight() -> bool {
    RUNNING.load(Ordering::SeqCst)
}

#[derive(Debug, Default)]
struct PassOutcome {
    npcs_considered: usize,
    entries_written: usize,
}

/// One transcript line past the watermark, normalized for the journal — a
/// spoken line OR a witnessed world beat (the SAME `is_system` narration the NPC
/// sees in their own history), so the journal reads the events woven into the
/// conversation exactly as the character experienced them.
#[derive(Debug, Clone)]
struct NewLine {
    scene: String,
    speaker_name: String,
    character_id: Option<String>,
    is_user: bool,
    text: String,
    /// Message timestamp (ISO-8601), used to interleave beats + spoken lines in
    /// true chronological order regardless of which session file they came from.
    send_date: String,
    /// A witnessed world beat (`is_system` + `chasm.witnessed`) — a thing that
    /// happened around the character, not a spoken line. Rendered as a bracketed
    /// beat and attributed only to the NPCs it was audible to.
    is_narration: bool,
    /// For a narration beat: the participant ids it was audible to (its
    /// witnesses). Empty for spoken lines.
    audience_participant_ids: Vec<String>,
}

async fn run_pass(state: &AppState) -> Result<PassOutcome, String> {
    let settings = AppSettings::load(&state.config.settings_path);
    if !settings.self_improvement.journaling_enabled {
        return Ok(PassOutcome::default());
    }
    let repo = &state.repository;
    let chats = repo
        .read_store()
        .map_err(|error| format!("live-chats store read failed: {error}"))?;
    let store = repo
        .read_journals()
        .map_err(|error| format!("journal store read failed: {error}"))?;

    // --- 1) Collect new transcript lines past each session watermark. --------
    let mut new_lines: Vec<NewLine> = Vec::new();
    let mut scanned: BTreeMap<String, u64> = BTreeMap::new();
    let mut seen: BTreeSet<(String, String, String)> = BTreeSet::new();
    // character id -> display name (card name wins later).
    let mut npc_names: BTreeMap<String, String> = BTreeMap::new();
    // participant id -> character id, so a witnessed beat (audible to a
    // participant) can be attributed to the right NPC's journal.
    let mut participant_to_character: BTreeMap<String, String> = BTreeMap::new();
    let mut player_name = String::new();

    for chat in chats.items.values() {
        let (_, macros) = crate::generate::latest_chat_macros(state, chat);
        if player_name.is_empty() {
            player_name = macros
                .get("player_name")
                .or_else(|| macros.get("Player_Name"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
        }
        for participant in chat.participants.values().chain(chat.presence.values()) {
            if let Some(id) = participant
                .character_id
                .as_deref()
                .filter(|id| !id.is_empty())
            {
                if !participant.participant_id.is_empty() {
                    participant_to_character
                        .entry(participant.participant_id.clone())
                        .or_insert_with(|| id.to_string());
                }
                if participant.kind == "npc" {
                    let display = if participant.name.is_empty() {
                        id.to_string()
                    } else {
                        participant.name.clone()
                    };
                    npc_names.entry(id.to_string()).or_insert(display);
                }
            }
        }
        for session_id in chat_session_ids(chat) {
            let messages = match repo.read_session_messages(&session_id) {
                Ok(messages) => messages,
                Err(_) => continue,
            };
            let watermark = store.watermarks.get(&session_id).copied().unwrap_or(0);
            let len = messages.len() as u64;
            let (fresh, _mark) = if watermark > len {
                (&messages[..], 0u64)
            } else {
                (&messages[watermark as usize..], watermark)
            };
            scanned.insert(session_id.clone(), len);
            let fresh = if fresh.len() > MAX_NEW_LINES_PER_SESSION {
                &fresh[fresh.len() - MAX_NEW_LINES_PER_SESSION..]
            } else {
                fresh
            };
            for message in fresh {
                let Some(line) = normalize_line(message, &session_id) else {
                    continue;
                };
                let key = (
                    message.send_date.clone().unwrap_or_default(),
                    line.speaker_name.clone(),
                    line.text.clone(),
                );
                if seen.insert(key) {
                    new_lines.push(line);
                }
            }
        }
    }
    if player_name.is_empty() {
        player_name = "the player".to_string();
    }

    // Nothing new anywhere → advance any shrink-reset watermarks and return.
    if new_lines.is_empty() {
        let needs_write = scanned
            .iter()
            .any(|(id, count)| store.watermarks.get(id).is_some_and(|w| w != count));
        if needs_write {
            let _ = repo.update_journals(|s| {
                for (id, count) in &scanned {
                    s.watermarks.insert(id.clone(), *count);
                }
            });
        }
        return Ok(PassOutcome::default());
    }

    // --- 2) Which NPCs have new material, and what did they see? -------------
    // Scene -> the set of NPC ids who spoke there (so an NPC's material is every
    // line from scenes they took part in — the player's lines and other NPCs'
    // lines included).
    let mut scene_npcs: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for line in &new_lines {
        if let Some(id) = line.character_id.as_deref() {
            scene_npcs.entry(line.scene.as_str()).or_default().insert(id);
        }
    }
    let ambient = ambient_events(state);
    let now = crate::persona::chrono_now_iso();
    let (game_time, game_day) = latest_game_clock(&ambient);

    // --- 3) One structured journal call per NPC with new material. -----------
    let mut outcome = PassOutcome::default();
    let mut written: Vec<(String, String, JournalEntry)> = Vec::new();
    let npc_ids: BTreeSet<String> = scene_npcs
        .values()
        .flat_map(|ids| ids.iter().map(|id| id.to_string()))
        .collect();

    let gm_settings = &settings;
    let sampling = crate::llm::Sampling::from_settings(&gm_settings.llm.sampling).with_overrides(
        crate::llm::GenerationOptions {
            temperature: Some(0.7),
            max_tokens: Some(JOURNAL_MAX_TOKENS),
        },
    );
    let target = crate::llm::LlmTarget::resolve(gm_settings, &state.config);

    for npc_id in npc_ids {
        // The lines this NPC saw: every spoken line from a scene they spoke in,
        // PLUS the witnessed world beats that were audible to THEM (only) —
        // exactly the mix that rides their own chat history. Sorted by timestamp
        // so beats and dialogue interleave in the order the character lived them
        // (e.g. "[Alex drew a weapon]" then "Alex: do push-ups", repeating).
        let mut material: Vec<&NewLine> = new_lines
            .iter()
            .filter(|line| {
                if line.is_narration {
                    line.audience_participant_ids.iter().any(|pid| {
                        participant_to_character.get(pid).is_some_and(|c| c == &npc_id)
                    })
                } else {
                    scene_npcs
                        .get(line.scene.as_str())
                        .is_some_and(|ids| ids.contains(npc_id.as_str()))
                }
            })
            .collect();
        material.sort_by(|a, b| a.send_date.cmp(&b.send_date));
        // Nothing to reflect on if the NPC only saw beats but never spoke.
        if material.is_empty() || material.iter().all(|line| line.is_narration) {
            continue;
        }
        outcome.npcs_considered += 1;

        let card = repo.read_character_card(&npc_id).ok().flatten();
        let display_name = card
            .as_ref()
            .map(|c| c.name.clone())
            .filter(|n| !n.is_empty())
            .or_else(|| npc_names.get(&npc_id).cloned())
            .unwrap_or_else(|| npc_id.clone());

        let persona_block = card
            .as_ref()
            .map(|c| persona_block(c))
            .unwrap_or_else(|| format!("You are {display_name}."));
        let prior = prior_entries_text(&store, &npc_id);
        let transcript = format_transcript(&material, &player_name);
        let ambient_text = ambient_text(&ambient, &display_name);
        let user_prompt = build_user_prompt(&display_name, &prior, &ambient_text, &transcript);

        let messages = vec![
            json!({ "role": "system", "content": format!("{JOURNAL_SYSTEM_PROMPT}\n\n{persona_block}") }),
            json!({ "role": "user", "content": user_prompt }),
        ];
        let (content, _metrics) = match crate::llm::chat_completion_capturing_sampled(
            &target,
            &messages,
            Some(&entry_response_format()),
            sampling.clone(),
        )
        .await
        {
            Ok(pair) => pair,
            Err(error) => {
                tracing::warn!(target: "chasm::journal", npc = %display_name, error = %error, "journal LLM call failed; skipping this NPC");
                continue;
            }
        };
        let text = parse_entry(&content);
        let text = bounded_text(&text);
        if text.is_empty() {
            continue;
        }
        written.push((
            npc_id.clone(),
            display_name.clone(),
            JournalEntry {
                created_at: now.clone(),
                game_time: game_time.clone(),
                game_day,
                text,
            },
        ));
    }

    // --- 4) Append entries + advance watermarks in ONE store write. ----------
    repo.update_journals(|s| {
        for (id, name, entry) in &written {
            s.append(id, name, entry.clone());
        }
        for (session_id, count) in &scanned {
            s.watermarks.insert(session_id.clone(), *count);
        }
        s.last_pass_at = Some(now.clone());
    })
    .map_err(|error| format!("journal store write failed: {error}"))?;
    outcome.entries_written = written.len();
    Ok(outcome)
}

/// The journal system prompt (the NPC's own card is appended after it). Elicits
/// pattern-noticing + a personality-driven intended response, WITHOUT naming
/// any specific example so the behaviour is general.
const JOURNAL_SYSTEM_PROMPT: &str = "You keep a private journal — your own inner voice, that no one else will ever read. You will be shown everything that has happened to you since your last entry, in order: what people said and did, and things that happened around you (events you saw are shown in [brackets]). Write ONE new journal entry in your own voice: how you feel, and — most importantly — any PATTERN you are starting to notice.

Look for CONNECTIONS, not just moods. The pattern that matters most is when one thing reliably leads to another: something the player (or someone else) keeps doing, and what keeps happening right after it. When you spot one, name it plainly to yourself — what keeps happening, and what follows — and then decide how you mean to handle it the NEXT time that thing happens.

This is about YOUR reaction, not about doing as you are told — and be specific about what you would actually DO. Let your character decide completely: you might get ahead of it and start doing it on your own before you are even asked, answer it your own way with something entirely different, go along with it, do it grudgingly, or refuse outright and push back. A proud, stubborn, or contrary character is every bit as likely to resolve on some response of their OWN choosing as an eager one is to simply comply. There is no right answer — resolve on whatever THIS character honestly would.

Rules:
- Write ONLY the new entry. Never repeat, summarize, or rewrite earlier entries — the journal is only ever added to.
- A few sentences, up to a short paragraph. First person, your natural voice.
- Stay fully in character. Never mention game mechanics, skills, triggers, saves, or that you are an AI.
- If truly nothing since the last entry is worth reflecting on, return an empty entry.";

/// llama.cpp-enforced response shape: a single `entry` string (empty = wrote
/// nothing this time).
fn entry_response_format() -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "chasm_journal_entry",
            "description": "One new private journal entry in the character's own voice.",
            "strict": true,
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "entry": {
                        "type": "string",
                        "description": "The new journal entry. Empty when nothing is worth reflecting on."
                    }
                },
                "required": ["entry"]
            }
        }
    })
}

fn parse_entry(content: &str) -> String {
    serde_json::from_str::<Value>(content.trim())
        .ok()
        .and_then(|v| v.get("entry").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// Compose an NPC's persona block from its card (name + system prompt +
/// description + personality), skipping empty fields.
fn persona_block(card: &chasm_st_compat::CharacterCard) -> String {
    let mut parts = vec![format!("You are {}.", card.name)];
    for (label, body) in [
        ("", card.system_prompt.trim()),
        ("Description: ", card.description.trim()),
        ("Personality: ", card.personality.trim()),
    ] {
        if !body.is_empty() {
            parts.push(format!("{label}{body}"));
        }
    }
    parts.join("\n")
}

fn prior_entries_text(store: &chasm_st_compat::JournalStore, character_id: &str) -> String {
    let entries = store.entries_for(character_id);
    if entries.is_empty() {
        return "(this is your first entry)".to_string();
    }
    let start = entries.len().saturating_sub(MAX_PRIOR_ENTRIES);
    entries[start..]
        .iter()
        .map(|e| format!("- {}", e.text.trim()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_transcript(lines: &[&NewLine], player_name: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for line in lines {
        if line.is_narration {
            // A witnessed world beat, exactly as it reads in the NPC's history.
            out.push(format!("[{}]", line.text.trim()));
        } else {
            let name = if line.is_user {
                player_name
            } else {
                line.speaker_name.as_str()
            };
            out.push(format!("{name}: {}", line.text));
        }
    }
    out.join("\n")
}

fn build_user_prompt(name: &str, prior: &str, ambient: &str, transcript: &str) -> String {
    format!(
        "You are {name}.\n\n\
         Your earlier journal entries (for context only — do NOT rewrite them):\n{prior}\n\n\
         Background — other things going on around you recently:\n{ambient}\n\n\
         What you saw and heard since your last entry, in order (events you \
         witnessed are shown in [brackets]; look for how they line up with what \
         was said):\n{transcript}\n\n\
         Write your one new journal entry now."
    )
}

/// Recent ambient events (newest last) that an NPC could plausibly have
/// witnessed — kept to the world-beat types (combat, gunfire, deaths, travel).
fn ambient_events(state: &AppState) -> Vec<Value> {
    let content_root = state.config.active_profile_paths().content_root();
    let all = crate::event_log::read_current_events(&content_root);
    let start = all.len().saturating_sub(MAX_AMBIENT_EVENTS);
    all[start..].to_vec()
}

fn ambient_text(events: &[Value], _name: &str) -> String {
    let lines: Vec<String> = events
        .iter()
        .filter_map(|e| e.get("summary").and_then(Value::as_str))
        .filter(|s| !s.trim().is_empty())
        .map(|s| format!("- {s}"))
        .collect();
    if lines.is_empty() {
        "(nothing notable)".to_string()
    } else {
        lines.join("\n")
    }
}

/// The newest event's in-game clock, for stamping the journal entry.
fn latest_game_clock(events: &[Value]) -> (Option<String>, Option<i64>) {
    for event in events.iter().rev() {
        let gt = event
            .get("gameTime")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        let gd = event.get("gameDay").and_then(Value::as_i64);
        if gt.is_some() || gd.is_some() {
            return (gt, gd);
        }
    }
    (None, None)
}

fn bounded_text(text: &str) -> String {
    let text = text.trim();
    if text.chars().count() <= MAX_ENTRY_CHARS {
        return text.to_string();
    }
    let truncated: String = text.chars().take(MAX_ENTRY_CHARS).collect();
    format!("{}…", truncated.trim_end())
}

// --- Transcript-gathering helpers (self-contained; mirror the GM pass). ------

/// Every transcript session backing one live chat: the shared segment streams
/// plus the per-participant projection sessions.
fn chat_session_ids(chat: &LiveChat) -> Vec<String> {
    let mut ids: Vec<String> = chat
        .segments
        .iter()
        .map(|segment| segment.session_id.clone())
        .filter(|id| !id.is_empty())
        .collect();
    if let Some(sessions) = chat.participant_sessions.as_object() {
        ids.extend(sessions.values().filter_map(|entry| {
            entry
                .get("sessionId")
                .and_then(Value::as_str)
                .or_else(|| entry.as_str())
                .filter(|id| !id.is_empty())
                .map(str::to_string)
        }));
    }
    ids.sort();
    ids.dedup();
    ids
}

/// One raw speech line: speaker + text + scene + player flag; no system lines,
/// no empties.
fn normalize_line(message: &STJsonlChatMessage, session_id: &str) -> Option<NewLine> {
    let text = message.mes.trim();
    if text.is_empty() {
        return None;
    }
    // A witnessed world beat is a system line the NPC DID see (it rides their
    // history). Keep those; drop every other system line (placeholders, etc.).
    let witnessed = message
        .extra
        .get("chasm")
        .and_then(|c| c.get("witnessed"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if message.is_system && !witnessed {
        return None;
    }
    let headless = message.extra.get("headless");
    let live = headless.and_then(|h| h.get("metadata")).and_then(|m| m.get("live"));
    let character_id = headless
        .and_then(|h| h.get("characterId"))
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_string);
    let scene = live
        .and_then(|l| l.get("segmentId"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(session_id)
        .to_string();
    let audience_participant_ids = if witnessed {
        live.and_then(|l| l.get("audibleTo").or_else(|| l.get("present")))
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    Some(NewLine {
        scene,
        speaker_name: message.name.clone(),
        character_id,
        is_user: message.is_user,
        text: text.to_string(),
        send_date: message.send_date.clone().unwrap_or_default(),
        is_narration: witnessed,
        audience_participant_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_entry_extracts_text_and_tolerates_junk() {
        assert_eq!(parse_entry(r#"{"entry":"He shot again today."}"#), "He shot again today.");
        assert_eq!(parse_entry(r#"{"entry":"   "}"#), "");
        assert_eq!(parse_entry("not json"), "");
        assert_eq!(parse_entry(r#"{"other":"x"}"#), "");
    }

    #[test]
    fn bounded_text_trims_and_caps() {
        assert_eq!(bounded_text("  hi  "), "hi");
        let long = "x".repeat(2000);
        let bounded = bounded_text(&long);
        assert!(bounded.chars().count() <= MAX_ENTRY_CHARS + 1);
        assert!(bounded.ends_with('…'));
    }

    #[test]
    fn latest_game_clock_reads_newest_with_a_clock() {
        let events = vec![
            json!({ "summary": "a", "gameTime": "08:00", "gameDay": 1 }),
            json!({ "summary": "b" }),
            json!({ "summary": "c", "gameTime": "12:00", "gameDay": 2 }),
            json!({ "summary": "d" }),
        ];
        // Newest-with-a-clock is event "c".
        let (gt, gd) = latest_game_clock(&events);
        assert_eq!(gt.as_deref(), Some("12:00"));
        assert_eq!(gd, Some(2));
    }

    #[test]
    fn ambient_text_lists_summaries_or_placeholder() {
        assert_eq!(ambient_text(&[], "Pete"), "(nothing notable)");
        let events = vec![json!({ "summary": "Fired a shot from your 10mm Pistol" })];
        assert_eq!(ambient_text(&events, "Pete"), "- Fired a shot from your 10mm Pistol");
    }

    fn msg(is_system: bool, mes: &str, send: Option<&str>, extra: Value) -> STJsonlChatMessage {
        STJsonlChatMessage {
            name: "Narrator".into(),
            is_user: false,
            is_system,
            send_date: send.map(str::to_string),
            mes: mes.into(),
            extra,
            original_avatar: None,
        }
    }

    #[test]
    fn normalize_keeps_witnessed_beats_but_drops_other_system_lines() {
        // A witnessed world beat is kept, flagged as narration, and carries its
        // audience (so it can be attributed to the right NPC's journal).
        let beat = msg(
            true,
            "Alex drew a weapon",
            Some("2026-01-01T00:00:01Z"),
            json!({
                "headless": { "metadata": { "live": { "segmentId": "seg1", "audibleTo": ["npc:chamzy"] } } },
                "chasm": { "witnessed": true }
            }),
        );
        let line = normalize_line(&beat, "sess").expect("witnessed beat is kept");
        assert!(line.is_narration);
        assert_eq!(line.audience_participant_ids, vec!["npc:chamzy".to_string()]);
        assert_eq!(line.scene, "seg1");
        assert_eq!(line.send_date, "2026-01-01T00:00:01Z");

        // A plain system line (no witnessed flag) is still dropped.
        assert!(normalize_line(&msg(true, "placeholder", None, json!({})), "sess").is_none());
    }

    #[test]
    fn format_transcript_brackets_beats_and_labels_speech() {
        let beat = NewLine {
            scene: "s".into(), speaker_name: "Narrator".into(), character_id: None,
            is_user: false, text: "Alex drew a weapon".into(), send_date: "t1".into(),
            is_narration: true, audience_participant_ids: vec![],
        };
        let user = NewLine {
            scene: "s".into(), speaker_name: String::new(), character_id: None,
            is_user: true, text: "do push-ups".into(), send_date: "t2".into(),
            is_narration: false, audience_participant_ids: vec![],
        };
        let npc = NewLine {
            scene: "s".into(), speaker_name: "chamzy".into(), character_id: Some("chamzy".into()),
            is_user: false, text: "Yes, master.".into(), send_date: "t3".into(),
            is_narration: false, audience_participant_ids: vec![],
        };
        let out = format_transcript(&[&beat, &user, &npc], "Alex");
        assert_eq!(out, "[Alex drew a weapon]\nAlex: do push-ups\nchamzy: Yes, master.");
    }
}
