//! The Gamemaster pass — the save-driven curator of NPC relationships.
//!
//! Every game save (the same `save`/`quicksave`/`autosave` capture trigger
//! that regenerates the player persona, see `persona::receive_capture`) spawns
//! one pass here. The pass reads only the conversation content that arrived
//! since the previous pass (per-session watermarks in the relationships
//! store), shows the GM model each involved character's EXISTING entries plus
//! the new raw transcript lines, and lets it upsert directional
//! `character → target` stance entries — or decide nothing notable happened.
//!
//! One structured-output LLM call per pass (the same llama.cpp
//! `response_format` enforcement NPC turns use), one store write at the end.
//! Watermarks only advance when the pass succeeds, so a failed LLM call is
//! retried on the next save with the same content.
//!
//! Like persona generation this is spawned on a background task behind a busy
//! flag: it can never delay the capture response, the persona generation, or
//! an NPC turn.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::{AtomicBool, Ordering},
    sync::Arc,
};

use serde_json::{json, Value};
use chasm_core::AppSettings;
use chasm_st_compat::{LiveChat, STJsonlChatMessage, PLAYER_TARGET_ID};

use crate::AppState;

/// One GM pass at a time, process-wide. A save arriving mid-pass is skipped —
/// its content stays beyond the watermark and the NEXT save picks it up.
static RUNNING: AtomicBool = AtomicBool::new(false);

/// `max_tokens` for the GM call: entries are 1–3 sentences and a pass rarely
/// touches more than a handful of pairs; this bounds a rambling model.
const GM_MAX_TOKENS: i64 = 1400;

/// Per-session cap of NEW lines fed to one pass. A first-ever pass over a long
/// history (or a watermark reset) could otherwise feed the model thousands of
/// lines; anything older than this window is skipped — the watermark still
/// advances past it, and the skip is logged, never silent.
const MAX_NEW_LINES_PER_SESSION: usize = 300;

/// A single upper bound on stored entry text. The prompt asks for 1–3
/// sentences; if the model ignores that, the entry is truncated at a char
/// boundary so injected prompt blocks stay bounded as entries evolve.
const MAX_ENTRY_CHARS: usize = 700;

/// Spawns a GM pass on a background task unless one is already running.
/// Returns whether a task was started. Mirrors `persona::spawn_generation`.
pub(crate) fn spawn_pass(state: Arc<AppState>) -> bool {
    if RUNNING.swap(true, Ordering::SeqCst) {
        return false;
    }
    tokio::spawn(async move {
        match run_pass(&state).await {
            Ok(outcome) => tracing::info!(
                target: "chasm::gamemaster",
                sessions = outcome.sessions_scanned,
                new_lines = outcome.new_lines,
                updates = outcome.updates_applied,
                "gamemaster pass complete"
            ),
            Err(error) => tracing::warn!(
                target: "chasm::gamemaster",
                error = %error,
                "gamemaster pass failed (content stays queued for the next save)"
            ),
        }
        RUNNING.store(false, Ordering::SeqCst);
    });
    true
}

/// True while a GM pass is in flight (for the UI, mirroring persona).
pub(crate) fn pass_in_flight() -> bool {
    RUNNING.load(Ordering::SeqCst)
}

#[derive(Debug, Default)]
pub(crate) struct PassOutcome {
    pub sessions_scanned: usize,
    pub new_lines: usize,
    pub updates_applied: usize,
}

/// One transcript line past the watermark, normalized for the GM.
#[derive(Debug, Clone)]
struct NewLine {
    /// Scene key (the live segment id, falling back to the session id) —
    /// NPC↔NPC pairs only form between characters who shared a scene.
    scene: String,
    speaker_name: String,
    /// The speaker's character id (`None` for the player).
    character_id: Option<String>,
    is_user: bool,
    text: String,
}

/// A directional pair the GM is allowed to write this pass.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct AllowedPair {
    character_id: String,
    character_name: String,
    target_id: String,
    target_name: String,
    target_kind: &'static str,
}

/// The full pass: scan → prompt → structured call → upsert → advance marks.
async fn run_pass(state: &AppState) -> Result<PassOutcome, String> {
    let repo = &state.repository;
    let chats = repo
        .read_store()
        .map_err(|error| format!("live-chats store read failed: {error}"))?;
    let store = repo
        .read_relationships()
        .map_err(|error| format!("relationships store read failed: {error}"))?;

    // --- 1) Collect new content past each session watermark. ----------------
    let mut outcome = PassOutcome::default();
    let mut new_lines: Vec<NewLine> = Vec::new();
    // session_id → message count read this pass (the watermark to advance to).
    let mut scanned: BTreeMap<String, u64> = BTreeMap::new();
    // Duplicate copies of one turn exist across segment + projection sessions
    // (see `messages_for_participant`); collapse them by identity so one line
    // never reads as two events.
    let mut seen: BTreeSet<(String, String, String)> = BTreeSet::new();
    // name/id → participant info for the name→id mapping later.
    let mut npc_names: BTreeMap<String, (String, String)> = BTreeMap::new(); // lookup key → (char id, display name)
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
        for participant in chat.participants.values() {
            if participant.kind == "npc" {
                if let Some(id) = participant
                    .character_id
                    .as_deref()
                    .filter(|id| !id.is_empty())
                {
                    let display = if participant.name.is_empty() {
                        id.to_string()
                    } else {
                        participant.name.clone()
                    };
                    npc_names.insert(lookup_key(id), (id.to_string(), display.clone()));
                    npc_names.insert(lookup_key(&display), (id.to_string(), display));
                }
            }
        }
        for session_id in chat_session_ids(chat) {
            let messages = match repo.read_session_messages(&session_id) {
                Ok(messages) => messages,
                Err(error) => {
                    tracing::warn!(
                        target: "chasm::gamemaster",
                        session = %session_id,
                        error = %error,
                        "session unreadable; skipped this pass"
                    );
                    continue;
                }
            };
            outcome.sessions_scanned += 1;
            let watermark = store.watermarks.get(&session_id).copied().unwrap_or(0);
            let (fresh, effective) = slice_since_watermark(&messages, watermark);
            if effective != watermark {
                tracing::info!(
                    target: "chasm::gamemaster",
                    session = %session_id,
                    stored = watermark,
                    now = effective,
                    "session shrank since last pass; watermark reset"
                );
            }
            scanned.insert(session_id.clone(), messages.len() as u64);
            let fresh = if fresh.len() > MAX_NEW_LINES_PER_SESSION {
                let dropped = fresh.len() - MAX_NEW_LINES_PER_SESSION;
                tracing::warn!(
                    target: "chasm::gamemaster",
                    session = %session_id,
                    dropped,
                    "backlog exceeds per-pass window; oldest {dropped} new lines skipped"
                );
                &fresh[dropped..]
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
    outcome.new_lines = new_lines.len();
    if player_name.is_empty() {
        player_name = "the player".to_string();
    }

    // Nothing new anywhere → nothing to do, nothing to write (all stored
    // watermarks already match), unless a shrink reset needs persisting.
    if new_lines.is_empty() {
        let needs_write = scanned
            .iter()
            .any(|(id, count)| store.watermarks.get(id).is_some_and(|w| w != count));
        if needs_write {
            advance_watermarks(repo, &scanned)?;
        }
        return Ok(outcome);
    }

    // --- 2) Which directional pairs may change this pass? -------------------
    let pairs = allowed_pairs(&new_lines, &npc_names, &player_name);
    if pairs.is_empty() {
        // New content, but no NPC with a character id spoke — advance past it.
        advance_watermarks(repo, &scanned)?;
        return Ok(outcome);
    }

    // --- 3) One structured GM call. ------------------------------------------
    let involved: BTreeSet<&str> = pairs.iter().map(|p| p.character_id.as_str()).collect();
    let existing = existing_entries_text(&store, &involved);
    let transcript = format_transcript(&new_lines, &player_name);
    let user_prompt = build_user_prompt(&player_name, &existing, &transcript, &pairs);
    let messages = vec![
        json!({ "role": "system", "content": GM_SYSTEM_PROMPT }),
        json!({ "role": "user", "content": user_prompt }),
    ];
    let sampling = crate::llm::Sampling::from_settings(
        &AppSettings::load(&state.config.settings_path).llm.sampling,
    )
    .with_overrides(crate::llm::GenerationOptions {
        temperature: Some(0.4),
        max_tokens: Some(GM_MAX_TOKENS),
    });
    let (content, _metrics) = crate::llm::chat_completion_capturing_sampled(
        &state.config.llm_endpoint,
        &messages,
        Some(&updates_response_format()),
        sampling,
    )
    .await
    .map_err(|error| format!("GM LLM call failed: {error}"))?;
    let updates = parse_updates(&content)?;

    // --- 4) Apply + advance in ONE store write. ------------------------------
    let now = crate::persona::chrono_now_iso();
    let applied = repo
        .update_relationships(|store| {
            let mut applied = 0usize;
            for update in &updates {
                let Some(pair) = match_pair(&pairs, &update.character, &update.target) else {
                    tracing::info!(
                        target: "chasm::gamemaster",
                        character = %update.character,
                        target = %update.target,
                        "GM update ignored: not an allowed pair this pass"
                    );
                    continue;
                };
                let text = bounded_text(&update.text);
                if text.is_empty() {
                    continue;
                }
                store.upsert(
                    &pair.character_id,
                    &pair.target_id,
                    &pair.target_name,
                    pair.target_kind,
                    &text,
                    &now,
                );
                applied += 1;
            }
            for (session_id, count) in &scanned {
                store.watermarks.insert(session_id.clone(), *count);
            }
            store.last_pass_at = Some(now.clone());
            applied
        })
        .map_err(|error| format!("relationships store write failed: {error}"))?;
    outcome.updates_applied = applied;
    Ok(outcome)
}

/// The GM's rulebook. Kept strict about grounding and neutral narrator voice;
/// the empty-updates path is called out explicitly so trivial smalltalk does
/// not manufacture relationships.
const GM_SYSTEM_PROMPT: &str = "You are the invisible Gamemaster of a Fallout: New Vegas roleplay. You maintain a private ledger of how each character currently regards specific people (the player, or another character). You are given each character's existing ledger entries and the newest conversation excerpts, and you decide which entries genuinely need creating or revising.

Rules:
- Only write an entry when the new excerpts contain something that would actually shape how that character sees that person: a favor, a threat, a promise, a revelation, an insult, shared danger, real warmth or friction. Routine greetings, small talk, or logistics do NOT justify an entry. When nothing notable happened, return an empty updates list — that is a correct and common answer.
- Relationships move BOTH ways: deepen or warm one when events earn it, cool or sour one after slights or betrayals, and rewrite one whose text no longer matches events.
- An entry is the character's CURRENT overall stance toward that person: concrete, specific, and grounded ONLY in what the excerpts and existing notes show. One to three short sentences. Never invent events.
- Write in a neutral narrator voice about the character's stance (for example: \"Pete sees the Courier as careful with dangerous things and has started trusting her word.\"). Never first person, never addressed to anyone, no lists, no headers.
- When revising an existing entry, REWRITE it as a single fresh current stance blending what still holds with what changed. Do not append. Keep it one to three sentences no matter how long the history gets.
- Refer to the player by name.
- Only use direction pairs from the allowed list. Never write entries from the player's perspective.";

/// llama.cpp-enforced response shape: a bare `updates` array of
/// `{character, target, text}` upserts. Empty array = nothing notable.
fn updates_response_format() -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "chasm_gamemaster_updates",
            "description": "Relationship ledger upserts decided by the Gamemaster.",
            "strict": true,
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "updates": {
                        "type": "array",
                        "description": "Entries to create or rewrite. Empty when nothing notable happened.",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "character": { "type": "string", "description": "Name of the character whose view changes (exactly as in the allowed pairs list)." },
                                "target": { "type": "string", "description": "Name of the person they regard (exactly as in the allowed pairs list)." },
                                "text": { "type": "string", "description": "The character's current stance toward the target: 1-3 short sentences, neutral narrator voice, grounded in the excerpts." }
                            },
                            "required": ["character", "target", "text"]
                        }
                    }
                },
                "required": ["updates"]
            }
        }
    })
}

#[derive(Debug)]
struct GmUpdate {
    character: String,
    target: String,
    text: String,
}

fn parse_updates(content: &str) -> Result<Vec<GmUpdate>, String> {
    let value: Value = serde_json::from_str(content.trim())
        .map_err(|error| format!("GM reply was not valid JSON: {error}"))?;
    let Some(items) = value.get("updates").and_then(Value::as_array) else {
        return Err("GM reply had no updates array".to_string());
    };
    Ok(items
        .iter()
        .filter_map(|item| {
            Some(GmUpdate {
                character: item.get("character")?.as_str()?.trim().to_string(),
                target: item.get("target")?.as_str()?.trim().to_string(),
                text: item.get("text")?.as_str()?.trim().to_string(),
            })
        })
        .collect())
}

/// Every transcript session backing one live chat: the shared segment streams
/// plus the per-participant projection sessions (the live game path writes to
/// projections and may leave the group segment empty — see
/// `messages_for_participant`). Duplicated copies are collapsed by the caller.
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

/// The messages past `watermark`, plus the EFFECTIVE watermark used. A session
/// that shrank below its stored watermark (history cleared / rewritten) resets
/// to the full current content rather than erroring or skipping forever.
fn slice_since_watermark(
    messages: &[STJsonlChatMessage],
    watermark: u64,
) -> (&[STJsonlChatMessage], u64) {
    let len = messages.len() as u64;
    if watermark > len {
        (messages, 0)
    } else {
        (&messages[watermark as usize..], watermark)
    }
}

/// One raw speech line for the GM: speaker + text, no retrieval context, no
/// system lines, no empties.
fn normalize_line(message: &STJsonlChatMessage, session_id: &str) -> Option<NewLine> {
    if message.is_system {
        return None;
    }
    let text = message.mes.trim();
    if text.is_empty() {
        return None;
    }
    let headless = message.extra.get("headless");
    let character_id = headless
        .and_then(|h| h.get("characterId"))
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_string);
    let scene = headless
        .and_then(|h| h.get("metadata"))
        .and_then(|m| m.get("live"))
        .and_then(|l| l.get("segmentId"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(session_id)
        .to_string();
    Some(NewLine {
        scene,
        speaker_name: message.name.clone(),
        character_id,
        is_user: message.is_user,
        text: text.to_string(),
    })
}

/// Case/format-insensitive lookup key for character names/ids.
fn lookup_key(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

/// The directional pairs eligible this pass:
/// * `NPC → player` for every NPC who spoke in a scene where the player spoke,
/// * `NPC → NPC` for every ordered pair of DISTINCT NPCs who both spoke in the
///   same scene (a shared group segment).
/// The player never appears as a source. Sorted + deduped for stable prompts.
fn allowed_pairs(
    lines: &[NewLine],
    npc_names: &BTreeMap<String, (String, String)>,
    player_name: &str,
) -> Vec<AllowedPair> {
    // scene → (npc character ids who spoke, player spoke?)
    let mut scenes: BTreeMap<&str, (BTreeSet<&str>, bool)> = BTreeMap::new();
    for line in lines {
        let entry = scenes.entry(line.scene.as_str()).or_default();
        if line.is_user {
            entry.1 = true;
        } else if let Some(id) = line.character_id.as_deref() {
            entry.0.insert(id);
        }
    }
    let display = |id: &str| {
        npc_names
            .get(&lookup_key(id))
            .map(|(_, name)| name.clone())
            .unwrap_or_else(|| id.to_string())
    };
    let mut pairs: BTreeSet<AllowedPair> = BTreeSet::new();
    for (npcs, player_spoke) in scenes.values() {
        for &npc in npcs {
            if *player_spoke {
                pairs.insert(AllowedPair {
                    character_id: npc.to_string(),
                    character_name: display(npc),
                    target_id: PLAYER_TARGET_ID.to_string(),
                    target_name: player_name.to_string(),
                    target_kind: "player",
                });
            }
            for &other in npcs {
                if other != npc {
                    pairs.insert(AllowedPair {
                        character_id: npc.to_string(),
                        character_name: display(npc),
                        target_id: other.to_string(),
                        target_name: display(other),
                        target_kind: "npc",
                    });
                }
            }
        }
    }
    pairs.into_iter().collect()
}

/// Existing entries of the involved characters, rendered for the GM. `(none)`
/// keeps the section present so the model knows the ledger starts empty.
fn existing_entries_text(
    store: &chasm_st_compat::RelationshipsStore,
    involved: &BTreeSet<&str>,
) -> String {
    let mut lines = Vec::new();
    for character_id in involved {
        for (_, entry) in store.entries_for(character_id) {
            lines.push(format!(
                "{character_id} → {}: {}",
                entry.target_name, entry.text
            ));
        }
    }
    if lines.is_empty() {
        "(none yet)".to_string()
    } else {
        lines.join("\n")
    }
}

/// The raw new transcript, grouped by scene, player lines under their real
/// name. Plain `Name: text` lines — no retrieval context, no metadata.
fn format_transcript(lines: &[NewLine], player_name: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut current_scene: Option<&str> = None;
    for line in lines {
        if current_scene != Some(line.scene.as_str()) {
            current_scene = Some(line.scene.as_str());
            out.push(format!("[Scene: {}]", line.scene));
        }
        let name = if line.is_user {
            player_name
        } else {
            line.speaker_name.as_str()
        };
        out.push(format!("{name}: {}", line.text));
    }
    out.join("\n")
}

fn build_user_prompt(
    player_name: &str,
    existing: &str,
    transcript: &str,
    pairs: &[AllowedPair],
) -> String {
    let allowed = pairs
        .iter()
        .map(|pair| format!("{} → {}", pair.character_name, pair.target_name))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "The player is named {player_name}.\n\n\
         Existing ledger entries (the current state before this update):\n{existing}\n\n\
         New conversation excerpts since the last update:\n{transcript}\n\n\
         Allowed direction pairs you may create or revise this time:\n{allowed}\n\n\
         Return the entries that genuinely need creating or rewriting, or an empty updates list if nothing notable happened."
    )
}

/// Resolves the GM's `(character, target)` names back onto an allowed pair,
/// loosely (case/punctuation-insensitive, matching either display name or id).
fn match_pair<'a>(
    pairs: &'a [AllowedPair],
    character: &str,
    target: &str,
) -> Option<&'a AllowedPair> {
    let character = lookup_key(character);
    let target = lookup_key(target);
    pairs.iter().find(|pair| {
        (lookup_key(&pair.character_name) == character || lookup_key(&pair.character_id) == character)
            && (lookup_key(&pair.target_name) == target
                || lookup_key(&pair.target_id) == target)
    })
}

/// Bounds one entry's stored text (see [`MAX_ENTRY_CHARS`]).
fn bounded_text(text: &str) -> String {
    let text = text.trim();
    if text.chars().count() <= MAX_ENTRY_CHARS {
        return text.to_string();
    }
    let truncated: String = text.chars().take(MAX_ENTRY_CHARS).collect();
    format!("{}…", truncated.trim_end())
}

/// Advances the stored watermarks (and only them) after a pass with nothing
/// to apply.
fn advance_watermarks(
    repo: &chasm_st_compat::LiveChatRepository,
    scanned: &BTreeMap<String, u64>,
) -> Result<(), String> {
    repo.update_relationships(|store| {
        for (session_id, count) in scanned {
            store.watermarks.insert(session_id.clone(), *count);
        }
    })
    .map_err(|error| format!("relationships store write failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(name: &str, is_user: bool, text: &str, character_id: Option<&str>) -> STJsonlChatMessage {
        STJsonlChatMessage {
            name: name.to_string(),
            is_user,
            is_system: false,
            send_date: Some("2026-07-02T10:00:00Z".to_string()),
            mes: text.to_string(),
            extra: json!({
                "headless": {
                    "characterId": character_id,
                    "metadata": { "live": { "segmentId": "saloon" } }
                }
            }),
            original_avatar: None,
        }
    }

    /// Watermark slicing: 0 → everything; mid → the tail; equal → empty;
    /// beyond the file (session shrank) → reset to everything.
    #[test]
    fn watermark_slices_and_resets_on_shrink() {
        let messages: Vec<STJsonlChatMessage> = (0..5)
            .map(|i| message("Easy Pete", false, &format!("line {i}"), Some("Easy Pete")))
            .collect();

        let (fresh, mark) = slice_since_watermark(&messages, 0);
        assert_eq!((fresh.len(), mark), (5, 0));

        let (fresh, mark) = slice_since_watermark(&messages, 3);
        assert_eq!((fresh.len(), mark), (2, 3));
        assert_eq!(fresh[0].mes, "line 3");

        let (fresh, mark) = slice_since_watermark(&messages, 5);
        assert_eq!((fresh.len(), mark), (0, 5));

        // Session shrank below the stored watermark → full reset, not a skip.
        let (fresh, mark) = slice_since_watermark(&messages, 9);
        assert_eq!((fresh.len(), mark), (5, 0));
    }

    /// Pair derivation: NPC→player only where the player spoke in the scene;
    /// NPC↔NPC only between NPCs sharing a scene; the player is never a source.
    #[test]
    fn allowed_pairs_follow_scene_participation() {
        let names: BTreeMap<String, (String, String)> = [
            ("easypete", ("Easy Pete", "Easy Pete")),
            ("sunnysmiles", ("Sunny Smiles", "Sunny Smiles")),
            ("trudy", ("Trudy", "Trudy")),
        ]
        .into_iter()
        .map(|(k, (id, name))| (k.to_string(), (id.to_string(), name.to_string())))
        .collect();

        // Scene A: player + Pete + Sunny. Scene B: Trudy alone (no player).
        let mut lines = vec![
            NewLine { scene: "a".into(), speaker_name: "Player".into(), character_id: None, is_user: true, text: "hi".into() },
            NewLine { scene: "a".into(), speaker_name: "Easy Pete".into(), character_id: Some("Easy Pete".into()), is_user: false, text: "howdy".into() },
            NewLine { scene: "a".into(), speaker_name: "Sunny Smiles".into(), character_id: Some("Sunny Smiles".into()), is_user: false, text: "hey".into() },
            NewLine { scene: "b".into(), speaker_name: "Trudy".into(), character_id: Some("Trudy".into()), is_user: false, text: "hmph".into() },
        ];
        let pairs = allowed_pairs(&lines, &names, "Courier");
        let as_tuples: Vec<(String, String)> = pairs
            .iter()
            .map(|p| (p.character_id.clone(), p.target_id.clone()))
            .collect();
        assert!(as_tuples.contains(&("Easy Pete".into(), PLAYER_TARGET_ID.into())));
        assert!(as_tuples.contains(&("Sunny Smiles".into(), PLAYER_TARGET_ID.into())));
        assert!(as_tuples.contains(&("Easy Pete".into(), "Sunny Smiles".into())));
        assert!(as_tuples.contains(&("Sunny Smiles".into(), "Easy Pete".into())));
        // Trudy shared no scene with the player or another NPC → no pairs.
        assert!(!as_tuples.iter().any(|(c, _)| c == "Trudy"));
        // The player is never a SOURCE.
        assert!(!as_tuples.iter().any(|(c, _)| c == PLAYER_TARGET_ID));

        // With the player absent from scene A too, only NPC↔NPC remains.
        lines.remove(0);
        let pairs = allowed_pairs(&lines, &names, "Courier");
        assert!(pairs.iter().all(|p| p.target_id != PLAYER_TARGET_ID));
        assert!(pairs.iter().any(|p| p.target_kind == "npc"));
    }

    /// GM name output maps back loosely (case, punctuation) and rejects pairs
    /// not allowed this pass.
    #[test]
    fn gm_names_map_back_to_allowed_pairs() {
        let pairs = vec![AllowedPair {
            character_id: "Easy Pete".into(),
            character_name: "Easy Pete".into(),
            target_id: PLAYER_TARGET_ID.into(),
            target_name: "Courier".into(),
            target_kind: "player",
        }];
        assert!(match_pair(&pairs, "easy pete", "COURIER").is_some());
        assert!(match_pair(&pairs, "Easy Pete", "Courier").is_some());
        assert!(match_pair(&pairs, "Easy Pete", "Sunny Smiles").is_none());
        assert!(match_pair(&pairs, "Trudy", "Courier").is_none());
    }

    /// System lines and blanks are dropped; speaker/scene/player attribution
    /// survives normalization; the transcript renders player lines under the
    /// player's real name.
    #[test]
    fn transcript_uses_player_name_and_scene_headers() {
        let mut system = message("narrator", false, "system note", None);
        system.is_system = true;
        assert!(normalize_line(&system, "s").is_none());
        assert!(normalize_line(&message("Easy Pete", false, "   ", Some("Easy Pete")), "s").is_none());

        let lines = vec![
            normalize_line(&message("Player", true, "Howdy Pete.", None), "s").unwrap(),
            normalize_line(&message("Easy Pete", false, "Dynamite's dangerous.", Some("Easy Pete")), "s").unwrap(),
        ];
        let transcript = format_transcript(&lines, "Courier");
        assert_eq!(
            transcript,
            "[Scene: saloon]\nCourier: Howdy Pete.\nEasy Pete: Dynamite's dangerous."
        );
    }

    /// Structured-output parsing: valid updates come through; an empty list is
    /// a valid no-op; junk is an error (watermarks must not advance on junk).
    #[test]
    fn parse_updates_accepts_empty_and_rejects_junk() {
        let updates = parse_updates(
            r#"{"updates":[{"character":"Easy Pete","target":"Courier","text":"Pete warms to her."}]}"#,
        )
        .unwrap();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].character, "Easy Pete");

        assert!(parse_updates(r#"{"updates":[]}"#).unwrap().is_empty());
        assert!(parse_updates("total junk").is_err());
        assert!(parse_updates(r#"{"something":"else"}"#).is_err());
    }

    #[test]
    fn entry_text_is_bounded() {
        let long = "x".repeat(2000);
        let bounded = bounded_text(&long);
        assert!(bounded.chars().count() <= MAX_ENTRY_CHARS + 1);
        assert!(bounded.ends_with('…'));
        assert_eq!(bounded_text("  short  "), "short");
    }
}
