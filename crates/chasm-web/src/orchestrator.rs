//! Live-chat speaker orchestrator — picks who speaks next (and in what order) in
//! a Live Chat. No LLM is involved: per-turn speaker selection used to be a
//! second model call (~0.8-2s on multi-NPC turns, generating reasoning we threw
//! away), and is now an instant weighted score over signals the game already
//! gives us.
//!
//! The flow:
//!
//! 1. **Eligible** = participants that are `present && audible && kind=="npc"`
//!    (with a `characterId`). `compute_eligible` returns them in a stable order.
//! 2. **Gating** (in `generate.rs::orchestrate`):
//!    - 0 eligible → no turns.
//!    - forced speaker (`forceParticipantId`/`forceCharacterId`) → that one.
//!    - orchestrator disabled, or exactly 1 eligible → that NPC, no scoring.
//!    - 2+ eligible → [`select_weighted_speakers`] scores and orders them.
//! 3. **The score** ([`select_weighted_speakers`]): a weighted sum of crosshair
//!    (is the player looking right at them), name mention, recency (who's being
//!    talked with), proximity, lexical topic overlap, and a little jitter so it
//!    never feels robotic. The top NPC speaks; close-behind NPCs can join, in
//!    score order. All the weights are constants at the top of that section.

use serde_json::Value;
use chasm_core::{
    ORCHESTRATOR_DEFAULT_SYSTEM_PROMPT, ORCHESTRATOR_MAX_SPEAKERS_MAX,
    ORCHESTRATOR_MAX_SPEAKERS_MIN, ORCHESTRATOR_TEMPERATURE_MAX, ORCHESTRATOR_TEMPERATURE_MIN,
};
use chasm_st_compat::{LiveChat, STJsonlChatMessage};

// --- Constants --------------------------------------------------------------

/// Recent-message window scored for the recency/topic signals.
const LIVE_SPEAKER_SELECTOR_CONTEXT_LIMIT: usize = 14;

// --- Settings ----------------------------------------------------------------

/// The (already-resolved, already-clamped) orchestrator knobs, sourced from the
/// GLOBAL `LlmSettings` at request time and passed into the orchestrator. We no
/// longer read any of these from the per-chat `live_chat.settings` Value.
#[derive(Debug, Clone)]
pub struct OrchestratorSettings {
    /// When false, skip the model call entirely and use the first eligible NPC.
    pub enabled: bool,
    /// Maximum speakers picked for one turn (clamped 1..=10).
    pub max_speakers: usize,
    /// Retained for config compatibility (and a possible future LLM selector);
    /// unused by the weighted scorer. Kept so the persisted orchestrator settings
    /// still round-trip without a schema change.
    #[allow(dead_code)]
    pub temperature: f64,
    /// Retained for config compatibility — see `temperature`.
    #[allow(dead_code)]
    pub system_prompt: String,
}

impl OrchestratorSettings {
    /// Builds the runtime settings from raw (persisted) values, applying the
    /// documented clamps and falling back to the default prompt when blank.
    pub fn new(enabled: bool, max_speakers: u32, temperature: f32, system_prompt: &str) -> Self {
        let max_speakers = max_speakers
            .clamp(ORCHESTRATOR_MAX_SPEAKERS_MIN, ORCHESTRATOR_MAX_SPEAKERS_MAX)
            as usize;
        let temperature = if temperature.is_finite() {
            temperature.clamp(ORCHESTRATOR_TEMPERATURE_MIN, ORCHESTRATOR_TEMPERATURE_MAX)
        } else {
            ORCHESTRATOR_TEMPERATURE_MIN
        } as f64;
        let prompt = system_prompt.trim();
        let system_prompt = if prompt.is_empty() {
            ORCHESTRATOR_DEFAULT_SYSTEM_PROMPT.to_string()
        } else {
            prompt.to_string()
        };
        Self {
            enabled,
            max_speakers,
            temperature,
            system_prompt,
        }
    }
}

// --- Eligible participant ----------------------------------------------------

/// An eligible NPC speaker candidate.
#[derive(Debug, Clone)]
pub struct EligibleParticipant {
    pub participant_id: String,
    pub character_id: String,
    pub name: String,
    /// Carried through for the turn pipeline's participant-view fallback.
    pub distance: Option<f64>,
}

/// A selected speaker plus the bookkeeping fields the turn pipeline attaches.
#[derive(Debug, Clone)]
pub struct SelectedSpeaker {
    pub participant: EligibleParticipant,
    pub reason: String,
    pub model_reason: Option<String>,
    pub confidence: Option<f64>,
    pub queue_index: usize,
}

/// The result of fallback (first-NPC / forced) or model selection.
#[derive(Debug, Clone)]
pub struct SpeakerSelection {
    pub speakers: Vec<SelectedSpeaker>,
    pub eligible: Vec<EligibleParticipant>,
    pub reason: String,
}

/// Input to the orchestrator for one generate call. The only request-level
/// knobs that survive are the forced-speaker ids (a direct API feature).
pub struct SelectionInput {
    pub force_participant_id: Option<String>,
    pub force_character_id: Option<String>,
}

// --- Eligibility -------------------------------------------------------------

/// Computes eligible NPC speakers from the live chat presence map.
///
/// Eligible = `present && audible && kind=="npc" && characterId`. Order follows
/// the presence map (participantId ascending, BTreeMap key order), which is a
/// stable, deterministic ordering.
pub fn compute_eligible(live_chat: &LiveChat) -> Vec<EligibleParticipant> {
    live_chat
        .presence
        .values()
        .filter(|participant| {
            participant.present.unwrap_or(false)
                && participant.audible.unwrap_or(false)
                && participant.kind == "npc"
                && participant
                    .character_id
                    .as_deref()
                    .map(|id| !id.is_empty())
                    .unwrap_or(false)
        })
        .map(|participant| {
            let character_id = participant.character_id.clone().unwrap_or_default();
            let name = if participant.name.is_empty() {
                character_id.clone()
            } else {
                participant.name.clone()
            };
            EligibleParticipant {
                participant_id: participant.participant_id.clone(),
                character_id,
                name,
                distance: participant.distance,
            }
        })
        .collect()
}

// --- Fallback / forced selection --------------------------------------------

/// Deterministic, no-LLM selection. Returns:
/// - a forced speaker when `forceParticipantId`/`forceCharacterId` matches an
///   eligible NPC,
/// - otherwise the FIRST eligible NPC (the cheap path used when the orchestrator
///   is disabled, when there is exactly one eligible NPC, or as the fallback for
///   any model-selection failure).
///
/// Returns `Err(message)` when there are no eligible NPCs, or when a forced
/// speaker was requested but is not an eligible NPC.
pub fn select_live_speaker_candidates(
    live_chat: &LiveChat,
    input: &SelectionInput,
) -> Result<SpeakerSelection, String> {
    let eligible = compute_eligible(live_chat);
    if eligible.is_empty() {
        return Err("Live Chat has no active audible NPC participants.".to_string());
    }

    // Forced short-circuit (forceParticipantId / forceCharacterId).
    let force_participant_id = input.force_participant_id.clone().unwrap_or_default();
    let force_character_id = input.force_character_id.clone().unwrap_or_default();
    if !force_participant_id.is_empty() || !force_character_id.is_empty() {
        let forced = eligible.iter().find(|participant| {
            participant.participant_id == force_participant_id
                || (!force_character_id.is_empty()
                    && participant.character_id == force_character_id)
        });
        return match forced {
            Some(participant) => Ok(single_speaker(participant.clone(), "forced", eligible)),
            None => Err("Forced speaker is not an active audible NPC participant.".to_string()),
        };
    }

    // Default cheap path: the first eligible NPC.
    let first = eligible[0].clone();
    Ok(single_speaker(first, "first_eligible", eligible))
}

/// Builds a one-speaker selection with the given reason.
fn single_speaker(
    participant: EligibleParticipant,
    reason: &str,
    eligible: Vec<EligibleParticipant>,
) -> SpeakerSelection {
    SpeakerSelection {
        speakers: vec![SelectedSpeaker {
            participant,
            reason: reason.to_string(),
            model_reason: None,
            confidence: None,
            queue_index: 0,
        }],
        eligible,
        reason: reason.to_string(),
    }
}

/// Gate: use the model selector only when enabled in settings, not forced, and
/// there is more than one eligible NPC. With 0/1 eligible NPCs (or when forced /
/// disabled) the deterministic single-speaker path is used instead.
pub fn should_use_model_speaker_selection(
    selection: &SpeakerSelection,
    settings: &OrchestratorSettings,
    forced: bool,
) -> bool {
    settings.enabled && !forced && selection.eligible.len() > 1
}

fn clamp_max_speakers(value: usize) -> usize {
    // mirror Math.max(0, Math.min(10, trunc(value)||3)).
    let effective = if value == 0 { 3 } else { value };
    effective.min(10)
}

/// The recent-message context limit used to assemble the selector transcript.
pub fn selector_context_limit() -> usize {
    LIVE_SPEAKER_SELECTOR_CONTEXT_LIMIT
}

// --- Weighted speaker scoring (replaces the per-turn LLM "director") ---------
//
// Picking who speaks next in a room is dominated by cheap, immediate signals the
// game already hands us, so we score each eligible NPC from those and take the
// top — no model call, effectively free. (The old LLM director spent ~0.8-2s per
// multi-NPC turn generating reasoning we threw away.)
//
//   * crosshair  — the player is looking straight at this NPC (strongest cue)
//   * name       — the player said this NPC's name
//   * recency    — this NPC is the one currently being talked with (continuity)
//   * proximity  — this NPC is physically close to the player
//   * topic      — the player's words overlap what this NPC's been talking about
//   * jitter     — a little organic randomness so identical moments don't always
//                  resolve the same robotic way
//
// A second/third NPC only joins when clearly engaged too (close to the top score
// AND above a floor), so most turns are one natural responder while a genuinely
// shared moment can still get a couple of voices, in order.

/// Feature weights. Each feature is computed in `0.0..=1.0` and multiplied by its
/// weight; the highest sum speaks first. Tuned so a direct cue (crosshair / name)
/// reliably wins, with proximity + recency + topic shaping ties and the ambient
/// case. All the knobs live here so the behaviour is easy to feel out and adjust.
const W_CROSSHAIR: f64 = 1.00;
const W_NAME: f64 = 0.90;
const W_RECENCY: f64 = 0.60;
const W_PROXIMITY: f64 = 0.45;
const W_TOPIC: f64 = 0.40;
const W_JITTER: f64 = 0.15;

/// A co-speaker (2nd/3rd NPC) joins when its score is within this ratio of the top
/// score AND clears the absolute floor. Tuned for lively group chatter — a second
/// NPC chimes in fairly readily; only clearly-disengaged bystanders stay quiet.
/// (An explicit group address like "you two" / "everyone" bypasses both gates —
/// see `is_group_address`.)
const CO_SPEAKER_RATIO: f64 = 0.50;
const CO_SPEAKER_FLOOR: f64 = 0.35;

/// Multi-word phrases that address the whole group; their presence makes every
/// eligible NPC (up to the speaker cap) answer this turn.
const GROUP_ADDRESS_PHRASES: &[&str] = &[
    "you two",
    "you three",
    "you both",
    "both of you",
    "the two of you",
    "all of you",
    "you all",
    "you guys",
    "you lot",
    "you folks",
    "you people",
];

/// Single words that, as a whole word, read as addressing the group.
const GROUP_ADDRESS_WORDS: &[&str] = &["everyone", "everybody", "yall", "guys", "folks"];

/// Distance (metres) at/above which proximity contributes ~nothing.
const PROXIMITY_FALLOFF_M: f64 = 12.0;
/// Proximity score when no distance was reported — neutral, not a penalty.
const PROXIMITY_UNKNOWN: f64 = 0.30;

/// Minimum length for a topic-matching token.
const TOPIC_MIN_TOKEN_LEN: usize = 4;

/// Scores each eligible NPC from game + conversation signals and returns who
/// speaks next (and in what order). Always returns at least the top speaker and
/// never makes a model call. `player_message` is the latest player utterance.
pub fn select_weighted_speakers(
    live_chat: &LiveChat,
    eligible: &[EligibleParticipant],
    recent_messages: &[STJsonlChatMessage],
    player_message: &str,
    max_speakers: usize,
) -> SpeakerSelection {
    let message_lc = player_message.to_lowercase();
    let query_tokens = topic_tokens(&message_lc);
    let recency = recency_scores(eligible, recent_messages);

    let mut scored: Vec<(usize, f64, String)> = eligible
        .iter()
        .enumerate()
        .map(|(index, npc)| {
            let metadata = live_chat
                .presence
                .get(&npc.participant_id)
                .map(|participant| &participant.metadata);
            let f_cross = crosshair_feature(metadata);
            let f_name = name_feature(&message_lc, &npc.name);
            let f_recent = recency.get(&npc.participant_id).copied().unwrap_or(0.0);
            let f_prox = proximity_feature(npc.distance);
            let f_topic = topic_feature(&query_tokens, npc, recent_messages);
            let f_jitter = rand::random::<f64>();
            let total = W_CROSSHAIR * f_cross
                + W_NAME * f_name
                + W_RECENCY * f_recent
                + W_PROXIMITY * f_prox
                + W_TOPIC * f_topic
                + W_JITTER * f_jitter;
            let reason = format!(
                "crosshair={f_cross:.2} name={f_name:.2} recency={f_recent:.2} proximity={f_prox:.2} topic={f_topic:.2}"
            );
            (index, total, reason)
        })
        .collect();

    // Highest score first; ties keep the original stable order.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let group_address = is_group_address(&message_lc);
    let cap = eligible.len().min(clamp_max_speakers(max_speakers)).max(1);
    let top_score = scored.first().map(|entry| entry.1).unwrap_or(0.0);

    let mut speakers: Vec<SelectedSpeaker> = Vec::new();
    for (rank, (index, score, reason)) in scored.into_iter().enumerate() {
        if rank > 0 {
            if speakers.len() >= cap {
                break;
            }
            // A co-speaker normally must be clearly engaged (near the top score AND
            // above the floor) so idle bystanders stay quiet. But when the player
            // addresses the group, fill up to the cap regardless so everyone answers.
            if !group_address && (score < CO_SPEAKER_FLOOR || score < top_score * CO_SPEAKER_RATIO)
            {
                break;
            }
        }
        let confidence = (top_score > 0.0).then(|| (score / top_score).clamp(0.0, 1.0));
        let queue_index = speakers.len();
        speakers.push(SelectedSpeaker {
            participant: eligible[index].clone(),
            reason: if group_address && rank > 0 {
                "group_address".to_string()
            } else {
                "weighted_score".to_string()
            },
            model_reason: Some(reason),
            confidence,
            queue_index,
        });
    }

    SpeakerSelection {
        speakers,
        eligible: eligible.to_vec(),
        reason: "weighted_score".to_string(),
    }
}

/// 1.0 when the player is looking straight at this NPC (the helper marks the
/// crosshair target in each presence participant's metadata), else 0.0.
fn crosshair_feature(metadata: Option<&Value>) -> f64 {
    let Some(metadata) = metadata else {
        return 0.0;
    };
    let flagged = metadata
        .get("underCrosshair")
        .or_else(|| metadata.get("under_crosshair"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if flagged {
        1.0
    } else {
        0.0
    }
}

/// 1.0 when any meaningful token of the NPC's name appears as a whole word in the
/// player's (already lowercased) line, else 0.0.
fn name_feature(message_lc: &str, name: &str) -> f64 {
    let hit = name
        .split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|token| token.chars().count() >= 3)
        .any(|token| contains_whole_word(message_lc, &token));
    if hit {
        1.0
    } else {
        0.0
    }
}

/// True when the player's (already lowercased) line addresses the group as a whole
/// ("hey you two", "what's up everyone", "you guys", ...). Such a turn fills up to
/// the speaker cap so the whole group answers, instead of just the top-scored NPC.
fn is_group_address(message_lc: &str) -> bool {
    if message_lc.contains("y'all") || message_lc.contains("y’all") {
        return true;
    }
    if GROUP_ADDRESS_PHRASES
        .iter()
        .any(|phrase| message_lc.contains(phrase))
    {
        return true;
    }
    GROUP_ADDRESS_WORDS
        .iter()
        .any(|word| contains_whole_word(message_lc, word))
}

/// Closer NPCs score higher; unknown distance is neutral. Linear falloff to ~0
/// at `PROXIMITY_FALLOFF_M` metres.
fn proximity_feature(distance: Option<f64>) -> f64 {
    match distance {
        Some(meters) if meters.is_finite() && meters >= 0.0 => {
            (1.0 - meters / PROXIMITY_FALLOFF_M).clamp(0.0, 1.0)
        }
        _ => PROXIMITY_UNKNOWN,
    }
}

/// Per-NPC continuity score: the NPC who spoke most recently scores 1.0, halving
/// for each older NPC line in the window (player lines don't count toward the
/// decay, so an answer right after the player still reads as "current").
fn recency_scores(
    eligible: &[EligibleParticipant],
    recent_messages: &[STJsonlChatMessage],
) -> std::collections::HashMap<String, f64> {
    let mut scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut npc_turns_back = 0i32;
    for message in recent_messages.iter().rev() {
        if message.is_user {
            continue;
        }
        let Some(pid) = message_speaker_participant_id(message) else {
            continue;
        };
        if !eligible.iter().any(|npc| npc.participant_id == pid) {
            continue;
        }
        let score = 0.5f64.powi(npc_turns_back);
        scores
            .entry(pid)
            .and_modify(|best| {
                if score > *best {
                    *best = score;
                }
            })
            .or_insert(score);
        npc_turns_back += 1;
    }
    scores
}

/// Lexical topic overlap: the share of the player's content tokens that also
/// appear in this NPC's recent spoken lines (plus their name). A cheap, instant
/// stand-in for embeddings — when retrieval is enabled this is the natural place
/// to swap in a semantic similarity score.
fn topic_feature(
    query_tokens: &std::collections::HashSet<String>,
    npc: &EligibleParticipant,
    recent_messages: &[STJsonlChatMessage],
) -> f64 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let mut corpus = npc.name.to_lowercase();
    for message in recent_messages {
        if message.is_user {
            continue;
        }
        if message_speaker_participant_id(message).as_deref() == Some(npc.participant_id.as_str()) {
            corpus.push(' ');
            corpus.push_str(&message.mes.to_lowercase());
        }
    }
    let corpus_tokens = topic_tokens(&corpus);
    if corpus_tokens.is_empty() {
        return 0.0;
    }
    let overlap = query_tokens
        .iter()
        .filter(|token| corpus_tokens.contains(*token))
        .count();
    (overlap as f64 / query_tokens.len() as f64).clamp(0.0, 1.0)
}

/// Content tokens for topic matching: lowercased alphanumeric runs of at least
/// `TOPIC_MIN_TOKEN_LEN` chars, minus a small stopword set. Input is expected to
/// already be lowercased.
fn topic_tokens(text: &str) -> std::collections::HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| token.chars().count() >= TOPIC_MIN_TOKEN_LEN)
        .filter(|token| !is_topic_stopword(token))
        .map(str::to_string)
        .collect()
}

/// Common 4+ letter words that carry no topical signal.
fn is_topic_stopword(token: &str) -> bool {
    matches!(
        token,
        "this"
            | "that"
            | "with"
            | "have"
            | "your"
            | "what"
            | "there"
            | "here"
            | "they"
            | "them"
            | "then"
            | "were"
            | "will"
            | "would"
            | "about"
            | "just"
            | "like"
            | "know"
            | "dont"
            | "cant"
            | "youre"
            | "going"
            | "really"
            | "think"
            | "want"
            | "could"
            | "should"
            | "been"
            | "from"
            | "into"
            | "over"
            | "very"
            | "much"
            | "when"
            | "where"
            | "which"
            | "because"
            | "their"
            | "whats"
            | "yeah"
            | "okay"
            | "well"
    )
}

/// Whole-word containment: true when `word` appears in `haystack` not flanked by
/// alphanumeric characters. Both are expected to be lowercased.
fn contains_whole_word(haystack: &str, word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    let mut search_from = 0;
    while let Some(offset) = haystack[search_from..].find(word) {
        let start = search_from + offset;
        let end = start + word.len();
        let before_ok = haystack[..start]
            .chars()
            .next_back()
            .map(|c| !c.is_alphanumeric())
            .unwrap_or(true);
        let after_ok = haystack[end..]
            .chars()
            .next()
            .map(|c| !c.is_alphanumeric())
            .unwrap_or(true);
        if before_ok && after_ok {
            return true;
        }
        search_from = end;
        if search_from >= haystack.len() {
            break;
        }
    }
    false
}

/// The live speaker participantId recorded on a message, when present (the
/// headless write tags each NPC line with who said it). `None` for player lines.
fn message_speaker_participant_id(message: &STJsonlChatMessage) -> Option<String> {
    message
        .extra
        .get("headless")
        .and_then(|headless| headless.get("metadata"))
        .and_then(|metadata| metadata.get("live"))
        .and_then(|live| live.get("speakerParticipantId"))
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use chasm_st_compat::LiveChatParticipant;
    use std::collections::HashSet;

    fn live_chat_with(participants: &[(&str, &str, bool, bool)]) -> LiveChat {
        let mut chat = LiveChat::default();
        for (id, name, present, audible) in participants {
            let participant = LiveChatParticipant {
                participant_id: id.to_string(),
                kind: "npc".to_string(),
                character_id: Some(format!("char-{name}")),
                name: name.to_string(),
                present: Some(*present),
                audible: Some(*audible),
                distance: None,
                metadata: Value::Null,
                updated_at: None,
                segment_id: None,
            };
            chat.presence.insert(id.to_string(), participant);
        }
        chat
    }

    fn default_settings() -> OrchestratorSettings {
        OrchestratorSettings::new(true, 3, 0.2, ORCHESTRATOR_DEFAULT_SYSTEM_PROMPT)
    }

    fn no_force() -> SelectionInput {
        SelectionInput {
            force_participant_id: None,
            force_character_id: None,
        }
    }

    // --- settings tests ---

    #[test]
    fn settings_apply_defaults_and_clamps() {
        let settings = default_settings();
        assert!(settings.enabled);
        assert_eq!(settings.max_speakers, 3);
        // f32 -> f64 widening, so compare with a tolerance.
        assert!((settings.temperature - 0.2).abs() < 1e-6);
        assert_eq!(settings.system_prompt, ORCHESTRATOR_DEFAULT_SYSTEM_PROMPT);

        // Clamps: max_speakers 0 -> 1, 99 -> 10; temperature 5.0 -> 2.0.
        let clamped = OrchestratorSettings::new(false, 99, 5.0, "  ");
        assert!(!clamped.enabled);
        assert_eq!(clamped.max_speakers, 10);
        assert_eq!(clamped.temperature, 2.0);
        // Blank prompt resets to the default.
        assert_eq!(clamped.system_prompt, ORCHESTRATOR_DEFAULT_SYSTEM_PROMPT);

        let low = OrchestratorSettings::new(true, 0, -1.0, "Custom prompt");
        assert_eq!(low.max_speakers, 1);
        assert_eq!(low.temperature, 0.0);
        assert_eq!(low.system_prompt, "Custom prompt");
    }

    // --- selection / gating tests ---

    #[test]
    fn forced_short_circuits() {
        let chat = live_chat_with(&[
            ("npc:a", "Alpha", true, true),
            ("npc:b", "Bravo", true, true),
        ]);
        let input = SelectionInput {
            force_participant_id: Some("npc:b".to_string()),
            force_character_id: None,
        };
        let result = select_live_speaker_candidates(&chat, &input).expect("ok");
        assert_eq!(result.reason, "forced");
        assert_eq!(result.speakers.len(), 1);
        assert_eq!(result.speakers[0].participant.participant_id, "npc:b");
        assert_eq!(result.speakers[0].reason, "forced");
    }

    #[test]
    fn forced_unknown_is_error() {
        let chat = live_chat_with(&[("npc:a", "Alpha", true, true)]);
        let input = SelectionInput {
            force_participant_id: Some("npc:ghost".to_string()),
            force_character_id: None,
        };
        assert!(select_live_speaker_candidates(&chat, &input).is_err());
    }

    #[test]
    fn no_eligible_is_error() {
        let chat = live_chat_with(&[("npc:a", "Alpha", false, true)]);
        assert!(select_live_speaker_candidates(&chat, &no_force()).is_err());
    }

    #[test]
    fn fallback_picks_first_eligible() {
        let chat = live_chat_with(&[
            ("npc:a", "Alpha", true, true),
            ("npc:b", "Bravo", true, true),
        ]);
        let result = select_live_speaker_candidates(&chat, &no_force()).expect("ok");
        // Presence map = participantId ascending, so npc:a is first.
        assert_eq!(result.reason, "first_eligible");
        assert_eq!(result.speakers.len(), 1);
        assert_eq!(result.speakers[0].participant.participant_id, "npc:a");
    }

    #[test]
    fn disabled_orchestrator_does_not_use_model() {
        let chat = live_chat_with(&[
            ("npc:a", "Alpha", true, true),
            ("npc:b", "Bravo", true, true),
        ]);
        let settings = OrchestratorSettings::new(false, 3, 0.2, ORCHESTRATOR_DEFAULT_SYSTEM_PROMPT);
        let result = select_live_speaker_candidates(&chat, &no_force()).expect("ok");
        assert_eq!(result.eligible.len(), 2);
        assert!(!should_use_model_speaker_selection(
            &result, &settings, false
        ));
    }

    #[test]
    fn single_eligible_does_not_use_model() {
        let chat = live_chat_with(&[
            ("npc:a", "Alpha", true, true),
            ("npc:b", "Bravo", false, true),
        ]);
        let settings = default_settings();
        let result = select_live_speaker_candidates(&chat, &no_force()).expect("ok");
        assert_eq!(result.eligible.len(), 1);
        assert!(!should_use_model_speaker_selection(
            &result, &settings, false
        ));
    }

    #[test]
    fn multi_eligible_uses_model_when_enabled() {
        let chat = live_chat_with(&[
            ("npc:a", "Alpha", true, true),
            ("npc:b", "Bravo", true, true),
        ]);
        let settings = default_settings();
        let result = select_live_speaker_candidates(&chat, &no_force()).expect("ok");
        assert!(should_use_model_speaker_selection(
            &result, &settings, false
        ));
        // Forced disables the model path even with 2+ eligible.
        assert!(!should_use_model_speaker_selection(
            &result, &settings, true
        ));
    }

    // --- weighted-scorer helpers (deterministic, no jitter) ---

    #[test]
    fn crosshair_feature_reads_metadata() {
        assert_eq!(
            crosshair_feature(Some(&json!({ "underCrosshair": true }))),
            1.0
        );
        assert_eq!(
            crosshair_feature(Some(&json!({ "underCrosshair": false }))),
            0.0
        );
        assert_eq!(crosshair_feature(Some(&json!({ "name": "x" }))), 0.0);
        assert_eq!(crosshair_feature(None), 0.0);
    }

    #[test]
    fn name_feature_matches_whole_words_only() {
        assert_eq!(name_feature("hey sunny, got a sec?", "Sunny Smiles"), 1.0);
        assert_eq!(name_feature("nice sunshine today", "Sunny"), 0.0);
        assert_eq!(name_feature("where is the doctor", "Doc Mitchell"), 0.0);
        assert_eq!(name_feature("any work, chet?", "Chet"), 1.0);
    }

    #[test]
    fn proximity_feature_falls_off_with_distance() {
        assert!(proximity_feature(Some(0.0)) > 0.99);
        assert!((proximity_feature(Some(6.0)) - 0.5).abs() < 0.01);
        assert_eq!(proximity_feature(Some(12.0)), 0.0);
        assert_eq!(proximity_feature(Some(50.0)), 0.0);
        assert_eq!(proximity_feature(None), PROXIMITY_UNKNOWN);
    }

    #[test]
    fn topic_tokens_drop_short_words_and_stopwords() {
        let tokens = topic_tokens("do you have any rifle ammunition");
        assert!(tokens.contains("rifle"));
        assert!(tokens.contains("ammunition"));
        assert!(!tokens.contains("you")); // too short
        assert!(!tokens.contains("have")); // stopword
    }

    #[test]
    fn topic_feature_scores_overlap_with_npc_lines() {
        let query = topic_tokens("do you sell weapons");
        let npc = scored("npc:b", "Bravo", None);
        let recent = vec![npc_line("npc:b", "I sell rifles and weapons")];
        assert!(topic_feature(&query, &npc, &recent) > 0.0);
        // An NPC who said nothing relevant scores nothing.
        let other = scored("npc:a", "Alpha", None);
        assert_eq!(topic_feature(&query, &other, &recent), 0.0);
    }

    // --- weighted-scorer selection (dominant signals beat the 0.15 jitter) ---

    #[test]
    fn crosshair_target_is_chosen_alone() {
        let chat = chat_with_crosshair(&[("npc:a", false), ("npc:b", true)]);
        let pool = vec![
            scored("npc:a", "Alpha", None),
            scored("npc:b", "Bravo", None),
        ];
        let result = select_weighted_speakers(&chat, &pool, &[], "", 3);
        assert_eq!(result.reason, "weighted_score");
        assert_eq!(result.speakers.len(), 1);
        assert_eq!(result.speakers[0].participant.participant_id, "npc:b");
    }

    #[test]
    fn named_npc_is_chosen() {
        let chat = chat_with_crosshair(&[("npc:a", false), ("npc:b", false)]);
        let pool = vec![
            scored("npc:a", "Alpha", None),
            scored("npc:b", "Bravo", None),
        ];
        let result = select_weighted_speakers(&chat, &pool, &[], "hey bravo", 3);
        assert_eq!(result.speakers[0].participant.participant_id, "npc:b");
    }

    #[test]
    fn most_recent_npc_keeps_the_floor() {
        let chat = chat_with_crosshair(&[("npc:a", false), ("npc:b", false)]);
        let pool = vec![
            scored("npc:a", "Alpha", None),
            scored("npc:b", "Bravo", None),
        ];
        let recent = vec![npc_line("npc:b", "good to see you")];
        let result = select_weighted_speakers(&chat, &pool, &recent, "", 3);
        assert_eq!(result.speakers[0].participant.participant_id, "npc:b");
    }

    #[test]
    fn two_engaged_npcs_both_speak_in_order() {
        // a is under the crosshair, b is named — both clear the co-speaker bar.
        let chat = chat_with_crosshair(&[("npc:a", true), ("npc:b", false)]);
        let pool = vec![
            scored("npc:a", "Alpha", None),
            scored("npc:b", "Bravo", None),
        ];
        let result = select_weighted_speakers(&chat, &pool, &[], "over here bravo", 3);
        assert_eq!(result.speakers.len(), 2);
        let ids: HashSet<&str> = result
            .speakers
            .iter()
            .map(|speaker| speaker.participant.participant_id.as_str())
            .collect();
        assert!(ids.contains("npc:a") && ids.contains("npc:b"));
        // queue_index is contiguous from 0 in score order.
        assert_eq!(result.speakers[0].queue_index, 0);
        assert_eq!(result.speakers[1].queue_index, 1);
    }

    #[test]
    fn group_address_pulls_in_the_bystander() {
        // Player looks at a and addresses the group; b has no individual cue and
        // would normally fall below the co-speaker bar, but "you two" pulls it in.
        let chat = chat_with_crosshair(&[("npc:a", true), ("npc:b", false)]);
        let pool = vec![
            scored("npc:a", "Alpha", None),
            scored("npc:b", "Bravo", None),
        ];
        let result = select_weighted_speakers(&chat, &pool, &[], "hey whats up you two?", 3);
        assert_eq!(result.speakers.len(), 2);
        let ids: HashSet<&str> = result
            .speakers
            .iter()
            .map(|speaker| speaker.participant.participant_id.as_str())
            .collect();
        assert!(ids.contains("npc:a") && ids.contains("npc:b"));
        assert!(result
            .speakers
            .iter()
            .any(|speaker| speaker.reason == "group_address"));
    }

    #[test]
    fn group_address_still_respects_the_cap() {
        let chat = chat_with_crosshair(&[("npc:a", true), ("npc:b", false), ("npc:c", false)]);
        let pool = vec![
            scored("npc:a", "Alpha", None),
            scored("npc:b", "Bravo", None),
            scored("npc:c", "Cara", None),
        ];
        // "everyone" addresses the group, but max_speakers=2 bounds the turn.
        let result = select_weighted_speakers(&chat, &pool, &[], "everyone listen up", 2);
        assert_eq!(result.speakers.len(), 2);
    }

    #[test]
    fn lone_bystander_stays_quiet_without_group_address() {
        // No group cue + no individual cue for b → b stays below the co-speaker bar.
        let chat = chat_with_crosshair(&[("npc:a", true), ("npc:b", false)]);
        let pool = vec![
            scored("npc:a", "Alpha", None),
            scored("npc:b", "Bravo", None),
        ];
        let result = select_weighted_speakers(&chat, &pool, &[], "what's going on?", 3);
        assert_eq!(result.speakers.len(), 1);
        assert_eq!(result.speakers[0].participant.participant_id, "npc:a");
    }

    fn scored(id: &str, name: &str, distance: Option<f64>) -> EligibleParticipant {
        EligibleParticipant {
            participant_id: id.to_string(),
            character_id: format!("char-{name}"),
            name: name.to_string(),
            distance,
        }
    }

    fn chat_with_crosshair(targets: &[(&str, bool)]) -> LiveChat {
        let mut chat = LiveChat::default();
        for (id, under_crosshair) in targets {
            let participant = LiveChatParticipant {
                participant_id: id.to_string(),
                kind: "npc".to_string(),
                present: Some(true),
                audible: Some(true),
                metadata: json!({ "underCrosshair": under_crosshair }),
                ..Default::default()
            };
            chat.presence.insert(id.to_string(), participant);
        }
        chat
    }

    fn npc_line(participant_id: &str, text: &str) -> STJsonlChatMessage {
        STJsonlChatMessage {
            is_user: false,
            mes: text.to_string(),
            extra: json!({
                "headless": { "metadata": { "live": { "speakerParticipantId": participant_id } } }
            }),
            ..Default::default()
        }
    }
}
