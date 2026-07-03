//! Rust port of the headless prompt assembler.
//!
//! Mirrors the prompt that `src/headless/generation.js` `prepareGenerationRun`
//! sends to the model: an ordered `system` block, the chat history, then the
//! pending player turn. This crate produces a [`PromptAssemblyView`] broken into
//! ordered, labeled components so the UI can show every piece in send order.
//!
//! Parity notes: the deterministic pieces (character card, constant + keyword
//! activation, the action/quest/structured formatters and constants, history)
//! are reproduced faithfully. The runtime-only pieces — vector activation and
//! chat-vector retrieval (need the embeddings backend), and per-request inputs
//! like the global scenario (resolves gamestate macros per turn) / world-state
//! scopes / TTS audio tags — are shown as correctly-ordered placeholders
//! rather than guessed at.
//!
//! Scenario note: since the Globals rework, the `Scenario` component comes
//! from the GLOBAL scenario template (Globals page, `{{macro}}` placeholders
//! resolved per turn — see [`scenario`]), NOT from the per-character card
//! `scenario` field. The card field is still parsed and stored for imported
//! SillyTavern cards, but it never reaches the prompt.

use std::cmp::Ordering;

mod lore_injection;
pub use lore_injection::*;

pub mod macros;
pub use macros::{apply_macros, macros_from_metadata, macros_from_value};

pub mod scenario;
pub use scenario::{participants_macro, DEFAULT_SCENARIO_TEMPLATE};

use regex::RegexBuilder;
use serde_json::Value;
use chasm_core::{
    ActionView, ActivatedActionView, CatalogItemView, InjectedEntryView, InjectedView, MessageView,
    ParticipantView, PromptAssemblyView, PromptComponentView, ScopedCatalogView,
};
use chasm_embed::{EmbeddingCache, Retriever};
use chasm_st_compat::{ActionEntry, CatalogItem, LiveChatRepository, LoreEntry, QuestEntry};

mod action_book_injection;
pub use action_book_injection::*;

/// The single instruction that tells the model the structured-reply shape AND how
/// to use actions. Deliberately ONE lean block with a POSITIVE bias (take an
/// action when it fits) — the old split across three instructions all said
/// "default to []", which suppressed actions entirely. An action is just its
/// alias string (e.g. `["attack"]`); the schema also allows an object with the
/// alias as `id` for actions that need parameters.
pub const STRUCTURED_OUTPUT_INSTRUCTION: &str = concat!(
    "Reply with one JSON object: \"speech\" (your spoken words only, no name or label), ",
    "\"stateUpdates\" ({} unless something must change), and \"actions\".\n",
    "\"actions\" is what you actually DO this turn: when a listed action fits the moment, add its ",
    "alias from the Activated Action Book entries to the array (e.g. [\"<alias>\"]); otherwise use []. ",
    "Use only listed aliases and keep them out of \"speech\". If an action needs details, use an ",
    "object with the alias as \"id\". When an entry is marked \"Needs a target\", you MUST use the ",
    "object form and add \"target\" naming who you do it to (the player, or a nearby person by name) — ",
    "e.g. {\"id\":\"attack\",\"target\":\"Easy Pete\"}; for actions with no target marker, just the alias is fine. ",
    "When an entry lists \"Spawnable now\" candidates, use the object form with \"entity\" (one of those ",
    "candidates), \"count\", and \"target\" — e.g. {\"id\":\"spawn\",\"entity\":\"Deathclaw\",\"count\":1,\"target\":\"player\"}."
);

/// Verbatim from `src/headless/quest-books.js` `QUEST_BOOK_STRUCTURED_OUTPUT_INSTRUCTION`.
pub const QUEST_BOOK_STRUCTURED_OUTPUT_INSTRUCTION: &str = concat!(
    "If a quest event is appropriate, return it as an abstract action id from the activated Quest Book event list.\n",
    "Never output raw game script, console commands, quest-stage commands, NVSE/xNVSE/JIP/JohnnyGuitar/ShowOff code, form ids, or execution bindings.\n",
    "Do not accept, decline, advance, or end a quest until the player clearly chooses that option in the current conversation.\n",
    "Keep quest talk in character. Do not mention quest books, stages, editor ids, form ids, action ids, structured output, or backend routing unless the user asks how the system works.\n",
    "Use an empty actions array when the character should only speak and no quest event is clearly warranted."
);

const HISTORY_LIMIT: usize = 40;
const LORE_LIMIT: usize = 10;
const ACTION_LIMIT: usize = 10;
const QUEST_LIMIT: usize = 5;

/// Number of most-recent history messages excluded from chat-vector recall, so
/// the "Relevant past chat context" block never echoes lines already visible in
/// the history window (mirrors the `protect` window in `chat-vectors.js`).
const CHAT_VECTOR_PROTECT: usize = 4;

// ---------------------------------------------------------------------------
// Retrieval context (semantic embed + rerank)
// ---------------------------------------------------------------------------

/// Per-request retrieval bundle handed to [`assemble_prompt_with_retrieval`]:
/// the loaded retriever, the persistent embedding cache, and the relevant
/// `RetrievalSettings` values (already resolved from the persisted settings by
/// the web layer). When `None`, the assembler runs the pure keyword path so the
/// static panel and the offline/degraded path keep working unchanged.
#[derive(Clone, Copy)]
pub struct RetrievalCtx<'a> {
    pub retriever: &'a Retriever,
    pub cache: &'a EmbeddingCache,
    /// Per-source toggles (master `enabled` is handled by the web layer: it only
    /// builds a ctx when retrieval is enabled).
    pub chat_memory_enabled: bool,
    pub lore_semantic_enabled: bool,
    pub action_semantic_enabled: bool,
    pub quest_semantic_enabled: bool,
    /// Retrieve -> rerank funnel sizing.
    pub candidates: usize,
    pub top_k: usize,
    pub min_score: f32,
    /// Separate, lower floor for actions (terse commands score below lore passages).
    pub action_min_score: f32,
    /// Per-source caps on how many hits each source contributes.
    pub chat_memory_limit: usize,
    pub lore_limit: usize,
    pub quest_limit: usize,
}

/// Builds `(id, vector, text)` candidates for [`search`], embedding each text
/// through the persistent cache so content embeds exactly once. Texts that are
/// empty after trimming are skipped. Best-effort: a cache/embed failure on one
/// item drops that item rather than failing the whole turn.
fn build_candidates(
    ctx: &RetrievalCtx,
    items: impl IntoIterator<Item = (String, String)>,
) -> Vec<(String, Vec<f32>, String)> {
    let kept: Vec<(String, String)> = items
        .into_iter()
        .filter(|(_, text)| !text.trim().is_empty())
        .collect();
    let trimmed: Vec<&str> = kept.iter().map(|(_, text)| text.trim()).collect();
    // One batched call: cache hits resolve individually, ALL misses embed in a
    // single model invocation (one GPU kernel launch instead of N).
    let mut probe = PhaseTimer::new();
    let vectors = ctx.cache.get_or_embed_batch(ctx.retriever, &trimmed);
    probe.mark(&format!("  candidates:embed_batch n={}", trimmed.len()));
    kept.into_iter()
        .zip(vectors)
        .filter_map(|((id, text), vector)| Some((id, vector?, text)))
        .collect()
}

/// Runs the two-stage retrieve -> rerank search for `query` over `candidates`,
/// returning at most `limit` ids in rank order. Empty query/candidates short
/// circuit; any retriever error degrades to an empty result.
fn retrieve_ids(
    ctx: &RetrievalCtx,
    query: &str,
    candidates: &[(String, Vec<f32>, String)],
    limit: usize,
    min_score: f32,
) -> Vec<String> {
    if query.trim().is_empty() || candidates.is_empty() || limit == 0 {
        return Vec::new();
    }
    // Embed the query THROUGH the cache: its in-process memo means the same
    // turn text is embedded once, not once per retrieval subsystem. The query
    // embeds with the model's retrieval-query prefix (BGE is asymmetric:
    // prefixed queries match ANSWERING passages instead of similar-shaped
    // questions); the prefixed string is the cache key, so it never collides
    // with the same text embedded as a passage.
    let mut probe = PhaseTimer::new();
    let query_text = ctx.retriever.query_text(query);
    let Ok(query_vec) = ctx.cache.get_or_embed(ctx.retriever, &query_text) else {
        return Vec::new();
    };
    probe.mark("  retrieve:query_embed");
    // `top_k` bounds the rerank stage; cap the per-source contribution to `limit`.
    let top_k = ctx.top_k.max(limit);
    let result = match chasm_embed::search_with_query_vec(
        ctx.retriever,
        &query_vec,
        query,
        candidates,
        ctx.candidates,
        top_k,
        min_score,
    ) {
        Ok(hits) => {
            dump_retrieval(ctx, query, &hits, candidates, limit, min_score);
            hits.into_iter()
                .take(limit)
                .map(|(id, _score)| id)
                .collect()
        }
        Err(_) => Vec::new(),
    };
    probe.mark("  retrieve:search+rerank");
    result
}

/// Env-gated (CHASM_RETRIEVAL_DUMP=1) content dump of what semantic retrieval
/// actually selected — query, per-hit score, kept/cut line, and a text snippet —
/// so ranking quality can be inspected offline. Appends to
/// %TEMP%\chasm-retrieval-dump.log.
fn dump_retrieval(
    ctx: &RetrievalCtx,
    query: &str,
    hits: &[(String, f32)],
    candidates: &[(String, Vec<f32>, String)],
    limit: usize,
    min_score: f32,
) {
    if std::env::var_os("CHASM_RETRIEVAL_DUMP").is_none() {
        return;
    }
    let Some(dir) = std::env::var_os("TEMP") else { return };
    let path = std::path::Path::new(&dir).join("chasm-retrieval-dump.log");
    let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    use std::io::Write as _;
    let reranker = if ctx.retriever.has_reranker() { "on" } else { "off" };
    let _ = writeln!(
        f,
        "== query [reranker {reranker}, cands {} min {min_score}]: {}",
        candidates.len(),
        query.replace('\n', " | ")
    );
    for (rank, (id, score)) in hits.iter().enumerate() {
        let text = candidates
            .iter()
            .find(|(cid, _, _)| cid == id)
            .map(|(_, _, t)| t.chars().take(110).collect::<String>().replace('\n', " "))
            .unwrap_or_default();
        let kept = if rank < limit { "KEEP" } else { "cut " };
        let _ = writeln!(f, "  {kept} #{rank} {score:.3} {id}: {text}");
    }
}

// ---------------------------------------------------------------------------
// Keyword activation (mirrors the keyword path of the JS resolvers)
// ---------------------------------------------------------------------------

/// Tests one activation key against the scan text. Tries a regex (case-insensitive
/// unless `case_sensitive`), falling back to a substring test, matching the JS
/// `keyMatches` try/catch.
pub(crate) fn key_matches(key: &str, text: &str, case_sensitive: bool) -> bool {
    let raw = key.trim();
    if raw.is_empty() {
        return false;
    }
    match RegexBuilder::new(raw)
        .case_insensitive(!case_sensitive)
        .size_limit(1 << 20)
        .build()
    {
        Ok(regex) => regex.is_match(text),
        Err(_) => {
            if case_sensitive {
                text.contains(raw)
            } else {
                text.to_lowercase().contains(&raw.to_lowercase())
            }
        }
    }
}

/// Constant-or-keyword activation shared by lore, action, and quest entries.
fn keyword_active(
    disable: bool,
    constant: bool,
    keys: &[String],
    case_sensitive: Option<bool>,
    text: &str,
) -> bool {
    if disable {
        return false;
    }
    if constant {
        return true;
    }
    let case_sensitive = case_sensitive.unwrap_or(false);
    keys.iter()
        .any(|key| key_matches(key, text, case_sensitive))
}

/// Mirrors `entryPassesCharacterFilter`: an entry with a character filter only
/// applies to the named characters (or everyone-but-named when `isExclude`).
fn lore_passes_character_filter(entry: &LoreEntry, character_name: Option<&str>) -> bool {
    if entry.filter_names.is_empty() {
        return true;
    }
    let matches = character_name
        .map(|name| {
            entry
                .filter_names
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(name))
        })
        .unwrap_or(false);
    if entry.filter_exclude {
        !matches
    } else {
        matches
    }
}

fn order_desc(left: f64, right: f64) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

// ---------------------------------------------------------------------------
// Action alias derivation (mirrors getStructuredActionAlias / slugActionAlias)
// ---------------------------------------------------------------------------

/// `(action_id, alias)` pairs for a set of action entries — the map the server
/// uses to resolve a model's emitted alias/id back to the canonical action id
/// (mirrors `getStructuredActionAliases`). Aliases are slugged + lowercased.
pub fn action_alias_pairs(actions: &[ActionEntry]) -> Vec<(String, String)> {
    actions
        .iter()
        .map(|action| (action.action_id.clone(), structured_action_alias(action)))
        .collect()
}

/// Flattens the structured output's normalized `actions` array into display
/// [`ActionView`]s for the per-message panel. Each action object is expected in
/// the post-normalization shape (`{ id, target, parameters, reason }`); the alias
/// is recovered from `aliases` (canonical id -> alias) so the panel can show the
/// short string the model was offered. Non-object / id-less entries are skipped.
pub fn turn_actions_from_structured(
    structured: &Value,
    aliases: &[(String, String)],
) -> Vec<ActionView> {
    let Some(items) = structured.get("actions").and_then(Value::as_array) else {
        return Vec::new();
    };
    let alias_by_id: std::collections::HashMap<&str, &str> = aliases
        .iter()
        .map(|(id, alias)| (id.as_str(), alias.as_str()))
        .collect();
    items
        .iter()
        .filter_map(|item| {
            let object = item.as_object()?;
            let id = object
                .get("id")
                .or_else(|| object.get("actionId"))
                .or_else(|| object.get("action_id"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            // An action with no resolvable id carries no display identity; skip it.
            if id.is_empty() {
                return None;
            }
            let alias = object
                .get("alias")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| {
                    alias_by_id
                        .get(id.as_str())
                        .map(|a| a.to_string())
                        .unwrap_or_default()
                });
            let target = object
                .get("target")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            let params = object
                .get("parameters")
                .filter(|params| !matches!(params, Value::Null) && schema_non_empty(params))
                .map(|params| serde_json::to_string(params).unwrap_or_default())
                .unwrap_or_default();
            let reason = object
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            Some(ActionView {
                id,
                alias,
                target,
                params,
                reason,
            })
        })
        .collect()
}

pub fn slug_action_alias(value: &str) -> String {
    let without_png = if value.to_ascii_lowercase().ends_with(".png") {
        &value[..value.len() - 4]
    } else {
        value
    };
    let mut result = String::with_capacity(without_png.len());
    let mut pending_underscore = false;
    for ch in without_png.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            result.push(ch);
            pending_underscore = false;
        } else if !pending_underscore {
            result.push('_');
            pending_underscore = true;
        }
    }
    result.trim_matches('_').to_string()
}

fn structured_action_alias(action: &ActionEntry) -> String {
    let action_id = action.action_id.trim();
    let explicit = action
        .alias
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            action
                .short_name
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
        });
    if let Some(explicit) = explicit {
        return slug_action_alias(explicit);
    }

    match action_id {
        "npc.gesture_wave" => return "wave".to_string(),
        "movement.follow_target" => return "follow".to_string(),
        "movement.stop_follow_target" | "movement.stop_following" | "movement.stop_follow" => {
            return "stop_follow".to_string()
        }
        "combat.start" => return "attack".to_string(),
        "combat.stop" => return "stop_combat".to_string(),
        "ai.wait_here" => return "wait".to_string(),
        "ai.sandbox_here" => return "sandbox".to_string(),
        "ai.resume_default" => return "resume".to_string(),
        "ai.sit_down" => return "sit".to_string(),
        "world.spawn_item" => return "spawn_item".to_string(),
        "world.spawn_entity" => return "spawn_entity".to_string(),
        _ => {}
    }

    if let Some(rest) = action_id.strip_prefix("npc.gesture_") {
        return slug_action_alias(rest);
    }

    let last = action_id.split('.').filter(|part| !part.is_empty()).last();
    let fallback = match last {
        Some(part) => part.to_string(),
        None if !action.title.is_empty() => action.title.clone(),
        None => action.action_id.clone(),
    };
    slug_action_alias(&fallback)
}

// ---------------------------------------------------------------------------
// Prompt-section formatters (verbatim ports)
// ---------------------------------------------------------------------------

fn schema_non_empty(value: &Value) -> bool {
    match value {
        Value::Object(map) => !map.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Null => false,
        _ => true,
    }
}

/// Port of `formatActionBookPrompt`. `spawn_candidates` maps an action_id to the
/// RAG-resolved spawnable names (nearby-NPC targets are still runtime-only).
pub fn format_action_book_prompt(
    actions: &[ActionEntry],
    spawn_candidates: &std::collections::HashMap<String, Vec<String>>,
) -> String {
    actions
        .iter()
        .map(|action| {
            let alias = structured_action_alias(action);
            let title = if action.title.is_empty() {
                action.action_id.clone()
            } else {
                action.title.clone()
            };
            let mut parts: Vec<String> = vec![
                format!("- {alias} => {}: {title}", action.action_id),
                format!("Action alias: {alias} (put this in \"actions\" to perform it)."),
            ];
            if let Some(names) = spawn_candidates.get(&action.action_id) {
                if !names.is_empty() {
                    parts.push(format!(
                        "Spawnable now (set \"entity\" to one of these): {}.",
                        names.join(", ")
                    ));
                    parts.push(format!(
                        "To spawn, use {{\"id\":\"{alias}\",\"entity\":\"<one of the above>\",\
                         \"count\":<how many>,\"target\":\"<who/where: the player or a nearby \
                         person by name>\"}}."
                    ));
                }
            }
            if action.requires_target {
                parts.push(format!(
                    "Needs a target: use {{\"id\":\"{alias}\",\"target\":\"<name>\"}} naming who \
                     (the player, or a nearby person by name)."
                ));
            }
            if !action.risk_tier.is_empty() {
                parts.push(format!("Risk: {}", action.risk_tier));
            }
            if !action.description.is_empty() {
                parts.push(format!("Description: {}", action.description));
            }
            if schema_non_empty(&action.parameters_schema) {
                parts.push(format!(
                    "Parameters: {}",
                    serde_json::to_string(&action.parameters_schema).unwrap_or_default()
                ));
            }
            if !action.preconditions.is_empty() {
                parts.push(format!(
                    "Preconditions: {}",
                    action.preconditions.join("; ")
                ));
            }
            if !action.effects.is_empty() {
                parts.push(format!("Effects: {}", action.effects.join("; ")));
            }
            if !action.examples_when_to_use.is_empty() {
                parts.push(format!(
                    "Use when: {}",
                    action.examples_when_to_use.join("; ")
                ));
            }
            if !action.examples_when_not_to_use.is_empty() {
                parts.push(format!(
                    "Do not use when: {}",
                    action.examples_when_not_to_use.join("; ")
                ));
            }
            parts.join("\n  ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Port of `formatStageHints`.
fn format_stage_hints(stages: &[chasm_st_compat::QuestStageHint]) -> String {
    stages
        .iter()
        .filter_map(|hint| {
            let label = if !hint.label.trim().is_empty() {
                hint.label.trim().to_string()
            } else {
                hint.description.trim().to_string()
            };
            let number = hint
                .stage
                .map(|stage| format!("stage {stage}"))
                .unwrap_or_default();
            let parts: Vec<String> = [number, label]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(": "))
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Port of `formatQuestEvents`.
fn format_quest_events(events: &[chasm_st_compat::QuestEvent]) -> String {
    events
        .iter()
        .map(|event| {
            let mut parts: Vec<String> = Vec::new();
            if !event.action_id.is_empty() {
                parts.push(event.action_id.clone());
            }
            if !event.label.is_empty() && event.label != event.action_id {
                parts.push(event.label.clone());
            }
            parts.push(
                if event.requires_player_consent {
                    "requires clear player consent"
                } else {
                    "may be used to offer only"
                }
                .to_string(),
            );
            if !event.description.is_empty() {
                parts.push(event.description.clone());
            }
            parts.join(" | ")
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Port of `formatQuestBookPrompt`.
pub fn format_quest_book_prompt(quests: &[QuestEntry]) -> String {
    quests
        .iter()
        .map(|quest| {
            let header_name = if !quest.quest_name.is_empty() {
                &quest.quest_name
            } else {
                &quest.quest_id
            };
            let header_title = if !quest.title.is_empty() {
                quest.title.clone()
            } else if !quest.quest_name.is_empty() {
                quest.quest_name.clone()
            } else {
                quest.quest_id.clone()
            };
            let mut parts: Vec<String> = vec![format!("- {header_name}: {header_title}")];
            if !quest.quest_id.is_empty() {
                parts.push(format!("Quest id: {}", quest.quest_id));
            }
            if !quest.quest_editor_id.is_empty() {
                parts.push(format!("Editor id: {}", quest.quest_editor_id));
            }
            if !quest.giver_character_ids.is_empty() {
                parts.push(format!("Given by: {}", quest.giver_character_ids.join(", ")));
            }
            if !quest.description.is_empty() {
                parts.push(format!("Context: {}", quest.description));
            }
            if !quest.offer_summary.is_empty() {
                parts.push(format!("Offer summary: {}", quest.offer_summary));
            }
            if !quest.pre_dialogue.is_empty() {
                parts.push(format!("Dialogue approach: {}", quest.pre_dialogue));
            }
            if !quest.objectives.is_empty() {
                parts.push(format!("Objectives: {}", quest.objectives.join("; ")));
            }
            if !quest.acceptance_cues.is_empty() {
                parts.push(format!(
                    "Player acceptance cues: {}",
                    quest.acceptance_cues.join("; ")
                ));
            }
            if !quest.refusal_cues.is_empty() {
                parts.push(format!("Player refusal cues: {}", quest.refusal_cues.join("; ")));
            }
            let stages = format_stage_hints(&quest.stage_hints);
            if !stages.is_empty() {
                parts.push(format!("Known stages: {stages}"));
            }
            let events = format_quest_events(&quest.quest_events);
            if !events.is_empty() {
                parts.push(format!("Available abstract quest events: {events}"));
            }
            parts.push(
                "Do not accept, decline, advance, or end this quest unless the player clearly chooses that option. Speak naturally as the character."
                    .to_string(),
            );
            parts.join("\n  ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

/// Accumulates ordered prompt components.
struct Builder {
    order: usize,
    components: Vec<PromptComponentView>,
}

impl Builder {
    fn new() -> Self {
        Self {
            order: 0,
            components: Vec::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        group: &str,
        key: &str,
        label: &str,
        role: &str,
        status: &str,
        note: impl Into<String>,
        content: impl Into<String>,
    ) {
        self.order += 1;
        let content = content.into();
        let char_count = content.chars().count();
        self.components.push(PromptComponentView {
            order: self.order,
            group: group.to_string(),
            key: key.to_string(),
            label: label.to_string(),
            role: role.to_string(),
            status: status.to_string(),
            note: note.into(),
            content,
            char_count,
        });
    }
}

/// Pushes the GLOBAL scenario component in the slot the per-character card
/// `scenario` field used to occupy (see [`assemble_prompt_with_retrieval_collect`]).
///
/// * `Some(text)` (non-blank) — included, formatted exactly like the old card
///   field (`"Scenario:\n{text}"`), so downstream `build_chat_messages`-style
///   consumers see an identical shape.
/// * `Some(blank)` — the user cleared the global template (or every macro in
///   it resolved empty): the component is omitted entirely.
/// * `None` — static/panel assembly with no live turn: a `generation-time`
///   placeholder keeps the send order visible without guessing at macros.
fn push_global_scenario(builder: &mut Builder, global_scenario: Option<&str>) {
    match global_scenario {
        Some(text) => {
            let text = text.trim();
            if !text.is_empty() {
                builder.push(
                    "system",
                    "scenario",
                    "Scenario",
                    "system",
                    "included",
                    "Global scenario template, gamestate macros resolved for this turn.",
                    format!("Scenario:\n{text}"),
                );
            }
        }
        None => builder.push(
            "system",
            "scenario",
            "Scenario",
            "system",
            "generation-time",
            "Global scenario template (Globals page) — {{macro}} placeholders resolve \
             against the turn's gamestate at generation time.",
            String::new(),
        ),
    }
}

/// Injects the generated player-persona description (chasm-web's persona
/// module writes it; [`LiveChatRepository::read_player_persona`] reads it) as
/// ONE clearly-named additive component — the chasm equivalent of
/// SillyTavern's user persona in the story string. Callers place it at ST's
/// default slot: directly after the character scenario, before example
/// dialogue. No persona generated yet (or empty description) → nothing is
/// pushed, mirroring ST's `{{#if persona}}`; a read error becomes a note, not
/// a failure.
fn push_player_persona(builder: &mut Builder, notes: &mut Vec<String>, repo: &LiveChatRepository) {
    match repo.read_player_persona() {
        Ok(Some(persona)) => {
            builder.push(
                "system",
                "player_persona",
                "Player persona",
                "system",
                "included",
                "",
                format!("Player persona:\n{persona}"),
            );
        }
        Ok(None) => {}
        Err(error) => notes.push(format!("Player persona read failed: {error}")),
    }
}

/// Normalizes a slug-style key for native-NPC matching: lowercase, non
/// alphanumeric collapsed to a single space (so `easy_pete` ~ `Easy Pete`),
/// mirroring the loose `normalizeSlugKey`/`normalizeLookupKey` comparison in
/// `quest-books.js`.
fn normalize_loose(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut pending_space = false;
    for ch in value.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            out.push(ch);
            pending_space = false;
        } else {
            pending_space = true;
        }
    }
    out
}

/// A quest with named giver characters only activates when the current speaker is
/// one of them — mirrors SillyTavern's giver/speaker gate in `quest-books.js`, so
/// (e.g.) Easy Pete does not offer a quest only Sunny Smiles can give. Quests with
/// no giver constraint stay always-relevant. Honors both `giverCharacterIds`
/// (names/ids) and `giverNpcKeys` (slug keys), matching the speaker or the scan
/// text loosely.
fn quest_giver_matches_speaker(quest: &QuestEntry, speaker: &ParticipantView, _scan: &str) -> bool {
    if quest.giver_character_ids.is_empty() && quest.giver_npc_keys.is_empty() {
        return true;
    }
    let speaker_keys: Vec<String> = [
        speaker.character_id.as_deref().unwrap_or_default(),
        speaker.name.as_str(),
    ]
    .into_iter()
    .filter(|value| !value.is_empty())
    .map(normalize_loose)
    .filter(|value| !value.is_empty())
    .collect();
    // Speaker-only gate: a quest with named givers activates ONLY when the current
    // speaker is one of them. We deliberately do NOT match a giver merely appearing
    // in the scan text — gamestate lists nearby NPCs, so a mentioned giver would
    // wrongly make any nearby NPC offer that giver's quest (e.g. Easy Pete offering
    // Sunny Smiles's quest just because she's standing nearby).
    quest
        .giver_character_ids
        .iter()
        .chain(quest.giver_npc_keys.iter())
        .any(|giver| {
            let giver = normalize_loose(giver);
            !giver.is_empty() && speaker_keys.iter().any(|key| key == &giver)
        })
}

/// Scopes gate (`entryMatchesFilters`): an entry with scopes only applies when
/// the request supplies an intersecting scope (or the entry is `global`). With no
/// requested scopes the entry passes (the live FNV path supplies none today).
fn quest_passes_scopes(quest: &QuestEntry, requested_scopes: &[String]) -> bool {
    if quest.scopes.is_empty() || requested_scopes.is_empty() {
        return true;
    }
    let requested: Vec<String> = requested_scopes
        .iter()
        .map(|scope| scope.trim().to_lowercase())
        .collect();
    quest.scopes.iter().any(|scope| {
        let scope = scope.trim().to_lowercase();
        scope == "global" || requested.iter().any(|r| r == &scope)
    })
}

/// Scopes gate for ACTIONS — STRICTER than quests: an action with scopes passes
/// ONLY if a requested scope matches (or it is `global`). Unlike quests there is no
/// "empty requested → pass" loophole, so an admin-only action (`scopes:["admin"]`)
/// is never offered to a request that lacks the `admin` scope (regular NPCs). An
/// action with no scopes is always available.
fn action_passes_scopes(action: &ActionEntry, requested_scopes: &[String]) -> bool {
    if action.scopes.is_empty() {
        return true;
    }
    let requested: std::collections::BTreeSet<String> = requested_scopes
        .iter()
        .map(|scope| scope.trim().to_lowercase())
        .filter(|scope| !scope.is_empty())
        .collect();
    action.scopes.iter().any(|scope| {
        let scope = scope.trim().to_lowercase();
        scope == "global" || requested.contains(&scope)
    })
}

/// Substring include/exclude gate against the scan text (`entryMatchesFilters`).
fn quest_passes_include_exclude(quest: &QuestEntry, scan: &str) -> bool {
    let haystack = scan.to_lowercase();
    if !quest.include.is_empty()
        && !quest
            .include
            .iter()
            .any(|needle| haystack.contains(&needle.to_lowercase()))
    {
        return false;
    }
    if quest
        .exclude
        .iter()
        .any(|needle| !needle.trim().is_empty() && haystack.contains(&needle.to_lowercase()))
    {
        return false;
    }
    true
}

/// Phase gate: completed/failed quests are dropped unless `include_completed`,
/// mirroring the status check in `entryMatchesQuestState`. We only have the
/// entry's own `phase` here (no per-request quest-state map yet — see TODO in the
/// report), so this checks the static phase value.
fn quest_passes_phase(quest: &QuestEntry) -> bool {
    if quest.include_completed {
        return true;
    }
    !matches!(
        quest.phase.trim().to_lowercase().as_str(),
        "complete" | "completed" | "done" | "failed"
    )
}

/// The full non-vector quest gate, combining the keyword/constant activation
/// (already applied by the caller) with the parity gates ported from
/// `quest-books.js` `entryMatchesFilters` + `entryMatchesGiver` +
/// `entryMatchesQuestState`.
fn quest_passes_gate(
    quest: &QuestEntry,
    speaker: &ParticipantView,
    scan: &str,
    requested_scopes: &[String],
) -> bool {
    quest_passes_scopes(quest, requested_scopes)
        && quest_passes_include_exclude(quest, scan)
        && quest_giver_matches_speaker(quest, speaker, scan)
        && quest_passes_phase(quest)
}

/// The text a quest entry is embedded under for vector retrieval. Prefers the
/// pre-baked `vectorizableText`, else builds a compact summary from the
/// human-meaningful fields (name/title/offer/objectives), mirroring the spirit
/// of `buildQuestVectorText`.
fn quest_vector_text(quest: &QuestEntry) -> String {
    if !quest.vectorizable_text.trim().is_empty() {
        return quest.vectorizable_text.clone();
    }
    let mut parts: Vec<String> = Vec::new();
    for value in [
        &quest.quest_name,
        &quest.title,
        &quest.description,
        &quest.offer_summary,
    ] {
        if !value.trim().is_empty() {
            parts.push(value.trim().to_string());
        }
    }
    if !quest.objectives.is_empty() {
        parts.push(quest.objectives.join("; "));
    }
    parts.join("\n")
}

/// The text a lore entry is embedded under for vector retrieval: its content,
/// optionally prefixed by the comment/title for a little extra signal.
fn lore_vector_text(entry: &LoreEntry) -> String {
    if entry.comment.trim().is_empty() {
        entry.content.clone()
    } else {
        format!("{}\n{}", entry.comment.trim(), entry.content)
    }
}

/// Embeddable text for an action: title + description + trigger synonyms + the
/// author's generic `vectorizableText` intent phrasing, so indirect phrasings
/// ("stick close" -> a follow action) match semantically. We deliberately use
/// `vectorizableText` (name-free intent) rather than the per-entry
/// `vectorSearchTexts` examples, which embed specific NPC names ("attack joe
/// cobb") and would make name queries ("tell me about joe cobb") cross-match the
/// wrong action.
fn action_vector_text(entry: &ActionEntry) -> String {
    let mut parts = Vec::new();
    if !entry.title.trim().is_empty() {
        parts.push(entry.title.trim().to_string());
    }
    if !entry.description.trim().is_empty() {
        parts.push(entry.description.trim().to_string());
    }
    if !entry.keys.is_empty() {
        parts.push(entry.keys.join(", "));
    }
    if !entry.vectorizable_text.trim().is_empty() {
        parts.push(entry.vectorizable_text.trim().to_string());
    }
    parts.join("\n")
}

/// Embedding text for a catalog item: the pre-baked `vectorizableText`, falling
/// back to the display name (so an item with no vector text still matches by name).
fn catalog_item_vector_text(item: &CatalogItem) -> String {
    if !item.vectorizable_text.trim().is_empty() {
        item.vectorizable_text.trim().to_string()
    } else {
        item.name.trim().to_string()
    }
}

/// Drops whole-word trigger words ("spawn", "summon", …) from the player message
/// so the catalog query is the *subject* ("a deathclaw on me" → "a deathclaw on
/// me", minus "spawn"). Falls back to the original text if stripping empties it.
fn strip_trigger_keys(text: &str, trigger_keys: &[String]) -> String {
    if trigger_keys.is_empty() {
        return text.to_string();
    }
    let triggers: std::collections::BTreeSet<String> = trigger_keys
        .iter()
        .map(|key| key.trim().to_lowercase())
        .filter(|key| !key.is_empty())
        .collect();
    let kept: Vec<&str> = text
        .split_whitespace()
        .filter(|word| {
            let normalized: String = word
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase();
            !triggers.contains(&normalized)
        })
        .collect();
    let stripped = kept.join(" ");
    if stripped.trim().is_empty() {
        text.to_string()
    } else {
        stripped
    }
}

/// RAG-ranks a scoped catalog's items against the player query and returns the top
/// candidates (in rank order, up to `limit`). Reuses the same vector pipeline as
/// action retrieval. Rank-only (min_score 0) so small curated catalogs always
/// surface their best matches; falls back to the catalog as-is when no retriever
/// is available, so spawn still works without RAG.
fn search_catalog_items(
    ctx: &RetrievalCtx,
    query: &str,
    items: &[CatalogItem],
    limit: usize,
) -> Vec<CatalogItem> {
    let limit = limit.max(1);
    let searchable: Vec<(usize, &CatalogItem)> = items
        .iter()
        .enumerate()
        .filter(|(_, it)| !it.disable)
        .collect();
    if searchable.is_empty() {
        return Vec::new();
    }
    let candidates = build_candidates(
        ctx,
        searchable
            .iter()
            .map(|(index, item)| (index.to_string(), catalog_item_vector_text(item))),
    );
    let ranked: Vec<CatalogItem> = retrieve_ids(ctx, query, &candidates, limit, 0.0)
        .iter()
        .filter_map(|id| id.parse::<usize>().ok())
        .filter_map(|index| items.get(index).cloned())
        .collect();
    if !ranked.is_empty() {
        return ranked;
    }
    // No retriever / empty vector text: show the catalog as-is (ranking unavailable).
    searchable
        .iter()
        .take(limit)
        .map(|(_, item)| (*item).clone())
        .collect()
}

/// For one action's scoped-catalog configs, resolve the matching catalog
/// candidates from `catalog_map` (RAG-ranked by `query`). Returns the relay views
/// (also used to render the candidate list in the prompt).
fn resolve_scoped_catalogs(
    ctx: Option<&RetrievalCtx>,
    repo: &LiveChatRepository,
    entry: &ActionEntry,
    catalog_map: &std::collections::HashMap<String, Vec<CatalogItem>>,
    query: &str,
) -> Vec<ScopedCatalogView> {
    let Some(ctx) = ctx else {
        return Vec::new();
    };
    let mut views = Vec::new();
    for config in &entry.scoped_catalogs {
        // Union the full standalone catalog file (the ~8k spawnable records) with
        // any inline book catalog of the same id (so a custom inline item still
        // works even when it isn't in the generated file). Only read for activated
        // scoped-catalog (spawn) actions, so non-spawn turns pay nothing.
        let mut by_id: std::collections::HashMap<String, CatalogItem> =
            std::collections::HashMap::new();
        if let Some(inline) = catalog_map.get(&config.catalog_id) {
            for item in inline {
                by_id.insert(item.id.clone(), item.clone());
            }
        }
        for item in repo.read_action_catalog(&config.catalog_id) {
            by_id.insert(item.id.clone(), item);
        }
        if by_id.is_empty() {
            continue;
        }
        let items: Vec<CatalogItem> = by_id.into_values().collect();
        let catalog_query = strip_trigger_keys(query, &config.trigger_keys);
        let matched = search_catalog_items(ctx, &catalog_query, &items, config.limit as usize);
        if matched.is_empty() {
            continue;
        }
        views.push(ScopedCatalogView {
            catalog_id: config.catalog_id.clone(),
            parameter_name: config.parameter_name.clone(),
            items: matched
                .into_iter()
                .map(|item| CatalogItemView {
                    id: item.id,
                    name: item.name,
                    aliases: item.aliases,
                    metadata: item.metadata,
                })
                .collect(),
        });
    }
    views
}

/// One line of the `Relevant past chat context` block, mirroring
/// `formatChatVectorPrompt`: the speaker name when known, else the role, then
/// `: <content>`.
fn format_chat_vector_line(message: &MessageView) -> String {
    let speaker = if message.speaker_name.trim().is_empty() {
        match message.role.as_str() {
            "player" => "user".to_string(),
            other => other.to_string(),
        }
    } else {
        message.speaker_name.clone()
    };
    format!("{speaker}: {}", message.content)
}

/// Assembles the full prompt for `participant` given their visible `messages`,
/// broken into ordered components. Reads are best-effort; failures become
/// `notes` rather than hard errors, mirroring the JS "best effort" resolvers.
pub fn assemble_prompt(
    repo: &LiveChatRepository,
    participant: &ParticipantView,
    messages: &[MessageView],
) -> PromptAssemblyView {
    // Static/panel preview has no live turn, so approximate the live path's
    // "scan the current turn" behavior with the most recent player line. Scanning
    // the whole conversation (the old behavior) over-activates keyword entries.
    let scan_text = messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .or_else(|| messages.last())
        .map(|message| message.content.clone())
        .unwrap_or_default();
    assemble_prompt_with_scan(repo, participant, messages, &scan_text)
}

/// Like [`assemble_prompt`] but with an explicit keyword-activation `scan_text`.
/// The live generation path passes the current player message + gamestate + extra
/// context (matching SillyTavern's `activationText` in `generation.js`), so
/// action / lore / quest entries activate by relevance to the *current turn*
/// rather than to the entire visible history.
pub fn assemble_prompt_with_scan(
    repo: &LiveChatRepository,
    participant: &ParticipantView,
    messages: &[MessageView],
    scan_text: &str,
) -> PromptAssemblyView {
    assemble_prompt_with_retrieval(repo, participant, messages, scan_text, &[], None)
}

/// Like [`assemble_prompt_with_scan`] but with an optional semantic-retrieval
/// [`RetrievalCtx`]. When `ctx` is `Some` and the relevant per-source toggle is
/// on, this additionally:
///
/// * fills the `Relevant past chat context` block from chat-vector recall over
///   the visible history (excluding the most-recent window),
/// * merges vector-matched lore entries into the keyword/constant lore set, and
/// * adds vector-matched quest entries to the gated quest set.
///
/// When `ctx` is `None` (the static panel, or retrieval disabled / model
/// unavailable) it is byte-for-byte the old keyword-only behavior.
///
/// This is the view-only wrapper kept for the static panel + back-compat;
/// [`assemble_prompt_with_retrieval_collect`] returns the same view PLUS the set
/// of injected lore/quest/action entries (which the generation path records onto
/// the produced message). The prompt TEXT is identical between the two.
pub fn assemble_prompt_with_retrieval(
    repo: &LiveChatRepository,
    participant: &ParticipantView,
    messages: &[MessageView],
    scan_text: &str,
    requested_scopes: &[String],
    ctx: Option<RetrievalCtx>,
) -> PromptAssemblyView {
    // The static/panel path has no live turn, so the GLOBAL scenario (a
    // per-request input: it resolves gamestate macros per turn) is shown as a
    // correctly-ordered placeholder (`None`), like the other runtime-only
    // pieces.
    assemble_prompt_with_retrieval_collect(
        repo,
        participant,
        messages,
        scan_text,
        scan_text,
        requested_scopes,
        ctx,
        None,
    )
    .0
}

/// Like [`assemble_prompt_with_retrieval`] but ALSO returns the identities of the
/// lore / quest / action entries that were actually injected into the prompt
/// this turn, each tagged with why it activated (`constant` / `keyword` /
/// `vector`). The generation path persists this onto the produced message's
/// `extra.chasm.injected` so the per-message panel can show exactly what
/// the model saw. The prompt TEXT (the returned [`PromptAssemblyView`]) is
/// unchanged — this only surfaces the entry identities that were previously
/// discarded after assembly.
///
/// `global_scenario` is the GLOBAL scenario for this turn — the Globals-page
/// template already resolved through the turn's gamestate macros (see
/// [`crate::scenario`]). It is injected at the exact slot the per-character
/// card `scenario` field used to occupy in the ST-style assembly order
/// (after `Personality`, before `Example dialogue`); the card field itself is
/// no longer injected (still parsed/stored for imported-card compat).
/// * `Some(text)`  — inject `text` as the `Scenario` component (skipped when
///   blank after trimming).
/// * `None`        — no live turn (the static panel): a correctly-ordered
///   `generation-time` placeholder is shown instead, like the other
///   per-request inputs.
#[allow(clippy::too_many_arguments)]

/// TEMPORARY perf probe: when `CHASM_RETRIEVAL_TIMING` is set, appends
/// "<phase>	<micros>" lines to %TEMP%/chasm-retrieval-timing.log.
struct PhaseTimer {
    enabled: bool,
    last: std::time::Instant,
}
impl PhaseTimer {
    fn new() -> Self {
        Self { enabled: std::env::var_os("CHASM_RETRIEVAL_TIMING").is_some(), last: std::time::Instant::now() }
    }
    fn mark(&mut self, phase: &str) {
        if !self.enabled { return; }
        let micros = self.last.elapsed().as_micros();
        self.last = std::time::Instant::now();
        if let Some(dir) = std::env::var_os("TEMP") {
            let path = std::path::Path::new(&dir).join("chasm-retrieval-timing.log");
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                use std::io::Write as _;
                let _ = writeln!(f, "{phase}	{micros}");
            }
        }
    }
}

pub fn assemble_prompt_with_retrieval_collect(
    repo: &LiveChatRepository,
    participant: &ParticipantView,
    messages: &[MessageView],
    scan_text: &str,
    action_scan: &str,
    requested_scopes: &[String],
    ctx: Option<RetrievalCtx>,
    global_scenario: Option<&str>,
) -> (PromptAssemblyView, InjectedView) {
    let mut timer = PhaseTimer::new();
    let mut notes: Vec<String> = Vec::new();
    let mut builder = Builder::new();
    let mut injected = InjectedView::default();

    // Lore / quest / chat-memory retrieve on the full scan (message + gamestate),
    // so location/NPC context still surfaces relevant lore. Actions retrieve on the
    // player MESSAGE ONLY: the gamestate (location + nearby-NPC list) otherwise
    // floods the action search and surfaces unrelated actions every turn.
    let activation_text = scan_text.to_string();
    let action_activation = action_scan.to_string();

    // --- Character card -----------------------------------------------------
    let card = match participant.character_id.as_deref() {
        Some(id) => match repo.read_character_card(id) {
            Ok(card) => card,
            Err(error) => {
                notes.push(format!("Character card read failed: {error}"));
                None
            }
        },
        None => None,
    };
    let character_found = card.is_some();

    if let Some(card) = &card {
        builder.push(
            "system",
            "character",
            "Character",
            "system",
            "included",
            "",
            format!("Character: {}", card.name),
        );
        // NOTE: `card.scenario` is deliberately NOT in this list any more. The
        // scenario slot is filled by the GLOBAL scenario template below
        // (resolved with gamestate macros per turn); the card field is only
        // tolerated for imported-ST-card storage compat (see
        // `chasm_st_compat::CharacterCard::scenario`).
        for (key, label, value) in [
            ("system_prompt", "System prompt", &card.system_prompt),
            ("description", "Description", &card.description),
            ("personality", "Personality", &card.personality),
        ] {
            if !value.is_empty() {
                builder.push(
                    "system",
                    key,
                    label,
                    "system",
                    "included",
                    "",
                    format!("{label}:\n{value}"),
                );
            }
        }
        push_global_scenario(&mut builder, global_scenario);
        // SillyTavern's default story string places the user persona
        // directly after the scenario slot (…{{scenario}} → {{persona}}),
        // before example dialogue — match that position exactly. The slot
        // fires whether or not scenario itself rendered (ST's template
        // slots are independent `{{#if}}` blocks), so this is unconditional:
        // a blank/omitted global scenario still gets the persona here.
        push_player_persona(&mut builder, &mut notes, repo);
        if !card.example_dialogue.is_empty() {
            builder.push(
                "system",
                "example_dialogue",
                "Example dialogue",
                "system",
                "included",
                "",
                format!("Example dialogue:\n{}", card.example_dialogue),
            );
        }
    } else {
        builder.push(
            "system",
            "character",
            "Character",
            "system",
            "unavailable",
            "No character card resolved for this participant.",
            String::new(),
        );
        // The scenario is global (scene, not persona), so a card-less
        // participant still gets it, at the same position in the order — but
        // only when a live turn supplies one (`Some`). The static-panel
        // placeholder (`None`) is skipped on this card-less path: there are no
        // Personality/Example anchors to order it against, and the persona
        // component below keeps its position directly after the character
        // block (scenario → persona stays the ST story-string order whenever
        // both render).
        if global_scenario.is_some() {
            push_global_scenario(&mut builder, global_scenario);
        }
        // The persona describes the PLAYER, so it still injects when the NPC's
        // card did not resolve (same relative position: end of the card block).
        push_player_persona(&mut builder, &mut notes, repo);
    }

    // --- Activated lore -----------------------------------------------------
    timer.mark("card+scenario+persona");
    let lorebooks = match card.as_ref().and_then(|card| card.world.clone()) {
        Some(world) => match repo.read_lorebook(&world) {
            Ok(Some(book)) => vec![book],
            Ok(None) => {
                notes.push(format!("Linked world '{world}' not found; lore omitted."));
                Vec::new()
            }
            Err(error) => {
                notes.push(format!("Lorebook read failed: {error}"));
                Vec::new()
            }
        },
        None => repo.list_lorebooks().unwrap_or_else(|error| {
            notes.push(format!("Lorebook read failed: {error}"));
            Vec::new()
        }),
    };
    let character_name = card.as_ref().map(|card| card.name.as_str());
    // Stable index id per entry (lorebook order) so keyword + vector sets dedup.
    let all_lore: Vec<&LoreEntry> = lorebooks
        .iter()
        .flat_map(|book| book.entries.iter())
        .collect();
    // Entries eligible for any activation (not disabled, passing char filter).
    let eligible_lore: Vec<(usize, &LoreEntry)> = all_lore
        .iter()
        .enumerate()
        .filter(|(_, entry)| !entry.disable && lore_passes_character_filter(entry, character_name))
        .map(|(index, entry)| (index, *entry))
        .collect();

    // Keyword/constant matches (the existing path).
    timer.mark("lorebook_load");
    let mut keyword_ids: Vec<usize> = eligible_lore
        .iter()
        .filter(|(_, entry)| {
            keyword_active(
                entry.disable,
                entry.constant,
                &entry.keys,
                entry.case_sensitive,
                &activation_text,
            )
        })
        .map(|(index, _)| *index)
        .collect();

    timer.mark("lore_keyword");
    // Snapshot the constant/keyword set BEFORE the vector merge so each injected
    // entry can be tagged with why it activated (constant vs keyword vs vector).
    let lore_keyword_set: std::collections::BTreeSet<usize> = keyword_ids.iter().copied().collect();

    // Semantic merge: vector-matched lore entries that the keyword path missed,
    // capped to the lore limit (mirrors generation.js merging vector lore into
    // the activated set). Dedup by entry index.
    let lore_note = if let Some(ctx) = ctx.as_ref().filter(|ctx| ctx.lore_semantic_enabled) {
        let candidates = build_candidates(
            ctx,
            eligible_lore
                .iter()
                .filter(|(index, _)| !lore_keyword_set.contains(index))
                .map(|(index, entry)| (index.to_string(), lore_vector_text(entry))),
        );
        let hits = retrieve_ids(
            ctx,
            &activation_text,
            &candidates,
            ctx.lore_limit,
            ctx.min_score,
        );
        for id in hits {
            if let Ok(index) = id.parse::<usize>() {
                keyword_ids.push(index);
            }
        }
        "Constant + keyword matches merged with vector-matched lore entries."
    } else {
        "Constant + keyword matches. The live path also adds vector-matched entries."
    };

    // Carry the entry index alongside each item so the recorded injected set can
    // name the activation reason; sort + truncate identically to before.
    timer.mark("lore_semantic");
    let mut lore_items: Vec<(usize, &LoreEntry)> = keyword_ids
        .iter()
        .filter_map(|index| all_lore.get(*index).copied().map(|entry| (*index, entry)))
        .collect();
    lore_items.sort_by(|(_, left), (_, right)| order_desc(left.order, right.order));
    lore_items.truncate(LORE_LIMIT);
    for (index, entry) in &lore_items {
        let reason = if entry.constant {
            "constant"
        } else if lore_keyword_set.contains(index) {
            "keyword"
        } else {
            "vector"
        };
        injected.lore.push(InjectedEntryView {
            source: "lore".to_string(),
            id: lore_entry_id(entry, *index),
            title: lore_entry_title(entry, *index),
            reason: reason.to_string(),
        });
    }
    if !lore_items.is_empty() {
        let body = lore_items
            .iter()
            .map(|(_, entry)| entry.content.clone())
            .filter(|content| !content.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        builder.push(
            "system",
            "lore",
            "Activated lore",
            "system",
            "included",
            lore_note,
            format!("Activated lore:\n{body}"),
        );
    }

    // --- Relevant past chat context (vectors) -------------------------------
    // Chat-vector memory: embed the visible history (excluding the most-recent
    // window already shown verbatim), retrieve the entries most relevant to the
    // scan text, and emit a `Relevant past chat context:` block formatted like
    // generation.js (`formatChatVectorPrompt`: "<name|role>: <content>").
    timer.mark("lore_format");
    let chat_block = ctx
        .as_ref()
        .filter(|ctx| ctx.chat_memory_enabled && ctx.chat_memory_limit > 0)
        .and_then(|ctx| {
            let indexable = messages.len().saturating_sub(CHAT_VECTOR_PROTECT);
            let candidates = build_candidates(
                ctx,
                messages[..indexable]
                    .iter()
                    .enumerate()
                    .map(|(index, msg)| (index.to_string(), msg.content.clone())),
            );
            let hits = retrieve_ids(
                ctx,
                &activation_text,
                &candidates,
                ctx.chat_memory_limit,
                ctx.min_score,
            );
            if hits.is_empty() {
                return None;
            }
            // Preserve chronological order of the matched messages.
            let mut indices: Vec<usize> = hits
                .iter()
                .filter_map(|id| id.parse::<usize>().ok())
                .collect();
            indices.sort_unstable();
            // Collapse identical recalled lines (e.g. a player who repeated
            // themselves) so the block never shows the same line twice.
            let mut seen = std::collections::HashSet::new();
            let body = indices
                .iter()
                .filter_map(|index| messages.get(*index))
                .map(format_chat_vector_line)
                .filter(|line| seen.insert(line.trim().to_lowercase()))
                .collect::<Vec<_>>()
                .join("\n");
            if body.trim().is_empty() {
                None
            } else {
                Some(body)
            }
        });
    match chat_block {
        Some(body) => builder.push(
            "system",
            "chat_vectors",
            "Relevant past chat context",
            "system",
            "included",
            "Vector-matched past chat lines relevant to the current turn.",
            format!("Relevant past chat context:\n{body}"),
        ),
        None => builder.push(
            "system",
            "chat_vectors",
            "Relevant past chat context",
            "system",
            "generation-time",
            "Vectorized past-chat retrieval needs the embeddings runtime; not reproduced in the static view.",
            String::new(),
        ),
    }

    // --- Activated Quest Book entries ---------------------------------------
    timer.mark("chat_memory");
    let quest_books = repo.list_quest_books().unwrap_or_else(|error| {
        notes.push(format!("Quest book read failed: {error}"));
        Vec::new()
    });
    // All entries that survive the always-on gate (giver + scopes + include/
    // exclude + phase) — this is the candidate pool for both keyword and vector.
    let gated_quests: Vec<QuestEntry> = quest_books
        .into_iter()
        .flat_map(|book| book.entries.into_iter())
        .filter(|entry| {
            !entry.disable
                && quest_passes_gate(entry, participant, &activation_text, requested_scopes)
        })
        .collect();

    // Keyword/constant activation within the gated pool.
    let mut quest_selected: Vec<usize> = gated_quests
        .iter()
        .enumerate()
        .filter(|(_, entry)| {
            keyword_active(
                entry.disable,
                entry.constant,
                &entry.keys,
                entry.case_sensitive,
                &activation_text,
            )
        })
        .map(|(index, _)| index)
        .collect();

    timer.mark("action_keyword_gate");
    // Snapshot the keyword/constant set before the vector merge (reason tagging).
    let quest_keyword_set: std::collections::BTreeSet<usize> =
        quest_selected.iter().copied().collect();

    // Semantic merge: vector-matched gated quests the keyword path missed.
    let quest_note = if let Some(ctx) = ctx.as_ref().filter(|ctx| ctx.quest_semantic_enabled) {
        let candidates = build_candidates(
            ctx,
            gated_quests
                .iter()
                .enumerate()
                .filter(|(index, _)| !quest_keyword_set.contains(index))
                .map(|(index, entry)| (index.to_string(), quest_vector_text(entry))),
        );
        let hits = retrieve_ids(
            ctx,
            &activation_text,
            &candidates,
            ctx.quest_limit,
            ctx.min_score,
        );
        for id in hits {
            if let Ok(index) = id.parse::<usize>() {
                quest_selected.push(index);
            }
        }
        "Gated (giver/scopes/include/phase) keyword matches merged with vector-matched quests."
    } else {
        "Gated (giver/scopes/include/phase) constant + keyword matches; vector-activated quests are not shown."
    };

    let mut quest_items: Vec<(usize, QuestEntry)> = quest_selected
        .iter()
        .filter_map(|index| {
            gated_quests
                .get(*index)
                .cloned()
                .map(|entry| (*index, entry))
        })
        .collect();
    quest_items.sort_by(|(_, left), (_, right)| order_desc(left.priority, right.priority));
    quest_items.truncate(QUEST_LIMIT);
    for (index, entry) in &quest_items {
        let reason = if entry.constant {
            "constant"
        } else if quest_keyword_set.contains(index) {
            "keyword"
        } else {
            "vector"
        };
        injected.quests.push(InjectedEntryView {
            source: "quest".to_string(),
            id: quest_entry_id(entry),
            title: quest_entry_title(entry),
            reason: reason.to_string(),
        });
    }
    let has_quests = !quest_items.is_empty();
    if has_quests {
        let quest_entries: Vec<QuestEntry> =
            quest_items.iter().map(|(_, entry)| entry.clone()).collect();
        builder.push(
            "system",
            "quest_books",
            "Activated Quest Book entries",
            "system",
            "included",
            quest_note,
            format!(
                "Activated Quest Book entries:\n{}",
                format_quest_book_prompt(&quest_entries)
            ),
        );
    }

    // --- Activated Action Book entries --------------------------------------
    timer.mark("quest_books");
    let action_books = repo.list_action_books().unwrap_or_else(|error| {
        notes.push(format!("Action book read failed: {error}"));
        Vec::new()
    });
    // Keep each book's catalogs (with items) so scoped-catalog (spawn) actions can
    // resolve candidates → FormIDs; the flatten below would otherwise drop them.
    timer.mark("action_book_read");
    let mut catalog_map: std::collections::HashMap<String, Vec<CatalogItem>> =
        std::collections::HashMap::new();
    let mut all_actions: Vec<ActionEntry> = Vec::new();
    for book in action_books {
        for catalog in book.catalogs {
            catalog_map
                .entry(catalog.id)
                .or_default()
                .extend(catalog.items);
        }
        all_actions.extend(book.entries);
    }
    let mut action_ids: Vec<usize> = all_actions
        .iter()
        .enumerate()
        .filter(|(_, entry)| {
            action_passes_scopes(entry, requested_scopes)
                && keyword_active(
                    entry.disable,
                    entry.constant,
                    &entry.keys,
                    entry.case_sensitive,
                    &action_activation,
                )
        })
        .map(|(index, _)| index)
        .collect();

    // Snapshot the keyword/constant set before the vector merge (reason tagging).
    let action_keyword_set: std::collections::BTreeSet<usize> =
        action_ids.iter().copied().collect();

    // Semantic merge: vector-matched actions the keyword path missed, so indirect
    // requests ("stick close", "patch me up") still surface the relevant action.
    // Bias toward over-inclusion — the model can still choose not to act, but a
    // never-injected action can never be chosen.
    let action_note = if let Some(ctx) = ctx.as_ref().filter(|ctx| ctx.action_semantic_enabled) {
        timer.mark("action_gates_pre_semantic");
        let candidates = build_candidates(
            ctx,
            all_actions
                .iter()
                .enumerate()
                // Respect the `vectorized` flag: involuntary idle gestures (sneeze,
                // pushups, look-around, ...) are keyword-only, never vector-retrieved
                // from player dialogue, so they don't pollute every turn.
                .filter(|(index, entry)| {
                    !entry.disable
                        && entry.vectorized
                        && action_passes_scopes(entry, requested_scopes)
                        && !action_keyword_set.contains(index)
                })
                .map(|(index, entry)| (index.to_string(), action_vector_text(entry))),
        );
        for id in retrieve_ids(
            ctx,
            &action_activation,
            &candidates,
            ACTION_LIMIT,
            ctx.action_min_score,
        ) {
            if let Ok(index) = id.parse::<usize>() {
                action_ids.push(index);
            }
        }
        "Constant + keyword matches merged with vector-matched actions."
    } else {
        "Constant + keyword matches; the live path also adds vector-matched actions."
    };

    let mut action_items: Vec<(usize, ActionEntry)> = action_ids
        .iter()
        .filter_map(|index| {
            all_actions
                .get(*index)
                .cloned()
                .map(|entry| (*index, entry))
        })
        .collect();
    action_items.sort_by(|(_, left), (_, right)| order_desc(left.order, right.order));
    action_items.truncate(ACTION_LIMIT);
    timer.mark("action_books+semantic");
    // Per-action spawn candidate names (action_id -> ["Deathclaw", ...]), rendered
    // in the prompt so the model picks a valid entity. Built from the same resolved
    // catalogs relayed to the helper.
    let mut spawn_render: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for (index, entry) in &action_items {
        let reason = if entry.constant {
            "constant"
        } else if action_keyword_set.contains(index) {
            "keyword"
        } else {
            "vector"
        };
        injected.actions.push(InjectedEntryView {
            source: "action".to_string(),
            id: if entry.action_id.is_empty() {
                structured_action_alias(entry)
            } else {
                entry.action_id.clone()
            },
            title: action_entry_title(entry),
            reason: reason.to_string(),
        });
        // Resolve scoped-catalog (spawn) candidates by RAG-ranking the catalog against
        // the player message — relayed to the helper (entity -> FormID) AND rendered.
        let scoped_catalogs =
            resolve_scoped_catalogs(ctx.as_ref(), repo, entry, &catalog_map, &action_activation);
        if !scoped_catalogs.is_empty() && !entry.action_id.is_empty() {
            let names: Vec<String> = scoped_catalogs
                .iter()
                .flat_map(|view| view.items.iter().map(|item| item.name.clone()))
                .filter(|name| !name.trim().is_empty())
                .collect();
            if !names.is_empty() {
                spawn_render.insert(entry.action_id.clone(), names);
            }
        }
        // Relay the trusted execution/binding so the helper can build the native
        // command for non-native actions (gestures, spawn). Without this the helper
        // only has id/title and silently drops everything but the 3 hardcoded natives.
        injected.activated_actions.push(ActivatedActionView {
            action_id: entry.action_id.clone(),
            alias: structured_action_alias(entry),
            binding: entry.binding.clone(),
            execution: entry.execution.clone(),
            requires_target: entry.requires_target,
            scoped_catalogs,
        });
    }
    let action_entries: Vec<ActionEntry> = action_items
        .iter()
        .map(|(_, entry)| entry.clone())
        .collect();
    let has_actions = !action_entries.is_empty();
    if has_actions {
        builder.push(
            "system",
            "action_books",
            "Activated Action Book entries",
            "system",
            "included",
            action_note,
            format!(
                "Activated Action Book entries:\n{}",
                format_action_book_prompt(&action_entries, &spawn_render)
            ),
        );
    }

    // --- External world state -----------------------------------------------
    builder.push(
        "system",
        "world_state",
        "External world state",
        "system",
        "generation-time",
        "Built per request from the requested world-state scopes; none are supplied in the static view.",
        String::new(),
    );

    // --- Gamestate / extra context / response instructions ------------------
    builder.push(
        "system",
        "gamestate",
        "Gamestate",
        "system",
        "generation-time",
        "Supplied per request from the game bridge (player location, nearby NPCs).",
        String::new(),
    );
    builder.push(
        "system",
        "extra_context",
        "Additional external context",
        "system",
        "generation-time",
        "Supplied per request (body.extraContext).",
        String::new(),
    );
    builder.push(
        "system",
        "response_instructions",
        "Response instructions",
        "system",
        "generation-time",
        "Supplied per request (body.responseInstructions).",
        String::new(),
    );

    // --- Structured-output rules (when responseFormat = structured) ---------
    builder.push(
        "system",
        "structured_output",
        "Structured output rules",
        "system",
        "conditional",
        "Included when responseFormat = structured (the FNV live/headless path uses structured output).",
        STRUCTURED_OUTPUT_INSTRUCTION,
    );
    if has_quests {
        builder.push(
            "system",
            "structured_output_quest",
            "Quest structured-output rules",
            "system",
            "conditional",
            "Included when structured output is on and quest entries are active.",
            QUEST_BOOK_STRUCTURED_OUTPUT_INSTRUCTION,
        );
    }
    // --- TTS audio tags -----------------------------------------------------
    builder.push(
        "system",
        "audio_tags",
        "TTS audio tags",
        "system",
        "generation-time",
        "Appended when SillyTavern TTS audio tags are enabled for the active provider.",
        String::new(),
    );

    // --- Chat history (last 40, mapped to chat-completion roles) ------------
    let start = messages.len().saturating_sub(HISTORY_LIMIT);
    for (index, message) in messages[start..].iter().enumerate() {
        let role = match message.role.as_str() {
            "player" => "user",
            "system" => "system",
            _ => "assistant",
        };
        builder.push(
            "history",
            &format!("history_{index}"),
            &message.speaker_name,
            role,
            "included",
            "",
            message.content.clone(),
        );
    }
    let history_count = messages.len() - start;

    // --- Pending player turn ------------------------------------------------
    builder.push(
        "input",
        "pending_user",
        "Pending player turn",
        "user",
        "generation-time",
        "The next player message is appended here as role: user before sending.",
        String::new(),
    );

    notes.push(
        "Vector-activated entries (lore/action/quest) and chat-vector retrieval require the embeddings runtime and are not included here.".to_string(),
    );

    let system_char_count = builder
        .components
        .iter()
        .filter(|component| component.group == "system" && component.status == "included")
        .map(|component| component.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n")
        .chars()
        .count();
    let total_char_count = builder
        .components
        .iter()
        .filter(|component| component.status == "included")
        .map(|component| component.char_count)
        .sum();

    let view = PromptAssemblyView {
        participant_id: participant.id.clone(),
        participant_name: participant.name.clone(),
        character_id: participant.character_id.clone(),
        character_found,
        system_char_count,
        history_count,
        total_char_count,
        components: builder.components,
        notes,
    };
    timer.mark("spawn_catalogs+format");
    (view, injected)
}

/// Display id for a lore entry: the comment (its title in ST) when present, else
/// a stable `lore-<index>` derived from the lorebook order. Lore entries have no
/// persistent uid in the normalized form, so the index keeps it deterministic.
fn lore_entry_id(entry: &LoreEntry, index: usize) -> String {
    if entry.comment.trim().is_empty() {
        format!("lore-{index}")
    } else {
        entry.comment.trim().to_string()
    }
}

/// Display title for a lore entry: the comment, else a short content preview, else
/// the `lore-<index>` fallback so a row is never blank.
fn lore_entry_title(entry: &LoreEntry, index: usize) -> String {
    if !entry.comment.trim().is_empty() {
        return entry.comment.trim().to_string();
    }
    let preview: String = entry.content.trim().chars().take(60).collect();
    if preview.is_empty() {
        format!("lore-{index}")
    } else {
        preview
    }
}

/// Display id for a quest entry: the quest id, else its editor id, else its name.
fn quest_entry_id(entry: &QuestEntry) -> String {
    for value in [&entry.quest_id, &entry.quest_editor_id, &entry.quest_name] {
        if !value.trim().is_empty() {
            return value.trim().to_string();
        }
    }
    String::new()
}

/// Display title for a quest entry: the quest name, else its title, else its id.
fn quest_entry_title(entry: &QuestEntry) -> String {
    for value in [&entry.quest_name, &entry.title, &entry.quest_id] {
        if !value.trim().is_empty() {
            return value.trim().to_string();
        }
    }
    String::new()
}

/// Display title for an action entry: its title, else its action id.
fn action_entry_title(entry: &ActionEntry) -> String {
    if entry.title.trim().is_empty() {
        entry.action_id.clone()
    } else {
        entry.title.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(action_id: &str, title: &str) -> ActionEntry {
        ActionEntry {
            keys: Vec::new(),
            title: title.to_string(),
            description: String::new(),
            constant: false,
            disable: false,
            vectorized: true,
            order: 0.0,
            case_sensitive: None,
            action_id: action_id.to_string(),
            alias: None,
            short_name: None,
            risk_tier: String::new(),
            parameters_schema: Value::Null,
            preconditions: Vec::new(),
            effects: Vec::new(),
            examples_when_to_use: Vec::new(),
            examples_when_not_to_use: Vec::new(),
            vectorizable_text: String::new(),
            execution: Value::Null,
            binding: Value::Null,
            requires_target: false,
            scoped_catalogs: Vec::new(),
            scopes: Vec::new(),
        }
    }

    #[test]
    fn aliases_match_known_action_ids() {
        assert_eq!(
            structured_action_alias(&action("movement.follow_target", "Follow")),
            "follow"
        );
        assert_eq!(
            structured_action_alias(&action("npc.gesture_wave", "Wave")),
            "wave"
        );
        // Unknown id falls back to the slugified last dotted segment.
        assert_eq!(
            structured_action_alias(&action("world.open_gate", "Open gate")),
            "open_gate"
        );
    }

    #[test]
    fn keyword_matching_respects_constant_and_keys() {
        assert!(keyword_active(false, true, &[], None, "anything"));
        assert!(!keyword_active(true, true, &[], None, "anything"));
        assert!(keyword_active(
            false,
            false,
            &["Goodsprings".to_string()],
            None,
            "We met in goodsprings yesterday."
        ));
        assert!(!keyword_active(
            false,
            false,
            &["Vegas".to_string()],
            None,
            "We met in goodsprings yesterday."
        ));
    }

    #[test]
    fn action_prompt_uses_alias_and_fields() {
        let mut entry = action("movement.follow_target", "Follow target");
        entry.risk_tier = "medium".to_string();
        entry.description = "Follow the player.".to_string();
        let prompt = format_action_book_prompt(
            std::slice::from_ref(&entry),
            &std::collections::HashMap::new(),
        );
        assert!(prompt.starts_with("- follow => movement.follow_target: Follow target"));
        assert!(prompt.contains("Risk: medium"));
        assert!(prompt.contains("Description: Follow the player."));
    }

    #[test]
    fn action_scopes_gate_admin_only() {
        let mut spawn = action("world.spawn_entity", "Spawn");
        spawn.scopes = vec!["admin".to_string()];
        // STRICT: an admin-scoped action is filtered with no requested scopes (no
        // empty-requested loophole) and with non-admin scopes, but passes with admin.
        assert!(!action_passes_scopes(&spawn, &[]));
        assert!(!action_passes_scopes(
            &spawn,
            &["global".to_string(), "game:fnv".to_string()]
        ));
        assert!(action_passes_scopes(&spawn, &["admin".to_string()]));
        // An unscoped action is always available.
        let wave = action("npc.gesture_wave", "Wave");
        assert!(action_passes_scopes(&wave, &[]));
        assert!(action_passes_scopes(&wave, &["admin".to_string()]));
        // A `global`-scoped action passes for anyone.
        let mut anyone = action("x.y", "Y");
        anyone.scopes = vec!["global".to_string()];
        assert!(action_passes_scopes(&anyone, &[]));
    }

    #[test]
    fn strip_trigger_keys_drops_trigger_words() {
        let keys = vec!["spawn".to_string(), "summon".to_string()];
        assert_eq!(
            strip_trigger_keys("spawn a deathclaw on me", &keys),
            "a deathclaw on me"
        );
        // Case-insensitive + punctuation-tolerant; non-trigger words kept verbatim.
        assert_eq!(strip_trigger_keys("Summon, a Gecko!", &keys), "a Gecko!");
        // Falls back to the original when stripping empties the query.
        assert_eq!(strip_trigger_keys("spawn", &keys), "spawn");
    }

    fn participant(name: &str, character_id: Option<&str>) -> ParticipantView {
        ParticipantView {
            id: name.to_string(),
            name: name.to_string(),
            initial: String::new(),
            kind: "npc".to_string(),
            character_id: character_id.map(str::to_string),
            present: true,
            audible: true,
            distance: None,
            distance_label: String::new(),
            message_count: 0,
            selected: false,
        }
    }

    fn quest() -> QuestEntry {
        QuestEntry {
            keys: Vec::new(),
            title: String::new(),
            description: String::new(),
            constant: false,
            disable: false,
            priority: 0.0,
            case_sensitive: None,
            quest_id: String::new(),
            quest_name: String::new(),
            quest_editor_id: String::new(),
            giver_character_ids: Vec::new(),
            offer_summary: String::new(),
            pre_dialogue: String::new(),
            objectives: Vec::new(),
            acceptance_cues: Vec::new(),
            refusal_cues: Vec::new(),
            stage_hints: Vec::new(),
            quest_events: Vec::new(),
            phase: "available".to_string(),
            giver_npc_keys: Vec::new(),
            scopes: Vec::new(),
            tags: Vec::new(),
            include: Vec::new(),
            exclude: Vec::new(),
            target_game: String::new(),
            available_when: Value::Null,
            include_completed: false,
            vectorizable_text: String::new(),
        }
    }

    #[test]
    fn quest_giver_gate_matches_name_or_npc_key() {
        let speaker = participant("Sunny Smiles", Some("Sunny Smiles"));
        let other = participant("Easy Pete", Some("Easy Pete"));

        let mut q = quest();
        q.giver_character_ids = vec!["Sunny Smiles".to_string()];
        assert!(quest_giver_matches_speaker(&q, &speaker, ""));
        assert!(!quest_giver_matches_speaker(&q, &other, ""));

        // Slug npc key matches loosely against the speaker name.
        let mut q2 = quest();
        q2.giver_npc_keys = vec!["sunny_smiles".to_string()];
        assert!(quest_giver_matches_speaker(&q2, &speaker, ""));
        assert!(!quest_giver_matches_speaker(&q2, &other, ""));

        // No giver constraint => always relevant.
        assert!(quest_giver_matches_speaker(&quest(), &other, ""));
    }

    #[test]
    fn quest_scopes_include_exclude_phase_gates() {
        let mut q = quest();
        q.scopes = vec!["goodsprings".to_string()];
        assert!(quest_passes_scopes(&q, &[])); // no requested scopes => pass
        assert!(quest_passes_scopes(&q, &["goodsprings".to_string()]));
        assert!(!quest_passes_scopes(&q, &["vegas".to_string()]));
        q.scopes = vec!["global".to_string()];
        assert!(quest_passes_scopes(&q, &["vegas".to_string()])); // global always passes

        let mut q2 = quest();
        q2.include = vec!["dog".to_string()];
        assert!(quest_passes_include_exclude(&q2, "the lost dog ran off"));
        assert!(!quest_passes_include_exclude(&q2, "a cat appeared"));
        q2.exclude = vec!["cat".to_string()];
        assert!(!quest_passes_include_exclude(&q2, "the dog and the cat"));

        let mut q3 = quest();
        q3.phase = "completed".to_string();
        assert!(!quest_passes_phase(&q3));
        q3.include_completed = true;
        assert!(quest_passes_phase(&q3));
    }

    #[test]
    fn chat_vector_line_uses_name_then_role() {
        let mut msg = MessageView {
            id: String::new(),
            role: "player".to_string(),
            speaker_participant_id: None,
            speaker_name: String::new(),
            speaker_initial: String::new(),
            content: "Where's the doctor?".to_string(),
            created_at: None,
            created_at_label: String::new(),
            segment_id: None,
            location: None,
            audible_to: Vec::new(),
            visible_reason: String::new(),
            injected: None,
            turn_actions: Vec::new(),
        };
        // Empty name + player role => "user: ...".
        assert_eq!(format_chat_vector_line(&msg), "user: Where's the doctor?");
        msg.speaker_name = "Doc Mitchell".to_string();
        assert_eq!(
            format_chat_vector_line(&msg),
            "Doc Mitchell: Where's the doctor?"
        );
    }

    #[test]
    fn normalize_loose_collapses_separators() {
        assert_eq!(normalize_loose("Easy Pete"), "easy pete");
        assert_eq!(normalize_loose("easy_pete"), "easy pete");
        assert_eq!(normalize_loose("  Sunny-Smiles! "), "sunny smiles");
    }

    #[test]
    fn turn_actions_flatten_resolves_alias_and_params() {
        // Post-normalization structured output: actions are objects keyed by `id`.
        let structured = serde_json::json!({
            "speech": "Sure, I'll come along.",
            "actions": [
                { "id": "movement.follow_target", "target": "player", "parameters": {}, "reason": "Player asked to be followed." },
                { "id": "world.spawn_item", "parameters": { "count": 3 } },
                { "target": "x" }, // id-less -> skipped
            ],
        });
        let aliases = vec![
            ("movement.follow_target".to_string(), "follow".to_string()),
            ("world.spawn_item".to_string(), "spawn_item".to_string()),
        ];
        let views = turn_actions_from_structured(&structured, &aliases);
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].id, "movement.follow_target");
        assert_eq!(views[0].alias, "follow");
        assert_eq!(views[0].target, "player");
        // Empty parameters object is omitted.
        assert!(views[0].params.is_empty());
        assert_eq!(views[0].reason, "Player asked to be followed.");
        assert_eq!(views[1].id, "world.spawn_item");
        assert_eq!(views[1].alias, "spawn_item");
        assert_eq!(views[1].params, "{\"count\":3}");
    }

    /// A unique scratch dir under the OS temp root (no extra dev-deps).
    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("chasm-test-{tag}-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The minimal PNG the compat card reader accepts: signature + one `chara`
    /// tEXt chunk (base64-encoded card JSON) + IEND. CRCs are not validated by
    /// the reader, so they are written as zeros.
    fn png_card(json: &str) -> Vec<u8> {
        use base64::{engine::general_purpose::STANDARD, Engine};
        let mut data = Vec::new();
        data.extend_from_slice(b"chara");
        data.push(0);
        data.extend_from_slice(STANDARD.encode(json.as_bytes()).as_bytes());

        let mut out = Vec::new();
        out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(b"tEXt");
        out.extend_from_slice(&data);
        out.extend_from_slice(&[0, 0, 0, 0]); // tEXt CRC (unchecked)
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(b"IEND");
        out.extend_from_slice(&[0, 0, 0, 0]); // IEND CRC (unchecked)
        out
    }

    #[test]
    fn scenario_component_comes_from_global_template_not_card() {
        // A card WITH a scenario field on disk: the assembler must inject the
        // GLOBAL scenario at that slot and never the card's own text.
        let root = scratch_dir("global-scenario");
        let characters_dir = root.join("characters");
        std::fs::create_dir_all(&characters_dir).unwrap();
        let card_json = serde_json::json!({
            "name": "Sunny Smiles",
            "personality": "Friendly scout.",
            "scenario": "CARD SCENARIO — must never be injected",
            "mes_example": "Example line.",
        });
        std::fs::write(
            characters_dir.join("Sunny Smiles.png"),
            png_card(&card_json.to_string()),
        )
        .unwrap();

        let repo = LiveChatRepository::new(&root);
        let speaker = participant("Sunny Smiles", Some("Sunny Smiles"));

        // Live turn: the resolved global scenario fills the card-scenario slot.
        let (view, _) = assemble_prompt_with_retrieval_collect(
            &repo,
            &speaker,
            &[],
            "hi",
            "hi",
            &[],
            None,
            Some("It is night. You are in Goodsprings."),
        );
        let scenario = view
            .components
            .iter()
            .find(|component| component.key == "scenario")
            .expect("scenario component present");
        assert_eq!(scenario.status, "included");
        assert_eq!(
            scenario.content,
            "Scenario:\nIt is night. You are in Goodsprings."
        );
        // Same position in the ST assembly order as the old card field:
        // after Personality, before Example dialogue.
        fn index_of(view: &PromptAssemblyView, key: &str) -> usize {
            view.components
                .iter()
                .position(|component| component.key == key)
                .unwrap_or_else(|| panic!("component {key} present"))
        }
        assert!(index_of(&view, "personality") < index_of(&view, "scenario"));
        assert!(index_of(&view, "scenario") < index_of(&view, "example_dialogue"));
        // The per-character card scenario is tolerated on disk but NEVER injected.
        assert!(view
            .components
            .iter()
            .all(|component| !component.content.contains("CARD SCENARIO")));

        // Blank global scenario (cleared template / all-empty macros) -> the
        // component is omitted entirely, not injected as an empty block.
        let (view, _) = assemble_prompt_with_retrieval_collect(
            &repo,
            &speaker,
            &[],
            "hi",
            "hi",
            &[],
            None,
            Some("   "),
        );
        assert!(view
            .components
            .iter()
            .all(|component| component.key != "scenario"));

        // Static panel (no live turn): a correctly-ordered placeholder.
        let (view, _) = assemble_prompt_with_retrieval_collect(
            &repo, &speaker, &[], "hi", "hi", &[], None, None,
        );
        let placeholder = view
            .components
            .iter()
            .find(|component| component.key == "scenario")
            .expect("scenario placeholder present");
        assert_eq!(placeholder.status, "generation-time");
        assert!(placeholder.content.is_empty());
        assert!(index_of(&view, "personality") < index_of(&view, "scenario"));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn player_persona_injects_at_st_position_when_present() {
        // No persona store → no component. With a persona store → ONE
        // "player_persona" component carrying the description, placed inside
        // the character block (ST's story-string slot). The no-card path here
        // exercises the else-branch placement; the exact after-scenario slot
        // is the same push, gated on key == "scenario".
        let root = scratch_dir("persona");
        let repo = LiveChatRepository::new(&root);
        let speaker = participant("Sunny Smiles", None);

        let view = assemble_prompt(&repo, &speaker, &[]);
        assert!(
            !view
                .components
                .iter()
                .any(|component| component.key == "player_persona"),
            "no persona store → no persona component"
        );

        let persona_dir = root.join("headless").join("persona");
        std::fs::create_dir_all(&persona_dir).unwrap();
        std::fs::write(
            persona_dir.join("persona.json"),
            serde_json::json!({
                "description": "A wiry courier with sun-scarred skin.",
                "generated_at": "2026-07-02T00:00:00Z",
            })
            .to_string(),
        )
        .unwrap();

        let view = assemble_prompt(&repo, &speaker, &[]);
        let persona = view
            .components
            .iter()
            .find(|component| component.key == "player_persona")
            .expect("persona component injected");
        assert_eq!(persona.label, "Player persona");
        assert_eq!(persona.status, "included");
        assert!(persona
            .content
            .contains("A wiry courier with sun-scarred skin."));
        // Position: directly after the character block (the no-card
        // "Character" component here), before lore/actions/etc.
        let character_order = view
            .components
            .iter()
            .find(|component| component.key == "character")
            .expect("character component present")
            .order;
        assert_eq!(persona.order, character_order + 1);

        // Empty description → treated as absent (nothing injected).
        std::fs::write(
            persona_dir.join("persona.json"),
            serde_json::json!({ "description": "   " }).to_string(),
        )
        .unwrap();
        let view = assemble_prompt(&repo, &speaker, &[]);
        assert!(!view
            .components
            .iter()
            .any(|component| component.key == "player_persona"));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn assembly_returns_injected_action_entries_with_reason() {
        // Lay down a single action book with one CONSTANT (always-on) action and
        // one KEYWORD action; assert the assembler returns both injected entries,
        // each tagged with the correct activation reason, with no retrieval ctx.
        let root = scratch_dir("inject");
        let actions_dir = root.join("headless").join("action-books");
        std::fs::create_dir_all(&actions_dir).unwrap();
        let book = serde_json::json!({
            "id": "fnv",
            "name": "FNV Actions",
            "entries": {
                "0": {
                    "key": [],
                    "comment": "Wave hello",
                    "constant": true,
                    "actionId": "npc.gesture_wave",
                },
                "1": {
                    "key": ["follow"],
                    "comment": "Follow target",
                    "constant": false,
                    "actionId": "movement.follow_target",
                },
            },
        });
        std::fs::write(
            actions_dir.join("fnv.json"),
            serde_json::to_string(&book).unwrap(),
        )
        .unwrap();

        let repo = LiveChatRepository::new(&root);
        let speaker = participant("Sunny Smiles", None);
        let (_view, injected) = assemble_prompt_with_retrieval_collect(
            &repo,
            &speaker,
            &[],
            "Hey, please follow me to town.",
            "Hey, please follow me to town.",
            &[],
            None,
            None,
        );

        assert_eq!(injected.actions.len(), 2, "both actions injected");
        let wave = injected
            .actions
            .iter()
            .find(|entry| entry.id == "npc.gesture_wave")
            .expect("wave injected");
        assert_eq!(wave.source, "action");
        assert_eq!(wave.reason, "constant");
        assert_eq!(wave.title, "Wave hello");
        let follow = injected
            .actions
            .iter()
            .find(|entry| entry.id == "movement.follow_target")
            .expect("follow injected");
        assert_eq!(follow.reason, "keyword");
        // Lore + quests empty (none on disk) — the view is the same shape as before.
        assert!(injected.lore.is_empty());
        assert!(injected.quests.is_empty());

        std::fs::remove_dir_all(&root).ok();
    }

    // ---------------------------------------------------------------------------
    // Prompt-injection battery — verifies which LORE + ACTIONS keyword-activate for
    // a message, now that the retrieval scan is MESSAGE-ONLY (no gamestate). Mirrors
    // the assembler's activation: lore = !disable && character-filter && keyword_active;
    // action = !disable && scopes && keyword_active. Vector/semantic activation is a
    // separate path (needs the embedder) and is not exercised here.
    // ---------------------------------------------------------------------------

    fn lore(comment: &str, constant: bool, keys: &[&str]) -> LoreEntry {
        LoreEntry {
            keys: keys.iter().map(|k| k.to_string()).collect(),
            comment: comment.to_string(),
            content: String::new(),
            constant,
            disable: false,
            order: 100.0,
            case_sensitive: None,
            filter_names: Vec::new(),
            filter_exclude: false,
        }
    }

    /// The FNV baseline lorebook (chasm-bridge-fnv v0.2.4): 6 pre-game entries.
    fn fnv_lore() -> Vec<LoreEntry> {
        vec![
            lore("Goodsprings", true, &["Goodsprings", "Good Springs", "town", "local town", "here"]),
            lore("The Mojave Wasteland", false, &["Mojave Wasteland", "Mojave", "wasteland", "desert", "the wastes"]),
            lore("New California Republic (NCR)", false, &["NCR", "New California Republic", "Republic", "troopers"]),
            lore("Caesar's Legion", false, &["Caesar", "Caesar's Legion", "Legion", "slavers", "legionaries"]),
            lore("New Vegas, the Strip & Mr. House", false, &["New Vegas", "The Strip", "Vegas", "Lucky 38", "Mr House", "Mr. House", "Robert House", "Securitron", "securitrons"]),
            lore("The Great War & the Old World", false, &["Great War", "Old World", "pre-War", "prewar", "the bombs", "before the war", "old world"]),
        ]
    }

    fn active_lore<'a>(book: &'a [LoreEntry], msg: &str) -> Vec<&'a str> {
        book.iter()
            .filter(|e| {
                !e.disable
                    && lore_passes_character_filter(e, None)
                    && keyword_active(e.disable, e.constant, &e.keys, e.case_sensitive, msg)
            })
            .map(|e| e.comment.as_str())
            .collect()
    }

    fn act(id: &str, scopes: &[&str], keys: &[&str]) -> ActionEntry {
        let mut a = action(id, id);
        a.scopes = scopes.iter().map(|s| s.to_string()).collect();
        a.keys = keys.iter().map(|k| k.to_string()).collect();
        a
    }

    // The action's ALLOWED-scopes field (what the book declares) vs the turn's
    // REQUESTED scopes (what a player / admin turn actually carries).
    const ALLOW_PUBLIC: &[&str] = &["global", "admin", "godmode", "game:fallout-new-vegas"];
    const ALLOW_ADMIN: &[&str] = &["admin", "godmode"];
    const REQ_PLAYER: &[&str] = &["global", "game:fallout-new-vegas"];
    const REQ_ADMIN: &[&str] = &["admin", "game:fallout-new-vegas"];

    /// A representative slice of the FNV action book (real ids/keys/scopes).
    fn fnv_actions() -> Vec<ActionEntry> {
        vec![
            act("movement.follow_target", ALLOW_PUBLIC, &[r"\bfollow\b", "follow me", "come with", "escort"]),
            act("movement.stop_follow_target", ALLOW_PUBLIC, &["stop following", "stop follow", "dismiss"]),
            act("combat.start", ALLOW_PUBLIC, &["attack", "start combat", "hostile", "fight"]),
            act("combat.stop", ALLOW_PUBLIC, &["stop attacking", "stop fighting", "stand down"]),
            act("ai.wait_here", ALLOW_PUBLIC, &["wait here", "stay here", "stand still", "hold position"]),
            act("ai.sit_down", ALLOW_PUBLIC, &["sit down", "take a seat", "sit here"]),
            act("npc.gesture_wave", ALLOW_PUBLIC, &["wave", "wave hello", "say hello", "greet"]),
            act("world.spawn_entity", ALLOW_ADMIN, &["spawn npc", "spawn character", "spawn creature", "spawn entity"]),
        ]
    }

    fn active_actions<'a>(book: &'a [ActionEntry], msg: &str, scopes: &[&str]) -> Vec<&'a str> {
        let scopes: Vec<String> = scopes.iter().map(|s| s.to_string()).collect();
        book.iter()
            .filter(|a| {
                !a.disable
                    && action_passes_scopes(a, &scopes)
                    && keyword_active(a.disable, a.constant, &a.keys, a.case_sensitive, msg)
            })
            .map(|a| a.action_id.as_str())
            .collect()
    }

    #[test]
    fn lore_injection_battery_message_only() {
        let book = fnv_lore();
        // (message, expected active lore comments) — Goodsprings is constant so it's
        // always present; nothing else should fire unless the MESSAGE names it.
        let cases: &[(&str, &[&str])] = &[
            ("hi there", &["Goodsprings"]),
            // The original bug: a trivial goodbye pulled in Legion/Mojave/opinions.
            ("Bye, that made me laugh", &["Goodsprings"]),
            ("What do you know about the Legion?", &["Goodsprings", "Caesar's Legion"]),
            ("Are the NCR troopers close?", &["Goodsprings", "New California Republic (NCR)"]),
            ("Tell me about New Vegas", &["Goodsprings", "New Vegas, the Strip & Mr. House"]),
            ("Who runs the Lucky 38?", &["Goodsprings", "New Vegas, the Strip & Mr. House"]),
            ("The Mojave is a harsh place", &["Goodsprings", "The Mojave Wasteland"]),
            ("What was the Great War?", &["Goodsprings", "The Great War & the Old World"]),
            // A message naming two topics activates both (plus constant).
            ("Did the Legion ever fight the NCR?", &["Goodsprings", "New California Republic (NCR)", "Caesar's Legion"]),
        ];
        for (msg, expected) in cases {
            let got = active_lore(&book, msg);
            assert_eq!(&got, expected, "lore for message {msg:?}");
        }
    }

    #[test]
    fn action_injection_battery_keyword_and_scopes() {
        let book = fnv_actions();
        // Public-scoped player turn.
        let cases: &[(&str, &[&str])] = &[
            ("follow me", &["movement.follow_target"]),
            // \bfollow\b means "following" no longer falsely fires the follow action;
            // only stop_follow does. (Was the over-match with a plain "follow" key.)
            ("stop following me", &["movement.stop_follow_target"]),
            ("attack that guy", &["combat.start"]),
            ("wait here for me", &["ai.wait_here"]),
            ("give me a wave", &["npc.gesture_wave"]),
            // A plain hello does NOT fire the wave (keys are specific).
            ("hello", &[]),
        ];
        for (msg, expected) in cases {
            let got = active_actions(&book, msg, REQ_PLAYER);
            assert_eq!(&got, expected, "actions for message {msg:?}");
        }
        // Admin-only actions are gated: a spawn request from a normal player turn
        // activates NOTHING; the same request from an admin turn activates it.
        assert_eq!(active_actions(&book, "spawn npc over there", REQ_PLAYER), Vec::<&str>::new());
        assert_eq!(active_actions(&book, "spawn npc over there", REQ_ADMIN), vec!["world.spawn_entity"]);
    }
}
