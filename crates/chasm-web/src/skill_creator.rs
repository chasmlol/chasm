//! The skill-creator pass — the persona-less curator of the self-improving-NPC
//! system, run right after the journal pass (chained from
//! [`crate::journal::spawn_pass`]).
//!
//! It is to skills what the Gamemaster is to relationships: a single
//! independent agent with NO character card of its own. It reads what each NPC
//! privately wrote in their journals and decides whether any NPC should START,
//! CHANGE, or STOP an automatic behaviour — a "skill" = owner + one trigger
//! event + one action, fired later with no LLM by [`crate::skill_executor`].
//!
//! One structured LLM call per NPC with NEW journal entries (the same
//! llama.cpp `response_format` enforcement NPC turns use), grammar-constrained
//! so a trigger is only ever one of the ALLOWED events and an action only ever
//! one of the ALLOWED actions. Deliberately conservative: it acts only on a
//! clearly repeated, settled intention, and does nothing for one-off moods.

use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicBool, Ordering},
    sync::Arc,
};

use serde_json::{json, Value};

use chasm_core::AppSettings;
use chasm_st_compat::{SkillOp, SkillOpKind};

use crate::AppState;

static RUNNING: AtomicBool = AtomicBool::new(false);

const CREATOR_MAX_TOKENS: i64 = 900;
/// How many of an NPC's newest journal entries to feed one pass (bounds the
/// prompt after a watermark reset / first run over a long journal).
const MAX_NEW_ENTRIES: usize = 12;

/// The ONLY game-event types a skill may trigger on — the merged event-log /
/// witness event vocabulary, led by the dedicated immediate `weapon_fire`
/// signal the mod emits from the reliable weapon-fire engine hook (the
/// `weapon_fire` is the immediate shot signal). `(type, description)` — the
/// description is shown to the model. Kept in one place so the executor and
/// creator agree on the vocabulary. This is the FULL event vocabulary the plugin
/// emits (not a curated subset) so the creator can always pick the event that
/// actually keeps happening instead of being forced onto a near-miss.
pub(crate) const ALLOWED_TRIGGER_EVENTS: &[(&str, &str)] = &[
    ("weapon_fire", "the player fires their weapon (immediately, once per shot or burst, out of combat)"),
    ("shooting", "the player takes a shot out of combat (a shot or burst)"),
    ("weapon", "the player draws or readies a weapon"),
    ("sneak", "the player starts sneaking"),
    ("combat", "a fight or combat encounter begins"),
    ("death", "someone dies nearby"),
    ("murder", "the player murders someone"),
    ("item", "the player picks up, buys, drops, equips, or uses an item"),
    ("theft", "the player steals something"),
    ("pickpocket", "the player pickpockets someone"),
    ("lockpick", "the player picks a lock"),
    ("hacking", "the player hacks a terminal"),
    ("location", "the player enters a new place"),
    ("arrival", "an NPC finishes traveling to meet the player"),
    ("trade", "the player buys or sells at a shop"),
    ("repair", "the player repairs equipment"),
    ("injury", "the player is badly injured (a limb is crippled)"),
    ("rads", "the player's radiation level rises"),
    ("day", "a new in-game day begins"),
    ("level", "the player levels up"),
    ("karma", "the player's karma shifts"),
    ("companion", "a companion joins or leaves the player"),
    ("quest", "a quest stage changes"),
    ("conversation", "a notable back-and-forth conversation happens"),
];

fn allowed_event_type(candidate: &str) -> bool {
    ALLOWED_TRIGGER_EVENTS
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case(candidate))
}

/// One allowed action the creator may attach to a skill: a safe, deterministic,
/// self-contained gesture from the Action Book.
#[derive(Debug, Clone)]
pub(crate) struct AllowedAction {
    pub action_id: String,
    pub title: String,
    pub description: String,
}

/// The action set the skill-creator may use: EVERY enabled entry in the Action
/// Book (gestures + the rest), so it can encode whatever response the character
/// resolved on — not just idle animations. Deduped by action id, sorted for a
/// stable prompt + grammar enum. (Gestures are self-contained and safest;
/// target-taking actions fire without an explicit target for now.)
pub(crate) fn allowed_actions(state: &AppState) -> Vec<AllowedAction> {
    let mut by_id: BTreeMap<String, AllowedAction> = BTreeMap::new();
    if let Ok(books) = state.repository.read_action_books() {
        for book in books {
            for entry in book.entries {
                let action_id = entry.action_id.trim();
                if action_id.is_empty() || entry.disable {
                    continue;
                }
                by_id.entry(action_id.to_string()).or_insert(AllowedAction {
                    action_id: action_id.to_string(),
                    title: entry.title.trim().to_string(),
                    description: entry.description.trim().to_string(),
                });
            }
        }
    }
    by_id.into_values().collect()
}

pub(crate) fn spawn_pass(state: Arc<AppState>) -> bool {
    if RUNNING.swap(true, Ordering::SeqCst) {
        return false;
    }
    tokio::spawn(async move {
        match run_pass(&state).await {
            Ok(applied) => tracing::info!(
                target: "chasm::skill_creator",
                applied,
                "skill-creator pass complete"
            ),
            Err(error) => tracing::warn!(
                target: "chasm::skill_creator",
                error = %error,
                "skill-creator pass failed"
            ),
        }
        RUNNING.store(false, Ordering::SeqCst);
    });
    true
}

/// True while a skill-creator pass is in flight (for the UI).
pub(crate) fn pass_in_flight() -> bool {
    RUNNING.load(Ordering::SeqCst)
}

async fn run_pass(state: &AppState) -> Result<usize, String> {
    let settings = AppSettings::load(&state.config.settings_path);
    if !settings.self_improvement.skill_creation_enabled {
        return Ok(0);
    }
    let repo = &state.repository;
    let journals = repo
        .read_journals()
        .map_err(|error| format!("journal store read failed: {error}"))?;
    let skills = repo
        .read_skills()
        .map_err(|error| format!("skill store read failed: {error}"))?;

    // Owners with journal entries past the skill-creator's cursor.
    let mut work: Vec<(String, String, usize)> = Vec::new(); // (owner, name, cursor)
    for (owner, journal) in &journals.characters {
        let cursor = skills.journal_cursors.get(owner).copied().unwrap_or(0);
        if journal.entries.len() > cursor {
            work.push((owner.clone(), journal.name.clone(), cursor));
        }
    }
    if work.is_empty() {
        return Ok(0);
    }

    let sampling = crate::llm::Sampling::from_settings(&settings.llm.sampling).with_overrides(
        crate::llm::GenerationOptions {
            temperature: Some(0.3),
            max_tokens: Some(CREATOR_MAX_TOKENS),
        },
    );
    let target = crate::llm::LlmTarget::resolve(&settings, &state.config);
    let response_format = operations_response_format();
    let now = crate::persona::chrono_now_iso();

    // Decide ops per owner up front (LLM calls), then apply them all in ONE
    // store write at the end so the store is touched once.
    let mut planned: Vec<(String, Vec<SkillOp>)> = Vec::new(); // (owner, ops)
    let mut cursors: BTreeMap<String, usize> = BTreeMap::new();
    for (owner, name, cursor) in &work {
        let display_name = if name.trim().is_empty() { owner.clone() } else { name.clone() };
        let entries = &journals.characters[owner].entries;
        let start = entries.len().saturating_sub(MAX_NEW_ENTRIES).max(*cursor);
        let new_entries: Vec<&str> = entries[start..].iter().map(|e| e.text.as_str()).collect();
        cursors.insert(owner.clone(), entries.len());
        if new_entries.is_empty() {
            continue;
        }

        let current = skills.skills_for(owner);
        let user_prompt = build_user_prompt(&display_name, &new_entries, &current);
        let messages = vec![
            json!({ "role": "system", "content": CREATOR_SYSTEM_PROMPT }),
            json!({ "role": "user", "content": user_prompt }),
        ];
        let content = match crate::llm::chat_completion_capturing_sampled(
            &target,
            &messages,
            Some(&response_format),
            sampling.clone(),
        )
        .await
        {
            Ok((content, _)) => content,
            Err(error) => {
                tracing::warn!(target: "chasm::skill_creator", owner = %display_name, error = %error, "creator LLM call failed; skipping this NPC");
                continue;
            }
        };
        let ops = parse_operations(&content, owner, &display_name);
        if !ops.is_empty() {
            planned.push((owner.clone(), ops));
        }
    }

    // --- Apply every op + advance every cursor in ONE store write. -----------
    let applied = repo
        .update_skills(|store| {
            let mut applied = 0usize;
            for (_owner, ops) in &planned {
                for op in ops {
                    match store.apply_op(op, &now) {
                        Ok(summary) => {
                            tracing::info!(target: "chasm::skill_creator", "{summary}");
                            applied += 1;
                        }
                        Err(reason) => {
                            tracing::info!(target: "chasm::skill_creator", "op rejected: {reason}");
                        }
                    }
                }
            }
            // Advance cursors for every owner considered this pass (even those
            // that produced no ops), so settled journals aren't re-read.
            for (owner, count) in &cursors {
                store.journal_cursors.insert(owner.clone(), *count);
            }
            store.last_pass_at = Some(now.clone());
            applied
        })
        .map_err(|error| format!("skill store write failed: {error}"))?;
    Ok(applied)
}

/// The skill-creator's rulebook. Persona-less and conservative. It authors a
/// character's HABITS: an event trigger + a first-person INTENTION (the
/// `thought`). It never picks a concrete action — the character acts the
/// intention out freely, in the moment, when it fires.
const CREATOR_SYSTEM_PROMPT: &str = "You manage the automatic HABITS of a cast of characters in a Fallout: New Vegas roleplay. You are NOT any of them — you read what one character privately wrote in their journal and decide whether they have settled into a habit worth making automatic.

A habit is: when a specific game EVENT happens, an INTENTION rises in the character — a first-person impulse, in their OWN voice, saying what they feel moved to do and why. It is NOT a fixed action or a game command and you do NOT choose any action; it is a natural-language thought the character will then act out however fits the moment. It is: owner + one trigger event + that thought.

You are given, for this one character: their newest journal entries and their current habits. You are also given the ONLY events that may be triggers — NEVER invent one.

Decide operations:
- CREATE when the journal shows the character has clearly noticed a RECURRING pattern and settled on how they mean to respond to it. Write the trigger event and the `thought` — their actual intention, personality included: if they resolved to go along with it, that; if they resolved to answer it their own way or dig in and refuse, THAT.
- EDIT an existing habit when a journal shows the same trigger should now stir a DIFFERENT intention (rewrite its thought).
- DELETE when the journal shows they have decided to STOP (e.g. they were asked to and agreed).
- Do NOTHING for one-off moods, vague feelings, or anything not clearly repeated and settled. An empty operations list is the correct and common answer.

Pick the trigger by what LITERALLY keeps happening in the journal — the most specific event the character actually reacted to. Never reach for a broad or umbrella event that only feels thematically related (e.g. do not choose a general \"combat\" or \"danger\" event when the concrete thing that recurred was the player drawing a weapon, entering a place, or picking something up). Match the exact recurring action.

Be conservative. One clear habit beats three speculative ones. For EDIT/DELETE, set skillId to an existing habit's id. Put a short human-facing reason in note.

The `thought` is the heart of it: the character's own intention in the FIRST PERSON, fully in their voice and personality — mirror how they wrote in their journal (devoted, resentful, proud, amused, whatever they are). One or two sentences, present tense, no game terms. It must read like a private impulse the character feels in the moment — what they intend to do and why — NOT a rule, a stage direction, or your description of them. It is slipped straight into their head each time the trigger fires, and they act on it.";

/// Grammar-constrained op list. `op` and `triggerEvent` are hard enums; the
/// model fills empty strings for fields an op doesn't use. There is no action —
/// the `thought` (a free-text first-person intention) is what fires.
fn operations_response_format() -> Value {
    let mut event_enum: Vec<Value> = vec![json!("")];
    event_enum.extend(ALLOWED_TRIGGER_EVENTS.iter().map(|(name, _)| json!(name)));

    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "chasm_skill_operations",
            "description": "Create/edit/delete operations for one character's automatic skills.",
            "strict": true,
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "operations": {
                        "type": "array",
                        "description": "The operations to apply. Empty when nothing should change.",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "op": { "type": "string", "enum": ["create", "edit", "delete"] },
                                "skillId": { "type": "string", "description": "Existing skill id for edit/delete; empty for create." },
                                "triggerEvent": { "type": "string", "enum": event_enum, "description": "The event that fires the habit (create/edit)." },
                                "note": { "type": "string", "description": "One-line reason, grounded in the journal (for the human-facing list)." },
                                "thought": { "type": "string", "description": "The character's own FIRST-PERSON intention — what they feel moved to do and why, in THEIR voice and personality (as they wrote in their journal). One or two sentences, present tense, no game terms; a private impulse, not a rule. This is slipped into their head each time the trigger fires and they act on it. Required for create/edit; empty for delete." }
                            },
                            "required": ["op", "skillId", "triggerEvent", "note", "thought"]
                        }
                    }
                },
                "required": ["operations"]
            }
        }
    })
}

fn build_user_prompt(
    name: &str,
    new_entries: &[&str],
    current: &[&chasm_st_compat::Skill],
) -> String {
    let entries = new_entries
        .iter()
        .map(|e| format!("- {}", e.trim()))
        .collect::<Vec<_>>()
        .join("\n");
    let skills = if current.is_empty() {
        "(none yet)".to_string()
    } else {
        current
            .iter()
            .map(|s| {
                format!(
                    "- id={} trigger={} enabled={} — thought: \"{}\"",
                    s.id, s.trigger_event, s.enabled, s.thought
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let events = ALLOWED_TRIGGER_EVENTS
        .iter()
        .map(|(name, desc)| format!("- {name}: {desc}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Character: {name}\n\n\
         Their newest journal entries (oldest first):\n{entries}\n\n\
         Their current habits:\n{skills}\n\n\
         ALLOWED trigger events (use only these):\n{events}\n\n\
         Return the operations that genuinely follow from what they wrote, or an empty list."
    )
}

/// Parse + validate the model's operations into structural [`SkillOp`]s. Only
/// ops that name an allowed event/action (for create/edit) and a plausible id
/// survive; the rest are dropped here (the store also re-checks id existence).
fn parse_operations(
    content: &str,
    owner: &str,
    owner_name: &str,
) -> Vec<SkillOp> {
    let Ok(value) = serde_json::from_str::<Value>(content.trim()) else {
        return Vec::new();
    };
    let Some(items) = value.get("operations").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut ops = Vec::new();
    for item in items {
        let op = item.get("op").and_then(Value::as_str).unwrap_or("").trim().to_lowercase();
        let skill_id = item.get("skillId").and_then(Value::as_str).unwrap_or("").trim().to_string();
        let event = item.get("triggerEvent").and_then(Value::as_str).unwrap_or("").trim().to_string();
        let note = item.get("note").and_then(Value::as_str).unwrap_or("").trim().to_string();
        let thought = item.get("thought").and_then(Value::as_str).unwrap_or("").trim().to_string();
        match op.as_str() {
            "create" => {
                // A habit needs an allowed trigger and an actual intention.
                if !allowed_event_type(&event) || thought.is_empty() {
                    continue;
                }
                ops.push(SkillOp {
                    kind: SkillOpKind::Create,
                    skill_id: None,
                    owner: owner.to_string(),
                    owner_name: owner_name.to_string(),
                    trigger_event: event,
                    trigger_filter: None,
                    actions: Vec::new(),
                    note,
                    thought,
                });
            }
            "edit" => {
                if skill_id.is_empty() {
                    continue;
                }
                // The trigger is optional on edit but, when present, must be allowed.
                if !event.is_empty() && !allowed_event_type(&event) {
                    continue;
                }
                ops.push(SkillOp {
                    kind: SkillOpKind::Edit,
                    skill_id: Some(skill_id),
                    owner: owner.to_string(),
                    owner_name: owner_name.to_string(),
                    trigger_event: event,
                    trigger_filter: None,
                    actions: Vec::new(),
                    note,
                    thought,
                });
            }
            "delete" => {
                if skill_id.is_empty() {
                    continue;
                }
                ops.push(SkillOp {
                    kind: SkillOpKind::Delete,
                    skill_id: Some(skill_id),
                    owner: owner.to_string(),
                    owner_name: owner_name.to_string(),
                    trigger_event: String::new(),
                    trigger_filter: None,
                    actions: Vec::new(),
                    note,
                    thought,
                });
            }
            _ => {}
        }
    }
    ops
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_needs_an_allowed_event_and_a_thought() {
        let content = r#"{"operations":[
            {"op":"create","skillId":"","triggerEvent":"weapon","note":"drew his gun again","thought":"He's got his weapon out — down I go for my push-ups, just as he likes."},
            {"op":"create","skillId":"","triggerEvent":"weapon","note":"no intention","thought":""},
            {"op":"create","skillId":"","triggerEvent":"explosion","note":"invented event","thought":"boom"}
        ]}"#;
        let ops = parse_operations(content, "Pete", "Pete");
        // Only the first survives (allowed event + a non-empty thought).
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].kind, SkillOpKind::Create);
        assert_eq!(ops[0].trigger_event, "weapon");
        assert!(ops[0].thought.starts_with("He's got his weapon out"));
        assert!(ops[0].actions.is_empty());
        assert_eq!(ops[0].owner, "Pete");
    }

    #[test]
    fn edit_and_delete_need_a_skill_id() {
        let content = r#"{"operations":[
            {"op":"edit","skillId":"skill-0","triggerEvent":"","note":"different intention","thought":"I'll have a smoke instead."},
            {"op":"edit","skillId":"","triggerEvent":"","note":"no id","thought":"x"},
            {"op":"delete","skillId":"skill-1","triggerEvent":"","note":"asked to stop","thought":""},
            {"op":"delete","skillId":"","triggerEvent":"","note":"no id","thought":""}
        ]}"#;
        let ops = parse_operations(content, "Pete", "Pete");
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].kind, SkillOpKind::Edit);
        assert_eq!(ops[0].skill_id.as_deref(), Some("skill-0"));
        assert_eq!(ops[0].thought, "I'll have a smoke instead.");
        assert_eq!(ops[1].kind, SkillOpKind::Delete);
        assert_eq!(ops[1].skill_id.as_deref(), Some("skill-1"));
    }

    #[test]
    fn junk_and_empty_yield_no_ops() {
        assert!(parse_operations("not json", "Pete", "Pete").is_empty());
        assert!(parse_operations(r#"{"operations":[]}"#, "Pete", "Pete").is_empty());
    }

    #[test]
    fn allowed_event_type_is_case_insensitive_and_bounded() {
        assert!(allowed_event_type("weapon_fire"));
        assert!(allowed_event_type("WEAPON_FIRE"));
        assert!(!allowed_event_type("teleport"));
    }
}
