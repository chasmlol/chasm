//! The skill executor — the runtime half of the self-improving-NPC system. When
//! the event log ingests a fresh batch (see [`crate::event_log::ingest_events`]),
//! each new event is matched against the enabled skills; every skill whose
//! trigger event matches — and whose owner actually WITNESSED it (the event's
//! `witnessedBy`/`witnesses` list) — FIRES, with a per-skill cooldown so a burst
//! of one event type cannot flood.
//!
//! A skill is a HABIT, not a fixed action: it carries the owner's first-person
//! INTENTION (`thought`). Firing does two things, ZERO plugin changes:
//!   1. Plants that intention into the owner's chat history as a private impulse
//!      (`witness::inject_skill_thought`) — passive context.
//!   2. Nudges the owner to take a TURN and act on it, freeform and adapting to
//!      the moment, via the idle-gated reaction queue
//!      (`witness::enqueue_skill_reaction`) — the NPC's own turn (find_action,
//!      gestures, a line, or nothing) decides what to do. No deterministic
//!      action is written.
//!
//! This module also owns the skill store's save-aware rollback (byte-copy
//! sidecars keyed by the save-sync checkpoint id, exactly like the scheduler
//! and journal stores).

use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use serde_json::Value;

use chasm_core::AppSettings;
use chasm_st_compat::Skill;

use crate::AppState;

// ---------------------------------------------------------------------------
// Save-aware rollback (sidecar keyed by the save-sync checkpoint id)
// ---------------------------------------------------------------------------

fn skills_store_path_at(content_root: &Path) -> std::path::PathBuf {
    content_root.join("headless").join("skills.json")
}

fn skills_checkpoint_path(content_root: &Path, checkpoint_id: &str) -> std::path::PathBuf {
    content_root
        .join("headless")
        .join("save-sync")
        .join("skills-checkpoints")
        .join(format!("{checkpoint_id}.json"))
}

/// Snapshot the skill store for a save checkpoint. A missing store writes an
/// EMPTY snapshot so a later restore correctly clears skills authored after
/// this checkpoint (rollback of a discarded branch).
pub fn checkpoint_skills_store(content_root: &Path, checkpoint_id: &str) {
    let dst = skills_checkpoint_path(content_root, checkpoint_id);
    if let Some(dir) = dst.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match std::fs::read(skills_store_path_at(content_root)) {
        Ok(bytes) => {
            let _ = std::fs::write(&dst, bytes);
        }
        Err(_) => {
            let _ = std::fs::write(&dst, b"{}");
        }
    }
    tracing::info!("skills: checkpointed store for {checkpoint_id}");
}

/// Restore the skill store from a checkpoint's sidecar on load. A missing
/// sidecar means the save predates any skill, so the live store is CLEARED.
pub fn restore_skills_store(content_root: &Path, checkpoint_id: &str) {
    let dst = skills_store_path_at(content_root);
    if let Some(dir) = dst.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match std::fs::read(skills_checkpoint_path(content_root, checkpoint_id)) {
        Ok(bytes) => {
            let _ = std::fs::write(&dst, bytes);
        }
        Err(_) => {
            tracing::info!("skills: cleared store (no sidecar for {checkpoint_id})");
            let _ = std::fs::write(&dst, b"{}");
        }
    }
}

// ---------------------------------------------------------------------------
// Matching + firing
// ---------------------------------------------------------------------------

/// Process-wide per-skill cooldown ledger: `skill id → last fire instant`.
fn cooldowns() -> &'static Mutex<HashMap<String, Instant>> {
    static C: OnceLock<Mutex<HashMap<String, Instant>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Match `events` against the enabled skills and fire the matches. Spawned
/// fire-and-forget from the event-log ingest so it never delays that response.
pub(crate) fn spawn_match(state: Arc<AppState>, events: Vec<Value>) {
    tokio::spawn(async move {
        // Blocking file I/O (store read + command writes) off the async pool.
        let _ = tokio::task::spawn_blocking(move || match_and_fire(&state, &events)).await;
    });
}

fn match_and_fire(state: &AppState, events: &[Value]) {
    let settings = AppSettings::load(&state.config.settings_path);
    if !settings.self_improvement.skill_execution_enabled {
        return;
    }
    let skills = match state.repository.read_skills() {
        Ok(store) => store.skills,
        Err(error) => {
            tracing::warn!(target: "chasm::skill_executor", error = %error, "skill store read failed");
            return;
        }
    };
    if skills.is_empty() {
        return;
    }
    let cooldown = Duration::from_secs(settings.self_improvement.effective_cooldown_secs());

    for event in events {
        let event_type = event.get("type").and_then(Value::as_str).unwrap_or("");
        if event_type.is_empty() {
            continue;
        }
        for skill in &skills {
            if !skill.enabled || !skill.trigger_event.eq_ignore_ascii_case(event_type) {
                continue;
            }
            if !filter_matches(skill.trigger_filter.as_deref(), event) {
                continue;
            }
            // Owner-witnessed gate: when the event carries a witness list (the
            // richer witness/triggers events do), the owner must actually have
            // seen it fire — more believable than radius-only, and it stops one
            // NPC's skill firing off an event they were nowhere near.
            if !owner_witnessed(skill, event) {
                tracing::debug!(target: "chasm::skill_executor", skill = %skill.id, "owner did not witness the trigger; skipped");
                continue;
            }
            if !cooldown_ready(&skill.id, cooldown) {
                tracing::debug!(target: "chasm::skill_executor", skill = %skill.id, "cooldown; skipped");
                continue;
            }
            fire_skill(state, skill, event);
        }
    }
}

/// True when the skill has never fired or its cooldown has elapsed. Records the
/// fire time as a side effect when it returns true.
fn cooldown_ready(skill_id: &str, cooldown: Duration) -> bool {
    let now = Instant::now();
    let mut map = match cooldowns().lock() {
        Ok(map) => map,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let Some(last) = map.get(skill_id) {
        if now.duration_since(*last) < cooldown {
            return false;
        }
    }
    map.insert(skill_id.to_string(), now);
    true
}

/// Whether the skill's owner witnessed the triggering event.
///
/// Prefers the event's `witnessedBy` (the EFFECTIVE list stamped at ingest —
/// after the sight/subject/companions-only filters), falling back to the raw
/// `witnesses` capture. When the event carries neither field (older events, or
/// event types the plugin does not attach witnesses to) the gate FAILS OPEN —
/// the plugin's own "owner must be loaded near the player to resolve" check is
/// then the only spatial guard, exactly as before witnessing existed.
///
/// Witness keys are native NPC keys — a SLUG of the NPC's name (the mod's
/// `Slugify`: lower-case, non-alphanumerics folded to `_`, e.g. `"Easy Pete"` →
/// `"easy_pete"`). The skill's `owner` is a character-card id and `owner_name`
/// its display name, so we compare on a NORMALIZED form (lower-case, only
/// ascii-alphanumerics kept) that collapses slug / id / display-name spelling
/// differences — the display name always normalizes to the same value as its
/// slug, so an NPC that witnessed the event matches whatever spelling the skill
/// stored.
fn owner_witnessed(skill: &Skill, event: &Value) -> bool {
    let has_witness_field = event.get("witnessedBy").and_then(Value::as_array).is_some()
        || event.get("witnesses").and_then(Value::as_array).is_some();
    if !has_witness_field {
        return true; // no witness data for this event — fail open.
    }
    matched_witness_key(skill, event).is_some()
}

/// The witness key (from `witnessedBy`, else the raw `witnesses`) that matches
/// the skill owner, or `None` when there is no witness data or no match. It IS
/// the exact native key the plugin resolves the NPC by, so it doubles as the
/// target key for the reaction turn.
fn matched_witness_key(skill: &Skill, event: &Value) -> Option<String> {
    let witnesses = event
        .get("witnessedBy")
        .and_then(Value::as_array)
        .or_else(|| event.get("witnesses").and_then(Value::as_array))?;
    let owner = normalize_npc_key(&skill.owner);
    let owner_name = normalize_npc_key(&skill.owner_name);
    if owner.is_empty() && owner_name.is_empty() {
        return None;
    }
    witnesses
        .iter()
        .filter_map(Value::as_str)
        .find(|w| {
            let n = normalize_npc_key(w);
            !n.is_empty() && (n == owner || n == owner_name)
        })
        .map(str::to_string)
}

/// Fold an NPC identity (native slug, card id, or display name) to a canonical
/// key for witness matching: lower-case, ascii-alphanumerics only. `"Easy
/// Pete"`, `"easy_pete"` and `"easy-pete"` all collapse to `"easypete"`.
fn normalize_npc_key(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// A `None` filter matches everything; otherwise the (case-insensitive)
/// substring must appear in the event's summary, location, or an actor name.
fn filter_matches(filter: Option<&str>, event: &Value) -> bool {
    let Some(filter) = filter.map(str::trim).filter(|f| !f.is_empty()) else {
        return true;
    };
    let needle = filter.to_lowercase();
    let hay = |v: Option<&str>| v.map(|s| s.to_lowercase().contains(&needle)).unwrap_or(false);
    if hay(event.get("summary").and_then(Value::as_str))
        || hay(event.get("location").and_then(Value::as_str))
    {
        return true;
    }
    event
        .get("actors")
        .and_then(Value::as_array)
        .map(|actors| {
            actors.iter().any(|a| {
                hay(a.get("name").and_then(Value::as_str)) || hay(a.as_str())
            })
        })
        .unwrap_or(false)
}

/// Fire a skill: plant the owner's INTENTION (their first-person `thought`) into
/// their chat history as a private impulse, then nudge them to take a turn and
/// act on it — freeform, adapting to the moment — via the idle-gated reaction
/// queue. No deterministic action is written; the NPC's own turn decides what to
/// do (find_action, a gesture, a line, or nothing if it doesn't suit them).
fn fire_skill(state: &AppState, skill: &Skill, event: &Value) {
    let owner = skill.owner.trim();
    let intention = skill.thought.trim();
    if owner.is_empty() || intention.is_empty() {
        return; // nothing to plant / act on
    }
    // Prefer the card display name (what the plugin resolves NPCs by), falling
    // back to the stored owner name / id.
    let display_name = state
        .repository
        .read_character_card(owner)
        .ok()
        .flatten()
        .map(|c| c.name)
        .filter(|n| !n.is_empty())
        .or_else(|| Some(skill.owner_name.clone()).filter(|n| !n.is_empty()))
        .unwrap_or_else(|| owner.to_string());

    // 1) Plant the intention as their own impulse — sits in history, passive.
    crate::witness::inject_skill_thought(state, owner, &display_name, intention);

    // 2) Nudge a turn to act on it now (idle-gated reaction). The plugin
    //    resolves the NPC by native key — use the exact witness key that matched
    //    the trigger, falling back to a name slug when the event carried no
    //    witness list.
    let native_key = matched_witness_key(skill, event).unwrap_or_else(|| slugify(&display_name));
    match crate::witness::enqueue_skill_reaction(state, &native_key, &display_name, intention) {
        Ok(true) => tracing::info!(
            target: "chasm::skill_executor",
            skill = %skill.id,
            owner = %display_name,
            trigger = %skill.trigger_event,
            "fired skill impulse (planted intention + nudged a turn)"
        ),
        Ok(false) => {}
        Err(error) => tracing::warn!(
            target: "chasm::skill_executor",
            skill = %skill.id,
            error = %error,
            "skill reaction enqueue failed"
        ),
    }
}

/// The mod's `Slugify`: lower-case, non-alphanumerics folded to a single `_`,
/// leading/trailing `_` trimmed — the native NPC key format, used as the
/// reaction target when an event carried no witness list to match against.
fn slugify(name: &str) -> String {
    let mut out = String::new();
    let mut last_underscore = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chasm_st_compat::SkillAction;
    use serde_json::json;

    fn skill(trigger: &str, filter: Option<&str>) -> Skill {
        Skill {
            id: "skill-0".into(),
            owner: "Easy Pete".into(),
            owner_name: "Easy Pete".into(),
            trigger_event: trigger.into(),
            trigger_filter: filter.map(str::to_string),
            actions: vec![SkillAction { action_id: "npc.gesture_pushups".into(), target: None }],
            note: String::new(),
            thought: String::new(),
            enabled: true,
            created_at: None,
            updated_at: None,
        }
    }

    #[test]
    fn filter_none_matches_any_event() {
        let event = json!({ "type": "weapon_fire", "summary": "Fired a shot" });
        assert!(filter_matches(skill("weapon_fire", None).trigger_filter.as_deref(), &event));
    }

    #[test]
    fn filter_matches_summary_location_or_actor_case_insensitively() {
        let event = json!({
            "type": "combat",
            "summary": "A fight broke out",
            "location": "Goodsprings Saloon",
            "actors": [{ "name": "Powder Ganger" }]
        });
        assert!(filter_matches(Some("saloon"), &event)); // location
        assert!(filter_matches(Some("FIGHT"), &event)); // summary, case-insensitive
        assert!(filter_matches(Some("powder"), &event)); // actor name
        assert!(!filter_matches(Some("brahmin"), &event)); // no match
    }

    #[test]
    fn owner_witnessed_gates_on_the_effective_or_raw_list() {
        let s = skill("weapon_fire", None); // owner + owner_name = "Easy Pete"
        // No witness data → fail open (fires; the plugin's spatial check guards).
        assert!(owner_witnessed(&s, &json!({ "type": "weapon_fire" })));
        // Owner is in the effective list (matched by name, case-insensitive).
        assert!(owner_witnessed(
            &s,
            &json!({ "type": "weapon_fire", "witnessedBy": ["sunny_smiles", "easy pete"] })
        ));
        // Owner is not in the effective list → gated.
        assert!(!owner_witnessed(
            &s,
            &json!({ "type": "weapon_fire", "witnessedBy": ["sunny_smiles"] })
        ));
        // witnessedBy present but empty (nobody saw it, e.g. hidden) → gated.
        assert!(!owner_witnessed(
            &s,
            &json!({ "type": "weapon_fire", "witnessedBy": [] })
        ));
        // Falls back to the raw `witnesses` capture when there is no witnessedBy.
        assert!(owner_witnessed(
            &s,
            &json!({ "type": "weapon_fire", "witnesses": ["Easy Pete"] })
        ));
    }

    #[test]
    fn owner_witnessed_matches_the_owner_id_key() {
        // A skill whose owner is a key that differs from the display name.
        let mut s = skill("weapon_fire", None);
        s.owner = "easy_pete".into();
        s.owner_name = "Easy Pete".into();
        assert!(owner_witnessed(
            &s,
            &json!({ "type": "weapon_fire", "witnessedBy": ["easy_pete"] })
        ));
    }

    #[test]
    fn cooldown_gates_repeat_fires() {
        let id = "cooldown-test-skill-unique";
        let cd = Duration::from_secs(60);
        assert!(cooldown_ready(id, cd)); // first fire allowed
        assert!(!cooldown_ready(id, cd)); // immediate refire blocked
        // A zero cooldown always allows (used when settings say 0 → default, but
        // the raw gate with Duration::ZERO must never block).
        assert!(cooldown_ready(id, Duration::ZERO));
    }

    #[test]
    fn slugify_matches_the_native_key_format() {
        assert_eq!(slugify("Easy Pete"), "easy_pete");
        assert_eq!(slugify("chamzy"), "chamzy");
        assert_eq!(slugify("  Mr. New Vegas!  "), "mr_new_vegas");
    }

    #[test]
    fn matched_witness_key_returns_the_native_key() {
        let s = skill("weapon", None); // owner + owner_name = "Easy Pete"
        assert_eq!(
            matched_witness_key(&s, &json!({ "witnessedBy": ["sunny_smiles", "easy_pete"] })),
            Some("easy_pete".to_string())
        );
        assert!(matched_witness_key(&s, &json!({ "witnessedBy": ["sunny_smiles"] })).is_none());
        // No witness field → no key (owner_witnessed fails open separately).
        assert!(matched_witness_key(&s, &json!({ "type": "weapon" })).is_none());
    }
}
