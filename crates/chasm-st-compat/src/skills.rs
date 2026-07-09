//! The skill store (`headless/skills.json`): event-triggered NPC "skills"
//! authored by the skill-creator pass and fired, with NO LLM, by the skill
//! executor when a matching game event arrives.
//!
//! A skill is a tiny automation: **owner + one trigger event + one or more
//! actions**. When the trigger event happens in-game the owner performs the
//! actions the instant it fires, no conversation involved — e.g. "when the
//! player fires their weapon, Easy Pete does push-ups". The skill-creator pass
//! (see the `chasm-web` `skill_creator` module) reads NPC journals and decides
//! which skills to CREATE, EDIT, or DELETE; this module is the durable store
//! plus the structural op application (id assignment, in-place edit, delete).
//!
//! Shape mirrors the other headless stores (relationships, scheduler): one JSON
//! file under the active profile's content root, camelCase keys, unknown keys
//! preserved through a read→write round-trip. Save-aware: the whole store rolls
//! back with the save exactly like the scheduler store.

use std::{collections::BTreeMap, fs};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{read_json_file, CompatError, LiveChatRepository, Result};

/// One action a skill performs — an Action-Book action id (usually a
/// `npc.gesture_*`) plus an optional explicit target for target-taking actions
/// (gestures leave it `None`; they act on the owner themselves).
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SkillAction {
    /// Action-Book `actionId`, e.g. `"npc.gesture_pushups"`.
    #[serde(default)]
    pub action_id: String,
    /// Explicit target for target-taking actions (e.g. an NPC name); `None`
    /// for self-gestures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
}

/// One event-triggered skill owned by a single character.
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Skill {
    /// Stable id (`skill-<n>`), assigned on create; never reused after delete.
    #[serde(default)]
    pub id: String,
    /// Owning character id (the character-card id, e.g. `"Easy Pete"`).
    #[serde(default)]
    pub owner: String,
    /// Owner display name at last write (card name, else the id).
    #[serde(default)]
    pub owner_name: String,
    /// The game-event type that fires this skill (one of the allowed events,
    /// e.g. `"weapon_fire"`, `"combat"`, `"death"`).
    #[serde(default)]
    pub trigger_event: String,
    /// Optional case-insensitive substring the event's summary / location /
    /// actor names must contain for the skill to fire. `None` = fire on any
    /// event of the trigger type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_filter: Option<String>,
    /// The actions performed, in order, when the skill fires.
    #[serde(default)]
    pub actions: Vec<SkillAction>,
    /// The skill-creator's short rationale / the journal observation behind it
    /// (shown on the Skills page; never sent in-game).
    #[serde(default)]
    pub note: String,
    /// The owner's own FIRST-PERSON thought about why they do this, in their
    /// voice (e.g. "The moment he draws his weapon, down I go — it's what he
    /// wants of me."). Injected into their chat history each time the skill
    /// fires, so their next words know WHY they acted. Authored by the
    /// skill-creator from the character's journal.
    #[serde(default)]
    pub thought: String,
    /// When off, the executor ignores the skill (user or skill-creator paused
    /// it) but it is kept for context/history.
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

fn default_true() -> bool {
    true
}

/// The whole skill store: the flat skill list, a monotonic id counter, the
/// skill-creator's per-owner journal cursors (how many of each owner's journal
/// entries it has already considered), and a last-run marker.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SkillsStore {
    #[serde(default)]
    pub skills: Vec<Skill>,
    /// Monotonic id source: only ever increases, so a deleted skill's id is
    /// never handed to a later create.
    #[serde(default)]
    pub next_seq: u64,
    /// `owner → journal entries already considered by the skill-creator`. Only
    /// journal entries past this cursor are fed to a pass, so a settled skill
    /// is not re-proposed every save.
    #[serde(default)]
    pub journal_cursors: BTreeMap<String, usize>,
    /// When the last skill-creator pass completed (RFC3339), for the UI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_pass_at: Option<String>,
    /// Forward-compat: unknown keys survive a round-trip.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The three operations the skill-creator can request.
#[derive(Debug, Clone, PartialEq)]
pub enum SkillOpKind {
    Create,
    Edit,
    Delete,
}

/// A structural op the store applies. Enum validation (trigger ∈ allowed
/// events, action ids ∈ allowed actions, owner ∈ known NPCs) is the pass's job;
/// the store only applies the op structurally and validates id existence.
#[derive(Debug, Clone)]
pub struct SkillOp {
    pub kind: SkillOpKind,
    /// Target skill id for EDIT / DELETE (ignored for CREATE).
    pub skill_id: Option<String>,
    pub owner: String,
    pub owner_name: String,
    pub trigger_event: String,
    pub trigger_filter: Option<String>,
    pub actions: Vec<SkillAction>,
    pub note: String,
    /// The owner's first-person thought (see [`Skill::thought`]).
    pub thought: String,
}

impl SkillsStore {
    /// The skills owned by `owner`, in stored order.
    pub fn skills_for(&self, owner: &str) -> Vec<&Skill> {
        self.skills.iter().filter(|s| s.owner == owner).collect()
    }

    fn find_index(&self, skill_id: &str) -> Option<usize> {
        self.skills.iter().position(|s| s.id == skill_id)
    }

    /// Applies one op, returning a short human description of what changed
    /// (`Ok`) or a reason it was rejected (`Err`). Rejections are non-fatal:
    /// the caller logs them and moves on to the next op.
    pub fn apply_op(&mut self, op: &SkillOp, now_iso: &str) -> std::result::Result<String, String> {
        match op.kind {
            SkillOpKind::Create => {
                if op.trigger_event.trim().is_empty() {
                    return Err("create with no trigger event".to_string());
                }
                if op.thought.trim().is_empty() {
                    return Err("create with no intention (thought)".to_string());
                }
                let id = format!("skill-{}", self.next_seq);
                self.next_seq += 1;
                let skill = Skill {
                    id: id.clone(),
                    owner: op.owner.clone(),
                    owner_name: op.owner_name.clone(),
                    trigger_event: op.trigger_event.clone(),
                    trigger_filter: normalize_filter(&op.trigger_filter),
                    actions: op.actions.clone(),
                    note: op.note.clone(),
                    thought: op.thought.trim().to_string(),
                    enabled: true,
                    created_at: Some(now_iso.to_string()),
                    updated_at: Some(now_iso.to_string()),
                };
                self.skills.push(skill);
                Ok(format!("created {id} for {}", op.owner))
            }
            SkillOpKind::Edit => {
                let skill_id = op
                    .skill_id
                    .as_deref()
                    .filter(|id| !id.trim().is_empty())
                    .ok_or_else(|| "edit with no skill id".to_string())?;
                let index = self
                    .find_index(skill_id)
                    .ok_or_else(|| format!("edit of unknown skill {skill_id}"))?;
                // Only the owner may edit their own skill (defense against a
                // cross-owner id collision in a global pass).
                if !op.owner.is_empty() && self.skills[index].owner != op.owner {
                    return Err(format!("edit of {skill_id} not owned by {}", op.owner));
                }
                let skill = &mut self.skills[index];
                if !op.trigger_event.trim().is_empty() {
                    skill.trigger_event = op.trigger_event.clone();
                }
                // A filter is replaced only when the op carried one; an explicit
                // empty string clears it.
                if op.trigger_filter.is_some() {
                    skill.trigger_filter = normalize_filter(&op.trigger_filter);
                }
                if !op.actions.is_empty() && op.actions.iter().any(|a| !a.action_id.trim().is_empty())
                {
                    skill.actions = op.actions.clone();
                }
                if !op.note.trim().is_empty() {
                    skill.note = op.note.clone();
                }
                if !op.thought.trim().is_empty() {
                    skill.thought = op.thought.trim().to_string();
                }
                skill.updated_at = Some(now_iso.to_string());
                Ok(format!("edited {skill_id}"))
            }
            SkillOpKind::Delete => {
                let skill_id = op
                    .skill_id
                    .as_deref()
                    .filter(|id| !id.trim().is_empty())
                    .ok_or_else(|| "delete with no skill id".to_string())?;
                let index = self
                    .find_index(skill_id)
                    .ok_or_else(|| format!("delete of unknown skill {skill_id}"))?;
                if !op.owner.is_empty() && self.skills[index].owner != op.owner {
                    return Err(format!("delete of {skill_id} not owned by {}", op.owner));
                }
                self.skills.remove(index);
                Ok(format!("deleted {skill_id}"))
            }
        }
    }
}

/// Trims a filter and folds a blank one to `None` ("no filter").
fn normalize_filter(filter: &Option<String>) -> Option<String> {
    filter
        .as_deref()
        .map(str::trim)
        .filter(|f| !f.is_empty())
        .map(str::to_string)
}

impl LiveChatRepository {
    /// Path to the skill store, resolved under the active profile's content
    /// root (`profiles/<id>/headless/skills.json`, legacy data-root fallback).
    pub fn skills_store_path(&self) -> std::path::PathBuf {
        self.paths().skills_store()
    }

    /// Reads the skill store. A missing file is the pristine default (no
    /// skills anywhere), not an error.
    pub fn read_skills(&self) -> Result<SkillsStore> {
        let path = self.skills_store_path();
        if !path.exists() {
            return Ok(SkillsStore::default());
        }
        read_json_file(&path)
    }

    /// Persists the skill store, pretty-printed like the other headless stores.
    pub fn write_skills(&self, store: &SkillsStore) -> Result<()> {
        let path = self.skills_store_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| CompatError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let text = serde_json::to_string_pretty(store).map_err(|source| CompatError::Json {
            path: path.clone(),
            source,
        })?;
        fs::write(&path, text).map_err(|source| CompatError::Io {
            path: path.clone(),
            source,
        })
    }

    /// Reads, mutates, and writes the skill store in one shot.
    pub fn update_skills<T>(&self, mutate: impl FnOnce(&mut SkillsStore) -> T) -> Result<T> {
        let mut store = self.read_skills()?;
        let out = mutate(&mut store);
        self.write_skills(&store)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_repo(tag: &str) -> (std::path::PathBuf, LiveChatRepository) {
        let root = std::env::temp_dir().join(format!("chasm-skills-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let repo = LiveChatRepository::new(&root);
        (root, repo)
    }

    fn create_op(owner: &str, event: &str, action_id: &str) -> SkillOp {
        SkillOp {
            kind: SkillOpKind::Create,
            skill_id: None,
            owner: owner.to_string(),
            owner_name: owner.to_string(),
            trigger_event: event.to_string(),
            trigger_filter: None,
            actions: vec![SkillAction {
                action_id: action_id.to_string(),
                target: None,
            }],
            note: "noticed a pattern".to_string(),
            thought: "and so I do it".to_string(),
        }
    }

    /// Round-trip: skills, the id counter, cursors, and unknown keys survive a
    /// write→read cycle; a missing file reads as the default.
    #[test]
    fn store_round_trips_through_disk() {
        let (root, repo) = temp_repo("roundtrip");
        let fresh = repo.read_skills().unwrap();
        assert!(fresh.skills.is_empty());

        let mut store = SkillsStore::default();
        store.apply_op(&create_op("Easy Pete", "weapon_fire", "npc.gesture_pushups"), "T1").unwrap();
        store.journal_cursors.insert("Easy Pete".into(), 3);
        store.last_pass_at = Some("T1".into());
        store.extra.insert("futureKey".into(), serde_json::json!(1));
        repo.write_skills(&store).unwrap();

        let back = repo.read_skills().unwrap();
        assert_eq!(back.skills.len(), 1);
        assert_eq!(back.skills[0].trigger_event, "weapon_fire");
        assert_eq!(back.skills[0].actions[0].action_id, "npc.gesture_pushups");
        assert_eq!(back.skills[0].thought, "and so I do it");
        assert!(back.skills[0].enabled);
        assert_eq!(back.next_seq, 1);
        assert_eq!(back.journal_cursors.get("Easy Pete"), Some(&3));
        assert_eq!(back.extra["futureKey"], serde_json::json!(1));

        let _ = fs::remove_dir_all(&root);
    }

    /// CREATE assigns a fresh non-reused id and stamps timestamps.
    #[test]
    fn create_assigns_ids_monotonically() {
        let mut store = SkillsStore::default();
        store.apply_op(&create_op("Pete", "weapon_fire", "npc.gesture_pushups"), "T1").unwrap();
        store.apply_op(&create_op("Sunny", "death", "npc.gesture_cry"), "T1").unwrap();
        assert_eq!(store.skills[0].id, "skill-0");
        assert_eq!(store.skills[1].id, "skill-1");

        // Delete the first, create a third: the third gets skill-2, NOT the
        // freed skill-0 (monotonic ids).
        store
            .apply_op(
                &SkillOp {
                    kind: SkillOpKind::Delete,
                    skill_id: Some("skill-0".into()),
                    owner: "Pete".into(),
                    owner_name: "Pete".into(),
                    trigger_event: String::new(),
                    trigger_filter: None,
                    actions: vec![],
                    note: String::new(),
                    thought: String::new(),
                },
                "T2",
            )
            .unwrap();
        store.apply_op(&create_op("Trudy", "combat", "npc.gesture_cower"), "T2").unwrap();
        assert!(store.find_index("skill-0").is_none());
        assert_eq!(store.skills.last().unwrap().id, "skill-2");
    }

    /// EDIT swaps the action in place, keeps created_at, bumps updated_at, and
    /// rejects a cross-owner or unknown-id edit.
    #[test]
    fn edit_swaps_action_in_place() {
        let mut store = SkillsStore::default();
        store.apply_op(&create_op("Pete", "weapon_fire", "npc.gesture_pushups"), "T1").unwrap();

        let edit = SkillOp {
            kind: SkillOpKind::Edit,
            skill_id: Some("skill-0".into()),
            owner: "Pete".into(),
            owner_name: "Pete".into(),
            trigger_event: String::new(),
            trigger_filter: None,
            actions: vec![SkillAction {
                action_id: "npc.gesture_smoke".into(),
                target: None,
            }],
            note: "he asked me to smoke instead".into(),
            thought: "I'll have a smoke instead".into(),
        };
        assert!(store.apply_op(&edit, "T2").is_ok());
        let skill = &store.skills[0];
        assert_eq!(skill.actions[0].action_id, "npc.gesture_smoke");
        assert_eq!(skill.trigger_event, "weapon_fire"); // untouched
        assert_eq!(skill.created_at.as_deref(), Some("T1"));
        assert_eq!(skill.updated_at.as_deref(), Some("T2"));
        assert_eq!(skill.note, "he asked me to smoke instead");
        assert_eq!(skill.thought, "I'll have a smoke instead");

        // Unknown id and cross-owner edit both reject.
        let mut bad = edit.clone();
        bad.skill_id = Some("skill-99".into());
        assert!(store.apply_op(&bad, "T3").is_err());
        let mut wrong_owner = edit.clone();
        wrong_owner.owner = "Sunny".into();
        assert!(store.apply_op(&wrong_owner, "T3").is_err());
    }

    /// DELETE removes the skill and rejects unknown ids.
    #[test]
    fn delete_removes_skill() {
        let mut store = SkillsStore::default();
        store.apply_op(&create_op("Pete", "weapon_fire", "npc.gesture_pushups"), "T1").unwrap();
        let del = SkillOp {
            kind: SkillOpKind::Delete,
            skill_id: Some("skill-0".into()),
            owner: "Pete".into(),
            owner_name: "Pete".into(),
            trigger_event: String::new(),
            trigger_filter: None,
            actions: vec![],
            note: String::new(),
            thought: String::new(),
        };
        assert!(store.apply_op(&del, "T2").is_ok());
        assert!(store.skills.is_empty());
        // Deleting again → error (already gone).
        assert!(store.apply_op(&del, "T3").is_err());
    }
}
