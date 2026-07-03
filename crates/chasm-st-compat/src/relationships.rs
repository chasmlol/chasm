//! The relationships store (`headless/relationships.json`): directional,
//! per-pair opinions each character holds about the player or another NPC,
//! written by the Gamemaster pass on every game save and read into the stable
//! head of that character's prompt.
//!
//! Shape mirrors the other headless stores (live-chats, globals): one JSON
//! file under the active profile's content root, camelCase keys, unknown keys
//! preserved through a read→write round-trip. One entry per directional pair
//! (`characters[characterId].entries[targetId]`) — the GM rewrites an entry in
//! place as events evolve; nothing is append-only.

use std::{collections::BTreeMap, fs};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{read_json_file, CompatError, LiveChatRepository, Result};

/// The stable key relationship entries use for the player target. The player
/// never has entries of their OWN (nothing is generated from the player's
/// perspective); they only ever appear as a target.
pub const PLAYER_TARGET_ID: &str = "player";

/// One directional relationship: how the owning character currently regards
/// `target` (short present-tense prose, rewritten — not appended — over time).
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RelationshipEntry {
    /// Display name of the target at last write (player name or NPC card name).
    #[serde(default)]
    pub target_name: String,
    /// `"player"` or `"npc"`.
    #[serde(default)]
    pub target_kind: String,
    /// The current stance, a few sentences of neutral-narrator prose.
    #[serde(default)]
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

/// All relationships one character holds, keyed by target id (an NPC's
/// character id, or [`PLAYER_TARGET_ID`]).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CharacterRelationships {
    #[serde(default)]
    pub entries: BTreeMap<String, RelationshipEntry>,
}

/// The whole store: per-character relationship maps plus the Gamemaster's
/// per-session watermarks (how many messages of each transcript session have
/// already been processed — the GM pass only reads content past its watermark).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct RelationshipsStore {
    /// Keyed by character id (the character-card id, e.g. `"Easy Pete"`).
    #[serde(default)]
    pub characters: BTreeMap<String, CharacterRelationships>,
    /// `session_id → messages already processed` for each transcript session
    /// the GM pass has seen. Advanced only after a successful pass.
    #[serde(default)]
    pub watermarks: BTreeMap<String, u64>,
    /// When the last GM pass completed (RFC3339), for the UI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_pass_at: Option<String>,
    /// Forward-compat: unknown keys survive a round-trip.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl RelationshipsStore {
    /// The entries character `character_id` holds, sorted by target id.
    /// Missing character → empty (the default state: nothing injected).
    pub fn entries_for(&self, character_id: &str) -> Vec<(&String, &RelationshipEntry)> {
        self.characters
            .get(character_id)
            .map(|c| c.entries.iter().collect())
            .unwrap_or_default()
    }

    /// Upserts one directional entry, preserving `created_at` on rewrite.
    /// An empty `text` removes the pair instead (and prunes the character's
    /// map when it was their last entry) — "cleared" and "never existed" are
    /// the same state, so cleared characters go back to injecting nothing.
    pub fn upsert(
        &mut self,
        character_id: &str,
        target_id: &str,
        target_name: &str,
        target_kind: &str,
        text: &str,
        now_iso: &str,
    ) {
        let text = text.trim();
        if text.is_empty() {
            if let Some(character) = self.characters.get_mut(character_id) {
                character.entries.remove(target_id);
                if character.entries.is_empty() {
                    self.characters.remove(character_id);
                }
            }
            return;
        }
        let character = self.characters.entry(character_id.to_string()).or_default();
        let entry = character.entries.entry(target_id.to_string()).or_default();
        if entry.created_at.is_none() {
            entry.created_at = Some(now_iso.to_string());
        }
        entry.target_name = target_name.to_string();
        entry.target_kind = target_kind.to_string();
        entry.text = text.to_string();
        entry.updated_at = Some(now_iso.to_string());
    }
}

impl LiveChatRepository {
    /// Path to the relationships store, resolved under the active profile's
    /// content root (`profiles/<id>/headless/relationships.json`, legacy
    /// data-root fallback) — the write-safe rule persona uses.
    pub fn relationships_store_path(&self) -> std::path::PathBuf {
        self.paths().relationships_store()
    }

    /// Reads the relationships store. A missing file is the pristine default
    /// (no relationships anywhere), not an error.
    pub fn read_relationships(&self) -> Result<RelationshipsStore> {
        let path = self.relationships_store_path();
        if !path.exists() {
            return Ok(RelationshipsStore::default());
        }
        read_json_file(&path)
    }

    /// Persists the relationships store, pretty-printed like the other
    /// headless stores.
    pub fn write_relationships(&self, store: &RelationshipsStore) -> Result<()> {
        let path = self.relationships_store_path();
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

    /// Reads, mutates, and writes the relationships store in one shot.
    pub fn update_relationships<T>(
        &self,
        mutate: impl FnOnce(&mut RelationshipsStore) -> T,
    ) -> Result<T> {
        let mut store = self.read_relationships()?;
        let out = mutate(&mut store);
        self.write_relationships(&store)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_repo(tag: &str) -> (std::path::PathBuf, LiveChatRepository) {
        let root = std::env::temp_dir().join(format!(
            "chasm-relationships-{tag}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let repo = LiveChatRepository::new(&root);
        (root, repo)
    }

    /// Full store round-trip: entries, watermarks, and unknown keys survive a
    /// write→read cycle byte-meaningfully; a missing file reads as the default.
    #[test]
    fn store_round_trips_through_disk() {
        let (root, repo) = temp_repo("roundtrip");

        // Missing file → pristine default.
        let fresh = repo.read_relationships().unwrap();
        assert!(fresh.characters.is_empty());
        assert!(fresh.watermarks.is_empty());

        let mut store = RelationshipsStore::default();
        store.upsert(
            "Easy Pete",
            PLAYER_TARGET_ID,
            "Courier",
            "player",
            "Pete finds the Courier level-headed after the dynamite talk.",
            "2026-07-02T10:00:00Z",
        );
        store.upsert(
            "Easy Pete",
            "Sunny Smiles",
            "Sunny Smiles",
            "npc",
            "Pete trusts Sunny's judgment around town defense.",
            "2026-07-02T10:00:00Z",
        );
        store.watermarks.insert("session-a".into(), 29);
        store.last_pass_at = Some("2026-07-02T10:00:00Z".into());
        store
            .extra
            .insert("futureKey".into(), serde_json::json!({ "keep": true }));
        repo.write_relationships(&store).unwrap();

        let back = repo.read_relationships().unwrap();
        let entries = back.entries_for("Easy Pete");
        assert_eq!(entries.len(), 2);
        let (_, player_entry) = entries
            .iter()
            .find(|(id, _)| id.as_str() == PLAYER_TARGET_ID)
            .unwrap();
        assert_eq!(player_entry.target_name, "Courier");
        assert_eq!(player_entry.target_kind, "player");
        assert_eq!(player_entry.created_at.as_deref(), Some("2026-07-02T10:00:00Z"));
        assert_eq!(back.watermarks.get("session-a"), Some(&29));
        assert_eq!(back.last_pass_at.as_deref(), Some("2026-07-02T10:00:00Z"));
        assert_eq!(back.extra["futureKey"]["keep"], serde_json::json!(true));

        // Unknown character → empty, no error (the default: inject nothing).
        assert!(back.entries_for("Trudy").is_empty());

        let _ = fs::remove_dir_all(&root);
    }

    /// Rewriting a pair keeps `created_at`, replaces the text, and bumps
    /// `updated_at`; clearing removes the pair and prunes the character.
    #[test]
    fn upsert_rewrites_in_place_and_clear_removes() {
        let mut store = RelationshipsStore::default();
        store.upsert("Trudy", PLAYER_TARGET_ID, "Courier", "player", "Wary.", "T1");
        store.upsert(
            "Trudy",
            PLAYER_TARGET_ID,
            "Courier",
            "player",
            "Warming up after the Cobb business.",
            "T2",
        );
        let entries = store.entries_for("Trudy");
        assert_eq!(entries.len(), 1);
        let entry = entries[0].1;
        assert_eq!(entry.text, "Warming up after the Cobb business.");
        assert_eq!(entry.created_at.as_deref(), Some("T1"));
        assert_eq!(entry.updated_at.as_deref(), Some("T2"));

        // Clearing (empty text) removes the pair AND the now-empty character.
        store.upsert("Trudy", PLAYER_TARGET_ID, "Courier", "player", "  ", "T3");
        assert!(store.entries_for("Trudy").is_empty());
        assert!(!store.characters.contains_key("Trudy"));
    }
}
