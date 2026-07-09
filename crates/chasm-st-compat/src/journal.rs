//! The journal store (`headless/journals.json`): each NPC's private,
//! APPEND-ONLY journal, written by the journal pass on every game save (right
//! after the Gamemaster relationships pass, see `crate::RelationshipsStore` and
//! the `chasm-web` `journal` module).
//!
//! Unlike the relationships store — whose entries the GM rewrites in place — a
//! journal is only ever ADDED to: each pass appends at most one new entry per
//! NPC and never touches earlier ones. This is the raw material the
//! skill-creator pass reads to decide which automatic behaviours ("skills") an
//! NPC should start, change, or stop.
//!
//! Shape mirrors the other headless stores (relationships, scheduler): one JSON
//! file under the active profile's content root, camelCase keys, unknown keys
//! preserved through a read→write round-trip. Per-session watermarks (how many
//! transcript messages of each session the pass has already read) are the
//! journal's OWN — independent of the GM's — so the two passes advance
//! separately even though they run back to back.

use std::{collections::BTreeMap, fs};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{read_json_file, CompatError, LiveChatRepository, Result};

/// A single journal entry — one NPC's inner-voice reflection written by one
/// pass. Never edited once written; the journal only grows.
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct JournalEntry {
    /// RFC3339 wall-clock time the entry was written.
    #[serde(default)]
    pub created_at: String,
    /// Pre-formatted in-game clock at write time, when available (from the
    /// event log / heartbeat). Purely for display; never parsed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub game_time: Option<String>,
    /// In-game day counter at write time, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub game_day: Option<i64>,
    /// The entry text — the NPC's own words about what happened since the last
    /// entry and any patterns they noticed.
    #[serde(default)]
    pub text: String,
}

/// One NPC's whole journal: their display name at last write plus the
/// append-only list of entries (oldest first).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CharacterJournal {
    /// Display name of the character at last write (card name, else the id).
    #[serde(default)]
    pub name: String,
    /// Every entry ever written, oldest first. Append-only.
    #[serde(default)]
    pub entries: Vec<JournalEntry>,
}

/// The whole journal store: per-character journals plus the pass's per-session
/// watermarks and last-run marker.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct JournalStore {
    /// Keyed by character id (the character-card id, e.g. `"Easy Pete"`).
    #[serde(default)]
    pub characters: BTreeMap<String, CharacterJournal>,
    /// `session_id → messages already read` for each transcript session the
    /// journal pass has processed. Advanced only after a successful pass.
    /// Independent of the GM's watermarks (a different store).
    #[serde(default)]
    pub watermarks: BTreeMap<String, u64>,
    /// When the last journal pass completed (RFC3339), for the UI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_pass_at: Option<String>,
    /// Forward-compat: unknown keys survive a round-trip.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl JournalStore {
    /// The entries character `character_id` holds, oldest first. Missing
    /// character → empty (the default state: nothing written yet).
    pub fn entries_for(&self, character_id: &str) -> &[JournalEntry] {
        self.characters
            .get(character_id)
            .map(|c| c.entries.as_slice())
            .unwrap_or(&[])
    }

    /// Appends one entry to `character_id`'s journal, creating the journal on
    /// first write. APPEND-ONLY: earlier entries are never touched. A blank
    /// `text` is ignored (the pass chose to write nothing this time).
    pub fn append(&mut self, character_id: &str, name: &str, entry: JournalEntry) {
        if entry.text.trim().is_empty() {
            return;
        }
        let journal = self.characters.entry(character_id.to_string()).or_default();
        if !name.trim().is_empty() {
            journal.name = name.to_string();
        }
        journal.entries.push(entry);
    }
}

impl LiveChatRepository {
    /// Path to the journal store, resolved under the active profile's content
    /// root (`profiles/<id>/headless/journals.json`, legacy data-root
    /// fallback) — the same write-safe rule persona / relationships use.
    pub fn journal_store_path(&self) -> std::path::PathBuf {
        self.paths().journal_store()
    }

    /// Reads the journal store. A missing file is the pristine default (no
    /// journals anywhere), not an error.
    pub fn read_journals(&self) -> Result<JournalStore> {
        let path = self.journal_store_path();
        if !path.exists() {
            return Ok(JournalStore::default());
        }
        read_json_file(&path)
    }

    /// Persists the journal store, pretty-printed like the other headless
    /// stores.
    pub fn write_journals(&self, store: &JournalStore) -> Result<()> {
        let path = self.journal_store_path();
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

    /// Reads, mutates, and writes the journal store in one shot.
    pub fn update_journals<T>(&self, mutate: impl FnOnce(&mut JournalStore) -> T) -> Result<T> {
        let mut store = self.read_journals()?;
        let out = mutate(&mut store);
        self.write_journals(&store)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_repo(tag: &str) -> (std::path::PathBuf, LiveChatRepository) {
        let root = std::env::temp_dir().join(format!("chasm-journal-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let repo = LiveChatRepository::new(&root);
        (root, repo)
    }

    fn entry(text: &str, at: &str) -> JournalEntry {
        JournalEntry {
            created_at: at.to_string(),
            game_time: None,
            game_day: None,
            text: text.to_string(),
        }
    }

    /// Full store round-trip: journals, watermarks, and unknown keys survive a
    /// write→read cycle; a missing file reads as the default.
    #[test]
    fn store_round_trips_through_disk() {
        let (root, repo) = temp_repo("roundtrip");

        let fresh = repo.read_journals().unwrap();
        assert!(fresh.characters.is_empty());
        assert!(fresh.watermarks.is_empty());

        let mut store = JournalStore::default();
        store.append("Easy Pete", "Easy Pete", entry("The Courier handled the dynamite well.", "T1"));
        store.watermarks.insert("session-a".into(), 12);
        store.last_pass_at = Some("T1".into());
        store
            .extra
            .insert("futureKey".into(), serde_json::json!({ "keep": true }));
        repo.write_journals(&store).unwrap();

        let back = repo.read_journals().unwrap();
        assert_eq!(back.entries_for("Easy Pete").len(), 1);
        assert_eq!(back.characters["Easy Pete"].name, "Easy Pete");
        assert_eq!(back.watermarks.get("session-a"), Some(&12));
        assert_eq!(back.last_pass_at.as_deref(), Some("T1"));
        assert_eq!(back.extra["futureKey"]["keep"], serde_json::json!(true));
        // Unknown character → empty, no error.
        assert!(back.entries_for("Trudy").is_empty());

        let _ = fs::remove_dir_all(&root);
    }

    /// The append-only invariant: appending NEVER mutates or removes earlier
    /// entries; blank entries are dropped, not stored.
    #[test]
    fn append_only_never_mutates_prior_entries() {
        let mut store = JournalStore::default();
        store.append("Trudy", "Trudy", entry("First day back.", "T1"));
        store.append("Trudy", "Trudy", entry("He keeps shooting near me.", "T2"));
        // A blank entry is a no-op (the pass wrote nothing).
        store.append("Trudy", "Trudy", entry("   ", "T3"));

        let entries = store.entries_for("Trudy");
        assert_eq!(entries.len(), 2);
        // Order preserved, oldest first, and the first entry is byte-identical
        // to what was written — nothing rewrote it.
        assert_eq!(entries[0].text, "First day back.");
        assert_eq!(entries[0].created_at, "T1");
        assert_eq!(entries[1].text, "He keeps shooting near me.");

        // A third real entry only ever grows the list.
        store.append("Trudy", "Trudy", entry("Decided I'll ignore it.", "T4"));
        let entries = store.entries_for("Trudy");
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].text, "First day back.");
        assert_eq!(entries[2].text, "Decided I'll ignore it.");
    }
}
