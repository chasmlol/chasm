//! Full-fidelity reader for SillyTavern lorebooks (World Info).
//!
//! SillyTavern stores each lorebook as `data/default-user/worlds/<id>.json`,
//! shape `{ "name"?, "entries": { "<uid>": Entry } }`. This module reads the
//! *complete* entry shape (every field the headless resolver consults) so the
//! web viewer can show entries faithfully and the prompt crate can run
//! activation against the real semantics.
//!
//! Field semantics mirror `src/headless/lorebooks.js`,
//! `src/endpoints/worldinfo.js`, and the `world_info_position` enum in
//! `public/scripts/world-info.js`.
//!
//! The minimal, activation-only [`crate::Lorebook`]/[`crate::LoreEntry`] types in
//! `sources.rs` remain the prompt assembler's stable inputs; this module adds a
//! richer, viewer-and-activation-oriented model (`LorebookFile` / `LoreEntryFull`)
//! without disturbing them.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{CompatError, LiveChatRepository, Result};

// ---------------------------------------------------------------------------
// world_info_position enum (public/scripts/world-info.js)
// ---------------------------------------------------------------------------

/// SillyTavern's `world_info_position`. The numeric discriminants are the wire
/// values stored in entry JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldInfoPosition {
    /// 0 — before the character definition.
    Before,
    /// 1 — after the character definition.
    After,
    /// 2 — top of the Author's Note.
    AuthorNoteTop,
    /// 3 — bottom of the Author's Note.
    AuthorNoteBottom,
    /// 4 — injected at a specific chat depth.
    AtDepth,
    /// 5 — top of the example messages block.
    ExampleMessagesTop,
    /// 6 — bottom of the example messages block.
    ExampleMessagesBottom,
    /// 7 — routed to a named outlet.
    Outlet,
    /// Any value outside the known enum.
    Unknown(i64),
}

impl WorldInfoPosition {
    pub fn from_i64(value: i64) -> Self {
        match value {
            0 => Self::Before,
            1 => Self::After,
            2 => Self::AuthorNoteTop,
            3 => Self::AuthorNoteBottom,
            4 => Self::AtDepth,
            5 => Self::ExampleMessagesTop,
            6 => Self::ExampleMessagesBottom,
            7 => Self::Outlet,
            other => Self::Unknown(other),
        }
    }

    /// Short human label used by the viewer.
    pub fn label(&self) -> String {
        match self {
            Self::Before => "Before char".to_string(),
            Self::After => "After char".to_string(),
            Self::AuthorNoteTop => "AN top".to_string(),
            Self::AuthorNoteBottom => "AN bottom".to_string(),
            Self::AtDepth => "At depth".to_string(),
            Self::ExampleMessagesTop => "EM top".to_string(),
            Self::ExampleMessagesBottom => "EM bottom".to_string(),
            Self::Outlet => "Outlet".to_string(),
            Self::Unknown(value) => format!("pos {value}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Raw on-disk shape
// ---------------------------------------------------------------------------

/// Accepts `key`/`keys` as either a `["a","b"]` array or a comma-joined string,
/// mirroring `getEntryKeys` (which also tolerates a bare string).
fn keys_from_value(value: &Value) -> Vec<String> {
    match value {
        Value::Array(items) => items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect(),
        Value::String(text) => text
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RawCharacterFilter {
    #[serde(default, alias = "isExclude")]
    is_exclude: bool,
    #[serde(default)]
    names: Vec<String>,
}

/// The full World Info entry shape. Field names follow ST's camelCase JSON;
/// `key`/`keys` and `extensions.vectorized` are reconciled in [`LoreEntryFull`].
#[derive(Debug, Clone, Deserialize, Default)]
struct RawLoreEntry {
    #[serde(default)]
    uid: Option<Value>,
    #[serde(default)]
    id: Option<Value>,
    #[serde(default)]
    key: Option<Value>,
    #[serde(default)]
    keys: Option<Value>,
    #[serde(default)]
    keysecondary: Option<Value>,
    #[serde(default)]
    comment: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    constant: bool,
    #[serde(default)]
    selective: bool,
    #[serde(default)]
    disable: bool,
    #[serde(default)]
    order: Option<f64>,
    #[serde(default)]
    priority: Option<f64>,
    #[serde(default)]
    position: Option<Value>,
    #[serde(default)]
    depth: Option<i64>,
    #[serde(default)]
    probability: Option<f64>,
    #[serde(rename = "caseSensitive", default)]
    case_sensitive: Option<bool>,
    #[serde(rename = "matchWholeWords", default)]
    match_whole_words: Option<bool>,
    #[serde(default)]
    vectorized: Option<bool>,
    #[serde(default)]
    role: Option<Value>,
    #[serde(rename = "characterFilter", alias = "character_filter", default)]
    character_filter: Option<RawCharacterFilter>,
    #[serde(default)]
    extensions: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RawLorebookFile {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    entries: serde_json::Map<String, Value>,
    #[serde(default)]
    extensions: Option<Value>,
}

// ---------------------------------------------------------------------------
// Normalized model
// ---------------------------------------------------------------------------

/// A fully-parsed World Info entry, normalized for activation and display.
#[derive(Debug, Clone, Serialize)]
pub struct LoreEntryFull {
    /// `uid`/`id`, falling back to the entry map key.
    pub uid: String,
    /// Primary activation keys (`key`/`keys`).
    pub keys: Vec<String>,
    /// Secondary keys (`keysecondary`) — used by selective AND/NOT logic in ST.
    pub keys_secondary: Vec<String>,
    /// Display title (`comment`).
    pub comment: String,
    pub content: String,
    pub constant: bool,
    pub selective: bool,
    pub disable: bool,
    /// Effective ordering weight (`order`, then `priority`).
    pub order: f64,
    pub position: WorldInfoPosition,
    pub depth: Option<i64>,
    pub probability: Option<f64>,
    pub case_sensitive: bool,
    pub match_whole_words: Option<bool>,
    /// Marked for vector retrieval (`vectorized` or `extensions.vectorized`).
    pub vectorized: bool,
    /// `role` (used at `atDepth`): 0 = system, 1 = user, 2 = assistant.
    pub role: Option<i64>,
    pub filter_names: Vec<String>,
    pub filter_exclude: bool,
}

/// A parsed lorebook file (`worlds/<id>.json`).
#[derive(Debug, Clone, Serialize)]
pub struct LorebookFile {
    /// File stem (the `worlds/<id>.json` id used by routes).
    pub id: String,
    /// `name` field, falling back to `id`.
    pub name: String,
    pub entries: Vec<LoreEntryFull>,
}

impl LorebookFile {
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

/// Lightweight listing row for the index page.
#[derive(Debug, Clone, Serialize)]
pub struct LorebookSummary {
    pub id: String,
    pub name: String,
    pub entry_count: usize,
}

fn value_to_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_f64().map(|number| number as i64))
        .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn entry_from_raw(map_key: &str, raw: RawLoreEntry) -> LoreEntryFull {
    let uid = raw
        .uid
        .as_ref()
        .filter(|value| !value.is_null())
        .or(raw.id.as_ref())
        .map(value_to_string)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| map_key.to_string());

    let keys = raw
        .key
        .as_ref()
        .map(keys_from_value)
        .filter(|keys| !keys.is_empty())
        .or_else(|| raw.keys.as_ref().map(keys_from_value))
        .unwrap_or_default();

    let keys_secondary = raw
        .keysecondary
        .as_ref()
        .map(keys_from_value)
        .unwrap_or_default();

    let position = raw
        .position
        .as_ref()
        .and_then(value_to_i64)
        .map(WorldInfoPosition::from_i64)
        .unwrap_or(WorldInfoPosition::Before);

    let extensions_vectorized = raw
        .extensions
        .as_ref()
        .and_then(|ext| ext.get("vectorized"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let (filter_names, filter_exclude) = raw
        .character_filter
        .map(|filter| (filter.names, filter.is_exclude))
        .unwrap_or_default();

    LoreEntryFull {
        uid,
        keys,
        keys_secondary,
        comment: raw.comment,
        content: raw.content,
        constant: raw.constant,
        selective: raw.selective,
        disable: raw.disable,
        order: raw.order.or(raw.priority).unwrap_or(0.0),
        position,
        depth: raw.depth,
        probability: raw.probability,
        case_sensitive: raw.case_sensitive.unwrap_or(false),
        match_whole_words: raw.match_whole_words,
        vectorized: raw.vectorized.unwrap_or(false) || extensions_vectorized,
        role: raw.role.as_ref().and_then(value_to_i64),
        filter_names,
        filter_exclude,
    }
}

fn file_from_raw(id: &str, raw: RawLorebookFile) -> LorebookFile {
    let name = raw
        .name
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| id.to_string());
    let entries = raw
        .entries
        .into_iter()
        .map(|(map_key, value)| {
            let raw_entry: RawLoreEntry = serde_json::from_value(value).unwrap_or_default();
            entry_from_raw(&map_key, raw_entry)
        })
        .collect();
    let _ = raw.extensions; // book-level extensions are not consumed by the viewer yet.
    LorebookFile {
        id: id.to_string(),
        name,
        entries,
    }
}

// ---------------------------------------------------------------------------
// Repository readers
// ---------------------------------------------------------------------------

impl LiveChatRepository {
    fn worlds_dir(&self) -> PathBuf {
        self.paths().worlds_dir()
    }

    /// Lists lorebooks under `worlds/` (id + name + entry count), sorted by id,
    /// mirroring `listLorebooks` in `src/headless/lorebooks.js`.
    pub fn list_lorebook_files(&self) -> Result<Vec<LorebookSummary>> {
        Ok(self
            .read_all_lorebook_files()?
            .into_iter()
            .map(|book| LorebookSummary {
                id: book.id,
                name: book.name,
                entry_count: book.entries.len(),
            })
            .collect())
    }

    /// Reads a single lorebook file (`worlds/<id>.json`) with full entry fields.
    pub fn read_lorebook_file(&self, id: &str) -> Result<Option<LorebookFile>> {
        let path = self.worlds_dir().join(format!("{id}.json"));
        if !path.exists() {
            return Ok(None);
        }
        let raw: RawLorebookFile = crate::read_json_file(&path)?;
        Ok(Some(file_from_raw(id, raw)))
    }

    /// Reads every lorebook file under `worlds/`, fully parsed, sorted by id.
    pub fn read_all_lorebook_files(&self) -> Result<Vec<LorebookFile>> {
        let dir = self.worlds_dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut ids: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(&dir).map_err(|source| CompatError::Io {
            path: dir.clone(),
            source,
        })? {
            let entry = entry.map_err(|source| CompatError::Io {
                path: dir.clone(),
                source,
            })?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) {
                    ids.push(stem.to_string());
                }
            }
        }
        ids.sort();
        let mut books = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(book) = self.read_lorebook_file(&id)? {
                books.push(book);
            }
        }
        Ok(books)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_accept_array_or_comma_string() {
        assert_eq!(
            keys_from_value(&serde_json::json!(["a", "b"])),
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(
            keys_from_value(&serde_json::json!("a, b ,c")),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn entry_reconciles_position_vectorized_and_keys() {
        let raw: RawLoreEntry = serde_json::from_value(serde_json::json!({
            "uid": 7,
            "key": ["Vegas"],
            "keysecondary": ["Strip"],
            "comment": "New Vegas",
            "content": "The Strip glitters.",
            "constant": false,
            "selective": true,
            "order": 100,
            "position": 4,
            "depth": 3,
            "caseSensitive": true,
            "role": 0,
            "extensions": { "vectorized": true }
        }))
        .unwrap();
        let entry = entry_from_raw("entry-key", raw);
        assert_eq!(entry.uid, "7");
        assert_eq!(entry.keys, vec!["Vegas".to_string()]);
        assert_eq!(entry.keys_secondary, vec!["Strip".to_string()]);
        assert_eq!(entry.position, WorldInfoPosition::AtDepth);
        assert_eq!(entry.depth, Some(3));
        assert!(entry.case_sensitive);
        assert!(entry.vectorized);
        assert_eq!(entry.role, Some(0));
        assert_eq!(entry.order, 100.0);
    }

    #[test]
    fn uid_falls_back_to_map_key() {
        let raw: RawLoreEntry = serde_json::from_value(serde_json::json!({
            "content": "no uid here",
        }))
        .unwrap();
        let entry = entry_from_raw("12", raw);
        assert_eq!(entry.uid, "12");
    }
}
