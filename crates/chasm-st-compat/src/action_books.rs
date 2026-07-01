//! Richer Action Book reader, ported from `src/headless/action-books.js`.
//!
//! `sources.rs` already exposes a slim [`crate::ActionBook`]/[`crate::ActionEntry`]
//! pair that the prompt assembler's keyword path consumes. This module is the
//! faithful, full-fidelity reader: it mirrors `normalizeActionBook` /
//! `normalizeActionEntry` (book-level `description`, `binding`, `catalogs`,
//! `settings`; entry-level `uid`, `keysecondary`, `riskTier`, `targetGame`,
//! `pluginSource`, `scopes`, `tags`, `scopedCatalogs`, `sourceLinks`, ...) and
//! produces the resolved shape `mapResolvedAction` hands to the formatter.
//!
//! It is additive: the existing names in `sources.rs` are untouched. Everything
//! here is prefixed `ActionBookDetail` / `ResolvedAction*` to avoid clashing.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{CompatError, LiveChatRepository, Result};

/// Default entry priority (`order`) when none is set — matches `normalizeActionEntry`.
pub const DEFAULT_ACTION_ORDER: f64 = 100.0;
/// Default risk tier when none is set — matches `normalizeActionEntry`.
pub const DEFAULT_RISK_TIER: &str = "low";

// ---------------------------------------------------------------------------
// Raw (on-disk) shape
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawActionEntry {
    #[serde(default)]
    uid: Option<Value>,
    #[serde(default)]
    key: Vec<String>,
    #[serde(default)]
    keysecondary: Vec<String>,
    #[serde(default)]
    comment: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    constant: bool,
    #[serde(default)]
    vectorized: bool,
    #[serde(default)]
    disable: bool,
    #[serde(default)]
    order: Option<f64>,
    #[serde(default)]
    priority: Option<f64>,
    #[serde(default)]
    case_sensitive: Option<bool>,
    #[serde(default)]
    action_id: String,
    #[serde(default)]
    risk_tier: Option<String>,
    #[serde(default)]
    target_game: String,
    #[serde(default)]
    plugin_source: String,
    #[serde(default)]
    parameters_schema: Value,
    #[serde(default)]
    preconditions: Vec<String>,
    #[serde(default)]
    effects: Vec<String>,
    #[serde(default)]
    command_template: String,
    #[serde(default)]
    binding: Value,
    #[serde(default)]
    execution: Value,
    #[serde(default)]
    examples_when_to_use: Vec<String>,
    #[serde(default)]
    examples_when_not_to_use: Vec<String>,
    #[serde(default)]
    scoped_catalogs: Vec<RawScopedCatalog>,
    #[serde(default)]
    scopes: Vec<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    source_links: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawScopedCatalog {
    #[serde(default)]
    id: String,
    #[serde(default)]
    catalog_id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    trigger_keys: Vec<String>,
    #[serde(default)]
    parameter_name: String,
    #[serde(default)]
    use_keywords: Option<bool>,
    #[serde(default)]
    use_vectors: Option<bool>,
    #[serde(default)]
    limit: Option<u32>,
    #[serde(default)]
    include_tags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawActionBook {
    #[serde(default)]
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    settings: Value,
    #[serde(default)]
    binding: Value,
    #[serde(default)]
    catalogs: BTreeMap<String, RawCatalog>,
    #[serde(default)]
    entries: BTreeMap<String, RawActionEntry>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawCatalog {
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    items: BTreeMap<String, Value>,
}

// ---------------------------------------------------------------------------
// Resolved (normalized) shape
// ---------------------------------------------------------------------------

/// A scoped-catalog config attached to an entry (`normalizeScopedCatalogConfigs`).
#[derive(Debug, Clone, Serialize)]
pub struct ScopedCatalogConfig {
    pub id: String,
    pub catalog_id: String,
    pub title: String,
    pub description: String,
    pub trigger_keys: Vec<String>,
    pub parameter_name: String,
    pub use_keywords: bool,
    pub use_vectors: bool,
    pub limit: u32,
    pub include_tags: Vec<String>,
}

/// A single Action Book entry, resolved to the shape `mapResolvedAction` exposes.
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedAction {
    /// `String(entry.uid)` — the resolved action id used by `mapResolvedAction`.
    pub id: String,
    pub uid: i64,
    pub action_id: String,
    /// `comment` — the action title.
    pub title: String,
    /// `content` — the description shown to the model.
    pub description: String,
    pub keys: Vec<String>,
    pub secondary_keys: Vec<String>,
    pub constant: bool,
    pub vectorized: bool,
    pub disable: bool,
    pub case_sensitive: Option<bool>,
    /// `order` (default 100) — priority, sorted descending at injection time.
    pub priority: f64,
    pub risk_tier: String,
    pub target_game: String,
    pub plugin_source: String,
    pub parameters_schema: Value,
    pub preconditions: Vec<String>,
    pub effects: Vec<String>,
    pub command_template: String,
    pub binding: Value,
    pub execution: Value,
    pub examples_when_to_use: Vec<String>,
    pub examples_when_not_to_use: Vec<String>,
    pub scoped_catalogs: Vec<ScopedCatalogConfig>,
    pub scopes: Vec<String>,
    pub tags: Vec<String>,
    pub source_links: Vec<String>,
}

/// A catalog of game records referenced by scoped catalogs (`catalogs.<id>`).
#[derive(Debug, Clone, Serialize)]
pub struct ActionCatalog {
    pub id: String,
    pub name: String,
    pub description: String,
    pub item_count: usize,
}

/// A fully-read Action Book (`normalizeActionBook`), with entries pre-sorted by
/// priority descending so callers see them in injection order.
#[derive(Debug, Clone, Serialize)]
pub struct ActionBookDetail {
    /// Sanitized filename stem (the storage id).
    pub id: String,
    pub name: String,
    pub description: String,
    pub settings: Value,
    pub binding: Value,
    pub target_game: String,
    pub catalogs: Vec<ActionCatalog>,
    pub entries: Vec<ResolvedAction>,
}

// ---------------------------------------------------------------------------
// Normalization (mirrors normalizeActionEntry / normalizeActionBook)
// ---------------------------------------------------------------------------

fn uid_to_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_f64().map(|number| number as i64))
        .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
}

fn scoped_catalog_from_raw(raw: RawScopedCatalog) -> ScopedCatalogConfig {
    ScopedCatalogConfig {
        id: raw.id,
        catalog_id: raw.catalog_id,
        title: raw.title,
        description: raw.description,
        trigger_keys: raw.trigger_keys,
        parameter_name: raw.parameter_name,
        // Mirrors `normalizeScopedCatalogConfigs`: both default true.
        use_keywords: raw.use_keywords.unwrap_or(true),
        use_vectors: raw.use_vectors.unwrap_or(true),
        limit: raw.limit.unwrap_or(8),
        include_tags: raw.include_tags,
    }
}

fn action_from_raw(map_key: &str, raw: RawActionEntry) -> ResolvedAction {
    let uid = raw
        .uid
        .as_ref()
        .and_then(uid_to_i64)
        .or_else(|| map_key.parse().ok())
        .unwrap_or(0);
    let risk_tier = raw
        .risk_tier
        .filter(|tier| !tier.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_RISK_TIER.to_string());
    ResolvedAction {
        id: uid.to_string(),
        uid,
        action_id: raw.action_id.trim().to_string(),
        title: raw.comment,
        description: raw.content,
        keys: raw.key,
        secondary_keys: raw.keysecondary,
        constant: raw.constant,
        vectorized: raw.vectorized,
        disable: raw.disable,
        case_sensitive: raw.case_sensitive,
        priority: raw.order.or(raw.priority).unwrap_or(DEFAULT_ACTION_ORDER),
        risk_tier,
        target_game: raw.target_game,
        plugin_source: raw.plugin_source,
        parameters_schema: raw.parameters_schema,
        preconditions: raw.preconditions,
        effects: raw.effects,
        command_template: raw.command_template,
        binding: raw.binding,
        execution: raw.execution,
        examples_when_to_use: raw.examples_when_to_use,
        examples_when_not_to_use: raw.examples_when_not_to_use,
        scoped_catalogs: raw
            .scoped_catalogs
            .into_iter()
            .map(scoped_catalog_from_raw)
            .collect(),
        scopes: raw.scopes,
        tags: raw.tags,
        source_links: raw.source_links,
    }
}

fn book_from_raw(id: String, raw: RawActionBook) -> ActionBookDetail {
    let target_game = raw
        .settings
        .get("targetGame")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let catalogs = raw
        .catalogs
        .into_iter()
        .map(|(key, catalog)| ActionCatalog {
            id: if catalog.id.is_empty() {
                key
            } else {
                catalog.id
            },
            name: catalog.name,
            description: catalog.description,
            item_count: catalog.items.len(),
        })
        .collect();
    let mut entries: Vec<ResolvedAction> = raw
        .entries
        .into_iter()
        .map(|(key, entry)| action_from_raw(&key, entry))
        .collect();
    // Pre-sort by priority descending (stable) so callers see injection order.
    entries.sort_by(|left, right| {
        right
            .priority
            .partial_cmp(&left.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let name = if raw.name.is_empty() {
        id.clone()
    } else {
        raw.name
    };
    ActionBookDetail {
        id,
        name,
        description: raw.description,
        settings: raw.settings,
        binding: raw.binding,
        target_game,
        catalogs,
        entries,
    }
}

// ---------------------------------------------------------------------------
// Repository readers
// ---------------------------------------------------------------------------

impl LiveChatRepository {
    fn action_books_dir(&self) -> std::path::PathBuf {
        self.paths().action_books_dir()
    }

    /// Reads and fully parses every Action Book under
    /// `headless/action-books/<id>.json`, sorted by storage id. The id is the
    /// sanitized filename stem.
    pub fn read_action_books(&self) -> Result<Vec<ActionBookDetail>> {
        let dir = self.action_books_dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut paths: Vec<std::path::PathBuf> = Vec::new();
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
                paths.push(path);
            }
        }
        paths.sort();

        let mut books = Vec::with_capacity(paths.len());
        for path in paths {
            let id = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or_default()
                .to_string();
            let raw: RawActionBook = crate::read_json_file(&path)?;
            books.push(book_from_raw(id, raw));
        }
        Ok(books)
    }

    /// Reads a single Action Book by its storage id (filename stem), returning
    /// `Ok(None)` when no matching file exists.
    pub fn read_action_book(&self, id: &str) -> Result<Option<ActionBookDetail>> {
        Ok(self
            .read_action_books()?
            .into_iter()
            .find(|book| book.id == id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_entry_defaults() {
        let raw = RawActionEntry {
            action_id: "  movement.follow_target  ".to_string(),
            comment: "Follow".to_string(),
            ..Default::default()
        };
        let action = action_from_raw("4", raw);
        assert_eq!(action.action_id, "movement.follow_target");
        assert_eq!(action.priority, DEFAULT_ACTION_ORDER);
        assert_eq!(action.risk_tier, DEFAULT_RISK_TIER);
        assert_eq!(action.uid, 4);
        assert_eq!(action.id, "4");
    }

    #[test]
    fn prefers_uid_field_over_map_key_and_sorts_by_priority() {
        let mut entries = BTreeMap::new();
        entries.insert(
            "a".to_string(),
            RawActionEntry {
                uid: Some(Value::from(7)),
                order: Some(100.0),
                action_id: "low.priority".to_string(),
                ..Default::default()
            },
        );
        entries.insert(
            "b".to_string(),
            RawActionEntry {
                uid: Some(Value::from(8)),
                order: Some(300.0),
                action_id: "high.priority".to_string(),
                ..Default::default()
            },
        );
        let book = book_from_raw(
            "Book".to_string(),
            RawActionBook {
                entries,
                ..Default::default()
            },
        );
        assert_eq!(book.entries[0].action_id, "high.priority");
        assert_eq!(book.entries[0].uid, 8);
        assert_eq!(book.entries[1].action_id, "low.priority");
    }

    #[test]
    fn scoped_catalog_use_flags_default_true() {
        let config = scoped_catalog_from_raw(RawScopedCatalog {
            id: "spawn.item".to_string(),
            catalog_id: "fallout-new-vegas.items".to_string(),
            ..Default::default()
        });
        assert!(config.use_keywords);
        assert!(config.use_vectors);
        assert_eq!(config.limit, 8);
    }
}
