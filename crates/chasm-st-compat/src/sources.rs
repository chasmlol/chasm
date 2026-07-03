//! Readers for the SillyTavern-compatible source data that feeds prompt
//! assembly: PNG character cards, lorebooks (world info), action books, quest
//! books, and the headless world-state store.
//!
//! These are faithful *readers* only. Activation and prompt formatting live in
//! the `chasm-prompt` crate so this crate stays compatibility-focused.

use std::{collections::BTreeMap, fs, path::PathBuf};

use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{action_books::ScopedCatalogConfig, CompatError, LiveChatRepository, Result};

// ---------------------------------------------------------------------------
// Character cards
// ---------------------------------------------------------------------------

/// The character fields used by prompt assembly, mirroring `mapCharacterDetails`
/// in `src/headless/characters.js`.
#[derive(Debug, Clone, Serialize, Default)]
pub struct CharacterCard {
    pub id: String,
    pub name: String,
    pub system_prompt: String,
    pub description: String,
    pub personality: String,
    /// The card's own scenario field. STORAGE COMPAT ONLY: imported SillyTavern
    /// cards may carry it and it round-trips through save/re-embed untouched,
    /// but prompt assembly no longer injects it — the Scenario slot is filled
    /// by the GLOBAL scenario template (Globals page; see
    /// `chasm_prompt::scenario`), resolved with gamestate macros per turn.
    pub scenario: String,
    pub example_dialogue: String,
    /// Linked world/lorebook name from `data.extensions.world`, when present.
    pub world: Option<String>,
}

const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// Extracts the embedded character JSON from a PNG card, preferring the V3
/// `ccv3` tEXt chunk and falling back to the V2 `chara` chunk (both hold
/// base64-encoded JSON). Mirrors `read` in `src/character-card-parser.js`.
fn read_png_character_json(bytes: &[u8]) -> Option<String> {
    if bytes.len() < 8 || bytes[..8] != PNG_SIGNATURE {
        return None;
    }
    let mut offset = 8usize;
    let mut chara: Option<String> = None;
    let mut ccv3: Option<String> = None;
    while offset + 8 <= bytes.len() {
        let length = u32::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        let kind = &bytes[offset + 4..offset + 8];
        let data_start = offset + 8;
        let Some(data_end) = data_start.checked_add(length) else {
            break;
        };
        // Each chunk is followed by a 4-byte CRC.
        if data_end + 4 > bytes.len() {
            break;
        }
        if kind == b"tEXt" {
            let data = &bytes[data_start..data_end];
            if let Some(zero) = data.iter().position(|byte| *byte == 0) {
                let keyword = String::from_utf8_lossy(&data[..zero]).to_ascii_lowercase();
                let text = String::from_utf8_lossy(&data[zero + 1..]).to_string();
                match keyword.as_str() {
                    "ccv3" => ccv3 = Some(text),
                    "chara" => chara = Some(text),
                    _ => {}
                }
            }
        }
        if kind == b"IEND" {
            break;
        }
        offset = data_end + 4;
    }
    let encoded = ccv3.or(chara)?;
    let decoded = STANDARD.decode(encoded.trim().as_bytes()).ok()?;
    Some(String::from_utf8_lossy(&decoded).to_string())
}

/// Maps a parsed card JSON to [`CharacterCard`], preferring V2/V3 `data.*`
/// fields and falling back to the legacy top-level fields.
fn card_from_json(id: &str, json: &str) -> Option<CharacterCard> {
    let value: Value = serde_json::from_str(json).ok()?;
    let data = value.get("data").cloned().unwrap_or(Value::Null);
    let field = |key: &str| -> String {
        data.get(key)
            .and_then(Value::as_str)
            .or_else(|| value.get(key).and_then(Value::as_str))
            .unwrap_or_default()
            .to_string()
    };
    let world = data
        .get("extensions")
        .and_then(|extensions| extensions.get("world"))
        .and_then(Value::as_str)
        .filter(|world| !world.is_empty())
        .map(str::to_string);
    let name = {
        let raw = field("name");
        if raw.is_empty() {
            id.to_string()
        } else {
            raw
        }
    };
    Some(CharacterCard {
        id: id.to_string(),
        name,
        system_prompt: field("system_prompt"),
        description: field("description"),
        personality: field("personality"),
        scenario: field("scenario"),
        example_dialogue: field("mes_example"),
        world,
    })
}

// ---------------------------------------------------------------------------
// Lorebooks (world info)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawCharacterFilter {
    #[serde(default)]
    is_exclude: bool,
    #[serde(default)]
    names: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawLoreEntry {
    #[serde(default)]
    key: Vec<String>,
    #[serde(default)]
    comment: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    constant: bool,
    #[serde(default)]
    disable: bool,
    #[serde(default)]
    order: Option<f64>,
    #[serde(default)]
    case_sensitive: Option<bool>,
    #[serde(default)]
    character_filter: Option<RawCharacterFilter>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RawLorebook {
    #[serde(default)]
    name: String,
    #[serde(default)]
    entries: BTreeMap<String, RawLoreEntry>,
}

/// A single world-info entry, normalized for activation/formatting.
#[derive(Debug, Clone, Serialize)]
pub struct LoreEntry {
    pub keys: Vec<String>,
    pub comment: String,
    pub content: String,
    pub constant: bool,
    pub disable: bool,
    pub order: f64,
    pub case_sensitive: Option<bool>,
    pub filter_names: Vec<String>,
    pub filter_exclude: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Lorebook {
    pub name: String,
    pub entries: Vec<LoreEntry>,
}

fn lore_from_raw(raw: RawLorebook) -> Lorebook {
    let entries = raw
        .entries
        .into_values()
        .map(|entry| {
            let (filter_names, filter_exclude) = entry
                .character_filter
                .map(|filter| (filter.names, filter.is_exclude))
                .unwrap_or_default();
            LoreEntry {
                keys: entry.key,
                comment: entry.comment,
                content: entry.content,
                constant: entry.constant,
                disable: entry.disable,
                order: entry.order.unwrap_or(0.0),
                case_sensitive: entry.case_sensitive,
                filter_names,
                filter_exclude,
            }
        })
        .collect();
    Lorebook {
        name: raw.name,
        entries,
    }
}

// ---------------------------------------------------------------------------
// Action books
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawActionEntry {
    #[serde(default)]
    key: Vec<String>,
    #[serde(default)]
    comment: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    constant: bool,
    #[serde(default)]
    disable: bool,
    #[serde(default = "default_true")]
    vectorized: bool,
    #[serde(default)]
    order: Option<f64>,
    #[serde(default)]
    case_sensitive: Option<bool>,
    #[serde(default)]
    action_id: String,
    #[serde(default)]
    alias: Option<String>,
    #[serde(default)]
    short_name: Option<String>,
    #[serde(default)]
    risk_tier: String,
    #[serde(default)]
    parameters_schema: Value,
    #[serde(default)]
    preconditions: Vec<String>,
    #[serde(default)]
    effects: Vec<String>,
    #[serde(default)]
    examples_when_to_use: Vec<String>,
    #[serde(default)]
    examples_when_not_to_use: Vec<String>,
    /// Pre-baked generic intent text for vector retrieval (`vectorizableText`).
    /// Name-free by design, so it never cross-matches NPC-name queries.
    #[serde(default)]
    vectorizable_text: String,
    /// Trusted GECK execution (script + arguments) + engine binding. Relayed to the
    /// FNV helper so it can build the native command for non-native actions.
    #[serde(default)]
    execution: Value,
    #[serde(default)]
    binding: Value,
    /// When true, the action acts on someone — the model must name a `target`.
    #[serde(default)]
    requires_target: bool,
    /// Scoped catalogs (e.g. spawnable-entity search config) attached to this entry.
    #[serde(default)]
    scoped_catalogs: Vec<RawScopedCatalog>,
    /// Availability scopes (e.g. `["admin"]` to restrict an action to Todd).
    #[serde(default)]
    scopes: Vec<String>,
}

/// Scoped-catalog config on an action entry (mirrors the full reader's parsing).
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

fn scoped_catalog_from_raw(raw: RawScopedCatalog) -> ScopedCatalogConfig {
    ScopedCatalogConfig {
        id: raw.id,
        catalog_id: raw.catalog_id,
        title: raw.title,
        description: raw.description,
        trigger_keys: raw.trigger_keys,
        parameter_name: raw.parameter_name,
        use_keywords: raw.use_keywords.unwrap_or(true),
        use_vectors: raw.use_vectors.unwrap_or(true),
        limit: raw.limit.unwrap_or(8),
        include_tags: raw.include_tags,
    }
}

/// A book's top-level catalog of spawnable game records (`catalogs.<id>`), kept
/// WITH its items so scoped-catalog searches can resolve candidates → FormIDs.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawCatalog {
    #[serde(default)]
    id: String,
    #[serde(default)]
    items: BTreeMap<String, RawCatalogItem>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawCatalogItem {
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default = "default_true")]
    vectorized: bool,
    #[serde(default)]
    disable: bool,
    #[serde(default)]
    vectorizable_text: String,
    #[serde(default)]
    metadata: Value,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RawActionBook {
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    catalogs: BTreeMap<String, RawCatalog>,
    #[serde(default)]
    entries: BTreeMap<String, RawActionEntry>,
}

/// A single action-book entry, normalized to the resolved shape that the
/// prompt formatter consumes (`mapResolvedAction` in `action-books.js`).
#[derive(Debug, Clone, Serialize)]
pub struct ActionEntry {
    pub keys: Vec<String>,
    pub title: String,
    pub description: String,
    pub constant: bool,
    pub disable: bool,
    /// SillyTavern `vectorized` flag — when false the entry only activates via
    /// keyword/constant, never semantic search (e.g. involuntary idle gestures).
    pub vectorized: bool,
    pub order: f64,
    pub case_sensitive: Option<bool>,
    pub action_id: String,
    pub alias: Option<String>,
    pub short_name: Option<String>,
    pub risk_tier: String,
    pub parameters_schema: Value,
    pub preconditions: Vec<String>,
    pub effects: Vec<String>,
    pub examples_when_to_use: Vec<String>,
    pub examples_when_not_to_use: Vec<String>,
    pub vectorizable_text: String,
    /// Trusted GECK execution (script + arguments) relayed to the FNV helper.
    pub execution: Value,
    /// Engine binding (e.g. `{ "engine": "fallout-new-vegas:xnvse" }`).
    pub binding: Value,
    /// True when the action acts on a target (player or NPC) and needs a name.
    pub requires_target: bool,
    /// Scoped catalogs (config) attached to this action — e.g. the spawnable-entity
    /// catalog a "spawn" action searches for candidates.
    pub scoped_catalogs: Vec<ScopedCatalogConfig>,
    /// Availability scopes. Empty = always available; otherwise the request must
    /// carry a matching scope (e.g. `["admin"]` restricts the action to Todd).
    pub scopes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActionBook {
    pub id: String,
    pub name: String,
    pub entries: Vec<ActionEntry>,
    /// Top-level catalogs (with items) referenced by entries' scoped catalogs.
    pub catalogs: Vec<ActionBookCatalog>,
}

/// A book catalog kept WITH its items (the full reader discards them, keeping only
/// a count; the live/admin path needs the items to resolve spawn candidates).
#[derive(Debug, Clone, Serialize)]
pub struct ActionBookCatalog {
    pub id: String,
    pub items: Vec<CatalogItem>,
}

/// One catalog record (e.g. a spawnable creature/item). `vectorizable_text` feeds
/// the candidate search; `metadata` carries the `formId` the helper spawns.
#[derive(Debug, Clone, Serialize)]
pub struct CatalogItem {
    pub id: String,
    pub name: String,
    pub aliases: Vec<String>,
    pub vectorized: bool,
    pub disable: bool,
    pub vectorizable_text: String,
    pub metadata: Value,
}

/// serde default for `vectorized`: actions are semantically retrievable unless a
/// book explicitly opts out (`"vectorized": false`).
fn default_true() -> bool {
    true
}

fn action_from_raw(raw: RawActionBook) -> ActionBook {
    let entries = raw
        .entries
        .into_values()
        .map(|entry| ActionEntry {
            keys: entry.key,
            title: entry.comment,
            description: entry.content,
            constant: entry.constant,
            disable: entry.disable,
            vectorized: entry.vectorized,
            order: entry.order.unwrap_or(0.0),
            case_sensitive: entry.case_sensitive,
            action_id: entry.action_id,
            alias: entry.alias,
            short_name: entry.short_name,
            risk_tier: entry.risk_tier,
            parameters_schema: entry.parameters_schema,
            preconditions: entry.preconditions,
            effects: entry.effects,
            examples_when_to_use: entry.examples_when_to_use,
            examples_when_not_to_use: entry.examples_when_not_to_use,
            vectorizable_text: entry.vectorizable_text,
            execution: entry.execution,
            binding: entry.binding,
            requires_target: entry.requires_target,
            scoped_catalogs: entry
                .scoped_catalogs
                .into_iter()
                .map(scoped_catalog_from_raw)
                .collect(),
            scopes: entry.scopes,
        })
        .collect();
    let catalogs = raw
        .catalogs
        .into_iter()
        .map(|(key, catalog)| {
            let id = if catalog.id.is_empty() {
                key
            } else {
                catalog.id.clone()
            };
            ActionBookCatalog {
                id,
                items: catalog_items_from_raw(catalog),
            }
        })
        .collect();
    ActionBook {
        id: raw.id,
        name: raw.name,
        entries,
        catalogs,
    }
}

/// Maps a raw catalog's `items` map to normalized [`CatalogItem`]s — shared by the
/// book's inline catalogs and the standalone `action-catalogs/<id>.json` files.
fn catalog_items_from_raw(catalog: RawCatalog) -> Vec<CatalogItem> {
    catalog
        .items
        .into_iter()
        .map(|(item_key, item)| CatalogItem {
            id: if item.id.is_empty() {
                item_key
            } else {
                item.id
            },
            name: item.name,
            aliases: item.aliases,
            vectorized: item.vectorized,
            disable: item.disable,
            vectorizable_text: item.vectorizable_text,
            metadata: item.metadata,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Quest books
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawQuestStageHint {
    #[serde(default)]
    stage: Option<Value>,
    #[serde(default)]
    label: String,
    #[serde(default)]
    description: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawQuestEvent {
    #[serde(default)]
    action_id: String,
    #[serde(default)]
    label: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    requires_player_consent: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawQuestEntry {
    #[serde(default)]
    key: Vec<String>,
    #[serde(default)]
    comment: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    constant: bool,
    #[serde(default)]
    disable: bool,
    #[serde(default)]
    order: Option<f64>,
    #[serde(default)]
    priority: Option<f64>,
    #[serde(default)]
    case_sensitive: Option<bool>,
    #[serde(default)]
    quest_id: String,
    #[serde(default)]
    quest_name: String,
    #[serde(default)]
    quest_editor_id: String,
    #[serde(default)]
    giver_character_ids: Vec<String>,
    #[serde(default)]
    offer_summary: String,
    #[serde(default)]
    pre_dialogue: String,
    #[serde(default)]
    objectives: Vec<String>,
    #[serde(default)]
    acceptance_cues: Vec<String>,
    #[serde(default)]
    refusal_cues: Vec<String>,
    #[serde(default)]
    stage_hints: Vec<RawQuestStageHint>,
    #[serde(default)]
    quest_events: Vec<RawQuestEvent>,
    // --- Gating fields (mirrors quest-books.js normalizeQuestEntry) ----------
    /// `phase` (default `available`), used by the availability gate.
    #[serde(default)]
    phase: Option<String>,
    /// Alternate native-NPC giver keys (`giverNpcKeys`), checked alongside names.
    #[serde(default)]
    giver_npc_keys: Vec<String>,
    /// Scope filter (`scopes`); `global` always passes.
    #[serde(default)]
    scopes: Vec<String>,
    /// Tag filter (`tags`).
    #[serde(default)]
    tags: Vec<String>,
    /// Substring include/exclude gates run against the scan text.
    #[serde(default)]
    include: Vec<String>,
    #[serde(default)]
    exclude: Vec<String>,
    /// `targetGame` filter (substring match against the requested game).
    #[serde(default)]
    target_game: String,
    /// Free `availableWhen`/`conditions` object (status/stage windows).
    #[serde(default, alias = "conditions")]
    available_when: Value,
    /// When true, completed/failed quests still pass the state gate.
    #[serde(default)]
    include_completed: bool,
    /// Pre-baked text used for vector retrieval (`vectorizableText`).
    #[serde(default)]
    vectorizable_text: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RawQuestBook {
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    entries: BTreeMap<String, RawQuestEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuestStageHint {
    pub stage: Option<i64>,
    pub label: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuestEvent {
    pub action_id: String,
    pub label: String,
    pub description: String,
    pub requires_player_consent: bool,
}

/// A single quest-book entry, normalized to the resolved shape that the prompt
/// formatter consumes (`mapResolvedQuest` in `quest-books.js`).
#[derive(Debug, Clone, Serialize)]
pub struct QuestEntry {
    pub keys: Vec<String>,
    pub title: String,
    pub description: String,
    pub constant: bool,
    pub disable: bool,
    pub priority: f64,
    pub case_sensitive: Option<bool>,
    pub quest_id: String,
    pub quest_name: String,
    pub quest_editor_id: String,
    pub giver_character_ids: Vec<String>,
    pub offer_summary: String,
    pub pre_dialogue: String,
    pub objectives: Vec<String>,
    pub acceptance_cues: Vec<String>,
    pub refusal_cues: Vec<String>,
    pub stage_hints: Vec<QuestStageHint>,
    pub quest_events: Vec<QuestEvent>,
    // --- Gating fields (additive; consumed by the prompt assembler's gate) ---
    /// Lifecycle phase (`available`, `active`, `complete`, …). Defaults to
    /// `available` per `normalizeQuestEntry`.
    pub phase: String,
    /// Native-NPC giver keys (`giverNpcKeys`), checked alongside character ids.
    pub giver_npc_keys: Vec<String>,
    /// Scope filter; an entry with scopes only applies when the request's scopes
    /// intersect (or the entry is `global`).
    pub scopes: Vec<String>,
    pub tags: Vec<String>,
    /// Substring include/exclude gates against the scan text.
    pub include: Vec<String>,
    pub exclude: Vec<String>,
    /// `targetGame` filter.
    pub target_game: String,
    /// `availableWhen`/`conditions` raw object (status + min/max stage windows).
    pub available_when: Value,
    /// When true, completed/failed quests still pass the state gate.
    pub include_completed: bool,
    /// Pre-baked vectorizable text; falls back to other fields when empty.
    pub vectorizable_text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuestBook {
    pub id: String,
    pub name: String,
    pub entries: Vec<QuestEntry>,
}

/// Lightweight listing row for the quest-book index page (mirrors
/// [`crate::LorebookSummary`]). `id` is the file stem used by the viewer routes.
#[derive(Debug, Clone, Serialize)]
pub struct QuestBookSummary {
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

fn quest_from_raw(raw: RawQuestBook) -> QuestBook {
    let entries = raw
        .entries
        .into_values()
        .map(|entry| {
            let title = if entry.comment.is_empty() {
                entry.quest_name.clone()
            } else {
                entry.comment.clone()
            };
            QuestEntry {
                keys: entry.key,
                title,
                description: entry.content,
                constant: entry.constant,
                disable: entry.disable,
                priority: entry.priority.or(entry.order).unwrap_or(0.0),
                case_sensitive: entry.case_sensitive,
                quest_id: entry.quest_id,
                quest_name: entry.quest_name,
                quest_editor_id: entry.quest_editor_id,
                giver_character_ids: entry.giver_character_ids,
                offer_summary: entry.offer_summary,
                pre_dialogue: entry.pre_dialogue,
                objectives: entry.objectives,
                acceptance_cues: entry.acceptance_cues,
                refusal_cues: entry.refusal_cues,
                stage_hints: entry
                    .stage_hints
                    .into_iter()
                    .map(|hint| QuestStageHint {
                        stage: hint.stage.as_ref().and_then(value_to_i64),
                        label: hint.label,
                        description: hint.description,
                    })
                    .collect(),
                quest_events: entry
                    .quest_events
                    .into_iter()
                    .map(|event| QuestEvent {
                        action_id: event.action_id,
                        label: event.label,
                        description: event.description,
                        requires_player_consent: event.requires_player_consent.unwrap_or(true),
                    })
                    .collect(),
                phase: entry
                    .phase
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| "available".to_string()),
                giver_npc_keys: entry.giver_npc_keys,
                scopes: entry.scopes,
                tags: entry.tags,
                include: entry.include,
                exclude: entry.exclude,
                target_game: entry.target_game,
                available_when: entry.available_when,
                include_completed: entry.include_completed,
                vectorizable_text: entry.vectorizable_text,
            }
        })
        .collect();
    QuestBook {
        id: raw.id,
        name: raw.name,
        entries,
    }
}

// ---------------------------------------------------------------------------
// Repository readers
// ---------------------------------------------------------------------------

impl LiveChatRepository {
    /// Reads and parses a PNG character card by id (the card file stem, e.g.
    /// `Easy Pete`). Returns `Ok(None)` when the card or its metadata is absent.
    pub fn read_character_card(&self, character_id: &str) -> Result<Option<CharacterCard>> {
        let id = character_id.trim_end_matches(".png");
        let path = self.paths().characters_dir().join(format!("{id}.png"));
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path).map_err(|source| CompatError::Io {
            path: path.clone(),
            source,
        })?;
        Ok(read_png_character_json(&bytes).and_then(|json| card_from_json(id, &json)))
    }

    /// Reads a single lorebook (`worlds/<name>.json`), resolved per-profile.
    pub fn read_lorebook(&self, name: &str) -> Result<Option<Lorebook>> {
        let path = self.paths().worlds_dir().join(format!("{name}.json"));
        if !path.exists() {
            return Ok(None);
        }
        let raw: RawLorebook = crate::read_json_file(&path)?;
        Ok(Some(lore_from_raw(raw)))
    }

    /// Reads every lorebook under the active profile's `worlds/`.
    pub fn list_lorebooks(&self) -> Result<Vec<Lorebook>> {
        self.read_json_dir(self.paths().worlds_dir(), lore_from_raw)
    }

    /// Reads every action book under the active profile's `headless/action-books/`.
    pub fn list_action_books(&self) -> Result<Vec<ActionBook>> {
        self.read_json_dir(self.paths().action_books_dir(), action_from_raw)
    }

    /// Reads ONE standalone catalog file (`headless/action-catalogs/<catalog_id>.json`)
    /// and returns its items — the full spawnable lists (creatures/NPCs/items) that
    /// scoped-catalog (spawn) actions search. Empty when the file is missing.
    pub fn read_action_catalog(&self, catalog_id: &str) -> Vec<CatalogItem> {
        let safe = catalog_id.trim();
        if safe.is_empty() || safe.contains(['/', '\\']) {
            return Vec::new();
        }
        let path = self
            .paths()
            .action_catalogs_dir()
            .join(format!("{safe}.json"));
        let Ok(bytes) = fs::read(&path) else {
            return Vec::new();
        };
        match serde_json::from_slice::<RawCatalog>(&bytes) {
            Ok(raw) => catalog_items_from_raw(raw),
            Err(_) => Vec::new(),
        }
    }

    /// Reads every standalone catalog file under `headless/action-catalogs/`. Used
    /// to pre-warm catalog embeddings.
    pub fn list_action_catalogs(&self) -> Vec<ActionBookCatalog> {
        self.read_json_dir(self.paths().action_catalogs_dir(), |raw: RawCatalog| {
            let id = raw.id.clone();
            ActionBookCatalog {
                id,
                items: catalog_items_from_raw(raw),
            }
        })
        .unwrap_or_default()
    }

    /// Reads every quest book under the active profile's `headless/quest-books/`.
    pub fn list_quest_books(&self) -> Result<Vec<QuestBook>> {
        self.read_json_dir(self.paths().quest_books_dir(), quest_from_raw)
    }

    /// Reads every quest book under `headless/quest-books/<id>.json` keyed by the
    /// file stem (so the viewer routes can address a book by a stable id even when
    /// the JSON omits its own `id`). Sorted by stem, mirroring
    /// [`Self::read_all_lorebook_files`].
    pub fn read_all_quest_books(&self) -> Result<Vec<QuestBook>> {
        let dir = self.paths().quest_books_dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut ids: Vec<String> = Vec::new();
        for entry in fs::read_dir(&dir).map_err(|source| CompatError::Io {
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
            if let Some(book) = self.read_quest_book(&id)? {
                books.push(book);
            }
        }
        Ok(books)
    }

    /// Lists quest books (id + name + entry count), sorted by file stem. The
    /// `name` falls back to the file-stem id when the JSON has no `name`.
    pub fn list_quest_book_files(&self) -> Result<Vec<QuestBookSummary>> {
        Ok(self
            .read_all_quest_books()?
            .into_iter()
            .map(|book| QuestBookSummary {
                id: book.id,
                name: book.name,
                entry_count: book.entries.len(),
            })
            .collect())
    }

    /// Reads a single quest book by its storage id (the `headless/quest-books/
    /// <id>.json` file stem). Returns `Ok(None)` when no matching file exists.
    /// The returned book's `id`/`name` are filled from the stem when the JSON
    /// omits them, so the viewer always has a title to show.
    pub fn read_quest_book(&self, id: &str) -> Result<Option<QuestBook>> {
        let path = self.paths().quest_books_dir().join(format!("{id}.json"));
        if !path.exists() {
            return Ok(None);
        }
        let raw: RawQuestBook = crate::read_json_file(&path)?;
        let mut book = quest_from_raw(raw);
        if book.id.is_empty() {
            book.id = id.to_string();
        }
        if book.name.is_empty() {
            book.name = id.to_string();
        }
        Ok(Some(book))
    }

    /// Reads the generated player-persona description (the SillyTavern "user
    /// persona" equivalent) from the active profile's persona store
    /// (`headless/persona/persona.json`, written by chasm-web's persona
    /// module). Returns `Ok(None)` when no persona has been generated yet or
    /// the stored description is empty — prompt assembly then injects nothing,
    /// mirroring SillyTavern's `{{#if persona}}` story-string slot.
    pub fn read_player_persona(&self) -> Result<Option<String>> {
        let path = self.paths().persona_dir().join("persona.json");
        if !path.exists() {
            return Ok(None);
        }
        let value: Value = crate::read_json_file(&path)?;
        Ok(value
            .get("description")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|description| !description.is_empty())
            .map(str::to_string))
    }

    /// Reads the headless world-state store (`headless/world-state.json`),
    /// returning an empty object when the file is absent. World-state is global
    /// (not per-profile), so it always reads from the legacy data root.
    pub fn read_world_state(&self) -> Result<Value> {
        let path = self.data_root().join("headless").join("world-state.json");
        if !path.exists() {
            return Ok(Value::Object(serde_json::Map::new()));
        }
        crate::read_json_file(&path)
    }

    /// Reads every `*.json` file directly under `dir` and maps each through
    /// `map`, sorted by file name for deterministic ordering.
    fn read_json_dir<Raw, Out>(&self, dir: PathBuf, map: impl Fn(Raw) -> Out) -> Result<Vec<Out>>
    where
        Raw: for<'de> Deserialize<'de>,
    {
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut paths: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(&dir).map_err(|source| CompatError::Io {
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
        let mut out = Vec::with_capacity(paths.len());
        for path in paths {
            let raw: Raw = crate::read_json_file(&path)?;
            out.push(map(raw));
        }
        Ok(out)
    }
}
