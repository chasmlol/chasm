//! UI books domain — Characters / Lore / Quest / Action book endpoints.
//!
//! The four book screens render via the shared React `<Book>` component and hit
//! `GET /api/ui/v1/books/:kind` + `POST /api/ui/v1/books/:kind/:id`. Each kind
//! projects its on-disk content into the SAME `{ entries: [{ id, title,
//! subtitle?, badge?, values }] }` shape (the `values` bag is the field schema
//! the matching screen declares), so the four screens are literally one
//! component fed four data sources.
//!
//! Read/save model (mirrors the Askama editors in [`crate::books`]):
//!   * Lore / Quest / Action are JSON books. We read the active profile's single
//!     book file as a `serde_json::Value` (this crate compiles `serde_json` with
//!     `preserve_order`, so object key + entry order round-trips), project the
//!     editable subset into `values`, and on save re-load the ORIGINAL file,
//!     overlay only the posted fields onto the matching entry IN PLACE (keeping
//!     unknown entry keys + top-level keys), and write pretty JSON back.
//!   * Characters are SillyTavern V2/V3 PNG cards. We read the embedded card
//!     JSON (preferring `ccv3`, falling back to `chara`) and expose the persona
//!     fields; on save we overlay the edited fields onto the embedded JSON and
//!     re-embed it into the PNG's `chara` + `ccv3` tEXt chunks (image bytes
//!     preserved), so the edit persists in the card itself.
//!
//! Stays under `/api/ui/v1` and reads/writes only book content; it never touches
//! the game transport or AI-stack lifecycle.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use axum::{
    extract::{Path as AxPath, State},
    Json,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::Serialize;
use serde_json::{Map, Value};
use chasm_core::{AppSettings, GameProfile};

use crate::{AppState, WebError, WebResult};

/// One book entry surfaced to the React `<Book>` (values are book-specific).
#[derive(Serialize)]
pub(crate) struct UiBookEntry {
    pub id: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    /// Optional small badge label (e.g. "Disabled", a phase, a scope) rendered
    /// on the collapsed row. The screens pass it straight to `<Book>`'s `badge`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub badge: Option<String>,
    /// Editable fields keyed by the field schema's keys.
    pub values: Value,
}

#[derive(Serialize)]
pub(crate) struct UiBookList {
    pub entries: Vec<UiBookEntry>,
}

/// `GET /api/ui/v1/books/:kind` — the entries for one book.
pub(crate) async fn list_book(
    State(state): State<Arc<AppState>>,
    AxPath(kind): AxPath<String>,
) -> WebResult<Json<UiBookList>> {
    let entries = match kind.as_str() {
        "characters" => list_characters(&state),
        "lore" => list_json_book(&state, BookKind::Lore),
        "quest" => list_json_book(&state, BookKind::Quest),
        "action" => list_json_book(&state, BookKind::Action),
        other => Err(WebError::from(anyhow::anyhow!(
            "unknown book kind '{other}'"
        ))),
    }?;
    Ok(Json(UiBookList { entries }))
}

/// `POST /api/ui/v1/books/:kind/:id` — save one entry's edited values, then
/// return the persisted entry (re-projected from disk so the UI re-syncs).
pub(crate) async fn save_book(
    State(state): State<Arc<AppState>>,
    AxPath((kind, id)): AxPath<(String, String)>,
    Json(body): Json<Value>,
) -> WebResult<Json<UiBookEntry>> {
    let values = body
        .get("values")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let entry = match kind.as_str() {
        "characters" => save_character(&state, &id, &values),
        "lore" => save_json_book(&state, BookKind::Lore, &id, &values),
        "quest" => save_json_book(&state, BookKind::Quest, &id, &values),
        "action" => save_json_book(&state, BookKind::Action, &id, &values),
        other => Err(WebError::from(anyhow::anyhow!(
            "unknown book kind '{other}'"
        ))),
    }?;
    Ok(Json(entry))
}

// ===========================================================================
// JSON books (Lore / Quest / Action) — shared read/overlay/write
// ===========================================================================

/// The three JSON book kinds. Each maps to a content dir + the editable field
/// set the matching React screen declares.
#[derive(Clone, Copy)]
enum BookKind {
    Lore,
    Quest,
    Action,
}

impl BookKind {
    /// The active profile's content dir for this book kind.
    fn dir(self, state: &AppState) -> PathBuf {
        let paths = state.config.active_profile_paths();
        match self {
            BookKind::Lore => paths.worlds_dir(),
            BookKind::Quest => paths.quest_books_dir(),
            BookKind::Action => paths.action_books_dir(),
        }
    }

    /// Human label used in the not-found error.
    fn label(self) -> &'static str {
        match self {
            BookKind::Lore => "lore book",
            BookKind::Quest => "quest book",
            BookKind::Action => "action book",
        }
    }
}

/// Resolves the active profile's single book file in `dir`: an explicit
/// `prefer_name` (the active profile/book name) wins, then the file with the
/// most top-level `entries`, then the first stem sorted ascending. Returns
/// `None` when the dir is missing/empty. Mirrors `resolve_single_book` in
/// [`crate::books`] (a real profile ships one file per kind; this only matters
/// for dev/legacy dirs that hold several).
fn resolve_single_book(dir: &Path, prefer_name: Option<&str>) -> Option<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("json") {
            files.push(path);
        }
    }
    if files.is_empty() {
        return None;
    }
    files.sort();
    if let Some(want) = prefer_name.map(str::trim).filter(|s| !s.is_empty()) {
        if let Some(hit) = files.iter().find(|path| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|stem| stem.eq_ignore_ascii_case(want))
        }) {
            return Some(hit.clone());
        }
    }
    files
        .iter()
        .max_by_key(|path| book_entry_count(path))
        .cloned()
        .or_else(|| files.first().cloned())
}

/// Counts top-level `entries` in a book file (best-effort; read/parse failure
/// counts as 0 so it loses the most-entries tiebreak gracefully).
fn book_entry_count(path: &Path) -> usize {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|value| {
            value
                .get("entries")
                .and_then(Value::as_object)
                .map(Map::len)
        })
        .unwrap_or(0)
}

/// The active profile's display name (the single-book name tiebreak).
fn active_profile_name(state: &AppState) -> Option<String> {
    let settings = AppSettings::load(&state.config.settings_path);
    let id = settings.active_profile_id(&state.config.profiles_dir);
    GameProfile::read(&state.config.profiles_dir, &id).map(|profile| profile.name)
}

/// Reads `key`/`keys` (array or comma-string) into a single comma-joined line
/// for the screen's "keys" text field.
fn keys_to_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(", "),
        Some(Value::String(text)) => text.trim().to_string(),
        _ => String::new(),
    }
}

/// Parses a comma-OR-newline separated keys string into a JSON string array,
/// trimming and dropping empties.
fn text_to_keys(text: &str) -> Value {
    let items: Vec<Value> = text
        .split([',', '\n'])
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| Value::String(part.to_string()))
        .collect();
    Value::Array(items)
}

fn str_field(entry: &Map<String, Value>, key: &str) -> String {
    entry
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn bool_field(entry: &Map<String, Value>, key: &str) -> bool {
    entry.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// The book's single file as a parsed `Value` + its path (`None` when absent).
fn load_book_value(state: &AppState, kind: BookKind) -> Option<(PathBuf, Value)> {
    let dir = kind.dir(state);
    let path = resolve_single_book(&dir, active_profile_name(state).as_deref())?;
    let text = fs::read_to_string(&path).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    Some((path, value))
}

/// The ordered `(map_key, entry_object)` list of a book's `entries`, using the
/// entry's `uid` as the stable id (falling back to the map key). Order is the
/// file's order (preserve_order is on).
fn book_entries(root: &Value) -> Vec<(String, Map<String, Value>)> {
    root.get("entries")
        .and_then(Value::as_object)
        .map(|entries| {
            entries
                .iter()
                .map(|(map_key, value)| {
                    let object = value.as_object().cloned().unwrap_or_default();
                    let id = object
                        .get("uid")
                        .map(value_to_id)
                        .filter(|id| !id.is_empty())
                        .unwrap_or_else(|| map_key.clone());
                    (id, object)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// A `uid` Value as a plain id string (numbers without quotes).
fn value_to_id(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn list_json_book(state: &AppState, kind: BookKind) -> WebResult<Vec<UiBookEntry>> {
    let Some((_, root)) = load_book_value(state, kind) else {
        // No book file under the active profile → empty list (the UI shows its
        // empty state). Not an error: a profile may legitimately lack a book.
        return Ok(Vec::new());
    };
    let entries = book_entries(&root)
        .into_iter()
        .map(|(id, entry)| project_json_entry(kind, id, &entry))
        .collect();
    Ok(entries)
}

/// Projects one book entry into the `<Book>` row for its kind. `values` keys
/// match the field schema the matching React screen declares.
fn project_json_entry(kind: BookKind, id: String, entry: &Map<String, Value>) -> UiBookEntry {
    let disabled = bool_field(entry, "disable");
    match kind {
        BookKind::Lore => {
            let comment = str_field(entry, "comment");
            let keys = keys_to_text(entry.get("key").or_else(|| entry.get("keys")));
            let title = pick_title(&[&comment, keys.split(',').next().unwrap_or("")], &id);
            let subtitle = one_line(&str_field(entry, "content"));
            UiBookEntry {
                title,
                subtitle: non_empty(subtitle),
                badge: disabled.then(|| "Disabled".to_string()),
                values: serde_json::json!({
                    "title": comment,
                    "keys": keys,
                    "content": str_field(entry, "content"),
                    "enabled": !disabled,
                }),
                id,
            }
        }
        BookKind::Quest => {
            let comment = str_field(entry, "comment");
            let quest_name = str_field(entry, "questName");
            let title = pick_title(&[&comment, &quest_name], &id);
            let phase = {
                let p = str_field(entry, "phase");
                if p.trim().is_empty() {
                    "available".to_string()
                } else {
                    p
                }
            };
            let subtitle = one_line(&first_non_empty(&[
                str_field(entry, "offerSummary"),
                str_field(entry, "content"),
            ]));
            UiBookEntry {
                title,
                subtitle: non_empty(subtitle),
                badge: Some(if disabled {
                    "Disabled".to_string()
                } else {
                    title_case(&phase)
                }),
                values: serde_json::json!({
                    "title": comment,
                    "questName": quest_name,
                    "questId": str_field(entry, "questId"),
                    "status": phase,
                    "keys": keys_to_text(entry.get("key")),
                    "offerSummary": str_field(entry, "offerSummary"),
                    "description": str_field(entry, "content"),
                    "enabled": !disabled,
                }),
                id,
            }
        }
        BookKind::Action => {
            let comment = str_field(entry, "comment");
            let action_id = str_field(entry, "actionId");
            let title = pick_title(&[&comment, &action_id], &id);
            // An action is admin-only when it has scopes that don't include the
            // public `global` scope (mirrors the runtime scope gate).
            let scopes: Vec<String> = entry
                .get("scopes")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            let admin_only = !scopes.is_empty() && !scopes.iter().any(|s| s == "global");
            let subtitle = one_line(&str_field(entry, "content"));
            let badge = if disabled {
                Some("Disabled".to_string())
            } else if admin_only {
                Some("Admin".to_string())
            } else {
                None
            };
            UiBookEntry {
                title,
                subtitle: non_empty(subtitle),
                badge,
                values: serde_json::json!({
                    "title": comment,
                    "actionId": action_id,
                    "riskTier": str_field(entry, "riskTier"),
                    "keys": keys_to_text(entry.get("key").or_else(|| entry.get("triggers"))),
                    "description": str_field(entry, "content"),
                    "scope": if admin_only { "admin" } else { "any" },
                    "enabled": !disabled,
                }),
                id,
            }
        }
    }
}

fn save_json_book(
    state: &AppState,
    kind: BookKind,
    id: &str,
    values: &Value,
) -> WebResult<UiBookEntry> {
    let dir = kind.dir(state);
    let path = resolve_single_book(&dir, active_profile_name(state).as_deref())
        .ok_or_else(|| WebError::from(anyhow::anyhow!("no {} file to save to", kind.label())))?;
    let text = fs::read_to_string(&path)?;
    let mut root: Value = serde_json::from_str(&text)?;
    if !root.is_object() {
        root = Value::Object(Map::new());
    }

    // Locate the entry whose uid (or map key) matches `id`, then overlay only the
    // posted fields onto it IN PLACE — unknown entry keys + entry order survive.
    {
        let entries = root
            .as_object_mut()
            .and_then(|obj| obj.get_mut("entries"))
            .and_then(Value::as_object_mut)
            .ok_or_else(|| {
                WebError::from(anyhow::anyhow!("{} has no entries to edit", kind.label()))
            })?;

        let map_key = entries
            .iter()
            .find(|(map_key, value)| {
                let uid = value
                    .as_object()
                    .and_then(|o| o.get("uid"))
                    .map(value_to_id)
                    .filter(|u| !u.is_empty());
                uid.as_deref() == Some(id) || map_key.as_str() == id
            })
            .map(|(map_key, _)| map_key.clone())
            .ok_or_else(|| {
                WebError::from(anyhow::anyhow!("{} entry '{}' not found", kind.label(), id))
            })?;

        let entry = entries
            .get_mut(&map_key)
            .and_then(Value::as_object_mut)
            .ok_or_else(|| WebError::from(anyhow::anyhow!("entry '{}' is not an object", id)))?;
        overlay_json_entry(kind, entry, values);
    }

    let pretty = serde_json::to_string_pretty(&root)?;
    fs::write(&path, pretty)?;
    tracing::info!(
        "ui: saved {} entry '{}' -> {}",
        kind.label(),
        id,
        path.display()
    );

    // Re-project from the just-written value so the UI re-syncs to disk truth.
    let entry = book_entries(&root)
        .into_iter()
        .find(|(eid, _)| eid == id)
        .map(|(eid, entry)| project_json_entry(kind, eid, &entry))
        .ok_or_else(|| WebError::from(anyhow::anyhow!("entry '{}' vanished after save", id)))?;
    Ok(entry)
}

/// Overlays the posted `values` onto one entry object in place. Only fields the
/// screen sent are touched; the `disable` flag is the inverse of `enabled`.
/// Text edits that match the screen's keys map back onto the on-disk field name.
fn overlay_json_entry(kind: BookKind, entry: &mut Map<String, Value>, values: &Value) {
    let get_str = |key: &str| values.get(key).and_then(Value::as_str).map(str::to_string);
    let get_bool = |key: &str| values.get(key).and_then(Value::as_bool);

    let mut set_str = |field: &str, value: Option<String>| {
        if let Some(value) = value {
            entry.insert(field.to_string(), Value::String(value));
        }
    };

    match kind {
        BookKind::Lore => {
            set_str("comment", get_str("title"));
            set_str("content", get_str("content"));
            if let Some(keys) = get_str("keys") {
                entry.insert("key".to_string(), text_to_keys(&keys));
            }
        }
        BookKind::Quest => {
            set_str("comment", get_str("title"));
            set_str("questName", get_str("questName"));
            set_str("questId", get_str("questId"));
            set_str("offerSummary", get_str("offerSummary"));
            set_str("content", get_str("description"));
            if let Some(status) = get_str("status") {
                entry.insert("phase".to_string(), Value::String(status));
            }
            if let Some(keys) = get_str("keys") {
                entry.insert("key".to_string(), text_to_keys(&keys));
            }
        }
        BookKind::Action => {
            set_str("comment", get_str("title"));
            set_str("actionId", get_str("actionId"));
            set_str("riskTier", get_str("riskTier"));
            set_str("content", get_str("description"));
            if let Some(keys) = get_str("keys") {
                entry.insert("key".to_string(), text_to_keys(&keys));
            }
            // The Scope select is a derived convenience: only act when the user
            // moves it AWAY from the entry's current effective scope, and only
            // by toggling the `global` membership of the existing scopes (never
            // inventing a scopes list where there wasn't one, so non-scoped
            // entries stay non-scoped).
            if let Some(scope) = get_str("scope") {
                apply_action_scope(entry, &scope);
            }
        }
    }

    // `enabled` (UI) is the inverse of `disable` (disk). Always written so a
    // toggle round-trips; the field exists on every entry in these books.
    if let Some(enabled) = get_bool("enabled") {
        entry.insert("disable".to_string(), Value::Bool(!enabled));
    }
}

/// Toggles an action entry's admin gating via the `global` scope. "any" ensures
/// `global` is present (only if the entry already had a scopes list); "admin"
/// removes `global`. Never creates a scopes array on an entry that had none.
fn apply_action_scope(entry: &mut Map<String, Value>, scope: &str) {
    let Some(arr) = entry.get_mut("scopes").and_then(Value::as_array_mut) else {
        return;
    };
    let has_global = arr.iter().any(|s| s.as_str() == Some("global"));
    match scope {
        "any" if !has_global => arr.insert(0, Value::String("global".to_string())),
        "admin" if has_global => arr.retain(|s| s.as_str() != Some("global")),
        _ => {}
    }
}

// ===========================================================================
// Characters — PNG card read + writeback
// ===========================================================================

const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// Lists every character card under the active profile's `characters/` dir,
/// projecting the editable persona fields. Cards that can't be parsed are
/// skipped (so one bad file never blanks the book).
fn list_characters(state: &AppState) -> WebResult<Vec<UiBookEntry>> {
    let dir = state.config.active_profile_paths().characters_dir();
    let mut cards: Vec<UiBookEntry> = Vec::new();
    let Ok(read) = fs::read_dir(&dir) else {
        return Ok(cards);
    };
    let mut paths: Vec<PathBuf> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("png"))
        .collect();
    paths.sort();
    for path in paths {
        let Some(id) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(bytes) = fs::read(&path) else { continue };
        let Some(json) = read_png_character_json(&bytes) else {
            continue;
        };
        let Ok(card) = serde_json::from_str::<Value>(&json) else {
            continue;
        };
        cards.push(project_character(id, &card));
    }
    Ok(cards)
}

/// Reads a card field, preferring `data.<key>` (V2/V3) then the legacy top-level
/// `<key>` (mirrors the compat reader's fallbacks).
fn card_field(card: &Value, key: &str) -> String {
    let data = card.get("data");
    data.and_then(|d| d.get(key))
        .and_then(Value::as_str)
        .or_else(|| card.get(key).and_then(Value::as_str))
        .unwrap_or("")
        .to_string()
}

/// Projects a parsed card into the Characters `<Book>` row. The editable
/// `values` are the persona/prompt fields; the system prompt is the key one.
fn project_character(id: &str, card: &Value) -> UiBookEntry {
    let name = {
        let raw = card_field(card, "name");
        if raw.trim().is_empty() {
            id.to_string()
        } else {
            raw
        }
    };
    let description = card_field(card, "description");
    let subtitle = one_line(&first_non_empty(&[
        card_field(card, "creator_notes"),
        description.clone(),
    ]));
    UiBookEntry {
        title: name.clone(),
        subtitle: non_empty(subtitle),
        badge: None,
        values: serde_json::json!({
            "name": name,
            "systemPrompt": card_field(card, "system_prompt"),
            "description": description,
            "personality": card_field(card, "personality"),
            "scenario": card_field(card, "scenario"),
            "firstMessage": card_field(card, "first_mes"),
            "exampleDialogue": card_field(card, "mes_example"),
        }),
        id: id.to_string(),
    }
}

/// The persona fields editable from the Characters screen, mapped to the card's
/// JSON key. The UI value key is the first element, the card key the second.
const CHARACTER_FIELDS: &[(&str, &str)] = &[
    ("name", "name"),
    ("systemPrompt", "system_prompt"),
    ("description", "description"),
    ("personality", "personality"),
    ("scenario", "scenario"),
    ("firstMessage", "first_mes"),
    ("exampleDialogue", "mes_example"),
];

fn save_character(state: &AppState, id: &str, values: &Value) -> WebResult<UiBookEntry> {
    // Resolve the card file; reject ids that escape the characters dir.
    let safe = id.trim().trim_end_matches(".png");
    if safe.is_empty() || safe.contains(['/', '\\']) || safe.contains("..") {
        return Err(WebError::from(anyhow::anyhow!(
            "invalid character id '{id}'"
        )));
    }
    let dir = state.config.active_profile_paths().characters_dir();
    let path = dir.join(format!("{safe}.png"));
    if !path.exists() {
        return Err(WebError::from(anyhow::anyhow!(
            "character card '{safe}.png' not found"
        )));
    }
    let bytes = fs::read(&path)?;
    let json = read_png_character_json(&bytes)
        .ok_or_else(|| WebError::from(anyhow::anyhow!("card '{safe}' has no embedded data")))?;
    let mut card: Value = serde_json::from_str(&json)?;
    if !card.is_object() {
        card = Value::Object(Map::new());
    }

    // Overlay each edited field onto BOTH `data.<key>` (the V2/V3 home the
    // reader prefers) and the legacy top-level `<key>` (kept in sync so V2-only
    // consumers see the edit too), but only for fields the form actually sent.
    {
        let obj = card.as_object_mut().expect("card is an object");
        // Ensure a `data` object exists so V3 readers stay authoritative.
        if !obj.get("data").map(Value::is_object).unwrap_or(false) {
            obj.insert("data".to_string(), Value::Object(Map::new()));
        }
        for (ui_key, card_key) in CHARACTER_FIELDS {
            if let Some(text) = values.get(*ui_key).and_then(Value::as_str) {
                let value = Value::String(text.to_string());
                obj.insert((*card_key).to_string(), value.clone());
                if let Some(data) = obj.get_mut("data").and_then(Value::as_object_mut) {
                    data.insert((*card_key).to_string(), value);
                }
            }
        }
    }

    let updated_json = serde_json::to_string(&card)?;
    let new_png = write_png_character_json(&bytes, &updated_json)
        .ok_or_else(|| WebError::from(anyhow::anyhow!("failed to re-embed card '{safe}'")))?;
    fs::write(&path, &new_png)?;
    tracing::info!("ui: saved character card '{}' -> {}", safe, path.display());

    Ok(project_character(safe, &card))
}

/// Extracts the embedded character JSON from a PNG card, preferring the V3
/// `ccv3` tEXt chunk and falling back to the V2 `chara` chunk (both base64).
/// Mirrors [`chasm_st_compat`]'s reader.
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

/// Re-embeds `json` into a PNG card's character chunks, returning new PNG bytes.
///
/// Strategy: copy the original PNG chunk-by-chunk, DROPPING the existing `chara`
/// and `ccv3` tEXt chunks, and insert freshly-built `chara` + `ccv3` tEXt chunks
/// (both holding the same base64 JSON, matching how cards are written) right
/// before `IEND`. The image data (`IHDR`, `IDAT`, palette, everything else) is
/// preserved byte-for-byte, so only the embedded persona changes. Returns `None`
/// if the input isn't a PNG with an `IEND`.
fn write_png_character_json(original: &[u8], json: &str) -> Option<Vec<u8>> {
    if original.len() < 8 || original[..8] != PNG_SIGNATURE {
        return None;
    }
    let encoded = STANDARD.encode(json.as_bytes());

    let mut out: Vec<u8> = Vec::with_capacity(original.len() + encoded.len() * 2 + 64);
    out.extend_from_slice(&PNG_SIGNATURE);

    let mut offset = 8usize;
    let mut wrote_cards = false;
    while offset + 8 <= original.len() {
        let length = u32::from_be_bytes([
            original[offset],
            original[offset + 1],
            original[offset + 2],
            original[offset + 3],
        ]) as usize;
        let kind = &original[offset + 4..offset + 8];
        let data_start = offset + 8;
        let data_end = data_start.checked_add(length)?;
        let crc_end = data_end.checked_add(4)?;
        if crc_end > original.len() {
            return None;
        }
        let chunk = &original[offset..crc_end];

        // Drop any pre-existing character chunks (we re-emit canonical ones).
        let is_card_text = kind == b"tEXt" && {
            let data = &original[data_start..data_end];
            let keyword = data
                .iter()
                .position(|b| *b == 0)
                .map(|z| String::from_utf8_lossy(&data[..z]).to_ascii_lowercase())
                .unwrap_or_default();
            keyword == "chara" || keyword == "ccv3"
        };

        if kind == b"IEND" {
            // Insert the new card chunks just before IEND.
            out.extend_from_slice(&text_chunk("chara", encoded.as_bytes()));
            out.extend_from_slice(&text_chunk("ccv3", encoded.as_bytes()));
            wrote_cards = true;
            out.extend_from_slice(chunk);
            break;
        }
        if !is_card_text {
            out.extend_from_slice(chunk);
        }
        offset = crc_end;
    }
    wrote_cards.then_some(out)
}

/// Builds one PNG `tEXt` chunk: `[len:4 BE][b"tEXt"][keyword\0text][crc:4 BE]`
/// where the CRC is over the type + data per the PNG spec.
fn text_chunk(keyword: &str, text: &[u8]) -> Vec<u8> {
    let mut data: Vec<u8> = Vec::with_capacity(keyword.len() + 1 + text.len());
    data.extend_from_slice(keyword.as_bytes());
    data.push(0);
    data.extend_from_slice(text);

    let mut chunk: Vec<u8> = Vec::with_capacity(12 + data.len());
    chunk.extend_from_slice(&(data.len() as u32).to_be_bytes());
    chunk.extend_from_slice(b"tEXt");
    chunk.extend_from_slice(&data);
    let mut crc_input: Vec<u8> = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(b"tEXt");
    crc_input.extend_from_slice(&data);
    chunk.extend_from_slice(&crc32(&crc_input).to_be_bytes());
    chunk
}

/// CRC-32 (IEEE 802.3, the PNG variant): reflected, poly 0xEDB88320, init/xor
/// 0xFFFFFFFF. Computed inline so the UI layer needs no extra dependency.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

// ===========================================================================
// Small text helpers (titles / subtitles)
// ===========================================================================

/// The first non-blank candidate, else `fallback` (used for row titles).
fn pick_title(candidates: &[&str], fallback: &str) -> String {
    candidates
        .iter()
        .map(|c| c.trim())
        .find(|c| !c.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| {
            if fallback.trim().is_empty() {
                "Untitled".to_string()
            } else {
                fallback.to_string()
            }
        })
}

fn first_non_empty(values: &[String]) -> String {
    values
        .iter()
        .find(|v| !v.trim().is_empty())
        .cloned()
        .unwrap_or_default()
}

fn non_empty(value: String) -> Option<String> {
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

/// Collapses whitespace/newlines and truncates to a one-line subtitle.
fn one_line(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX: usize = 140;
    if collapsed.chars().count() > MAX {
        let truncated: String = collapsed.chars().take(MAX).collect();
        format!("{}…", truncated.trim_end())
    } else {
        collapsed
    }
}

/// Capitalizes the first letter (for the quest phase badge).
fn title_case(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}
