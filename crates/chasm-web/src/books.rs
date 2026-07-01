//! Book editor + viewer routes (Library: Lorebook / Quest Book / Action Book).
//!
//! Each game profile ships exactly ONE file in each of `worlds/`,
//! `headless/quest-books/`, `headless/action-books/`, so the rail links go
//! straight to that single book — there is no index page.
//!
//! Phase A implements the **full Lorebook editor** here (`GET`/`POST /lorebook`)
//! and leaves Quest/Action as the existing read-only detail views
//! (`GET /questbook`, `GET /actionbook`). The lorebook editor establishes the
//! save PATTERN phase B copies for the other two:
//!
//!   * READ the file as a `serde_json::Value` (preserve_order is on for this
//!     crate, so object key order is retained) — never the typed reader, which
//!     can drop fields the editor must round-trip.
//!   * Build a per-entry view exposing every known field with a real control,
//!     plus an `__extra` JSON blob for any field without one, so nothing is
//!     un-editable.
//!   * On SAVE, re-load the ORIGINAL `Value`, overlay the edited fields onto
//!     each entry object IN PLACE (keeping unknown keys + top-level keys like
//!     `extensions` + entry order), apply adds/deletes, and write pretty JSON
//!     back to the active profile's file. Parse errors keep the prior value
//!     rather than corrupting it.

use std::{path::PathBuf, sync::Arc};

use askama::Template;
use axum::{
    extract::{Form, Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Redirect, Response},
};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{html_escape, AppState, WebResult};

// ---------------------------------------------------------------------------
// Single-book resolution
// ---------------------------------------------------------------------------

/// Resolves the active profile's single book file in `dir`, returning
/// `(id, path)` where `id` is the file stem (used by routes). Selection, when
/// more than one `*.json` is present (a dev/legacy artifact — a real profile
/// ships one), is deterministic: an explicit `prefer_id` wins, then a file whose
/// stem case-insensitively equals `prefer_name` (the active profile/book name),
/// then the file with the most top-level `entries`, then the first stem sorted
/// ascending. Returns `None` when the dir is missing/empty.
fn resolve_single_book(
    dir: &PathBuf,
    prefer_id: Option<&str>,
    prefer_name: Option<&str>,
) -> Option<(String, PathBuf)> {
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    let read = std::fs::read_dir(dir).ok()?;
    for entry in read.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("json") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                files.push((stem.to_string(), path));
            }
        }
    }
    if files.is_empty() {
        return None;
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));

    if let Some(want) = prefer_id.map(str::trim).filter(|s| !s.is_empty()) {
        if let Some(hit) = files.iter().find(|(id, _)| id == want) {
            return Some(hit.clone());
        }
    }
    if let Some(want) = prefer_name.map(str::trim).filter(|s| !s.is_empty()) {
        if let Some(hit) = files.iter().find(|(id, _)| id.eq_ignore_ascii_case(want)) {
            return Some(hit.clone());
        }
    }
    // Most-entries tiebreak, keeping the sorted order stable for equal counts.
    files
        .iter()
        .max_by_key(|(_, path)| book_entry_count(path))
        .cloned()
        .or_else(|| files.first().cloned())
}

/// Counts top-level `entries` in a book file without a full parse (best-effort;
/// a read/parse failure counts as 0 so it loses the tiebreak gracefully).
fn book_entry_count(path: &PathBuf) -> usize {
    std::fs::read_to_string(path)
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

/// The active profile's display name, used as the single-book name tiebreak.
fn active_profile_name(state: &AppState) -> Option<String> {
    use chasm_core::{AppSettings, GameProfile};
    let settings = AppSettings::load(&state.config.settings_path);
    let id = settings.active_profile_id(&state.config.profiles_dir);
    GameProfile::read(&state.config.profiles_dir, &id).map(|profile| profile.name)
}

// ---------------------------------------------------------------------------
// Lorebook editor — view model
// ---------------------------------------------------------------------------

/// Every entry field that gets a dedicated control. Anything NOT in here is
/// surfaced via the per-entry `__extra` raw-JSON box, so the editor stays
/// total (no field is silently un-editable) even if ST adds new fields.
const STRUCTURED_FIELDS: &[&str] = &[
    "uid",
    "key",
    "keysecondary",
    "comment",
    "content",
    "constant",
    "selective",
    "vectorized",
    "disable",
    "order",
    "position",
    "depth",
    "probability",
    "useProbability",
    "role",
    "scanDepth",
    "caseSensitive",
    "matchWholeWords",
    "excludeRecursion",
    "preventRecursion",
    "delayUntilRecursion",
    "group",
    "groupOverride",
    "groupWeight",
    "sticky",
    "cooldown",
    "delay",
    "displayIndex",
    "addMemo",
    "useGroupScoring",
    "automationId",
];

/// The nullable tri-state bool fields rendered as `(unset)/true/false` selects.
const TRISTATE_FIELDS: &[(&str, &str)] = &[
    ("caseSensitive", "Case sensitive"),
    ("matchWholeWords", "Match whole words"),
    ("useGroupScoring", "Use group scoring"),
];

struct SelectOption {
    value: String,
    label: String,
    selected: bool,
}

struct TriState {
    field: String,
    label: String,
    /// "unset" | "true" | "false".
    state: String,
}

/// One lorebook entry, every field pre-rendered for the template.
struct LoreEntryEdit {
    uid: String,
    title: String,
    comment: String,
    content: String,
    /// `key` / `keysecondary` joined newline-per-key for the textarea.
    key: String,
    keysecondary: String,
    constant: bool,
    selective: bool,
    vectorized: bool,
    disable: bool,
    use_probability: bool,
    exclude_recursion: bool,
    prevent_recursion: bool,
    delay_until_recursion: bool,
    group_override: bool,
    add_memo: bool,
    order: String,
    depth: String,
    probability: String,
    scan_depth: String,
    group_weight: String,
    sticky: String,
    cooldown: String,
    delay: String,
    display_index: String,
    group: String,
    automation_id: String,
    position_options: Vec<SelectOption>,
    role_options: Vec<SelectOption>,
    tristates: Vec<TriState>,
    /// Pretty JSON of the fields without a dedicated control ("" if none).
    extra_json: String,
}

struct LorebookEditView {
    id: String,
    name: String,
    path: String,
    entries: Vec<LoreEntryEdit>,
    saved: bool,
}

#[derive(Template)]
#[template(path = "lorebook_editor.html")]
struct LorebookEditorTemplate {
    book: LorebookEditView,
}

/// Reads `key`/`keys` as an array or comma-string into newline-joined text for
/// the textarea (one key per line reads cleanly for long keys).
fn keys_to_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::String(text)) => text
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// A number field as a display string ("" for null/absent so the input is blank).
fn num_to_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::Number(number)) => number.to_string(),
        Some(Value::String(text)) => text.clone(),
        _ => String::new(),
    }
}

/// Reads a string-array field as newline-joined text for an editable textarea.
/// A comma-string is split into lines too; anything else is empty. Shared by the
/// quest/action editors for their many `string[]` fields.
fn string_array_to_text(value: Option<&Value>) -> String {
    keys_to_text(value)
}

/// One labelled string-array textarea field (quest/action editors). `value` is
/// the newline-joined contents; `field` is the entry sub-field name (e.g. `tags`).
struct ArrayField {
    field: String,
    label: String,
    value: String,
}

impl ArrayField {
    fn build(entry: &Map<String, Value>, field: &str, label: &str) -> Self {
        Self {
            field: field.to_string(),
            label: label.to_string(),
            value: string_array_to_text(entry.get(field)),
        }
    }
}

/// One labelled raw-JSON textarea field for a nested object / array-of-objects
/// that has no structured control (quest/action editors). Empty `value` ("")
/// means the field was absent — the template hides empties by default but still
/// lets the field be added via the per-entry `__extra` box.
struct JsonField {
    field: String,
    label: String,
    value: String,
}

impl JsonField {
    fn build(entry: &Map<String, Value>, field: &str, label: &str) -> Self {
        let value = match entry.get(field) {
            Some(Value::Null) | None => String::new(),
            Some(other) => serde_json::to_string_pretty(other).unwrap_or_default(),
        };
        Self {
            field: field.to_string(),
            label: label.to_string(),
            value,
        }
    }
}

/// Applies a submitted string-array textarea (split on comma OR newline, trims,
/// drops empties) onto `entry[field]`, but ONLY when the form carried the field
/// (so an un-rendered field is never blanked). Always writes an array (possibly
/// empty) when present, matching how these fields are stored.
fn apply_array_field(
    entry: &mut Map<String, Value>,
    uid: &str,
    field: &str,
    fields: &std::collections::HashMap<String, String>,
) {
    if let Some(raw) = fields.get(&format!("entry.{uid}.{field}")) {
        let items: Vec<Value> = raw
            .split('\n')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(|part| Value::String(part.to_string()))
            .collect();
        // Don't materialise an absent field as an empty array: a field that
        // wasn't on the original entry and is still empty stays absent, so a
        // save stays surgical/idempotent and sparse entries stay sparse.
        if entry.contains_key(field) || !items.is_empty() {
            entry.insert(field.to_string(), Value::Array(items));
        }
    }
}

/// Applies a submitted raw-JSON textarea onto `entry[field]`. An empty box that
/// was rendered for an EXISTING field clears it to `null` (lets you empty a
/// nested object); an empty box for an absent field is ignored. Invalid JSON is
/// ignored (keeps the prior value — never corrupts).
fn apply_json_field(
    entry: &mut Map<String, Value>,
    uid: &str,
    field: &str,
    fields: &std::collections::HashMap<String, String>,
) {
    let key = format!("entry.{uid}.{field}");
    let Some(raw) = fields.get(&key) else {
        return;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        // Only clear a field that already existed; never invent a null key.
        if entry.contains_key(field) {
            entry.insert(field.to_string(), Value::Null);
        }
        return;
    }
    if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
        entry.insert(field.to_string(), parsed);
    }
    // else: keep the prior value (bad JSON should not nuke the field).
}

/// Applies a plain text field onto `entry[field]` as a string, only when the
/// form carried it. Used for scalar string fields in the quest/action editors.
fn apply_text_field(
    entry: &mut Map<String, Value>,
    uid: &str,
    field: &str,
    fields: &std::collections::HashMap<String, String>,
) {
    if let Some(value) = fields.get(&format!("entry.{uid}.{field}")) {
        if entry.contains_key(field) || !value.is_empty() {
            entry.insert(field.to_string(), Value::String(value.clone()));
        }
    }
}

/// Applies an integer field onto `entry[field]`. A blank/unparsable value keeps
/// the prior value (never corrupts). Used for the quest/action numeric scalars.
fn apply_int_field(
    entry: &mut Map<String, Value>,
    uid: &str,
    field: &str,
    fields: &std::collections::HashMap<String, String>,
) {
    if let Some(raw) = fields.get(&format!("entry.{uid}.{field}")) {
        if let Ok(parsed) = raw.trim().parse::<i64>() {
            entry.insert(field.to_string(), Value::Number(parsed.into()));
        }
    }
}

/// Applies a NULLABLE integer field onto `entry[field]`: blank -> `null`, a valid
/// int -> that number, anything else keeps the prior value. Used for the action
/// editor's legitimately-nullable ints (`scanDepth`, `sticky`, `cooldown`,
/// `delay`).
fn apply_nullable_int_field(
    entry: &mut Map<String, Value>,
    uid: &str,
    field: &str,
    fields: &std::collections::HashMap<String, String>,
) {
    if let Some(raw) = fields.get(&format!("entry.{uid}.{field}")) {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            // Blank clears an EXISTING field to null, but never invents one.
            if entry.contains_key(field) {
                entry.insert(field.to_string(), Value::Null);
            }
        } else if let Ok(parsed) = trimmed.parse::<i64>() {
            entry.insert(field.to_string(), Value::Number(parsed.into()));
        }
    }
}

/// Applies a plain bool checkbox onto `entry[field]` (present == true). Only call
/// for entries whose card was rendered, so an absent box is a real `false`.
fn apply_bool_field(
    entry: &mut Map<String, Value>,
    uid: &str,
    field: &str,
    fields: &std::collections::HashMap<String, String>,
) {
    let present = fields.contains_key(&format!("entry.{uid}.{field}"));
    // Don't add a `false` for a checkbox absent from the original entry — only
    // write when the field already existed or the user actually ticked it.
    if entry.contains_key(field) || present {
        entry.insert(field.to_string(), Value::Bool(present));
    }
}

/// Merges the per-entry `__extra` raw-JSON object over `entry`, skipping `uid`.
/// Invalid/empty JSON is ignored. Shared finaliser for all three editors.
fn apply_extra_json(
    entry: &mut Map<String, Value>,
    uid: &str,
    fields: &std::collections::HashMap<String, String>,
) {
    if let Some(raw) = fields.get(&format!("entry.{uid}.__extra")) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            if let Ok(Value::Object(extra)) = serde_json::from_str::<Value>(trimmed) {
                for (extra_key, extra_value) in extra {
                    if extra_key == "uid" {
                        continue;
                    }
                    entry.insert(extra_key, extra_value);
                }
            }
        }
    }
}

/// Computes the pretty-JSON `__extra` blob for an entry: every field NOT in
/// `structured` (the fields that already have a dedicated control) so nothing is
/// silently un-editable. Returns "" when nothing is left over.
fn extra_json_for(entry: &Map<String, Value>, structured: &[&str]) -> String {
    let mut extra = Map::new();
    for (field_key, field_value) in entry {
        if !structured.contains(&field_key.as_str()) {
            extra.insert(field_key.clone(), field_value.clone());
        }
    }
    if extra.is_empty() {
        String::new()
    } else {
        serde_json::to_string_pretty(&Value::Object(extra)).unwrap_or_default()
    }
}

fn bool_field(entry: &Map<String, Value>, key: &str) -> bool {
    entry.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// Tri-state read: explicit `true`/`false` map to those, anything else (null /
/// absent / non-bool) is "unset".
fn tristate_of(entry: &Map<String, Value>, key: &str) -> String {
    match entry.get(key) {
        Some(Value::Bool(true)) => "true".to_string(),
        Some(Value::Bool(false)) => "false".to_string(),
        _ => "unset".to_string(),
    }
}

/// SillyTavern `world_info_position` options (value = wire int).
fn position_options(current: i64) -> Vec<SelectOption> {
    const POSITIONS: &[(i64, &str)] = &[
        (0, "Before char"),
        (1, "After char"),
        (2, "Author's Note top"),
        (3, "Author's Note bottom"),
        (4, "At depth"),
        (5, "Example messages top"),
        (6, "Example messages bottom"),
        (7, "Outlet"),
    ];
    let mut options: Vec<SelectOption> = POSITIONS
        .iter()
        .map(|(value, label)| SelectOption {
            value: value.to_string(),
            label: (*label).to_string(),
            selected: *value == current,
        })
        .collect();
    if !POSITIONS.iter().any(|(value, _)| *value == current) {
        options.push(SelectOption {
            value: current.to_string(),
            label: format!("Position {current}"),
            selected: true,
        });
    }
    options
}

/// `role` options (used at At-depth). Null/absent = "(default)".
fn role_options(current: Option<i64>) -> Vec<SelectOption> {
    const ROLES: &[(&str, &str)] = &[
        ("", "(default)"),
        ("0", "System"),
        ("1", "User"),
        ("2", "Assistant"),
    ];
    let current_str = current.map(|value| value.to_string()).unwrap_or_default();
    ROLES
        .iter()
        .map(|(value, label)| SelectOption {
            value: (*value).to_string(),
            label: (*label).to_string(),
            selected: *value == current_str,
        })
        .collect()
}

/// Builds the editable view from one entry object + its map key (uid fallback).
fn entry_edit_from(map_key: &str, entry: &Map<String, Value>) -> LoreEntryEdit {
    let uid = entry
        .get("uid")
        .map(value_to_plain_string)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| map_key.to_string());

    let comment = entry
        .get("comment")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let key = keys_to_text(entry.get("key").or_else(|| entry.get("keys")));
    let first_key = key.lines().next().unwrap_or("").to_string();
    let title = if !comment.trim().is_empty() {
        comment.clone()
    } else if !first_key.is_empty() {
        first_key
    } else {
        format!("Entry {uid}")
    };

    let position = entry.get("position").and_then(value_to_i64).unwrap_or(0);
    let role = entry.get("role").and_then(value_to_i64);

    // Leftover fields (no dedicated control) -> pretty JSON for the raw box.
    let mut extra = Map::new();
    for (field_key, field_value) in entry {
        if !STRUCTURED_FIELDS.contains(&field_key.as_str()) {
            extra.insert(field_key.clone(), field_value.clone());
        }
    }
    let extra_json = if extra.is_empty() {
        String::new()
    } else {
        serde_json::to_string_pretty(&Value::Object(extra)).unwrap_or_default()
    };

    let tristates = TRISTATE_FIELDS
        .iter()
        .map(|(field, label)| TriState {
            field: (*field).to_string(),
            label: (*label).to_string(),
            state: tristate_of(entry, field),
        })
        .collect();

    LoreEntryEdit {
        title,
        comment,
        content: entry
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        key,
        keysecondary: keys_to_text(entry.get("keysecondary")),
        constant: bool_field(entry, "constant"),
        selective: bool_field(entry, "selective"),
        vectorized: bool_field(entry, "vectorized"),
        disable: bool_field(entry, "disable"),
        use_probability: bool_field(entry, "useProbability"),
        exclude_recursion: bool_field(entry, "excludeRecursion"),
        prevent_recursion: bool_field(entry, "preventRecursion"),
        delay_until_recursion: bool_field(entry, "delayUntilRecursion"),
        group_override: bool_field(entry, "groupOverride"),
        add_memo: bool_field(entry, "addMemo"),
        order: num_to_text(entry.get("order")),
        depth: num_to_text(entry.get("depth")),
        probability: num_to_text(entry.get("probability")),
        scan_depth: num_to_text(entry.get("scanDepth")),
        group_weight: num_to_text(entry.get("groupWeight")),
        sticky: num_to_text(entry.get("sticky")),
        cooldown: num_to_text(entry.get("cooldown")),
        delay: num_to_text(entry.get("delay")),
        display_index: num_to_text(entry.get("displayIndex")),
        group: entry
            .get("group")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        automation_id: entry
            .get("automationId")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        position_options: position_options(position),
        role_options: role_options(role),
        tristates,
        extra_json,
        uid,
    }
}

fn value_to_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_f64().map(|number| number as i64))
        .or_else(|| value.as_str().and_then(|text| text.trim().parse().ok()))
}

/// A scalar as a plain string (no surrounding quotes for strings).
fn value_to_plain_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Reads the lorebook file as a `Value` and builds the editor view. The file is
/// the active profile's single `worlds/*.json` (legacy `<root>/worlds` fallback).
fn build_lorebook_view(
    state: &AppState,
    prefer_id: Option<&str>,
    saved: bool,
) -> Option<LorebookEditView> {
    let dir = state.config.active_profile_paths().worlds_dir();
    let name_hint = active_profile_name(state);
    let (id, path) = resolve_single_book(&dir, prefer_id, name_hint.as_deref())?;
    let text = std::fs::read_to_string(&path).ok()?;
    let root: Value = serde_json::from_str(&text).ok()?;

    let name = root
        .get("name")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or(&id)
        .to_string();

    let entries = root
        .get("entries")
        .and_then(Value::as_object)
        .map(|entries| {
            entries
                .iter()
                .map(|(map_key, value)| {
                    let empty = Map::new();
                    let object = value.as_object().unwrap_or(&empty);
                    entry_edit_from(map_key, object)
                })
                .collect()
        })
        .unwrap_or_default();

    Some(LorebookEditView {
        id,
        name,
        path: path.display().to_string(),
        entries,
        saved,
    })
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct LoreQuery {
    /// Optional explicit book id (file stem) when several books exist.
    id: Option<String>,
    /// Set to "1" after a save so the page shows the "Saved" confirmation.
    saved: Option<String>,
}

/// `GET /lorebook` — the full editor for the active profile's single lorebook.
pub async fn lorebook_editor(
    State(state): State<Arc<AppState>>,
    Query(query): Query<LoreQuery>,
) -> WebResult<Response> {
    let saved = query.saved.as_deref() == Some("1");
    let Some(book) = build_lorebook_view(&state, query.id.as_deref(), saved) else {
        return Ok(no_book_page(
            "Lorebook",
            "No lorebook file was found under the active profile (worlds/<id>.json).",
        ));
    };
    Ok(Html(LorebookEditorTemplate { book }.render()?).into_response())
}

/// `POST /lorebook` — overlay the submitted fields onto the ORIGINAL file Value
/// and write it back, preserving entry order, unknown entry fields, and
/// top-level keys (`extensions`, etc.). Redirects back to the editor with a
/// "Saved" flag.
pub async fn lorebook_save(
    State(state): State<Arc<AppState>>,
    Form(form): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    // Collapse the form pairs into a single map. Checkboxes only appear when
    // checked, so an absent key means "false"; the `__present` marker per uid
    // tells us which entries the form rendered (so we know an absent checkbox is
    // a real `false`, not just an un-rendered entry).
    let mut fields: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for (key, value) in &form {
        fields.insert(key.clone(), value.clone());
    }

    let prefer_id = fields.get("id").map(String::as_str);
    let dir = state.config.active_profile_paths().worlds_dir();
    let name_hint = active_profile_name(&state);
    let Some((id, path)) = resolve_single_book(&dir, prefer_id, name_hint.as_deref()) else {
        return Ok(no_book_page("Lorebook", "No lorebook file to save to."));
    };

    let text = std::fs::read_to_string(&path)?;
    let mut root: Value = serde_json::from_str(&text)?;
    if !root.is_object() {
        root = Value::Object(Map::new());
    }
    let root_obj = root.as_object_mut().expect("root is an object");

    // Book name.
    if let Some(name) = fields.get("name") {
        root_obj.insert("name".to_string(), Value::String(name.clone()));
    }

    // Ensure an `entries` object exists, preserving any existing entries+order.
    if !root_obj
        .get("entries")
        .map(Value::is_object)
        .unwrap_or(false)
    {
        root_obj.insert("entries".to_string(), Value::Object(Map::new()));
    }

    // Which uids were present in the form, and which were deleted.
    let deleted: std::collections::HashSet<String> = fields
        .get("deleted")
        .map(|csv| {
            csv.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    let present_uids: Vec<String> = fields
        .keys()
        .filter_map(|key| {
            key.strip_prefix("entry.")
                .and_then(|rest| rest.strip_suffix(".__present"))
                .map(str::to_string)
        })
        .collect();

    {
        let entries = root_obj
            .get_mut("entries")
            .and_then(Value::as_object_mut)
            .expect("entries is an object");

        // Apply deletes first (by uid == map key, matching how we render keys).
        for uid in &deleted {
            entries.remove(uid);
        }

        // Overlay edited fields onto each present entry (existing or new),
        // mutating the entry object in place so unknown fields + order survive.
        for uid in &present_uids {
            if deleted.contains(uid) {
                continue;
            }
            let entry = entries
                .entry(uid.clone())
                .or_insert_with(|| Value::Object(default_entry(uid)));
            if !entry.is_object() {
                *entry = Value::Object(default_entry(uid));
            }
            let object = entry.as_object_mut().expect("entry is an object");
            apply_entry_fields(object, uid, &fields);
        }
    }

    let pretty = serde_json::to_string_pretty(&root)?;
    std::fs::write(&path, pretty)?;
    tracing::info!(
        "saved lorebook '{}' ({} present, {} deleted) -> {}",
        id,
        present_uids.len(),
        deleted.len(),
        path.display()
    );

    let redirect = match prefer_id {
        Some(want) if !want.is_empty() => {
            format!("/lorebook?id={}&saved=1", urlencoding::encode(want))
        }
        _ => "/lorebook?saved=1".to_string(),
    };
    Ok(Redirect::to(&redirect).into_response())
}

/// A minimal entry object for a freshly-added card (uid set; everything else
/// gets filled by [`apply_entry_fields`] from the submitted form).
fn default_entry(uid: &str) -> Map<String, Value> {
    let mut map = Map::new();
    let uid_value = uid
        .parse::<i64>()
        .map(|n| Value::Number(n.into()))
        .unwrap_or_else(|_| Value::String(uid.to_string()));
    map.insert("uid".to_string(), uid_value);
    map
}

/// Overlays the submitted fields for one uid onto its entry object. Parse
/// failures keep the prior value (never corrupt). Field name scheme:
/// `entry.<uid>.<field>`; checkboxes present-only; tri-states a select with
/// `unset`/`true`/`false`; `__extra` a JSON object merged for the un-modelled
/// fields.
fn apply_entry_fields(
    entry: &mut Map<String, Value>,
    uid: &str,
    fields: &std::collections::HashMap<String, String>,
) {
    let get = |name: &str| fields.get(&format!("entry.{uid}.{name}"));
    let present = |name: &str| fields.contains_key(&format!("entry.{uid}.{name}"));

    // Text fields. Never invent an absent field as an empty string.
    for name in ["comment", "content", "group", "automationId"] {
        if let Some(value) = get(name) {
            if entry.contains_key(name) || !value.is_empty() {
                entry.insert(name.to_string(), Value::String(value.clone()));
            }
        }
    }

    // Key arrays (split on comma OR newline; trims; drops empties). An absent
    // field that is still empty stays absent.
    for name in ["key", "keysecondary"] {
        if let Some(raw) = get(name) {
            let keys: Vec<Value> = raw
                .split('\n')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(|part| Value::String(part.to_string()))
                .collect();
            if entry.contains_key(name) || !keys.is_empty() {
                entry.insert(name.to_string(), Value::Array(keys));
            }
        }
    }

    // Plain bool checkboxes (present == true, absent == false). Only written
    // when the entry was rendered (its `__present` marker is in the form), which
    // is always true here since we only call this for present uids.
    for name in [
        "constant",
        "selective",
        "vectorized",
        "disable",
        "useProbability",
        "excludeRecursion",
        "preventRecursion",
        "delayUntilRecursion",
        "groupOverride",
        "addMemo",
    ] {
        if entry.contains_key(name) || present(name) {
            entry.insert(name.to_string(), Value::Bool(present(name)));
        }
    }

    // Integer fields. Blank clears `scanDepth` to null (it is legitimately
    // nullable); the others keep their prior value when blank/unparsable.
    for name in [
        "order",
        "depth",
        "probability",
        "groupWeight",
        "sticky",
        "cooldown",
        "delay",
        "displayIndex",
    ] {
        if let Some(raw) = get(name) {
            let trimmed = raw.trim();
            if let Ok(parsed) = trimmed.parse::<i64>() {
                entry.insert(name.to_string(), Value::Number(parsed.into()));
            }
            // else: keep prior value (do not corrupt on a bad/blank int).
        }
    }
    if let Some(raw) = get("scanDepth") {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            if entry.contains_key("scanDepth") {
                entry.insert("scanDepth".to_string(), Value::Null);
            }
        } else if let Ok(parsed) = trimmed.parse::<i64>() {
            entry.insert("scanDepth".to_string(), Value::Number(parsed.into()));
        }
    }

    // Position (int enum). Keeps prior value on a bad parse; an absent field at
    // the default (0 = before char) stays absent.
    if let Some(raw) = get("position") {
        if let Ok(parsed) = raw.trim().parse::<i64>() {
            if entry.contains_key("position") || parsed != 0 {
                entry.insert("position".to_string(), Value::Number(parsed.into()));
            }
        }
    }

    // Role select: blank -> null (only for an existing field); otherwise the int.
    if let Some(raw) = get("role") {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            if entry.contains_key("role") {
                entry.insert("role".to_string(), Value::Null);
            }
        } else if let Ok(parsed) = trimmed.parse::<i64>() {
            entry.insert("role".to_string(), Value::Number(parsed.into()));
        }
    }

    // Tri-state selects -> null / true / false.
    for (name, _) in TRISTATE_FIELDS {
        if let Some(raw) = get(name) {
            let value = match raw.as_str() {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                _ => Value::Null,
            };
            // Don't invent an absent tri-state as null (unset == absent).
            if entry.contains_key(*name) || !value.is_null() {
                entry.insert((*name).to_string(), value);
            }
        }
    }

    // Raw-JSON "other fields" box: merge a submitted JSON object over the entry
    // (so the un-modelled fields are editable). Invalid JSON is ignored.
    if let Some(raw) = get("__extra") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            if let Ok(Value::Object(extra)) = serde_json::from_str::<Value>(trimmed) {
                for (extra_key, extra_value) in extra {
                    // Never let the raw box overwrite the identity field.
                    if extra_key == "uid" {
                        continue;
                    }
                    entry.insert(extra_key, extra_value);
                }
            }
        }
    }

    // Make sure uid is set (new entries; or one whose uid field was absent).
    entry
        .entry("uid".to_string())
        .or_insert_with(|| uid_value(uid));
}

fn uid_value(uid: &str) -> Value {
    uid.parse::<i64>()
        .map(|n| Value::Number(n.into()))
        .unwrap_or_else(|_| Value::String(uid.to_string()))
}

/// A small HTML page shown when a book file can't be resolved (200, themed).
fn no_book_page(kind: &str, message: &str) -> Response {
    let body = format!(
        "<main class=\"error-page\"><h1>chasm</h1><h2>{}</h2><p>{}</p>\
<p><a class=\"button\" href=\"/\">Back to chat</a></p></main>",
        html_escape(kind),
        html_escape(message)
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        body,
    )
        .into_response()
}

// ===========================================================================
// Quest Book editor — view model
// ===========================================================================

/// Quest entry fields with a dedicated control (structured input, bool/tri-state
/// select, labelled string-array textarea, or labelled JSON box). Everything NOT
/// here surfaces through the per-entry `__extra` raw-JSON box, keeping the editor
/// total even if new fields appear.
const QUEST_STRUCTURED_FIELDS: &[&str] = &[
    // identity + WI base
    "uid",
    "key",
    "keysecondary",
    "comment",
    "content",
    "constant",
    "vectorized",
    "selective",
    "selectiveLogic",
    "order",
    "priority",
    "disable",
    "caseSensitive",
    "matchWholeWords",
    // quest scalars
    "questId",
    "questName",
    "questEditorId",
    "formId",
    "targetGame",
    "questType",
    "phase",
    "offerSummary",
    "preDialogue",
    "availableWhen",
    "includeCompleted",
    "vectorizableText",
    // string arrays
    "giverCharacterIds",
    "giverNpcKeys",
    "relatedNpcKeys",
    "acceptanceCues",
    "refusalCues",
    "stageHints",
    "questEvents",
    "scopes",
    "tags",
    "include",
    "exclude",
    // nested JSON boxes
    "objectives",
    "sourceLinks",
    "extensions",
];

/// Quest entry string-array fields (field, label) rendered as textareas.
const QUEST_ARRAY_FIELDS: &[(&str, &str)] = &[
    ("giverCharacterIds", "Giver character ids"),
    ("giverNpcKeys", "Giver NPC keys"),
    ("relatedNpcKeys", "Related NPC keys"),
    ("acceptanceCues", "Acceptance cues"),
    ("refusalCues", "Refusal cues"),
    ("scopes", "Scopes"),
    ("tags", "Tags"),
    ("include", "Include"),
    ("exclude", "Exclude"),
];

/// Quest entry nested-JSON fields (field, label) rendered as JSON textareas.
/// `stageHints`/`questEvents` are arrays-of-objects; the rest are objects/arrays.
const QUEST_JSON_FIELDS: &[(&str, &str)] = &[
    ("objectives", "Objectives (string array)"),
    ("stageHints", "Stage hints (array of objects)"),
    ("questEvents", "Quest events (array of objects)"),
    ("availableWhen", "Available when (object)"),
    ("sourceLinks", "Source links (string array)"),
    ("extensions", "Extensions (object)"),
];

/// One quest entry, every field pre-rendered for the template.
struct QuestEntryEdit {
    uid: String,
    title: String,
    // Body
    comment: String,
    content: String,
    offer_summary: String,
    pre_dialogue: String,
    vectorizable_text: String,
    // Quest scalars
    quest_id: String,
    quest_name: String,
    quest_editor_id: String,
    form_id: String,
    target_game: String,
    quest_type: String,
    phase: String,
    include_completed: bool,
    // Matching & scope
    key: String,
    keysecondary: String,
    constant: bool,
    vectorized: bool,
    selective: bool,
    disable: bool,
    selective_logic: String,
    order: String,
    priority: String,
    tristates: Vec<TriState>,
    // String-array + JSON fields
    arrays: Vec<ArrayField>,
    json_fields: Vec<JsonField>,
    // Anything with no dedicated control.
    extra_json: String,
}

struct QuestBookEditView {
    id: String,
    name: String,
    description: String,
    path: String,
    entries: Vec<QuestEntryEdit>,
    saved: bool,
}

#[derive(Template)]
#[template(path = "quest_book_editor.html")]
struct QuestBookEditorTemplate {
    book: QuestBookEditView,
}

/// The nullable tri-state bools shared by quest entries.
const QUEST_TRISTATE_FIELDS: &[(&str, &str)] = &[
    ("caseSensitive", "Case sensitive"),
    ("matchWholeWords", "Match whole words"),
];

fn quest_entry_edit_from(map_key: &str, entry: &Map<String, Value>) -> QuestEntryEdit {
    let uid = entry
        .get("uid")
        .map(value_to_plain_string)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| map_key.to_string());

    let comment = entry
        .get("comment")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let quest_name = entry
        .get("questName")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let title = if !comment.trim().is_empty() {
        comment.clone()
    } else if !quest_name.trim().is_empty() {
        quest_name.clone()
    } else {
        format!("Quest {uid}")
    };

    let tristates = QUEST_TRISTATE_FIELDS
        .iter()
        .map(|(field, label)| TriState {
            field: (*field).to_string(),
            label: (*label).to_string(),
            state: tristate_of(entry, field),
        })
        .collect();

    let arrays = QUEST_ARRAY_FIELDS
        .iter()
        .map(|(field, label)| ArrayField::build(entry, field, label))
        .collect();
    let json_fields = QUEST_JSON_FIELDS
        .iter()
        .map(|(field, label)| JsonField::build(entry, field, label))
        .collect();

    let str_field = |name: &str| {
        entry
            .get(name)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };

    QuestEntryEdit {
        title,
        comment,
        content: str_field("content"),
        offer_summary: str_field("offerSummary"),
        pre_dialogue: str_field("preDialogue"),
        vectorizable_text: str_field("vectorizableText"),
        quest_id: str_field("questId"),
        quest_name,
        quest_editor_id: str_field("questEditorId"),
        form_id: str_field("formId"),
        target_game: str_field("targetGame"),
        quest_type: str_field("questType"),
        phase: str_field("phase"),
        include_completed: bool_field(entry, "includeCompleted"),
        key: keys_to_text(entry.get("key")),
        keysecondary: keys_to_text(entry.get("keysecondary")),
        constant: bool_field(entry, "constant"),
        vectorized: bool_field(entry, "vectorized"),
        selective: bool_field(entry, "selective"),
        disable: bool_field(entry, "disable"),
        selective_logic: num_to_text(entry.get("selectiveLogic")),
        order: num_to_text(entry.get("order")),
        priority: num_to_text(entry.get("priority")),
        tristates,
        arrays,
        json_fields,
        extra_json: extra_json_for(entry, QUEST_STRUCTURED_FIELDS),
        uid,
    }
}

/// A one-line summary of an object/array value for a read-only hint (the field
/// itself is edited via its JSON box). Empty for null/absent.
fn build_quest_book_view(
    state: &AppState,
    prefer_id: Option<&str>,
    saved: bool,
) -> Option<QuestBookEditView> {
    let dir = state.config.active_profile_paths().quest_books_dir();
    let name_hint = active_profile_name(state);
    let (id, path) = resolve_single_book(&dir, prefer_id, name_hint.as_deref())?;
    let text = std::fs::read_to_string(&path).ok()?;
    let root: Value = serde_json::from_str(&text).ok()?;

    let name = root
        .get("name")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or(&id)
        .to_string();
    let description = root
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let entries = root
        .get("entries")
        .and_then(Value::as_object)
        .map(|entries| {
            entries
                .iter()
                .map(|(map_key, value)| {
                    let empty = Map::new();
                    let object = value.as_object().unwrap_or(&empty);
                    quest_entry_edit_from(map_key, object)
                })
                .collect()
        })
        .unwrap_or_default();

    Some(QuestBookEditView {
        id,
        name,
        description,
        path: path.display().to_string(),
        entries,
        saved,
    })
}

/// `GET /questbook` — full editor for the active profile's single quest book.
pub async fn questbook_editor(
    State(state): State<Arc<AppState>>,
    Query(query): Query<LoreQuery>,
) -> WebResult<Response> {
    let saved = query.saved.as_deref() == Some("1");
    let Some(book) = build_quest_book_view(&state, query.id.as_deref(), saved) else {
        return Ok(no_book_page(
            "Quest Book",
            "No quest book file was found under the active profile (headless/quest-books/<id>.json).",
        ));
    };
    Ok(Html(QuestBookEditorTemplate { book }.render()?).into_response())
}

/// `POST /questbook` — overlay submitted fields onto the ORIGINAL file Value and
/// write it back, preserving entry order, unknown entry fields, and top-level
/// keys (`settings`, `extensions`, etc.). Redirects with a "Saved" flag.
pub async fn questbook_save(
    State(state): State<Arc<AppState>>,
    Form(form): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    let mut fields: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for (key, value) in &form {
        fields.insert(key.clone(), value.clone());
    }

    let prefer_id = fields.get("id").map(String::as_str);
    let dir = state.config.active_profile_paths().quest_books_dir();
    let name_hint = active_profile_name(&state);
    let Some((id, path)) = resolve_single_book(&dir, prefer_id, name_hint.as_deref()) else {
        return Ok(no_book_page("Quest Book", "No quest book file to save to."));
    };

    let text = std::fs::read_to_string(&path)?;
    let mut root: Value = serde_json::from_str(&text)?;
    if !root.is_object() {
        root = Value::Object(Map::new());
    }
    let root_obj = root.as_object_mut().expect("root is an object");

    // Book name + description (top-level, editable).
    if let Some(name) = fields.get("name") {
        root_obj.insert("name".to_string(), Value::String(name.clone()));
    }
    if let Some(description) = fields.get("description") {
        root_obj.insert(
            "description".to_string(),
            Value::String(description.clone()),
        );
    }

    if !root_obj
        .get("entries")
        .map(Value::is_object)
        .unwrap_or(false)
    {
        root_obj.insert("entries".to_string(), Value::Object(Map::new()));
    }

    let (deleted, present_uids) = deleted_and_present(&fields);

    {
        let entries = root_obj
            .get_mut("entries")
            .and_then(Value::as_object_mut)
            .expect("entries is an object");
        for uid in &deleted {
            entries.remove(uid);
        }
        for uid in &present_uids {
            if deleted.contains(uid) {
                continue;
            }
            let entry = entries
                .entry(uid.clone())
                .or_insert_with(|| Value::Object(default_entry(uid)));
            if !entry.is_object() {
                *entry = Value::Object(default_entry(uid));
            }
            let object = entry.as_object_mut().expect("entry is an object");
            apply_quest_entry_fields(object, uid, &fields);
        }
    }

    let pretty = serde_json::to_string_pretty(&root)?;
    std::fs::write(&path, pretty)?;
    tracing::info!(
        "saved quest book '{}' ({} present, {} deleted) -> {}",
        id,
        present_uids.len(),
        deleted.len(),
        path.display()
    );

    Ok(Redirect::to(&saved_redirect("/questbook", prefer_id)).into_response())
}

/// Overlays the submitted fields for one quest uid onto its entry object.
fn apply_quest_entry_fields(
    entry: &mut Map<String, Value>,
    uid: &str,
    fields: &std::collections::HashMap<String, String>,
) {
    // Scalar text fields.
    for name in [
        "comment",
        "content",
        "offerSummary",
        "preDialogue",
        "vectorizableText",
        "questId",
        "questName",
        "questEditorId",
        "formId",
        "targetGame",
        "questType",
        "phase",
    ] {
        apply_text_field(entry, uid, name, fields);
    }

    // Key arrays (WI base).
    for name in ["key", "keysecondary"] {
        apply_array_field(entry, uid, name, fields);
    }
    // Quest string arrays.
    for (name, _) in QUEST_ARRAY_FIELDS {
        apply_array_field(entry, uid, name, fields);
    }

    // Bool checkboxes.
    for name in [
        "constant",
        "vectorized",
        "selective",
        "disable",
        "includeCompleted",
    ] {
        apply_bool_field(entry, uid, name, fields);
    }

    // Integer scalars (blank keeps prior value).
    for name in ["selectiveLogic", "order", "priority"] {
        apply_int_field(entry, uid, name, fields);
    }

    // Tri-state selects.
    for (name, _) in QUEST_TRISTATE_FIELDS {
        if let Some(raw) = fields.get(&format!("entry.{uid}.{name}")) {
            let value = match raw.as_str() {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                _ => Value::Null,
            };
            // Don't invent an absent tri-state as null (unset == absent).
            if entry.contains_key(*name) || !value.is_null() {
                entry.insert((*name).to_string(), value);
            }
        }
    }

    // Nested-JSON boxes.
    for (name, _) in QUEST_JSON_FIELDS {
        apply_json_field(entry, uid, name, fields);
    }

    // Catch-all raw-JSON box, then ensure uid.
    apply_extra_json(entry, uid, fields);
    entry
        .entry("uid".to_string())
        .or_insert_with(|| uid_value(uid));
}

// ===========================================================================
// Action Book editor — view model
// ===========================================================================

/// Action entry fields with a dedicated control. Everything NOT here surfaces
/// through the per-entry `__extra` raw-JSON box.
const ACTION_STRUCTURED_FIELDS: &[&str] = &[
    // identity + WI base
    "uid",
    "key",
    "keysecondary",
    "comment",
    "content",
    "constant",
    "vectorized",
    "selective",
    "selectiveLogic",
    "addMemo",
    "order",
    "position",
    "disable",
    "ignoreBudget",
    "excludeRecursion",
    "preventRecursion",
    "delayUntilRecursion",
    "probability",
    "useProbability",
    "depth",
    "group",
    "groupOverride",
    "groupWeight",
    "scanDepth",
    "caseSensitive",
    "matchWholeWords",
    "useGroupScoring",
    "automationId",
    "role",
    "sticky",
    "cooldown",
    "delay",
    // match* family
    "matchPersonaDescription",
    "matchCharacterDescription",
    "matchCharacterPersonality",
    "matchCharacterDepthPrompt",
    "matchScenario",
    "matchCreatorNotes",
    // action scalars
    "actionId",
    "riskTier",
    "targetGame",
    "pluginSource",
    "commandTemplate",
    "vectorizableText",
    // string arrays
    "triggers",
    "examplesWhenToUse",
    "examplesWhenNotToUse",
    "scopes",
    "tags",
    "include",
    "exclude",
    "vectorSearchTexts",
    // nested JSON
    "parametersSchema",
    "preconditions",
    "effects",
    "sourceLinks",
    "execution",
    "scopedCatalogs",
    "binding",
    "extensions",
];

/// Action string-array fields (field, label) rendered as textareas.
const ACTION_ARRAY_FIELDS: &[(&str, &str)] = &[
    ("triggers", "Triggers"),
    ("examplesWhenToUse", "Examples — when to use"),
    ("examplesWhenNotToUse", "Examples — when NOT to use"),
    ("scopes", "Scopes"),
    ("tags", "Tags"),
    ("include", "Include"),
    ("exclude", "Exclude"),
    ("vectorSearchTexts", "Vector search texts"),
];

/// Action nested-JSON fields (field, label) rendered as JSON textareas.
const ACTION_JSON_FIELDS: &[(&str, &str)] = &[
    ("parametersSchema", "Parameters schema (object)"),
    ("preconditions", "Preconditions (array)"),
    ("effects", "Effects (array)"),
    ("execution", "Execution (object)"),
    ("scopedCatalogs", "Scoped catalogs (array)"),
    ("binding", "Binding (object)"),
    ("sourceLinks", "Source links (string array)"),
    ("extensions", "Extensions (object)"),
];

/// The plain match*/recursion/budget bools shown in the Advanced section.
const ACTION_ADVANCED_BOOLS: &[(&str, &str)] = &[
    ("matchPersonaDescription", "Match persona description"),
    ("matchCharacterDescription", "Match character description"),
    ("matchCharacterPersonality", "Match character personality"),
    ("matchCharacterDepthPrompt", "Match character depth prompt"),
    ("matchScenario", "Match scenario"),
    ("matchCreatorNotes", "Match creator notes"),
    ("ignoreBudget", "Ignore budget"),
    ("excludeRecursion", "Exclude recursion"),
    ("preventRecursion", "Prevent recursion"),
];

/// Action tri-state nullable bools.
const ACTION_TRISTATE_FIELDS: &[(&str, &str)] = &[
    ("caseSensitive", "Case sensitive"),
    ("matchWholeWords", "Match whole words"),
    ("useGroupScoring", "Use group scoring"),
];

/// `riskTier` select options (the value is shown plainly; unknown values are
/// preserved as their own option so nothing is lost).
fn risk_tier_options(current: &str) -> Vec<SelectOption> {
    const TIERS: &[&str] = &["low", "medium", "high"];
    let mut options: Vec<SelectOption> = TIERS
        .iter()
        .map(|tier| SelectOption {
            value: (*tier).to_string(),
            label: (*tier).to_string(),
            selected: tier.eq_ignore_ascii_case(current),
        })
        .collect();
    if !current.is_empty() && !TIERS.iter().any(|tier| tier.eq_ignore_ascii_case(current)) {
        options.push(SelectOption {
            value: current.to_string(),
            label: current.to_string(),
            selected: true,
        });
    }
    options
}

/// One action entry, every field pre-rendered for the template.
struct ActionEntryEdit {
    uid: String,
    title: String,
    risk_tier: String,
    // Body
    comment: String,
    content: String,
    command_template: String,
    vectorizable_text: String,
    // Action scalars
    action_id: String,
    target_game: String,
    plugin_source: String,
    risk_tier_options: Vec<SelectOption>,
    // Matching & scope
    key: String,
    keysecondary: String,
    constant: bool,
    vectorized: bool,
    selective: bool,
    disable: bool,
    add_memo: bool,
    use_probability: bool,
    group_override: bool,
    selective_logic: String,
    order: String,
    probability: String,
    depth: String,
    group: String,
    group_weight: String,
    automation_id: String,
    position_options: Vec<SelectOption>,
    role_options: Vec<SelectOption>,
    // Advanced
    delay_until_recursion: String,
    scan_depth: String,
    sticky: String,
    cooldown: String,
    delay: String,
    advanced_bools: Vec<BoolField>,
    tristates: Vec<TriState>,
    // Arrays + JSON
    arrays: Vec<ArrayField>,
    json_fields: Vec<JsonField>,
    extra_json: String,
}

/// A plain bool field (label + checked) for the Advanced section's many bools.
struct BoolField {
    field: String,
    label: String,
    checked: bool,
}

struct ActionBookEditView {
    id: String,
    name: String,
    description: String,
    path: String,
    entries: Vec<ActionEntryEdit>,
    saved: bool,
}

#[derive(Template)]
#[template(path = "action_book_editor.html")]
struct ActionBookEditorTemplate {
    book: ActionBookEditView,
}

fn action_entry_edit_from(map_key: &str, entry: &Map<String, Value>) -> ActionEntryEdit {
    let uid = entry
        .get("uid")
        .map(value_to_plain_string)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| map_key.to_string());

    let comment = entry
        .get("comment")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let action_id = entry
        .get("actionId")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let title = if !comment.trim().is_empty() {
        comment.clone()
    } else if !action_id.trim().is_empty() {
        action_id.clone()
    } else {
        format!("Action {uid}")
    };
    let risk_tier = entry
        .get("riskTier")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let position = entry.get("position").and_then(value_to_i64).unwrap_or(0);
    let role = entry.get("role").and_then(value_to_i64);

    let advanced_bools = ACTION_ADVANCED_BOOLS
        .iter()
        .map(|(field, label)| BoolField {
            field: (*field).to_string(),
            label: (*label).to_string(),
            checked: bool_field(entry, field),
        })
        .collect();
    let tristates = ACTION_TRISTATE_FIELDS
        .iter()
        .map(|(field, label)| TriState {
            field: (*field).to_string(),
            label: (*label).to_string(),
            state: tristate_of(entry, field),
        })
        .collect();
    let arrays = ACTION_ARRAY_FIELDS
        .iter()
        .map(|(field, label)| ArrayField::build(entry, field, label))
        .collect();
    let json_fields = ACTION_JSON_FIELDS
        .iter()
        .map(|(field, label)| JsonField::build(entry, field, label))
        .collect();

    let str_field = |name: &str| {
        entry
            .get(name)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };

    ActionEntryEdit {
        title,
        comment,
        content: str_field("content"),
        command_template: str_field("commandTemplate"),
        vectorizable_text: str_field("vectorizableText"),
        action_id,
        target_game: str_field("targetGame"),
        plugin_source: str_field("pluginSource"),
        risk_tier_options: risk_tier_options(&risk_tier),
        key: keys_to_text(entry.get("key")),
        keysecondary: keys_to_text(entry.get("keysecondary")),
        constant: bool_field(entry, "constant"),
        vectorized: bool_field(entry, "vectorized"),
        selective: bool_field(entry, "selective"),
        disable: bool_field(entry, "disable"),
        add_memo: bool_field(entry, "addMemo"),
        use_probability: bool_field(entry, "useProbability"),
        group_override: bool_field(entry, "groupOverride"),
        selective_logic: num_to_text(entry.get("selectiveLogic")),
        order: num_to_text(entry.get("order")),
        probability: num_to_text(entry.get("probability")),
        depth: num_to_text(entry.get("depth")),
        group: str_field("group"),
        group_weight: num_to_text(entry.get("groupWeight")),
        automation_id: str_field("automationId"),
        position_options: position_options(position),
        role_options: role_options(role),
        delay_until_recursion: num_to_text(entry.get("delayUntilRecursion")),
        scan_depth: num_to_text(entry.get("scanDepth")),
        sticky: num_to_text(entry.get("sticky")),
        cooldown: num_to_text(entry.get("cooldown")),
        delay: num_to_text(entry.get("delay")),
        advanced_bools,
        tristates,
        arrays,
        json_fields,
        risk_tier,
        extra_json: extra_json_for(entry, ACTION_STRUCTURED_FIELDS),
        uid,
    }
}

fn build_action_book_view(
    state: &AppState,
    prefer_id: Option<&str>,
    saved: bool,
) -> Option<ActionBookEditView> {
    let dir = state.config.active_profile_paths().action_books_dir();
    let name_hint = active_profile_name(state);
    let (id, path) = resolve_single_book(&dir, prefer_id, name_hint.as_deref())?;
    let text = std::fs::read_to_string(&path).ok()?;
    let root: Value = serde_json::from_str(&text).ok()?;

    let name = root
        .get("name")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or(&id)
        .to_string();
    let description = root
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let entries = root
        .get("entries")
        .and_then(Value::as_object)
        .map(|entries| {
            entries
                .iter()
                .map(|(map_key, value)| {
                    let empty = Map::new();
                    let object = value.as_object().unwrap_or(&empty);
                    action_entry_edit_from(map_key, object)
                })
                .collect()
        })
        .unwrap_or_default();

    Some(ActionBookEditView {
        id,
        name,
        description,
        path: path.display().to_string(),
        entries,
        saved,
    })
}

/// `GET /actionbook` — full editor for the active profile's single action book.
pub async fn actionbook_editor(
    State(state): State<Arc<AppState>>,
    Query(query): Query<LoreQuery>,
) -> WebResult<Response> {
    let saved = query.saved.as_deref() == Some("1");
    let Some(book) = build_action_book_view(&state, query.id.as_deref(), saved) else {
        return Ok(no_book_page(
            "Action Book",
            "No action book file was found under the active profile (headless/action-books/<id>.json).",
        ));
    };
    Ok(Html(ActionBookEditorTemplate { book }.render()?).into_response())
}

/// `POST /actionbook` — overlay submitted fields onto the ORIGINAL file Value and
/// write it back, preserving entry order, unknown entry fields, and ALL top-level
/// keys (`settings`, `extensions`, `binding`, `catalogs`). Redirects with a
/// "Saved" flag.
pub async fn actionbook_save(
    State(state): State<Arc<AppState>>,
    Form(form): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    let mut fields: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for (key, value) in &form {
        fields.insert(key.clone(), value.clone());
    }

    let prefer_id = fields.get("id").map(String::as_str);
    let dir = state.config.active_profile_paths().action_books_dir();
    let name_hint = active_profile_name(&state);
    let Some((id, path)) = resolve_single_book(&dir, prefer_id, name_hint.as_deref()) else {
        return Ok(no_book_page(
            "Action Book",
            "No action book file to save to.",
        ));
    };

    let text = std::fs::read_to_string(&path)?;
    let mut root: Value = serde_json::from_str(&text)?;
    if !root.is_object() {
        root = Value::Object(Map::new());
    }
    let root_obj = root.as_object_mut().expect("root is an object");

    if let Some(name) = fields.get("name") {
        root_obj.insert("name".to_string(), Value::String(name.clone()));
    }
    if let Some(description) = fields.get("description") {
        root_obj.insert(
            "description".to_string(),
            Value::String(description.clone()),
        );
    }

    if !root_obj
        .get("entries")
        .map(Value::is_object)
        .unwrap_or(false)
    {
        root_obj.insert("entries".to_string(), Value::Object(Map::new()));
    }

    let (deleted, present_uids) = deleted_and_present(&fields);

    {
        let entries = root_obj
            .get_mut("entries")
            .and_then(Value::as_object_mut)
            .expect("entries is an object");
        for uid in &deleted {
            entries.remove(uid);
        }
        for uid in &present_uids {
            if deleted.contains(uid) {
                continue;
            }
            let entry = entries
                .entry(uid.clone())
                .or_insert_with(|| Value::Object(default_entry(uid)));
            if !entry.is_object() {
                *entry = Value::Object(default_entry(uid));
            }
            let object = entry.as_object_mut().expect("entry is an object");
            apply_action_entry_fields(object, uid, &fields);
        }
    }

    let pretty = serde_json::to_string_pretty(&root)?;
    std::fs::write(&path, pretty)?;
    tracing::info!(
        "saved action book '{}' ({} present, {} deleted) -> {}",
        id,
        present_uids.len(),
        deleted.len(),
        path.display()
    );

    Ok(Redirect::to(&saved_redirect("/actionbook", prefer_id)).into_response())
}

/// Overlays the submitted fields for one action uid onto its entry object.
fn apply_action_entry_fields(
    entry: &mut Map<String, Value>,
    uid: &str,
    fields: &std::collections::HashMap<String, String>,
) {
    // Scalar text fields.
    for name in [
        "comment",
        "content",
        "commandTemplate",
        "vectorizableText",
        "actionId",
        "riskTier",
        "targetGame",
        "pluginSource",
        "group",
        "automationId",
    ] {
        apply_text_field(entry, uid, name, fields);
    }

    // Key arrays (WI base) + action string arrays.
    for name in ["key", "keysecondary"] {
        apply_array_field(entry, uid, name, fields);
    }
    for (name, _) in ACTION_ARRAY_FIELDS {
        apply_array_field(entry, uid, name, fields);
    }

    // Plain bool checkboxes: the WI-base bools, plus every Advanced bool.
    for name in [
        "constant",
        "vectorized",
        "selective",
        "disable",
        "addMemo",
        "useProbability",
        "groupOverride",
    ] {
        apply_bool_field(entry, uid, name, fields);
    }
    for (name, _) in ACTION_ADVANCED_BOOLS {
        apply_bool_field(entry, uid, name, fields);
    }

    // Integer scalars (blank keeps prior value). `delayUntilRecursion` is an int
    // in the data (not a bool), so it lives here.
    for name in [
        "selectiveLogic",
        "order",
        "probability",
        "depth",
        "groupWeight",
        "delayUntilRecursion",
    ] {
        apply_int_field(entry, uid, name, fields);
    }

    // Position (int enum): keep prior value on a bad parse; an absent field at
    // the default (0) stays absent (the select always submits a value).
    if let Some(raw) = fields.get(&format!("entry.{uid}.position")) {
        if let Ok(parsed) = raw.trim().parse::<i64>() {
            if entry.contains_key("position") || parsed != 0 {
                entry.insert("position".to_string(), Value::Number(parsed.into()));
            }
        }
    }

    // Nullable ints: blank -> null.
    for name in ["scanDepth", "sticky", "cooldown", "delay"] {
        apply_nullable_int_field(entry, uid, name, fields);
    }

    // Role select: blank -> null (only for an existing field), else the int.
    if let Some(raw) = fields.get(&format!("entry.{uid}.role")) {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            if entry.contains_key("role") {
                entry.insert("role".to_string(), Value::Null);
            }
        } else if let Ok(parsed) = trimmed.parse::<i64>() {
            entry.insert("role".to_string(), Value::Number(parsed.into()));
        }
    }

    // Tri-state selects.
    for (name, _) in ACTION_TRISTATE_FIELDS {
        if let Some(raw) = fields.get(&format!("entry.{uid}.{name}")) {
            let value = match raw.as_str() {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                _ => Value::Null,
            };
            // Don't invent an absent tri-state as null (unset == absent).
            if entry.contains_key(*name) || !value.is_null() {
                entry.insert((*name).to_string(), value);
            }
        }
    }

    // Nested-JSON boxes.
    for (name, _) in ACTION_JSON_FIELDS {
        apply_json_field(entry, uid, name, fields);
    }

    // Catch-all raw-JSON box, then ensure uid.
    apply_extra_json(entry, uid, fields);
    entry
        .entry("uid".to_string())
        .or_insert_with(|| uid_value(uid));
}

// ---------------------------------------------------------------------------
// Shared save helpers (deletes/present parse + redirect)
// ---------------------------------------------------------------------------

/// Parses the `deleted` CSV and the set of `__present` uids out of a submitted
/// form. Shared by all three editors' save handlers.
fn deleted_and_present(
    fields: &std::collections::HashMap<String, String>,
) -> (std::collections::HashSet<String>, Vec<String>) {
    let deleted: std::collections::HashSet<String> = fields
        .get("deleted")
        .map(|csv| {
            csv.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let present_uids: Vec<String> = fields
        .keys()
        .filter_map(|key| {
            key.strip_prefix("entry.")
                .and_then(|rest| rest.strip_suffix(".__present"))
                .map(str::to_string)
        })
        .collect();
    (deleted, present_uids)
}

/// Builds the post-save redirect target, carrying `?id=` when one was submitted.
fn saved_redirect(base: &str, prefer_id: Option<&str>) -> String {
    match prefer_id {
        Some(want) if !want.is_empty() => {
            format!("{base}?id={}&saved=1", urlencoding::encode(want))
        }
        _ => format!("{base}?saved=1"),
    }
}
