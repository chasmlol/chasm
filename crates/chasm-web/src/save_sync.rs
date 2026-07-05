//! Save-sync subsystem, ported faithfully from the Node headless runtime
//! (`src/headless/save-sync.js`). It checkpoints bridge-owned chat/game state
//! when the player saves and restores it when they load.
//!
//! The FNV helper (`tools/fnv/nvbridge-helper.mjs`) only ever calls
//! `POST /save-sync/events` with a `{event, gameId, saveId, ...}` body, so that
//! single route is the public surface. Internally it dispatches:
//!
//! * `save`/`saved`/`checkpoint`/`autosave`/`quicksave` -> create a checkpoint.
//! * `load`/`loaded`/`restore`/`reload`                 -> restore one.
//!
//! ## On-disk layout (under `<data_root>/headless/save-sync/`)
//! * `index.json`               — the store: `{version, current, items, events}`.
//! * `checkpoints/<id>.json`    — one full snapshot per checkpoint id.
//! * `restore-backups/<id>.json`— safety backups written before a restore.
//!
//! This mirrors ST's location/filenames exactly so a checkpoint written by Rust
//! round-trips with the Node implementation (JSON object key order is not
//! semantically meaningful; ST and Rust both look entries up by key).
//!
//! ## Safety
//! Restore *writes* files under the data root. Every session-file path is
//! re-derived from the snapshot's session id and verified to resolve **inside**
//! the data root before any write/delete, guarding against path traversal.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use axum::{extract::State, Json};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::Serialize as _;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::{AppState, WebError, WebResult};

const SAVE_SYNC_DIRECTORY: &str = "save-sync";
const SAVE_SYNC_INDEX_FILE: &str = "index.json";
const CHECKPOINTS_DIRECTORY: &str = "checkpoints";
const RESTORE_BACKUPS_DIRECTORY: &str = "restore-backups";
const SAVE_SYNC_VERSION: u64 = 1;
const MAX_EVENTS: usize = 500;

// ---------------------------------------------------------------------------
// Options (mirrors DEFAULT_SAVE_SYNC_OPTIONS + normalizeSaveSyncOptions)
// ---------------------------------------------------------------------------

/// Effective save-sync options. Mirrors ST's `DEFAULT_SAVE_SYNC_OPTIONS` shape
/// and defaults (everything enabled, retention 50).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SaveSyncOptions {
    enabled: bool,
    auto_checkpoint: bool,
    auto_restore: bool,
    include_live_chats: bool,
    include_world_state: bool,
    include_chat_files: bool,
    include_participant_projections: bool,
    create_safety_backup: bool,
    delete_newer_chat_files: bool,
    retention_limit: i64,
}

impl Default for SaveSyncOptions {
    fn default() -> Self {
        // == DEFAULT_SAVE_SYNC_OPTIONS (save-sync.js:22).
        Self {
            enabled: true,
            auto_checkpoint: true,
            auto_restore: true,
            include_live_chats: true,
            include_world_state: true,
            include_chat_files: true,
            include_participant_projections: true,
            create_safety_backup: true,
            delete_newer_chat_files: true,
            retention_limit: 50,
        }
    }
}

impl SaveSyncOptions {
    /// Serializes to the JSON shape ST stores/returns (camelCase keys).
    fn to_json(self) -> Value {
        json!({
            "enabled": self.enabled,
            "autoCheckpoint": self.auto_checkpoint,
            "autoRestore": self.auto_restore,
            "includeLiveChats": self.include_live_chats,
            "includeWorldState": self.include_world_state,
            "includeChatFiles": self.include_chat_files,
            "includeParticipantProjections": self.include_participant_projections,
            "createSafetyBackup": self.create_safety_backup,
            "deleteNewerChatFiles": self.delete_newer_chat_files,
            "retentionLimit": self.retention_limit,
        })
    }
}

/// Reads a boolean option accepting both camelCase and snake_case aliases,
/// falling back to `fallback` when neither is present. Mirrors
/// `normalizeBooleanOption`: any present value is coerced via JS `Boolean(...)`.
fn normalize_boolean_option(source: &Value, camel: &str, snake: &str, fallback: bool) -> bool {
    let value = source.get(camel).or_else(|| source.get(snake));
    match value {
        None | Some(Value::Null) => fallback,
        Some(v) => js_truthy(v),
    }
}

/// JS `Boolean(value)` semantics for the JSON values an option may hold.
fn js_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0 && !f.is_nan()).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        // Objects and arrays are always truthy in JS.
        Value::Object(_) | Value::Array(_) => true,
    }
}

/// Mirrors `normalizeRetentionLimit`: truncate to integer, clamp to `[0, 500]`,
/// and fall back when not finite.
fn normalize_retention_limit(value: Option<&Value>, fallback: i64) -> i64 {
    let raw = match value {
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
        _ => None,
    };
    match raw {
        Some(f) if f.is_finite() => (f.trunc() as i64).clamp(0, 500),
        _ => fallback,
    }
}

/// Normalizes raw options over a fallback set. Mirrors
/// `normalizeSaveSyncOptions(input, fallback)`.
fn normalize_save_sync_options(input: &Value, fallback: SaveSyncOptions) -> SaveSyncOptions {
    let source = if input.is_object() {
        input.clone()
    } else {
        Value::Object(Map::new())
    };
    SaveSyncOptions {
        enabled: normalize_boolean_option(&source, "enabled", "enabled", fallback.enabled),
        auto_checkpoint: normalize_boolean_option(
            &source,
            "autoCheckpoint",
            "auto_checkpoint",
            fallback.auto_checkpoint,
        ),
        auto_restore: normalize_boolean_option(
            &source,
            "autoRestore",
            "auto_restore",
            fallback.auto_restore,
        ),
        include_live_chats: normalize_boolean_option(
            &source,
            "includeLiveChats",
            "include_live_chats",
            fallback.include_live_chats,
        ),
        include_world_state: normalize_boolean_option(
            &source,
            "includeWorldState",
            "include_world_state",
            fallback.include_world_state,
        ),
        include_chat_files: normalize_boolean_option(
            &source,
            "includeChatFiles",
            "include_chat_files",
            fallback.include_chat_files,
        ),
        include_participant_projections: normalize_boolean_option(
            &source,
            "includeParticipantProjections",
            "include_participant_projections",
            fallback.include_participant_projections,
        ),
        create_safety_backup: normalize_boolean_option(
            &source,
            "createSafetyBackup",
            "create_safety_backup",
            fallback.create_safety_backup,
        ),
        delete_newer_chat_files: normalize_boolean_option(
            &source,
            "deleteNewerChatFiles",
            "delete_newer_chat_files",
            fallback.delete_newer_chat_files,
        ),
        retention_limit: normalize_retention_limit(
            source
                .get("retentionLimit")
                .or_else(|| source.get("retention_limit")),
            fallback.retention_limit,
        ),
    }
}

/// Reads the configured save-sync settings, then layers per-request `overrides`
/// over them. Mirrors `getEffectiveSaveSyncOptions`: ST reads the live-chat
/// extension settings (`extension_settings.live_chat.save_sync`) from the user's
/// `settings.json`; we read the same file under the data root when present,
/// otherwise fall back to `DEFAULT_SAVE_SYNC_OPTIONS`.
fn effective_save_sync_options(data_root: &Path, overrides: &Value) -> SaveSyncOptions {
    let configured = read_configured_save_sync_options(data_root);
    normalize_save_sync_options(overrides, configured)
}

/// Reads `extension_settings.live_chat.save_sync` from `<data_root>/settings.json`
/// and normalizes it over the hard defaults. Mirrors
/// `readLiveChatExtensionSettings(...).saveSync`. A missing/invalid file yields
/// the defaults (same as ST's try/catch fallback).
fn read_configured_save_sync_options(data_root: &Path) -> SaveSyncOptions {
    let settings_path = data_root.join("settings.json");
    let Ok(text) = fs::read_to_string(&settings_path) else {
        return SaveSyncOptions::default();
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return SaveSyncOptions::default();
    };
    let save_sync = value
        .get("extension_settings")
        .and_then(|v| v.get("live_chat"))
        .and_then(|v| v.get("save_sync"))
        .cloned()
        .unwrap_or(Value::Null);
    normalize_save_sync_options(&save_sync, SaveSyncOptions::default())
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// `<data_root>/headless` — mirrors `getHeadlessDataDirectory`.
fn headless_dir(data_root: &Path) -> PathBuf {
    data_root.join("headless")
}

/// `<data_root>/headless/save-sync` — mirrors `getSaveSyncDirectory`.
fn save_sync_dir(data_root: &Path) -> PathBuf {
    headless_dir(data_root).join(SAVE_SYNC_DIRECTORY)
}

fn save_sync_index_path(data_root: &Path) -> PathBuf {
    save_sync_dir(data_root).join(SAVE_SYNC_INDEX_FILE)
}

/// Validates a checkpoint id is filesystem-safe (mirrors `validateSafeId`):
/// non-empty, no NUL/`/`/`\`/`..`, at most 256 chars.
fn validate_safe_id(value: &str, name: &str) -> WebResult<String> {
    let id = value.trim();
    if id.is_empty() {
        return Err(web_err(&format!("{name} must be a non-empty string.")));
    }
    if id.contains('\0') || id.contains('/') || id.contains('\\') || id.contains("..") {
        return Err(web_err(&format!(
            "{name} contains invalid path characters."
        )));
    }
    if id.chars().count() > 256 {
        return Err(web_err(&format!("{name} is too long.")));
    }
    Ok(id.to_string())
}

fn checkpoint_file_path(data_root: &Path, checkpoint_id: &str) -> WebResult<PathBuf> {
    let id = validate_safe_id(checkpoint_id, "checkpointId")?;
    Ok(save_sync_dir(data_root)
        .join(CHECKPOINTS_DIRECTORY)
        .join(format!("{id}.json")))
}

fn restore_backup_file_path(data_root: &Path, file_id: &str) -> WebResult<PathBuf> {
    let id = validate_safe_id(file_id, "backupId")?;
    Ok(save_sync_dir(data_root)
        .join(RESTORE_BACKUPS_DIRECTORY)
        .join(format!("{id}.json")))
}

// ---------------------------------------------------------------------------
// JSON write helpers (4-space indent, mirroring ST's JSON.stringify(_, null, 4))
// ---------------------------------------------------------------------------

/// Serializes `value` with a 4-space indent to match ST's on-disk formatting.
/// (Indentation is irrelevant to round-tripping, but we mirror it for parity.)
fn to_pretty_json(value: &Value) -> String {
    let mut buf = Vec::new();
    let indent = b"    ";
    let formatter = serde_json::ser::PrettyFormatter::with_indent(indent);
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
    value
        .serialize(&mut ser)
        .expect("serializing a serde_json::Value never fails");
    String::from_utf8(buf).expect("serde_json emits valid UTF-8")
}

/// Writes `value` as pretty JSON to `path`, creating parent dirs.
fn write_json_file(path: &Path, value: &Value) -> WebResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, to_pretty_json(value))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Store (index.json) — mirrors readSaveSyncStore / writeSaveSyncStore
// ---------------------------------------------------------------------------

/// In-memory view of `index.json`. `items` is keyed by checkpoint id.
#[derive(Debug, Default)]
struct SaveSyncStore {
    current: Value,
    items: Map<String, Value>,
    events: Vec<Value>,
}

/// Reads the store, normalizing missing/invalid files to an empty store.
/// Mirrors `readSaveSyncStore` (truncates events to the last `MAX_EVENTS`).
fn read_save_sync_store(data_root: &Path) -> SaveSyncStore {
    let path = save_sync_index_path(data_root);
    let Ok(text) = fs::read_to_string(&path) else {
        return SaveSyncStore::default();
    };
    let Ok(parsed) = serde_json::from_str::<Value>(&text) else {
        return SaveSyncStore::default();
    };
    let Value::Object(obj) = parsed else {
        return SaveSyncStore::default();
    };
    let current = obj.get("current").cloned().unwrap_or(Value::Null);
    let items = match obj.get("items") {
        Some(Value::Object(map)) => map.clone(),
        _ => Map::new(),
    };
    let events = match obj.get("events") {
        Some(Value::Array(arr)) => last_n(arr, MAX_EVENTS),
        _ => Vec::new(),
    };
    SaveSyncStore {
        current: if current.is_object() {
            current
        } else {
            Value::Null
        },
        items,
        events,
    }
}

/// Writes the store back to `index.json`. Mirrors `writeSaveSyncStore`.
fn write_save_sync_store(data_root: &Path, store: &SaveSyncStore) -> WebResult<()> {
    let events = last_n(&store.events, MAX_EVENTS);
    let value = json!({
        "version": SAVE_SYNC_VERSION,
        "current": if store.current.is_object() { store.current.clone() } else { Value::Null },
        "items": Value::Object(store.items.clone()),
        "events": Value::Array(events),
    });
    write_json_file(&save_sync_index_path(data_root), &value)
}

/// Appends an event (stamped with `createdAt`) and truncates to `MAX_EVENTS`.
/// Mirrors `appendStoreEvent`.
fn append_store_event(store: &mut SaveSyncStore, mut event: Map<String, Value>) {
    event.insert("createdAt".into(), json!(now_iso()));
    store.events.push(Value::Object(event));
    let trimmed = last_n(&store.events, MAX_EVENTS);
    store.events = trimmed;
}

/// Returns the last `n` elements of `items`, cloned (mirrors JS `slice(-n)`).
fn last_n(items: &[Value], n: usize) -> Vec<Value> {
    let start = items.len().saturating_sub(n);
    items[start..].to_vec()
}

// ---------------------------------------------------------------------------
// Identity (mirrors getSaveIdentity / getSaveSyncCheckpointId)
// ---------------------------------------------------------------------------

struct SaveIdentity {
    game_id: String,
    game_name: String,
    save_id: String,
    save_name: String,
    save_file: String,
    save_fingerprint: String,
}

/// Reads `value` as a plain object, else an empty object. Mirrors
/// `getPlainObject`.
fn plain_object(value: Option<&Value>) -> Value {
    match value {
        Some(v) if v.is_object() => v.clone(),
        _ => Value::Object(Map::new()),
    }
}

/// First present, non-null, string-coercible field among `keys` of `body`,
/// then the nested `save`/`game` object. Used to mirror the long `??` chains.
fn first_str(body: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(value) = body.get(*key) {
            if let Some(text) = value_as_string(value) {
                return Some(text);
            }
        }
    }
    None
}

/// JS string coercion for the field types these bodies carry. Returns `None`
/// for null/undefined so the `??` fallthrough continues.
fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Validates a non-empty identity string (mirrors `validateIdentityString`):
/// trims, rejects NUL, enforces `max_length`.
fn validate_identity_string(
    value: Option<String>,
    field: &str,
    max_length: usize,
) -> WebResult<String> {
    let text = value.unwrap_or_default();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(web_err(&format!("{field} must be a non-empty string.")));
    }
    if trimmed.contains('\0') {
        return Err(web_err(&format!("{field} contains invalid characters.")));
    }
    if trimmed.chars().count() > max_length {
        return Err(web_err(&format!(
            "{field} must be at most {max_length} characters."
        )));
    }
    Ok(trimmed.to_string())
}

/// Truncates a string to at most `max` chars (mirrors JS `String(x).slice(0, max)`).
fn truncate_chars(value: Option<String>, max: usize) -> String {
    value
        .map(|s| s.chars().take(max).collect::<String>())
        .unwrap_or_default()
}

/// Extracts the normalized save identity from a request body. Mirrors
/// `getSaveIdentity` including its alias chains and per-field length caps.
fn get_save_identity(body: &Value) -> WebResult<SaveIdentity> {
    let save = plain_object(body.get("save"));
    let game = plain_object(body.get("game"));

    let game_id_raw = first_str(body, &["gameId", "game_id"])
        .or_else(|| first_str(&game, &["id", "gameId"]))
        .or(Some("default-game".to_string()));
    let game_id = validate_identity_string(game_id_raw, "gameId", 160)?;

    let save_id_raw = first_str(body, &["saveId", "save_id"])
        .or_else(|| first_str(&save, &["id", "saveId", "slot", "file"]))
        .or_else(|| first_str(body, &["saveFile", "save_file"]));
    let save_id = validate_identity_string(save_id_raw, "saveId", 300)?;

    let game_name = truncate_chars(
        first_str(body, &["gameName", "game_name"]).or_else(|| first_str(&game, &["name"])),
        160,
    );
    let save_name = truncate_chars(
        first_str(body, &["saveName", "save_name"]).or_else(|| first_str(&save, &["name"])),
        200,
    );
    let save_file = truncate_chars(
        first_str(body, &["saveFile", "save_file"]).or_else(|| first_str(&save, &["file"])),
        500,
    );
    let save_fingerprint = truncate_chars(
        first_str(body, &["saveFingerprint", "save_fingerprint"])
            .or_else(|| first_str(&save, &["fingerprint", "modifiedAt", "modified_at"])),
        300,
    );

    Ok(SaveIdentity {
        game_id,
        game_name,
        save_id,
        save_name,
        save_file,
        save_fingerprint,
    })
}

/// Stable checkpoint id for a `(gameId, saveId)` pair: lowercased, NUL-joined,
/// sha256, first 48 hex chars. Mirrors `getSaveSyncCheckpointId` exactly so ids
/// match the Node implementation byte-for-byte.
fn save_sync_checkpoint_id(game_id: &str, save_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(game_id.trim().to_lowercase().as_bytes());
    hasher.update([0u8]);
    hasher.update(save_id.trim().to_lowercase().as_bytes());
    let digest = hasher.finalize();
    let hex = digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    hex.chars().take(48).collect()
}

/// Resolves the checkpoint id from the body: explicit `checkpointId` if present
/// (validated safe), otherwise derived from the save identity. Mirrors
/// `resolveCheckpointId`.
fn resolve_checkpoint_id(body: &Value) -> WebResult<String> {
    if let Some(explicit) =
        first_str(body, &["checkpointId", "checkpoint_id"]).filter(|s| !s.trim().is_empty())
    {
        return validate_safe_id(&explicit, "checkpointId");
    }
    let identity = get_save_identity(body)?;
    Ok(save_sync_checkpoint_id(
        &identity.game_id,
        &identity.save_id,
    ))
}

// ---------------------------------------------------------------------------
// Session-file path resolution + path-traversal guard
// ---------------------------------------------------------------------------

/// The chat-session roots the REPOSITORY actually reads/writes, resolved with
/// the same profile-else-legacy rule as `ProfilePaths::{chats_dir,
/// group_chats_dir}` (profile subdir when it exists, else the legacy data
/// root). Joining these subdirs onto the profile content root blindly was the
/// save-sync chat-rollback bug: imported profiles ship `headless/` but no
/// `chats`/`group chats`, so the real session files live under the legacy data
/// root while save-sync captured/restored phantom files under the profile —
/// checkpoints recorded empty content and restores rewrote a location nothing
/// reads, leaving NPCs with post-save memory after a load.
struct ChatRoots {
    /// `single`-mode sessions: `<chats>/<characterId>/<chatId>.jsonl`.
    chats: PathBuf,
    /// `group`-mode sessions: `<group chats>/<chatId>.jsonl`.
    group_chats: PathBuf,
}

/// Sanitizes one path segment by dropping characters illegal on Windows/ST.
/// Mirrors st-compat's `sanitize_path_segment` (which mirrors `sanitize-filename`).
fn sanitize_path_segment(value: &str) -> String {
    value
        .chars()
        .filter(|ch| !matches!(ch, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'))
        .collect()
}

fn sanitize_file_name(value: &str) -> String {
    sanitize_path_segment(value)
        .trim_end_matches('.')
        .to_string()
}

/// Resolves a session id to its backing JSONL file under the repository's
/// resolved chat roots, with a path-traversal guard. Mirrors st-compat
/// `session_file_path` + ST's `isPathUnderParent` check.
/// `single` -> `<chats>/<characterId>/<chatId>.jsonl`;
/// `group` -> `<group chats>/<chatId>.jsonl`. Returns `None` for an undecodable
/// id (ST would 400; here we treat it as "no file", matching the snapshot's
/// best-effort handling).
fn session_file_path(chat_roots: &ChatRoots, session_id: &str) -> Option<PathBuf> {
    let payload = chasm_st_compat::decode_session_id(session_id).ok()?;
    let mode = payload.get("mode").and_then(Value::as_str)?;
    let chat_id = payload.get("chatId").and_then(Value::as_str)?;
    let file = sanitize_file_name(&format!("{chat_id}.jsonl"));

    let (base, full) = match mode {
        "single" => {
            let character_id =
                sanitize_path_segment(payload.get("characterId").and_then(Value::as_str)?);
            if character_id.is_empty() {
                return None;
            }
            let base = chat_roots.chats.clone();
            let full = base.join(&character_id).join(&file);
            (base, full)
        }
        "group" => {
            let base = chat_roots.group_chats.clone();
            let full = base.join(&file);
            (base, full)
        }
        _ => return None,
    };

    if !is_path_under(&base, &full) {
        return None;
    }
    Some(full)
}

/// Returns whether `child` resolves inside `parent`. Compares lexically on
/// normalized components (no canonicalization, so it works for not-yet-existing
/// files). Mirrors the intent of ST's `isPathUnderParent`. The earlier
/// segment sanitization already strips `..`, but this is the load-bearing guard
/// for restore writes.
fn is_path_under(parent: &Path, child: &Path) -> bool {
    let parent_norm = normalize_lexically(parent);
    let child_norm = normalize_lexically(child);
    child_norm.starts_with(&parent_norm)
}

/// Lexical normalization: resolves `.`/`..` components without touching the
/// filesystem. A `..` that would escape the root is dropped (clamped), so a
/// crafted id can never climb above `data_root`.
fn normalize_lexically(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Live-chat session refs (mirrors getLiveChatSessionRefs)
// ---------------------------------------------------------------------------

struct SessionRef {
    session_id: String,
    owner: String,
    kind: &'static str,
    live_chat_id: String,
}

/// Collects unique session ids referenced by a live chat: one per segment, plus
/// per-participant projection sessions when `includeParticipantProjections` is
/// on. Mirrors `getLiveChatSessionRefs` (de-dupes by session id, first wins).
fn live_chat_session_refs(
    live_chat: &Value,
    live_chat_id: &str,
    include_participant_projections: bool,
) -> Vec<SessionRef> {
    let mut refs = Vec::new();

    if let Some(segments) = live_chat.get("segments").and_then(Value::as_array) {
        for segment in segments {
            if let Some(session_id) = segment.get("sessionId").and_then(Value::as_str) {
                if !session_id.is_empty() {
                    refs.push(SessionRef {
                        session_id: session_id.to_string(),
                        owner: segment
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        kind: "segment",
                        live_chat_id: live_chat_id.to_string(),
                    });
                }
            }
        }
    }

    if include_participant_projections {
        if let Some(sessions) = live_chat
            .get("participantSessions")
            .and_then(Value::as_object)
        {
            for (participant_id, projection) in sessions {
                if let Some(session_id) = projection.get("sessionId").and_then(Value::as_str) {
                    if !session_id.is_empty() {
                        refs.push(SessionRef {
                            session_id: session_id.to_string(),
                            owner: participant_id.clone(),
                            kind: "participantProjection",
                            live_chat_id: live_chat_id.to_string(),
                        });
                    }
                }
            }
        }
    }

    let mut seen = BTreeSet::new();
    refs.into_iter()
        .filter(|r| !r.session_id.is_empty() && seen.insert(r.session_id.clone()))
        .collect()
}

/// Reads a session file into a snapshot entry (base64 content + sha256), or an
/// `exists: false` placeholder. Mirrors `readSessionFileSnapshot`.
fn read_session_file_snapshot(chat_roots: &ChatRoots, reference: &SessionRef) -> Value {
    let path = session_file_path(chat_roots, &reference.session_id);
    let base = json!({
        "sessionId": reference.session_id,
        "owner": reference.owner,
        "kind": reference.kind,
        "liveChatId": reference.live_chat_id,
    });

    let Some(path) = path else {
        return merge_file_entry(base, false, 0, "", "");
    };
    match fs::read(&path) {
        Ok(content) => {
            let mut hasher = Sha256::new();
            hasher.update(&content);
            let sha = hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            merge_file_entry(
                base,
                true,
                content.len() as u64,
                &sha,
                &STANDARD.encode(&content),
            )
        }
        Err(_) => merge_file_entry(base, false, 0, "", ""),
    }
}

fn merge_file_entry(
    mut base: Value,
    exists: bool,
    size: u64,
    sha256: &str,
    content_b64: &str,
) -> Value {
    if let Value::Object(map) = &mut base {
        map.insert("exists".into(), json!(exists));
        map.insert("size".into(), json!(size));
        map.insert("sha256".into(), json!(sha256));
        map.insert("contentBase64".into(), json!(content_b64));
    }
    base
}

/// Writes or deletes a session file from a snapshot entry. Returns the action
/// taken. Mirrors `restoreSessionFileSnapshot`. Guarded: writes only inside the
/// resolved chat roots (an unresolvable/escaping path is a no-op `missing`).
fn restore_session_file_snapshot(
    chat_roots: &ChatRoots,
    file_entry: &Value,
) -> WebResult<&'static str> {
    let session_id = file_entry
        .get("sessionId")
        .and_then(Value::as_str)
        .unwrap_or("");
    let Some(path) = session_file_path(chat_roots, session_id) else {
        return Ok("missing");
    };
    let exists = file_entry
        .get("exists")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if !exists {
        if path.exists() {
            fs::remove_file(&path)?;
            return Ok("deleted");
        }
        return Ok("missing");
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let content_b64 = file_entry
        .get("contentBase64")
        .and_then(Value::as_str)
        .unwrap_or("");
    let bytes = STANDARD.decode(content_b64).unwrap_or_default();
    fs::write(&path, bytes)?;
    Ok("restored")
}

/// Deletes a session file if it exists (guarded). Mirrors `deleteSessionFile`.
fn delete_session_file(chat_roots: &ChatRoots, session_id: &str) -> WebResult<bool> {
    let Some(path) = session_file_path(chat_roots, session_id) else {
        return Ok(false);
    };
    if !path.exists() {
        return Ok(false);
    }
    fs::remove_file(&path)?;
    Ok(true)
}

/// Removes a participant's messages from every save-sync checkpoint snapshot for
/// `live_chat_id`, so a later restore (on game load / quickload) can't bring a
/// cleared conversation back. Rewrites each affected checkpoint's embedded
/// session-file snapshot (`contentBase64` + `size` + `sha256`) in place, using
/// the SAME removal rule as the live participant clear. Best-effort: a missing
/// checkpoints dir or an unreadable/malformed checkpoint is skipped, not fatal.
/// Returns the total number of messages removed across all checkpoints.
///
/// `store_root` is the save-sync content root (the active profile's content
/// root), matching where `save_sync_event` reads/writes checkpoints.
pub(crate) fn scrub_participant_from_checkpoints(
    store_root: &Path,
    live_chat_id: &str,
    participant_id: &str,
) -> usize {
    let dir = save_sync_dir(store_root).join(CHECKPOINTS_DIRECTORY);
    let Ok(entries) = fs::read_dir(&dir) else {
        return 0;
    };
    let mut removed_total = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(mut checkpoint) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let Some(files) = checkpoint.get_mut("files").and_then(Value::as_array_mut) else {
            continue;
        };
        let mut removed_here = 0usize;
        for file in files.iter_mut() {
            if file.get("liveChatId").and_then(Value::as_str) != Some(live_chat_id) {
                continue;
            }
            if !file.get("exists").and_then(Value::as_bool).unwrap_or(false) {
                continue;
            }
            let Some(decoded) = file
                .get("contentBase64")
                .and_then(Value::as_str)
                .and_then(|b64| STANDARD.decode(b64).ok())
                .and_then(|bytes| String::from_utf8(bytes).ok())
            else {
                continue;
            };
            let (out, removed) =
                chasm_st_compat::strip_participant_from_jsonl(&decoded, participant_id);
            if removed == 0 {
                continue;
            }
            let bytes = out.into_bytes();
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            let sha = hasher
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            if let Value::Object(map) = file {
                map.insert("size".into(), json!(bytes.len() as u64));
                map.insert("sha256".into(), json!(sha));
                map.insert("contentBase64".into(), json!(STANDARD.encode(&bytes)));
            }
            removed_here += removed;
        }
        if removed_here > 0 && write_json_file(&path, &checkpoint).is_ok() {
            removed_total += removed_here;
        }
    }
    removed_total
}

// ---------------------------------------------------------------------------
// World-state + live-chat stores (under <data_root>/headless)
// ---------------------------------------------------------------------------

/// Reads `<world_root>/headless/world-state.json` as an object (mirrors
/// `readWorldStateStore`; missing/invalid -> empty object). World-state is GLOBAL
/// (not per-profile), so `world_root` is the legacy data root, distinct from the
/// per-profile content root used for live-chats / chats / save-sync.
fn read_world_state_store(world_root: &Path) -> Map<String, Value> {
    let path = headless_dir(world_root).join("world-state.json");
    let Ok(text) = fs::read_to_string(&path) else {
        return Map::new();
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    }
}

/// Writes the (global) world-state store (mirrors `writeWorldStateStore`).
fn write_world_state_store(world_root: &Path, store: &Map<String, Value>) -> WebResult<()> {
    let path = headless_dir(world_root).join("world-state.json");
    write_json_file(&path, &Value::Object(store.clone()))
}

/// Reads `<data_root>/headless/live-chats.json` raw (as a `Value`) so we can
/// snapshot/restore arbitrary live-chat JSON faithfully without the typed
/// st-compat model dropping unknown fields. Mirrors `readLiveChatStore` shape
/// `{items: {...}}`.
fn read_live_chat_store_raw(data_root: &Path) -> Value {
    let path = headless_dir(data_root).join("live-chats.json");
    let Ok(text) = fs::read_to_string(&path) else {
        return json!({ "items": {} });
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(value) if value.is_object() => value,
        _ => json!({ "items": {} }),
    }
}

fn write_live_chat_store_raw(data_root: &Path, store: &Value) -> WebResult<()> {
    let path = headless_dir(data_root).join("live-chats.json");
    write_json_file(&path, store)
}

/// Selected live-chat ids: explicit from the body, else every id in the store.
/// Mirrors `getSelectedLiveChatIds`.
fn selected_live_chat_ids(body: &Value, live_chat_store: &Value) -> WebResult<(Vec<String>, bool)> {
    let raw = body
        .get("liveChatIds")
        .or_else(|| body.get("live_chat_ids"))
        .or_else(|| body.get("liveChats"))
        .or_else(|| body.get("live_chats"));
    if let Some(raw) = raw.filter(|v| !v.is_null()) {
        let Value::Array(arr) = raw else {
            return Err(web_err("liveChatIds must be an array."));
        };
        if arr.len() > 200 {
            return Err(web_err("liveChatIds may contain at most 200 items."));
        }
        let mut ids = Vec::with_capacity(arr.len());
        for (index, item) in arr.iter().enumerate() {
            let Some(text) = item.as_str() else {
                return Err(web_err(&format!("liveChatIds[{index}] must be a string.")));
            };
            ids.push(validate_safe_id(text, "liveChatId")?);
        }
        return Ok((ids, true));
    }
    let ids = live_chat_store
        .get("items")
        .and_then(Value::as_object)
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default();
    Ok((ids, false))
}

/// Selected world-state scopes: explicit from the body, else every scope key.
/// Mirrors `getSelectedWorldStateScopes`.
fn selected_world_state_scopes(
    body: &Value,
    world_state: &Map<String, Value>,
) -> WebResult<(Vec<String>, bool)> {
    let raw = body
        .get("worldStateScopes")
        .or_else(|| body.get("world_state_scopes"));
    if let Some(raw) = raw.filter(|v| !v.is_null()) {
        let Value::Array(arr) = raw else {
            return Err(web_err("worldStateScopes must be an array."));
        };
        if arr.len() > 500 {
            return Err(web_err("worldStateScopes may contain at most 500 items."));
        }
        let mut scopes = Vec::with_capacity(arr.len());
        for (index, item) in arr.iter().enumerate() {
            let Some(text) = item.as_str() else {
                return Err(web_err(&format!(
                    "worldStateScopes[{index}] must be a string."
                )));
            };
            scopes.push(text.to_string());
        }
        return Ok((scopes, false));
    }
    Ok((world_state.keys().cloned().collect(), true))
}

// ---------------------------------------------------------------------------
// Snapshot build (mirrors buildSaveSyncSnapshot)
// ---------------------------------------------------------------------------

/// Builds a full snapshot of the bridge-owned state selected by `options`.
/// Mirrors `buildSaveSyncSnapshot` field-for-field so the JSON round-trips with
/// ST.
fn build_save_sync_snapshot(
    data_root: &Path,
    world_root: &Path,
    chat_roots: &ChatRoots,
    body: &Value,
    options: SaveSyncOptions,
) -> WebResult<Value> {
    let identity = get_save_identity(body)?;
    let checkpoint_id = save_sync_checkpoint_id(&identity.game_id, &identity.save_id);
    let live_chat_store = read_live_chat_store_raw(data_root);
    let world_state = read_world_state_store(world_root);
    let now = now_iso();

    let source = body
        .get("source")
        .and_then(Value::as_str)
        .map(|s| s.chars().take(160).collect::<String>())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "headless-save-sync".to_string());
    let metadata = plain_object(body.get("metadata"));

    let mut live_chats_items = Map::new();
    let mut live_chats_tracked: Vec<String> = Vec::new();
    let mut live_chats_absent: Vec<String> = Vec::new();
    let mut live_chats_explicit = false;
    let mut files: Vec<Value> = Vec::new();

    if options.include_live_chats {
        let (ids, explicit) = selected_live_chat_ids(body, &live_chat_store)?;
        live_chats_explicit = explicit;
        live_chats_tracked = ids.clone();
        let items = live_chat_store.get("items").and_then(Value::as_object);
        for live_chat_id in &ids {
            let live_chat = items.and_then(|m| m.get(live_chat_id));
            let Some(live_chat) = live_chat else {
                live_chats_absent.push(live_chat_id.clone());
                continue;
            };
            live_chats_items.insert(live_chat_id.clone(), live_chat.clone());
            if options.include_chat_files {
                for reference in live_chat_session_refs(
                    live_chat,
                    live_chat_id,
                    options.include_participant_projections,
                ) {
                    files.push(read_session_file_snapshot(chat_roots, &reference));
                }
            }
        }
    }

    let mut world_scopes = Map::new();
    let mut world_tracked: Vec<String> = Vec::new();
    let mut world_absent: Vec<String> = Vec::new();
    let mut world_all = false;

    if options.include_world_state {
        let (scopes, all) = selected_world_state_scopes(body, &world_state)?;
        world_all = all;
        world_tracked = scopes.clone();
        for scope in &scopes {
            if let Some(value) = world_state.get(scope) {
                world_scopes.insert(scope.clone(), value.clone());
            } else {
                world_absent.push(scope.clone());
            }
        }
    }

    let snapshot = json!({
        "version": SAVE_SYNC_VERSION,
        "checkpointId": checkpoint_id,
        "identity": {
            "gameId": identity.game_id,
            "saveId": identity.save_id,
            "saveName": identity.save_name,
            "gameName": identity.game_name,
            "saveFile": identity.save_file,
            "saveFingerprint": identity.save_fingerprint,
        },
        "createdAt": now,
        "updatedAt": now,
        "source": source,
        "metadata": metadata,
        "options": options.to_json(),
        "liveChats": {
            "enabled": options.include_live_chats,
            "explicit": live_chats_explicit,
            "trackedIds": live_chats_tracked,
            "absentIds": live_chats_absent,
            "items": Value::Object(live_chats_items),
        },
        "worldState": {
            "enabled": options.include_world_state,
            "all": world_all,
            "trackedScopes": world_tracked,
            "absentScopes": world_absent,
            "scopes": Value::Object(world_scopes),
        },
        "files": Value::Array(files),
    });
    Ok(snapshot)
}

/// Snapshot counts for the public summary. Mirrors `getSnapshotCounts`.
fn snapshot_counts(snapshot: &Value) -> Value {
    let live_chats = snapshot
        .pointer("/liveChats/items")
        .and_then(Value::as_object)
        .map(Map::len)
        .unwrap_or(0);
    let absent_live_chats = snapshot
        .pointer("/liveChats/absentIds")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let world_scopes = snapshot
        .pointer("/worldState/scopes")
        .and_then(Value::as_object)
        .map(Map::len)
        .unwrap_or(0);
    let absent_world = snapshot
        .pointer("/worldState/absentScopes")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let files = snapshot.get("files").and_then(Value::as_array);
    let file_count = files.map(Vec::len).unwrap_or(0);
    let existing_files = files
        .map(|arr| {
            arr.iter()
                .filter(|f| f.get("exists").and_then(Value::as_bool).unwrap_or(false))
                .count()
        })
        .unwrap_or(0);

    json!({
        "liveChats": live_chats,
        "absentLiveChats": absent_live_chats,
        "worldStateScopes": world_scopes,
        "absentWorldStateScopes": absent_world,
        "files": file_count,
        "existingFiles": existing_files,
    })
}

/// Maps a snapshot to its public checkpoint summary (stored in `items`).
/// Mirrors `mapCheckpointSummary`.
fn map_checkpoint_summary(snapshot: &Value) -> Value {
    let identity = snapshot.get("identity").cloned().unwrap_or(json!({}));
    json!({
        "checkpointId": snapshot.get("checkpointId").cloned().unwrap_or(Value::Null),
        "gameId": identity.get("gameId").cloned().unwrap_or(json!("")),
        "gameName": identity.get("gameName").cloned().unwrap_or(json!("")),
        "saveId": identity.get("saveId").cloned().unwrap_or(json!("")),
        "saveName": identity.get("saveName").cloned().unwrap_or(json!("")),
        "saveFile": identity.get("saveFile").cloned().unwrap_or(json!("")),
        "saveFingerprint": identity.get("saveFingerprint").cloned().unwrap_or(json!("")),
        "source": snapshot.get("source").cloned().unwrap_or(json!("")),
        "liveChatIds": snapshot.pointer("/liveChats/trackedIds").cloned().unwrap_or(json!([])),
        "absentLiveChatIds": snapshot.pointer("/liveChats/absentIds").cloned().unwrap_or(json!([])),
        "worldStateScopes": snapshot.pointer("/worldState/trackedScopes").cloned().unwrap_or(json!([])),
        "createdAt": snapshot.get("createdAt").cloned().unwrap_or(Value::Null),
        "updatedAt": snapshot.get("updatedAt").cloned().unwrap_or(Value::Null),
        "metadata": snapshot.get("metadata").cloned().unwrap_or(json!({})),
        "counts": snapshot_counts(snapshot),
    })
}

// ---------------------------------------------------------------------------
// Checkpoint create (mirrors createSaveSyncCheckpoint)
// ---------------------------------------------------------------------------

fn create_save_sync_checkpoint(
    data_root: &Path,
    world_root: &Path,
    chat_roots: &ChatRoots,
    body: &Value,
) -> WebResult<Value> {
    let options = effective_save_sync_options(data_root, &plain_object(body.get("options")));
    if !options.enabled {
        return Ok(json!({
            "status": "disabled",
            "checkpoint": Value::Null,
            "options": options.to_json(),
        }));
    }

    let mut snapshot = build_save_sync_snapshot(data_root, world_root, chat_roots, body, options)?;
    let mut store = read_save_sync_store(data_root);

    let checkpoint_id = snapshot
        .get("checkpointId")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    // Preserve the original createdAt when updating an existing checkpoint.
    let existing = store.items.get(&checkpoint_id).cloned();
    if let Some(created_at) = existing
        .as_ref()
        .and_then(|e| e.get("createdAt"))
        .filter(|v| !v.is_null())
        .cloned()
    {
        if let Value::Object(map) = &mut snapshot {
            map.insert("createdAt".into(), created_at);
        }
    }

    // Write the full snapshot file, then update the index.
    let path = checkpoint_file_path(data_root, &checkpoint_id)?;
    write_json_file(&path, &snapshot)?;

    let summary = map_checkpoint_summary(&snapshot);
    store.items.insert(checkpoint_id.clone(), summary);
    let identity = snapshot.get("identity").cloned().unwrap_or(json!({}));
    store.current = json!({
        "checkpointId": checkpoint_id,
        "gameId": identity.get("gameId").cloned().unwrap_or(json!("")),
        "saveId": identity.get("saveId").cloned().unwrap_or(json!("")),
        "saveName": identity.get("saveName").cloned().unwrap_or(json!("")),
        "updatedAt": snapshot.get("updatedAt").cloned().unwrap_or(Value::Null),
    });
    append_store_event(
        &mut store,
        json_object(json!({
            "type": "checkpoint.created",
            "checkpointId": checkpoint_id,
            "gameId": identity.get("gameId").cloned().unwrap_or(json!("")),
            "saveId": identity.get("saveId").cloned().unwrap_or(json!("")),
            "saveName": identity.get("saveName").cloned().unwrap_or(json!("")),
            "source": snapshot.get("source").cloned().unwrap_or(json!("")),
        })),
    );
    prune_save_sync_checkpoints(data_root, &mut store, options.retention_limit)?;
    write_save_sync_store(data_root, &store)?;

    Ok(json!({
        "status": if existing.is_some() { "checkpoint_updated" } else { "checkpoint_created" },
        "checkpoint": store.items.get(&checkpoint_id).cloned().unwrap_or(Value::Null),
        "options": options.to_json(),
        "counts": snapshot_counts(&snapshot),
    }))
}

/// Prunes checkpoints beyond the retention limit, keeping the newest by
/// `updatedAt` plus the `current` one. Mirrors `pruneSaveSyncCheckpoints`.
fn prune_save_sync_checkpoints(
    data_root: &Path,
    store: &mut SaveSyncStore,
    retention_limit: i64,
) -> WebResult<()> {
    if retention_limit <= 0 {
        return Ok(());
    }
    let limit = retention_limit as usize;

    // Sort entries by updatedAt descending (string compare, like ST).
    let mut entries: Vec<(String, String)> = store
        .items
        .iter()
        .map(|(id, item)| {
            let updated = item
                .get("updatedAt")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            (id.clone(), updated)
        })
        .collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));

    let mut keep: BTreeSet<String> = entries
        .iter()
        .take(limit)
        .map(|(id, _)| id.clone())
        .collect();
    if let Some(current_id) = store.current.get("checkpointId").and_then(Value::as_str) {
        keep.insert(current_id.to_string());
    }

    for (id, _) in &entries {
        if keep.contains(id) {
            continue;
        }
        store.items.remove(id);
        if let Ok(path) = checkpoint_file_path(data_root, id) {
            let _ = fs::remove_file(path);
        }
    }
    Ok(())
}

/// Reads a full checkpoint snapshot file. Mirrors `readCheckpointSnapshot`
/// (not-found / invalid both surface as errors).
fn read_checkpoint_snapshot(data_root: &Path, checkpoint_id: &str) -> WebResult<Value> {
    let path = checkpoint_file_path(data_root, checkpoint_id)?;
    let text = fs::read_to_string(&path).map_err(|_| web_err("Save-sync checkpoint not found."))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|_| web_err("Save-sync checkpoint file is invalid."))?;
    if !value.is_object() {
        return Err(web_err("Save-sync checkpoint file is invalid."));
    }
    Ok(value)
}

// ---------------------------------------------------------------------------
// Safety backup (mirrors writeRestoreSafetyBackup)
// ---------------------------------------------------------------------------

/// Writes a "before restore" safety backup snapshot and returns its file path
/// (as a string). Mirrors `writeRestoreSafetyBackup`.
fn write_restore_safety_backup(
    data_root: &Path,
    world_root: &Path,
    chat_roots: &ChatRoots,
    restore_snapshot: &Value,
    options: SaveSyncOptions,
) -> WebResult<String> {
    let identity = restore_snapshot
        .get("identity")
        .cloned()
        .unwrap_or(json!({}));
    let game_id = identity.get("gameId").and_then(Value::as_str).unwrap_or("");
    let save_id = identity.get("saveId").and_then(Value::as_str).unwrap_or("");
    let save_name = identity
        .get("saveName")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(save_id);
    let checkpoint_id = restore_snapshot
        .get("checkpointId")
        .and_then(Value::as_str)
        .unwrap_or("");

    let world_all = restore_snapshot
        .pointer("/worldState/all")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let tracked_scopes = restore_snapshot
        .pointer("/worldState/trackedScopes")
        .cloned()
        .unwrap_or(json!([]));
    let tracked_ids = restore_snapshot
        .pointer("/liveChats/trackedIds")
        .cloned()
        .unwrap_or(json!([]));

    let mut body = json!({
        "gameId": game_id,
        "saveId": format!("before-restore-{save_id}-{}", epoch_millis()),
        "saveName": format!("Before restore: {save_name}"),
        "liveChatIds": tracked_ids,
        "source": "headless-save-sync-safety-backup",
        "metadata": { "restoringCheckpointId": checkpoint_id },
    });
    // ST sets worldStateScopes to undefined when `all` (so the snapshot selects
    // all scopes); only set it when NOT all.
    if !world_all {
        if let Value::Object(map) = &mut body {
            map.insert("worldStateScopes".into(), tracked_scopes);
        }
    }

    let snapshot = build_save_sync_snapshot(data_root, world_root, chat_roots, &body, options)?;
    let stamp = sanitize_backup_stamp(&now_iso());
    let file_id = format!("{stamp}-{checkpoint_id}");
    let path = restore_backup_file_path(data_root, &file_id)?;
    write_json_file(&path, &snapshot)?;
    Ok(path.display().to_string())
}

/// Mirrors ST's `new Date().toISOString().replace(/[^0-9A-Za-z]+/g, '-')`.
fn sanitize_backup_stamp(iso: &str) -> String {
    let mut out = String::with_capacity(iso.len());
    let mut prev_dash = false;
    for ch in iso.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Restore (mirrors restoreSaveSyncCheckpoint)
// ---------------------------------------------------------------------------

fn restore_save_sync_checkpoint(
    data_root: &Path,
    world_root: &Path,
    chat_roots: &ChatRoots,
    body: &Value,
) -> WebResult<Value> {
    let checkpoint_id = resolve_checkpoint_id(body)?;
    let snapshot = read_checkpoint_snapshot(data_root, &checkpoint_id)?;

    let snapshot_options = snapshot.get("options").cloned().unwrap_or(Value::Null);
    let fallback = if snapshot_options.is_object() {
        normalize_save_sync_options(&snapshot_options, SaveSyncOptions::default())
    } else {
        effective_save_sync_options(data_root, &Value::Null)
    };
    let options = normalize_save_sync_options(&plain_object(body.get("options")), fallback);

    if !options.enabled {
        return Ok(json!({
            "status": "disabled",
            "checkpoint": map_checkpoint_summary(&snapshot),
            "restored": false,
            "options": options.to_json(),
        }));
    }

    let dry_run = body.get("dryRun").and_then(Value::as_bool).unwrap_or(false)
        || body
            .get("dry_run")
            .and_then(Value::as_bool)
            .unwrap_or(false);

    let safety_backup_file = if !dry_run && options.create_safety_backup {
        write_restore_safety_backup(data_root, world_root, chat_roots, &snapshot, options)?
    } else {
        String::new()
    };

    let mut live_chats_restored = 0u64;
    let mut live_chats_deleted = 0u64;
    let mut world_scopes_restored = 0u64;
    let mut world_scopes_deleted = 0u64;
    let mut files_restored = 0u64;
    let mut files_deleted = 0u64;
    let mut extra_files_deleted = 0u64;
    let mut missing_files = 0u64;

    let live_chats_enabled = snapshot
        .pointer("/liveChats/enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if !dry_run && live_chats_enabled {
        let mut live_chat_store = read_live_chat_store_raw(data_root);
        let tracked_ids: Vec<String> = snapshot
            .pointer("/liveChats/trackedIds")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let snapshot_items = snapshot.pointer("/liveChats/items");

        // Capture current session ids per tracked live chat BEFORE overwriting,
        // so we can delete now-orphaned chat files after restore.
        let mut current_session_ids: Vec<(String, Vec<String>)> = Vec::new();
        let store_items = live_chat_store
            .get("items")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        for live_chat_id in &tracked_ids {
            let session_ids = store_items
                .get(live_chat_id)
                .map(|lc| {
                    live_chat_session_refs(
                        lc,
                        live_chat_id,
                        options.include_participant_projections,
                    )
                    .into_iter()
                    .map(|r| r.session_id)
                    .collect()
                })
                .unwrap_or_default();
            current_session_ids.push((live_chat_id.clone(), session_ids));
        }

        // Restore (or delete) each tracked live chat in the store.
        if let Some(Value::Object(items)) = live_chat_store.get_mut("items") {
            for live_chat_id in &tracked_ids {
                let snap_item = snapshot_items.and_then(|m| m.get(live_chat_id));
                if let Some(item) = snap_item {
                    items.insert(live_chat_id.clone(), item.clone());
                    live_chats_restored += 1;
                } else {
                    items.remove(live_chat_id);
                    live_chats_deleted += 1;
                }
            }
        } else if let Value::Object(root) = &mut live_chat_store {
            // No items object yet: build one from restored entries.
            let mut items = Map::new();
            for live_chat_id in &tracked_ids {
                if let Some(item) = snapshot_items.and_then(|m| m.get(live_chat_id)) {
                    items.insert(live_chat_id.clone(), item.clone());
                    live_chats_restored += 1;
                } else {
                    live_chats_deleted += 1;
                }
            }
            root.insert("items".into(), Value::Object(items));
        }
        write_live_chat_store_raw(data_root, &live_chat_store)?;

        // Restore the snapshot's chat files.
        let snapshot_files = snapshot
            .get("files")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let snapshot_session_ids: BTreeSet<String> = snapshot_files
            .iter()
            .filter_map(|f| {
                f.get("sessionId")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect();
        for file in &snapshot_files {
            match restore_session_file_snapshot(chat_roots, file)? {
                "restored" => files_restored += 1,
                "deleted" => files_deleted += 1,
                "missing" => missing_files += 1,
                _ => {}
            }
        }

        // Delete chat files that are now newer/orphaned (present before, absent
        // from snapshot), when the option is on.
        if options.delete_newer_chat_files {
            for (_id, session_ids) in &current_session_ids {
                for session_id in session_ids {
                    if !snapshot_session_ids.contains(session_id)
                        && delete_session_file(chat_roots, session_id)?
                    {
                        extra_files_deleted += 1;
                    }
                }
            }
        }
    }

    let world_enabled = snapshot
        .pointer("/worldState/enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if !dry_run && world_enabled {
        let world_all = snapshot
            .pointer("/worldState/all")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let snapshot_scopes = snapshot
            .pointer("/worldState/scopes")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();

        if world_all {
            let previous = read_world_state_store(world_root);
            write_world_state_store(world_root, &snapshot_scopes)?;
            world_scopes_restored = snapshot_scopes.len() as u64;
            world_scopes_deleted = (previous.len() as u64).saturating_sub(world_scopes_restored);
        } else {
            let mut world_state = read_world_state_store(world_root);
            let tracked_scopes: Vec<String> = snapshot
                .pointer("/worldState/trackedScopes")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            for scope in &tracked_scopes {
                if let Some(value) = snapshot_scopes.get(scope) {
                    world_state.insert(scope.clone(), value.clone());
                    world_scopes_restored += 1;
                } else {
                    world_state.remove(scope);
                    world_scopes_deleted += 1;
                }
            }
            write_world_state_store(world_root, &world_state)?;
        }
    }

    let counts = json!({
        "liveChatsRestored": live_chats_restored,
        "liveChatsDeleted": live_chats_deleted,
        "worldStateScopesRestored": world_scopes_restored,
        "worldStateScopesDeleted": world_scopes_deleted,
        "filesRestored": files_restored,
        "filesDeleted": files_deleted,
        "extraFilesDeleted": extra_files_deleted,
        "missingFiles": missing_files,
    });

    let mut store = read_save_sync_store(data_root);
    let summary = store
        .items
        .get(&checkpoint_id)
        .cloned()
        .unwrap_or_else(|| map_checkpoint_summary(&snapshot));
    if !dry_run {
        let identity = snapshot.get("identity").cloned().unwrap_or(json!({}));
        store.current = json!({
            "checkpointId": checkpoint_id,
            "gameId": identity.get("gameId").cloned().unwrap_or(json!("")),
            "saveId": identity.get("saveId").cloned().unwrap_or(json!("")),
            "saveName": identity.get("saveName").cloned().unwrap_or(json!("")),
            "restoredAt": now_iso(),
        });
        append_store_event(
            &mut store,
            json_object(json!({
                "type": "checkpoint.restored",
                "checkpointId": checkpoint_id,
                "gameId": identity.get("gameId").cloned().unwrap_or(json!("")),
                "saveId": identity.get("saveId").cloned().unwrap_or(json!("")),
                "saveName": identity.get("saveName").cloned().unwrap_or(json!("")),
                "safetyBackupFile": safety_backup_file.clone(),
            })),
        );
        write_save_sync_store(data_root, &store)?;
    }

    Ok(json!({
        "status": if dry_run { "restore_preview" } else { "restored" },
        "restored": !dry_run,
        "checkpoint": summary,
        "counts": counts,
        "options": options.to_json(),
        "safetyBackupFile": safety_backup_file,
    }))
}

// ---------------------------------------------------------------------------
// Event dispatch (mirrors handleSaveSyncEvent) + the route handler
// ---------------------------------------------------------------------------

/// `POST /save-sync/events` — the single public route the FNV helper calls.
/// Normalizes the event, gates on options, and dispatches save -> checkpoint /
/// load -> restore. Mirrors `handleSaveSyncEvent`.
pub async fn save_sync_event(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> WebResult<Json<Value>> {
    if !body.is_object() {
        return Err(web_err("body must be an object."));
    }
    // Content root = the ACTIVE profile's folder (live-chats store + save-sync
    // snapshots resolve here). The chat-session files are resolved SEPARATELY
    // via ProfilePaths::{chats_dir, group_chats_dir} — the same
    // profile-else-legacy rule the repository writes them with — because a
    // profile that ships `headless/` but no `chats`/`group chats` keeps its
    // session files under the legacy data root. The world-state store stays
    // GLOBAL, so it reads/writes from the legacy data root. All resolved per
    // request so a profile switch takes effect live.
    let paths = state.config.active_profile_paths();
    let content_root = paths.content_root();
    let chat_roots = ChatRoots {
        chats: paths.chats_dir(),
        group_chats: paths.group_chats_dir(),
    };
    let world_root = state.config.data_root.clone();
    // The filesystem work is synchronous; run it on a blocking thread so we
    // don't stall the async runtime on disk I/O.
    let result = tokio::task::spawn_blocking(move || {
        handle_save_sync_event(&content_root, &world_root, &chat_roots, &body)
    })
    .await??;
    Ok(Json(result))
}

fn handle_save_sync_event(
    data_root: &Path,
    world_root: &Path,
    chat_roots: &ChatRoots,
    body: &Value,
) -> WebResult<Value> {
    let event = first_str(body, &["event", "type", "eventType", "event_type"])
        .unwrap_or_default()
        .trim()
        .to_lowercase();

    let options = effective_save_sync_options(data_root, &plain_object(body.get("options")));
    let mut store = read_save_sync_store(data_root);

    if !options.enabled {
        append_store_event(
            &mut store,
            json_object(json!({ "type": "event.ignored", "reason": "disabled", "event": event })),
        );
        write_save_sync_store(data_root, &store)?;
        return Ok(json!({ "status": "disabled", "event": event, "options": options.to_json() }));
    }

    // Build the body forwarded to create/restore with normalized options merged
    // in (ST passes `{ ...body, options }`).
    let mut forwarded = body.clone();
    if let Value::Object(map) = &mut forwarded {
        map.insert("options".into(), options.to_json());
    }

    if matches!(
        event.as_str(),
        "save" | "saved" | "checkpoint" | "autosave" | "quicksave"
    ) {
        if !options.auto_checkpoint {
            append_store_event(
                &mut store,
                json_object(json!({
                    "type": "event.ignored",
                    "reason": "autoCheckpoint disabled",
                    "event": event,
                })),
            );
            write_save_sync_store(data_root, &store)?;
            return Ok(json!({
                "status": "ignored",
                "reason": "autoCheckpoint disabled",
                "event": event,
                "options": options.to_json(),
            }));
        }
        let result = create_save_sync_checkpoint(data_root, world_root, chat_roots, &forwarded)?;
        // Save-aware sidecars checkpoint alongside the chat state, keyed by the
        // same checkpoint id, so a later load rolls them all back together.
        // Never fatal: a failed sidecar snapshot must not void the chat
        // checkpoint — the event log warns, the scheduler/movement stores log
        // internally.
        if let Ok(checkpoint_id) = resolve_checkpoint_id(body) {
            if let Err(e) = crate::event_log::checkpoint_event_log(
                data_root,
                &checkpoint_id,
                options.retention_limit,
            ) {
                tracing::warn!("event-log checkpoint {checkpoint_id}: {e}");
            }
            // Scheduler tasks + active NPC journeys roll back with the save
            // exactly like chat history (a task scheduled in a branch discarded
            // by that load vanishes).
            crate::scheduler::checkpoint_scheduler_store(data_root, &checkpoint_id);
            crate::movement::checkpoint_movement_store(data_root, &checkpoint_id);
        }
        return Ok(result);
    }

    if matches!(event.as_str(), "load" | "loaded" | "restore" | "reload") {
        let checkpoint_id = resolve_checkpoint_id(body)?;
        if !store.items.contains_key(&checkpoint_id) {
            let identity = get_save_identity(body)?;
            append_store_event(
                &mut store,
                json_object(json!({
                    "type": "restore.missing",
                    "checkpointId": checkpoint_id,
                    "gameId": identity.game_id,
                    "saveId": identity.save_id,
                })),
            );
            write_save_sync_store(data_root, &store)?;
            return Ok(json!({
                "status": "snapshot_missing",
                "restored": false,
                "checkpointId": checkpoint_id,
                "event": event,
                "options": options.to_json(),
            }));
        }
        if !options.auto_restore {
            append_store_event(
                &mut store,
                json_object(json!({
                    "type": "event.ignored",
                    "reason": "autoRestore disabled",
                    "event": event,
                    "checkpointId": checkpoint_id,
                })),
            );
            write_save_sync_store(data_root, &store)?;
            return Ok(json!({
                "status": "ignored",
                "reason": "autoRestore disabled",
                "event": event,
                "checkpoint": store.items.get(&checkpoint_id).cloned().unwrap_or(Value::Null),
                "options": options.to_json(),
            }));
        }
        if let Value::Object(map) = &mut forwarded {
            map.insert("checkpointId".into(), json!(checkpoint_id));
        }
        let result = restore_save_sync_checkpoint(data_root, world_root, chat_roots, &forwarded)?;
        // Roll every save-aware sidecar back with the chat state (see the save
        // arm): event log, scheduler tasks, and active NPC journeys all match
        // the loaded save.
        if let Err(e) = crate::event_log::restore_event_log(data_root, &checkpoint_id) {
            tracing::warn!("event-log restore {checkpoint_id}: {e}");
        }
        crate::scheduler::restore_scheduler_store(data_root, &checkpoint_id);
        crate::movement::restore_movement_store(data_root, &checkpoint_id);
        return Ok(result);
    }

    Err(web_err("event must be one of: save, load."))
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Current time as an ISO-8601 UTC string with millisecond precision and a `Z`
/// suffix, matching JS `new Date().toISOString()`.
pub(crate) fn now_iso() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format_iso_millis(now.as_millis() as i64)
}

pub(crate) fn epoch_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Formats epoch milliseconds as `YYYY-MM-DDTHH:MM:SS.mmmZ` (UTC), avoiding a
/// chrono dependency. Uses the civil-from-days algorithm (Howard Hinnant).
fn format_iso_millis(millis: i64) -> String {
    let total_secs = millis.div_euclid(1000);
    let ms = millis.rem_euclid(1000);
    let days = total_secs.div_euclid(86_400);
    let secs_of_day = total_secs.rem_euclid(86_400);
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;

    // days is days since 1970-01-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{ms:03}Z")
}

/// Unwraps a `json!({...})` literal into its object map (the literals here are
/// always objects, so this never panics in practice).
fn json_object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

/// Builds a `WebError` carrying `message` (rendered as the error page body).
fn web_err(message: &str) -> WebError {
    WebError::from(anyhow::anyhow!(message.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A throwaway data root under the OS temp dir. NEVER touches the real one.
    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new(tag: &str) -> Self {
            let mut dir = std::env::temp_dir();
            let unique = format!(
                "sb-save-sync-test-{tag}-{}-{}",
                std::process::id(),
                epoch_millis()
            );
            dir.push(unique);
            fs::create_dir_all(dir.join("headless")).unwrap();
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Seeds a minimal live-chat store with one live chat + a group segment, and
    /// the backing JSONL chat file, so checkpoints capture real chat content.
    fn seed_live_chat(root: &Path) -> String {
        // session id == base64url(JSON) of a group session, as ST encodes.
        let payload = json!({ "mode": "group", "groupId": "g1", "chatId": "Goodsprings" });
        let session_id = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        let store = json!({
            "items": {
                "lc1": {
                    "id": "lc1",
                    "title": "Goodsprings",
                    "segments": [ { "id": "seg1", "sessionId": session_id } ],
                    "participantSessions": {}
                }
            }
        });
        write_live_chat_store_raw(root, &store).unwrap();

        // Write the JSONL chat file the segment points at.
        let chat_dir = root.join("group chats");
        fs::create_dir_all(&chat_dir).unwrap();
        fs::write(
            chat_dir.join("Goodsprings.jsonl"),
            "{\"chat_metadata\":{}}\n{\"name\":\"Sunny\",\"mes\":\"Howdy.\"}\n",
        )
        .unwrap();
        session_id
    }

    fn save_body() -> Value {
        json!({
            "event": "save",
            "gameId": "fallout-new-vegas",
            "gameName": "Fallout: New Vegas",
            "saveId": "Save7",
            "saveName": "Quicksave",
            "liveChatIds": ["lc1"],
        })
    }

    /// Chat roots colocated with the given root — the legacy single-root layout
    /// most tests exercise (matches `ProfilePaths` resolution when no profile
    /// subdirs exist).
    fn chat_roots(root: &Path) -> ChatRoots {
        ChatRoots {
            chats: root.join("chats"),
            group_chats: root.join("group chats"),
        }
    }

    #[test]
    fn checkpoint_and_restore_use_repository_chat_roots() {
        // Chat files under a DIFFERENT root than the save-sync content root —
        // the imported-profile layout that broke rollback (the profile ships
        // `headless/` but no `group chats/`, so the repository writes session
        // files under the legacy data root while save-sync state lives in the
        // profile folder).
        let content = TempRoot::new("split-content");
        let legacy = TempRoot::new("split-legacy");
        let roots = chat_roots(legacy.path());

        // Store under the CONTENT root; the backing chat file under LEGACY.
        let payload = json!({ "mode": "group", "groupId": "g1", "chatId": "Goodsprings" });
        let session_id = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        let store = json!({
            "items": {
                "lc1": { "id": "lc1", "segments": [ { "id": "seg1", "sessionId": session_id } ] }
            }
        });
        write_live_chat_store_raw(content.path(), &store).unwrap();
        fs::create_dir_all(legacy.path().join("group chats")).unwrap();
        let chat_path = legacy.path().join("group chats").join("Goodsprings.jsonl");
        fs::write(
            &chat_path,
            "{\"chat_metadata\":{}}\n{\"name\":\"Sunny\",\"mes\":\"Howdy.\"}\n",
        )
        .unwrap();

        // Checkpoint captures the REAL file content, not an exists:false phantom.
        handle_save_sync_event(content.path(), content.path(), &roots, &save_body()).unwrap();
        let checkpoint_id = save_sync_checkpoint_id("fallout-new-vegas", "Save7");
        let snapshot = read_checkpoint_snapshot(content.path(), &checkpoint_id).unwrap();
        let files = snapshot["files"].as_array().unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0]["exists"].as_bool().unwrap());
        assert!(files[0]["size"].as_u64().unwrap() > 0);

        // Talk after the save, then load: the REAL file rolls back.
        fs::write(
            &chat_path,
            "{\"chat_metadata\":{}}\n{\"name\":\"Sunny\",\"mes\":\"I remember Charlie.\"}\n",
        )
        .unwrap();
        let load_body = json!({
            "event": "load",
            "gameId": "fallout-new-vegas",
            "saveId": "Save7",
        });
        let result =
            handle_save_sync_event(content.path(), content.path(), &roots, &load_body).unwrap();
        assert_eq!(result["status"], "restored");
        let restored = fs::read_to_string(&chat_path).unwrap();
        assert!(restored.contains("Howdy."));
        assert!(!restored.contains("Charlie"));
    }

    #[test]
    fn save_and_load_carry_the_event_log_along() {
        // End-to-end through handle_save_sync_event: the event log checkpoints
        // with the chat state and rolls back with it on load.
        let root = TempRoot::new("event-log");
        seed_live_chat(root.path());

        let evt = |id: &str, summary: &str| json!({ "id": id, "type": "combat", "summary": summary });
        crate::event_log::append_events(root.path(), &[evt("a", "Shot a gecko")]).unwrap();
        handle_save_sync_event(root.path(), root.path(), &chat_roots(root.path()), &save_body()).unwrap();

        // The doomed branch after the save.
        crate::event_log::append_events(root.path(), &[evt("b", "Died horribly")]).unwrap();

        let load_body = json!({ "event": "load", "gameId": "fallout-new-vegas", "saveId": "Save7" });
        let result =
            handle_save_sync_event(root.path(), root.path(), &chat_roots(root.path()), &load_body).unwrap();
        assert_eq!(result["status"], "restored");

        let log = fs::read_to_string(
            root.path().join("headless").join("event-log").join("current.jsonl"),
        )
        .unwrap();
        assert!(log.contains("Shot a gecko"));
        assert!(!log.contains("Died horribly"));
    }

    #[test]
    fn save_event_creates_checkpoint_in_store() {
        let root = TempRoot::new("create");
        seed_live_chat(root.path());

        let result = handle_save_sync_event(root.path(), root.path(), &chat_roots(root.path()), &save_body()).unwrap();
        assert_eq!(result["status"], "checkpoint_created");

        let checkpoint_id = save_sync_checkpoint_id("fallout-new-vegas", "Save7");
        assert_eq!(result["checkpoint"]["checkpointId"], checkpoint_id);

        // Store index records the checkpoint and marks it current.
        let store = read_save_sync_store(root.path());
        assert!(store.items.contains_key(&checkpoint_id));
        assert_eq!(store.current["checkpointId"], checkpoint_id);

        // The full snapshot file exists and captured the live chat + chat file.
        let snapshot = read_checkpoint_snapshot(root.path(), &checkpoint_id).unwrap();
        assert_eq!(snapshot["liveChats"]["items"]["lc1"]["id"], "lc1");
        let files = snapshot["files"].as_array().unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0]["exists"].as_bool().unwrap());
        assert!(files[0]["size"].as_u64().unwrap() > 0);

        // A second save with the same identity updates (not duplicates) it.
        let result2 = handle_save_sync_event(root.path(), root.path(), &chat_roots(root.path()), &save_body()).unwrap();
        assert_eq!(result2["status"], "checkpoint_updated");
        let store2 = read_save_sync_store(root.path());
        assert_eq!(store2.items.len(), 1);
    }

    #[test]
    fn load_event_restores_checkpoint() {
        let root = TempRoot::new("restore");
        let session_id = seed_live_chat(root.path());

        // Checkpoint the seeded state.
        handle_save_sync_event(root.path(), root.path(), &chat_roots(root.path()), &save_body()).unwrap();

        // Mutate live state AFTER the checkpoint: change the chat file + drop the
        // live chat from the store.
        let chat_path = session_file_path(&chat_roots(root.path()), &session_id).unwrap();
        fs::write(
            &chat_path,
            "{\"chat_metadata\":{}}\n{\"name\":\"X\",\"mes\":\"changed\"}\n",
        )
        .unwrap();
        write_live_chat_store_raw(root.path(), &json!({ "items": {} })).unwrap();

        // Load restores the snapshot.
        let load_body = json!({
            "event": "load",
            "gameId": "fallout-new-vegas",
            "saveId": "Save7",
        });
        let result = handle_save_sync_event(root.path(), root.path(), &chat_roots(root.path()), &load_body).unwrap();
        assert_eq!(result["status"], "restored");
        assert_eq!(result["restored"], true);
        assert!(result["counts"]["liveChatsRestored"].as_u64().unwrap() >= 1);

        // Live chat is back in the store, and the chat file content is restored.
        let store = read_live_chat_store_raw(root.path());
        assert_eq!(store["items"]["lc1"]["id"], "lc1");
        let restored = fs::read_to_string(&chat_path).unwrap();
        assert!(restored.contains("Howdy."));
        assert!(!restored.contains("changed"));

        // A safety backup was written before the restore.
        let safety = result["safetyBackupFile"].as_str().unwrap();
        assert!(!safety.is_empty());
        assert!(Path::new(safety).exists());
    }

    #[test]
    fn disabled_option_gates_event() {
        let root = TempRoot::new("disabled");
        seed_live_chat(root.path());

        let mut body = save_body();
        body["options"] = json!({ "enabled": false });
        let result = handle_save_sync_event(root.path(), root.path(), &chat_roots(root.path()), &body).unwrap();
        assert_eq!(result["status"], "disabled");

        // Nothing was checkpointed.
        let store = read_save_sync_store(root.path());
        assert!(store.items.is_empty());
    }

    #[test]
    fn auto_checkpoint_disabled_ignores_save() {
        let root = TempRoot::new("noauto");
        seed_live_chat(root.path());

        let mut body = save_body();
        body["options"] = json!({ "autoCheckpoint": false });
        let result = handle_save_sync_event(root.path(), root.path(), &chat_roots(root.path()), &body).unwrap();
        assert_eq!(result["status"], "ignored");
        assert_eq!(result["reason"], "autoCheckpoint disabled");
        assert!(read_save_sync_store(root.path()).items.is_empty());
    }

    #[test]
    fn load_missing_checkpoint_reports_snapshot_missing() {
        let root = TempRoot::new("missing");
        let body = json!({ "event": "load", "gameId": "g", "saveId": "nope" });
        let result = handle_save_sync_event(root.path(), root.path(), &chat_roots(root.path()), &body).unwrap();
        assert_eq!(result["status"], "snapshot_missing");
        assert_eq!(result["restored"], false);
    }

    #[test]
    fn unknown_event_is_an_error() {
        let root = TempRoot::new("unknown");
        let body = json!({ "event": "frobnicate", "gameId": "g", "saveId": "s" });
        let err = handle_save_sync_event(root.path(), root.path(), &chat_roots(root.path()), &body);
        assert!(err.is_err());
    }

    #[test]
    fn retention_prunes_oldest_checkpoints() {
        let root = TempRoot::new("retention");

        // retentionLimit=2: create three distinct checkpoints; the oldest two by
        // updatedAt should be pruned down to the limit (plus current).
        for index in 0..3u32 {
            let body = json!({
                "event": "save",
                "gameId": "g",
                "saveId": format!("save-{index}"),
                "options": { "retentionLimit": 2, "includeLiveChats": false, "includeWorldState": false },
            });
            handle_save_sync_event(root.path(), root.path(), &chat_roots(root.path()), &body).unwrap();
            // Ensure strictly increasing updatedAt timestamps (ms precision).
            std::thread::sleep(std::time::Duration::from_millis(3));
        }

        let store = read_save_sync_store(root.path());
        assert!(
            store.items.len() <= 2,
            "expected retention to cap items at 2, got {}",
            store.items.len()
        );
        // The newest (save-2) is current and must survive.
        let newest = save_sync_checkpoint_id("g", "save-2");
        assert!(store.items.contains_key(&newest));
        // Its checkpoint file must still be on disk.
        assert!(checkpoint_file_path(root.path(), &newest).unwrap().exists());
        // The oldest file should be gone.
        let oldest = save_sync_checkpoint_id("g", "save-0");
        assert!(!checkpoint_file_path(root.path(), &oldest).unwrap().exists());
    }

    #[test]
    fn checkpoint_id_matches_node_algorithm() {
        // sha256(lower(gameId) + \0 + lower(saveId)).slice(0,48). Stable hash, so
        // a fixed input must produce a fixed 48-char hex id (round-trips w/ Node).
        let id = save_sync_checkpoint_id("Fallout-NV", "Save7");
        assert_eq!(id.len(), 48);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
        // Case-insensitivity: differing case yields the same id.
        assert_eq!(id, save_sync_checkpoint_id("fallout-nv", "save7"));
    }

    #[test]
    fn path_traversal_is_blocked() {
        let root = TempRoot::new("traversal");
        // A crafted group session id whose chatId tries to climb out via `..`.
        let payload = json!({ "mode": "group", "groupId": "g", "chatId": "../../escape" });
        let session_id = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        let resolved = session_file_path(&chat_roots(root.path()), &session_id);
        // Either rejected outright, or sanitized to stay under the data root.
        if let Some(path) = resolved {
            assert!(
                is_path_under(&root.path().join("group chats"), &path),
                "resolved path escaped the data root: {}",
                path.display()
            );
        }
    }

    #[test]
    fn iso_timestamp_format_is_correct() {
        // 2021-01-01T00:00:00.000Z is 1609459200000 ms since epoch.
        assert_eq!(
            format_iso_millis(1_609_459_200_000),
            "2021-01-01T00:00:00.000Z"
        );
        // A value with sub-second precision.
        assert_eq!(
            format_iso_millis(1_609_459_200_123),
            "2021-01-01T00:00:00.123Z"
        );
    }
}
