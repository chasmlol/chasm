//! Game event log — the save-aware, per-playthrough record of notable gameplay
//! events (combat encounters, deaths, travel, loot, conversations, quest/world
//! beats). The NVSE plugin extracts events in-game and drops JSONL batches into
//! the bridge's `control/gameevents/`; the bridge relays them to
//! `POST /event-log/events` (this module), and the React UI reads the current
//! log from `GET /api/ui/v1/events`.
//!
//! The log follows the player's SAVES the same way chat history does (see
//! [`crate::save_sync`]): every save-sync checkpoint also snapshots the event
//! log, and every restore rolls the log back to that snapshot — events from the
//! discarded timeline disappear (a copy is kept under `discarded/`). The hooks
//! live in `save_sync::handle_save_sync_event`, so the event log can never drift
//! from the chat checkpoints.
//!
//! ## On-disk layout (under `<data_root>/headless/event-log/`)
//! * `current.jsonl`            — the live log, one JSON event per line.
//! * `checkpoints/<id>.jsonl`   — the log as of save-sync checkpoint `<id>`.
//! * `discarded/<ms>-<id>.jsonl`— pre-restore backups (the abandoned branch).
//!
//! Events are append-only between saves; `seq` is assigned here and is strictly
//! increasing within the current timeline. Ingest dedups by event `id` so a
//! re-delivered bridge batch (crash between POST and archive) is harmless.

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use axum::{extract::State, Json};
use serde_json::{json, Map, Value};

use crate::save_sync::{epoch_millis, now_iso};
use crate::{AppState, WebError, WebResult};

const EVENT_LOG_DIRECTORY: &str = "event-log";
const CURRENT_FILE: &str = "current.jsonl";
const CHECKPOINTS_DIRECTORY: &str = "checkpoints";
const DISCARDED_DIRECTORY: &str = "discarded";
/// Ingest guards: a single bridge batch is small (the plugin flushes every few
/// seconds); anything bigger is malformed input, not gameplay.
const MAX_BATCH_EVENTS: usize = 500;
const MAX_SUMMARY_CHARS: usize = 400;
/// The UI view returns at most this many trailing events.
const MAX_VIEW_EVENTS: usize = 2000;

fn web_err(message: impl Into<String>) -> WebError {
    WebError::from(anyhow::anyhow!(message.into()))
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

fn event_log_dir(data_root: &Path) -> PathBuf {
    data_root.join("headless").join(EVENT_LOG_DIRECTORY)
}

fn current_file(data_root: &Path) -> PathBuf {
    event_log_dir(data_root).join(CURRENT_FILE)
}

fn checkpoint_file(data_root: &Path, checkpoint_id: &str) -> PathBuf {
    event_log_dir(data_root)
        .join(CHECKPOINTS_DIRECTORY)
        .join(format!("{}.jsonl", safe_id(checkpoint_id)))
}

/// Checkpoint ids are save-sync sha256 hex slices; anything else is rejected at
/// the filename level (defense in depth — ids never come from the network raw).
fn safe_id(id: &str) -> String {
    id.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(64)
        .collect()
}

// ---------------------------------------------------------------------------
// Store primitives
// ---------------------------------------------------------------------------

/// Read every event in a JSONL file (missing file = empty log). Unparseable
/// lines are skipped rather than poisoning the whole log.
fn read_events_file(path: &Path) -> Vec<Value> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v.is_object())
        .collect()
}

fn write_events_file(path: &Path, events: &[Value]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut body = String::new();
    for event in events {
        body.push_str(&serde_json::to_string(event)?);
        body.push('\n');
    }
    fs::write(path, body)?;
    Ok(())
}

/// Normalize one incoming event object: enforce the known fields, cap the
/// summary, drop unknown top-level keys into place (extra data rides in
/// `data`). Returns `None` for non-objects and events with no usable summary.
fn normalize_event(raw: &Value, seq: u64, now: &str) -> Option<Value> {
    let obj = raw.as_object()?;
    let str_field = |key: &str| -> String {
        obj.get(key)
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string()
    };
    let summary: String = str_field("summary").chars().take(MAX_SUMMARY_CHARS).collect();
    if summary.is_empty() {
        return None;
    }
    let id = {
        let raw_id = str_field("id");
        if raw_id.is_empty() {
            format!("evt-{seq}")
        } else {
            safe_id(&raw_id)
        }
    };
    let event_type = {
        let t = str_field("type").to_lowercase();
        if t.is_empty() {
            "world".to_string()
        } else {
            t
        }
    };
    let real_time = {
        let t = str_field("realTime");
        if t.is_empty() {
            now.to_string()
        } else {
            t
        }
    };
    let mut out = Map::new();
    out.insert("id".into(), json!(id));
    out.insert("seq".into(), json!(seq));
    out.insert("type".into(), json!(event_type));
    out.insert("summary".into(), json!(summary));
    out.insert("realTime".into(), json!(real_time));
    let game_time = str_field("gameTime");
    if !game_time.is_empty() {
        out.insert("gameTime".into(), json!(game_time));
    }
    // In-game day counter (GameDaysPassed, 1-based). Accept a JSON number or a
    // numeric string, keep it as a number.
    if let Some(day) = obj
        .get("gameDay")
        .and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())))
    {
        if day >= 0 {
            out.insert("gameDay".into(), json!(day));
        }
    }
    let location = str_field("location");
    if !location.is_empty() {
        out.insert("location".into(), json!(location));
    }
    if let Some(actors) = obj.get("actors").and_then(Value::as_array) {
        let actors: Vec<Value> = actors
            .iter()
            .filter(|a| a.is_object() || a.is_string())
            .take(16)
            .cloned()
            .collect();
        if !actors.is_empty() {
            out.insert("actors".into(), json!(actors));
        }
    }
    if let Some(data) = obj.get("data").filter(|d| d.is_object()) {
        out.insert("data".into(), data.clone());
    }
    Some(Value::Object(out))
}

/// Append a batch to the current log. Dedups by event id against the existing
/// log, assigns `seq`, and returns the number of events actually appended.
pub(crate) fn append_events(data_root: &Path, incoming: &[Value]) -> anyhow::Result<usize> {
    let path = current_file(data_root);
    let mut events = read_events_file(&path);
    let mut seen: HashSet<String> = events
        .iter()
        .filter_map(|e| e.get("id").and_then(Value::as_str).map(str::to_string))
        .collect();
    let mut next_seq = events
        .iter()
        .filter_map(|e| e.get("seq").and_then(Value::as_u64))
        .max()
        .map(|s| s + 1)
        .unwrap_or(1);
    let now = now_iso();
    let mut appended = 0usize;
    for raw in incoming.iter().take(MAX_BATCH_EVENTS) {
        let Some(event) = normalize_event(raw, next_seq, &now) else {
            continue;
        };
        let id = event
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if !id.is_empty() && !seen.insert(id) {
            continue; // Redelivered batch — already logged.
        }
        events.push(event);
        next_seq += 1;
        appended += 1;
    }
    if appended > 0 {
        write_events_file(&path, &events)?;
    }
    Ok(appended)
}

// ---------------------------------------------------------------------------
// Checkpoint / restore (called from save_sync::handle_save_sync_event)
// ---------------------------------------------------------------------------

/// Snapshot the current log as checkpoint `checkpoint_id` (the save-sync id for
/// this save slot). Overwrites any previous snapshot for the same slot, exactly
/// like re-saving over a slot overwrites its chat checkpoint. Prunes old
/// snapshots beyond `retention_limit` (newest by mtime survive; <= 0 keeps all).
pub(crate) fn checkpoint_event_log(
    data_root: &Path,
    checkpoint_id: &str,
    retention_limit: i64,
) -> anyhow::Result<()> {
    let events = read_events_file(&current_file(data_root));
    write_events_file(&checkpoint_file(data_root, checkpoint_id), &events)?;
    prune_checkpoints(data_root, retention_limit);
    Ok(())
}

/// Roll the current log back to checkpoint `checkpoint_id`. The replaced log
/// (the now-abandoned branch) is copied to `discarded/` first. A checkpoint
/// with no event-log snapshot (saved before this feature existed) restores to
/// an empty log — that IS the log's state at that save point.
pub(crate) fn restore_event_log(data_root: &Path, checkpoint_id: &str) -> anyhow::Result<Value> {
    let current = current_file(data_root);
    let before = read_events_file(&current);
    let snapshot = read_events_file(&checkpoint_file(data_root, checkpoint_id));

    // Only archive when the rollback actually discards something.
    if before.len() > snapshot.len() {
        let discarded_dir = event_log_dir(data_root).join(DISCARDED_DIRECTORY);
        fs::create_dir_all(&discarded_dir)?;
        let name = format!("{}-{}.jsonl", epoch_millis(), safe_id(checkpoint_id));
        write_events_file(&discarded_dir.join(name), &before)?;
    }
    write_events_file(&current, &snapshot)?;
    Ok(json!({
        "restored": true,
        "checkpointId": checkpoint_id,
        "events": snapshot.len(),
        "discarded": before.len().saturating_sub(snapshot.len()),
    }))
}

fn prune_checkpoints(data_root: &Path, retention_limit: i64) {
    if retention_limit <= 0 {
        return;
    }
    let dir = event_log_dir(data_root).join(CHECKPOINTS_DIRECTORY);
    let Ok(entries) = fs::read_dir(&dir) else {
        return;
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            let modified = e.metadata().ok()?.modified().ok()?;
            Some((modified, path))
        })
        .collect();
    if files.len() <= retention_limit as usize {
        return;
    }
    files.sort_by(|a, b| b.0.cmp(&a.0)); // newest first
    for (_, path) in files.into_iter().skip(retention_limit as usize) {
        let _ = fs::remove_file(path);
    }
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/// `POST /api/headless/v1/event-log/events` — the bridge's ingest route.
/// Body: `{ "events": [ {...}, ... ] }`. Appends to the ACTIVE profile's
/// current log; returns `{ appended, total? }` counts.
pub async fn ingest_events(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> WebResult<Json<Value>> {
    let Some(events) = body.get("events").and_then(Value::as_array).cloned() else {
        return Err(web_err("body.events must be an array."));
    };
    let data_root = state.config.active_profile_paths().content_root();
    let appended = tokio::task::spawn_blocking(move || append_events(&data_root, &events))
        .await
        .map_err(|e| web_err(e.to_string()))??;
    Ok(Json(json!({ "status": "ok", "appended": appended })))
}

/// `GET /api/ui/v1/events` — the Events page projection: the current log's
/// trailing events (ascending `seq`) plus a total count.
pub(crate) async fn events_view(
    State(state): State<Arc<AppState>>,
) -> WebResult<Json<Value>> {
    let data_root = state.config.active_profile_paths().content_root();
    let (events, total) = tokio::task::spawn_blocking(move || {
        let all = read_events_file(&current_file(&data_root));
        let total = all.len();
        let skip = total.saturating_sub(MAX_VIEW_EVENTS);
        (all.into_iter().skip(skip).collect::<Vec<_>>(), total)
    })
    .await
    .map_err(|e| web_err(e.to_string()))?;
    Ok(Json(json!({ "events": events, "total": total })))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new(tag: &str) -> Self {
            let mut dir = std::env::temp_dir();
            dir.push(format!(
                "sb-event-log-test-{tag}-{}-{}",
                std::process::id(),
                epoch_millis()
            ));
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

    fn event(id: &str, summary: &str) -> Value {
        json!({ "id": id, "type": "combat", "summary": summary, "location": "Goodsprings" })
    }

    fn summaries(root: &Path) -> Vec<String> {
        read_events_file(&current_file(root))
            .iter()
            .map(|e| e["summary"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn append_assigns_increasing_seq_and_dedups_by_id() {
        let root = TempRoot::new("append");
        let n = append_events(root.path(), &[event("a", "First"), event("b", "Second")]).unwrap();
        assert_eq!(n, 2);
        // Redelivered batch (bridge crash between POST and archive) is a no-op.
        let n = append_events(root.path(), &[event("b", "Second"), event("c", "Third")]).unwrap();
        assert_eq!(n, 1);
        let events = read_events_file(&current_file(root.path()));
        let seqs: Vec<u64> = events.iter().map(|e| e["seq"].as_u64().unwrap()).collect();
        assert_eq!(seqs, vec![1, 2, 3]);
        assert_eq!(summaries(root.path()), vec!["First", "Second", "Third"]);
    }

    #[test]
    fn append_preserves_game_day_and_structured_data() {
        let root = TempRoot::new("gameday");
        let incoming = json!({
            "id": "x",
            "type": "location",
            "summary": "Entered Prospector Saloon",
            "gameDay": 3,
            "location": "Prospector Saloon, Goodsprings",
            "data": { "locationMajor": "Goodsprings", "locationMinor": "Prospector Saloon" }
        });
        append_events(root.path(), &[incoming]).unwrap();
        let stored = &read_events_file(&current_file(root.path()))[0];
        assert_eq!(stored["gameDay"], 3);
        assert_eq!(stored["data"]["locationMajor"], "Goodsprings");
        assert_eq!(stored["data"]["locationMinor"], "Prospector Saloon");
    }

    #[test]
    fn append_skips_events_without_summary() {
        let root = TempRoot::new("nosummary");
        let n = append_events(
            root.path(),
            &[json!({ "id": "x", "type": "combat" }), event("y", "Real")],
        )
        .unwrap();
        assert_eq!(n, 1);
        assert_eq!(summaries(root.path()), vec!["Real"]);
    }

    #[test]
    fn checkpoint_then_restore_rolls_back_the_branch() {
        let root = TempRoot::new("rollback");
        append_events(root.path(), &[event("a", "Before save")]).unwrap();
        checkpoint_event_log(root.path(), "cp1", 50).unwrap();

        // The doomed branch: events after the save.
        append_events(root.path(), &[event("b", "After save"), event("c", "Also after")]).unwrap();
        assert_eq!(summaries(root.path()).len(), 3);

        let result = restore_event_log(root.path(), "cp1").unwrap();
        assert_eq!(result["discarded"], 2);
        assert_eq!(summaries(root.path()), vec!["Before save"]);

        // The discarded branch is archived, not lost.
        let discarded_dir = event_log_dir(root.path()).join(DISCARDED_DIRECTORY);
        let backups: Vec<_> = fs::read_dir(&discarded_dir).unwrap().flatten().collect();
        assert_eq!(backups.len(), 1);
        let archived = read_events_file(&backups[0].path());
        assert_eq!(archived.len(), 3);
    }

    #[test]
    fn restore_after_rollback_continues_seq_without_collisions() {
        let root = TempRoot::new("reseq");
        append_events(root.path(), &[event("a", "One")]).unwrap();
        checkpoint_event_log(root.path(), "cp1", 50).unwrap();
        append_events(root.path(), &[event("b", "Two")]).unwrap();
        restore_event_log(root.path(), "cp1").unwrap();
        // New branch after the rollback: seq restarts after the restored tail.
        append_events(root.path(), &[event("d", "New branch")]).unwrap();
        let events = read_events_file(&current_file(root.path()));
        let seqs: Vec<u64> = events.iter().map(|e| e["seq"].as_u64().unwrap()).collect();
        assert_eq!(seqs, vec![1, 2]);
        assert_eq!(summaries(root.path()), vec!["One", "New branch"]);
    }

    #[test]
    fn restore_with_no_snapshot_empties_the_log_but_archives_it() {
        let root = TempRoot::new("missing");
        append_events(root.path(), &[event("a", "Orphan")]).unwrap();
        let result = restore_event_log(root.path(), "never-saved").unwrap();
        assert_eq!(result["events"], 0);
        assert_eq!(result["discarded"], 1);
        assert!(summaries(root.path()).is_empty());
        let discarded_dir = event_log_dir(root.path()).join(DISCARDED_DIRECTORY);
        assert_eq!(fs::read_dir(&discarded_dir).unwrap().flatten().count(), 1);
    }

    #[test]
    fn resaving_a_slot_overwrites_its_checkpoint() {
        let root = TempRoot::new("resave");
        append_events(root.path(), &[event("a", "One")]).unwrap();
        checkpoint_event_log(root.path(), "cp1", 50).unwrap();
        append_events(root.path(), &[event("b", "Two")]).unwrap();
        checkpoint_event_log(root.path(), "cp1", 50).unwrap();
        restore_event_log(root.path(), "cp1").unwrap();
        assert_eq!(summaries(root.path()), vec!["One", "Two"]);
    }

    #[test]
    fn checkpoint_pruning_keeps_newest() {
        let root = TempRoot::new("prune");
        append_events(root.path(), &[event("a", "One")]).unwrap();
        for i in 0..5 {
            checkpoint_event_log(root.path(), &format!("cp{i}"), 3).unwrap();
        }
        let dir = event_log_dir(root.path()).join(CHECKPOINTS_DIRECTORY);
        assert_eq!(fs::read_dir(&dir).unwrap().flatten().count(), 3);
    }
}
