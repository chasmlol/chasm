//! Game event-log relay — ship the plugin's gameplay-event batches to chasm.
//!
//! The NVSE plugin aggregates notable gameplay events (combat encounters,
//! deaths, travel, loot, conversations, quest/world beats) in-game and writes
//! them as JSONL batch files to `{root}/control/gameevents/` — one complete
//! JSON event object per line (see `native/nvse-plugin/main.cpp`, the event
//! extraction section). This module reads each batch, POSTs it to chasm's
//! `POST /event-log/events` (which owns the save-aware store), and archives the
//! file to `processed/game-events/`.
//!
//! Unlike save-state events there is no ack: the plugin fires and forgets.
//! Delivery is at-least-once — a crash between POST and archive re-delivers the
//! batch, and chasm dedups by event id. A batch that keeps failing (chasm down,
//! endpoint erroring) is retried a few polls and then parked as `.failed` so it
//! never wedges the loop.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde_json::{json, Value};
use tracing::{error, info};

use crate::chasm::ChasmClient;
use crate::config::BridgeConfig;
use crate::protocol::{now_epoch_millis, safe_file_id};

const MAX_DELIVERY_ATTEMPTS: u32 = 5;

pub(crate) fn game_event_dir(root: &Path) -> PathBuf {
    root.join("control").join("gameevents")
}

/// Per-file delivery attempts, so a persistently failing batch is eventually
/// parked instead of retried forever at poll cadence.
fn attempts() -> &'static Mutex<HashMap<PathBuf, u32>> {
    static ATTEMPTS: OnceLock<Mutex<HashMap<PathBuf, u32>>> = OnceLock::new();
    ATTEMPTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Process every pending game-event batch under `{root}/control/gameevents/`.
/// Self-contained like `saves::process_save_state_events`: never returns Err —
/// a bad batch is logged and parked, not allowed to break the poll loop.
pub async fn process_game_events(_config: &BridgeConfig, client: &dyn ChasmClient, root: &Path) {
    let directory = game_event_dir(root);
    let entries = match std::fs::read_dir(&directory) {
        Ok(e) => e,
        Err(_) => return,
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("txt") || e.eq_ignore_ascii_case("jsonl"))
                .unwrap_or(false)
        })
        .filter(|p| {
            !p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("__"))
                .unwrap_or(false)
        })
        .collect();
    files.sort();

    for path in files {
        match deliver_batch(client, &path).await {
            Ok(count) => {
                attempts().lock().unwrap().remove(&path);
                archive_batch(root, &path, "");
                if count > 0 {
                    info!("event-log: delivered {count} event(s) from {}", file_name(&path));
                }
            }
            Err(e) => {
                let n = {
                    let mut map = attempts().lock().unwrap();
                    let n = map.entry(path.clone()).or_insert(0);
                    *n += 1;
                    *n
                };
                if n >= MAX_DELIVERY_ATTEMPTS {
                    attempts().lock().unwrap().remove(&path);
                    archive_batch(root, &path, ".failed");
                    error!("event-log batch {} failed {n} time(s), parked: {e}", file_name(&path));
                } else {
                    error!("event-log batch {} attempt {n}: {e}", file_name(&path));
                }
            }
        }
    }
}

/// Parse one JSONL batch file and POST it to chasm. Unparseable lines are
/// skipped (the plugin writes each line atomically, but be forgiving). An empty
/// batch is a successful no-op delivery.
async fn deliver_batch(client: &dyn ChasmClient, path: &Path) -> anyhow::Result<usize> {
    let text = std::fs::read_to_string(path)?;
    let events: Vec<Value> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v.is_object())
        .collect();
    if events.is_empty() {
        return Ok(0);
    }
    let count = events.len();
    client.event_log_ingest(&json!({ "events": events })).await?;
    Ok(count)
}

fn archive_batch(root: &Path, path: &Path, suffix: &str) {
    let dir = root.join("processed").join("game-events");
    let _ = std::fs::create_dir_all(&dir);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let name = format!("{}-{}{suffix}.txt", now_epoch_millis(), safe_file_id(stem));
    if std::fs::rename(path, dir.join(&name)).is_err() {
        let _ = std::fs::remove_file(path);
    }
}

fn file_name(path: &Path) -> String {
    path.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string()
}
