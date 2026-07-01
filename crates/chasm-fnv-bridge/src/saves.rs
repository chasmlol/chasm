//! Save-sync — checkpoint/restore the conversation on game save/load. Port of the
//! Node helper's save-sync path: detect save/load events (inbox requests + the
//! plugin's `control/events` files), call chasm `POST /save-sync/events`, and write
//! acks to `control/acks`. chasm owns the actual checkpoint store; we relay events.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::{json, Value};
use tracing::{error, info};

use crate::chasm::ChasmClient;
use crate::config::BridgeConfig;
use crate::npc::location_string;
use crate::protocol::{now_epoch_millis, safe_file_id, sanitize_bridge_line, split_crlf, NativeRequest};

/// The normalized inputs for a single save-sync event.
pub struct SaveContext {
    pub event: &'static str, // "save" | "load"
    pub save_id: String,
    pub save_name: String,
    pub save_file: String,
    pub save_fingerprint: String,
    pub request_id: String,
    pub location: String,
    pub source: String,
}

/// What chasm reported for a save-sync event.
pub struct SaveSyncOutcome {
    pub status: String,
    pub checkpoint_id: String,
    pub checkpoint_save_name: String,
}

/// `isNativeSaveSyncRequest`.
pub fn is_native_save_sync_request(request: &NativeRequest) -> bool {
    normalize_event(&event_field(request)).is_some()
}

fn normalize_event(raw: &str) -> Option<&'static str> {
    match raw.trim().to_lowercase().as_str() {
        "save" | "saved" | "checkpoint" | "autosave" | "quicksave" => Some("save"),
        "load" | "loaded" | "restore" | "reload" => Some("load"),
        _ => None,
    }
}

fn event_field(request: &NativeRequest) -> String {
    for key in ["save_event", "saveEvent", "event", "event_type", "eventType", "type"] {
        if let Some(s) = request.metadata.get(key).and_then(value_str) {
            if !s.trim().is_empty() {
                return s;
            }
        }
    }
    String::new()
}

/// Build a [`SaveContext`] from an inbox save-sync request (fields in metadata).
pub fn save_context_from_request(request: &NativeRequest) -> anyhow::Result<SaveContext> {
    let event = normalize_event(&event_field(request))
        .ok_or_else(|| anyhow::anyhow!("not a save-sync request"))?;
    let save = request.metadata.get("save").filter(|v| v.is_object());
    let meta = |top: &[&str], nested: &[&str]| -> String {
        for k in top {
            if let Some(s) = request.metadata.get(*k).and_then(value_str) {
                if !s.is_empty() {
                    return s;
                }
            }
        }
        if let Some(obj) = save {
            for k in nested {
                if let Some(s) = obj.get(*k).and_then(value_str) {
                    if !s.is_empty() {
                        return s;
                    }
                }
            }
        }
        String::new()
    };

    let save_id = sanitize_bridge_line(&meta(
        &["save_id", "saveId", "save_slot", "saveSlot", "save_file", "saveFile", "file"],
        &["id", "saveId", "slot", "file"],
    ));
    if save_id.is_empty() {
        anyhow::bail!("Save-sync request did not include save_id/saveId/save_file.");
    }
    let save_name = sanitize_bridge_line(&meta(&["save_name", "saveName"], &["name"]));
    let save_name = if save_name.is_empty() { save_id.clone() } else { save_name };
    Ok(SaveContext {
        event,
        save_id,
        save_name,
        save_file: sanitize_bridge_line(&meta(&["save_file", "saveFile"], &["file"])),
        save_fingerprint: sanitize_bridge_line(&meta(
            &["save_fingerprint", "saveFingerprint"],
            &["fingerprint", "modified_at", "modifiedAt"],
        )),
        request_id: request.request_id.clone(),
        location: location_string(request),
        source: "fallout-new-vegas-native".into(),
    })
}

/// `POST /save-sync/events` and parse the outcome.
pub async fn call_save_sync_event(
    config: &BridgeConfig,
    client: &dyn ChasmClient,
    ctx: &SaveContext,
    extra_meta: &[(&str, String)],
) -> anyhow::Result<SaveSyncOutcome> {
    let mut metadata = json!({
        "requestId": ctx.request_id,
        "location": ctx.location,
        "nativeEvent": ctx.event,
    });
    if let Some(obj) = metadata.as_object_mut() {
        for (k, v) in extra_meta {
            obj.insert((*k).to_string(), json!(v));
        }
    }
    let body = json!({
        "event": ctx.event,
        "gameId": "fallout-new-vegas",
        "gameName": "Fallout: New Vegas",
        "saveId": ctx.save_id,
        "saveName": ctx.save_name,
        "saveFile": ctx.save_file,
        "saveFingerprint": ctx.save_fingerprint,
        "liveChatIds": [config.live_chat_id],
        "source": ctx.source,
        "metadata": metadata,
    });
    let result = client.save_sync_event(&body).await?;
    let default_status = if ctx.event == "save" { "checkpoint_created" } else { "restored" };
    let status = result
        .get("status")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(default_status)
        .to_string();
    let checkpoint = result.get("checkpoint");
    let checkpoint_id = result
        .get("checkpointId")
        .and_then(Value::as_str)
        .or_else(|| checkpoint.and_then(|c| c.get("checkpointId")).and_then(Value::as_str))
        .unwrap_or("")
        .to_string();
    let checkpoint_save_name = checkpoint
        .and_then(|c| c.get("saveName"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    Ok(SaveSyncOutcome {
        status,
        checkpoint_id,
        checkpoint_save_name,
    })
}

// ---------------------------------------------------------------------------
// Save-state event files (control/events → control/acks)
// ---------------------------------------------------------------------------

/// Process the plugin's save-state event files (written on save/load). Self-
/// contained: calls chasm, writes an ack, archives the event.
pub async fn process_save_state_events(config: &BridgeConfig, client: &dyn ChasmClient, root: &Path) {
    let directory = event_dir(root);
    let entries = match std::fs::read_dir(&directory) {
        Ok(e) => e,
        Err(_) => return,
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("txt")).unwrap_or(false))
        .collect();
    files.sort();
    for path in files {
        if let Err(e) = process_one_event(config, client, root, &path).await {
            let event_id = file_stem(&path);
            let _ = write_save_state_ack(root, &event_id, false, &e.to_string(), &[]);
            archive_event(root, &path, ".failed");
            error!("save-sync event {}: {e}", file_stem(&path));
        }
    }
}

async fn process_one_event(
    config: &BridgeConfig,
    client: &dyn ChasmClient,
    root: &Path,
    path: &Path,
) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(path)?;
    let ctx = parse_save_state_event(path, &text);

    let normalized = normalize_event(&ctx.event);
    let Some(event) = normalized else {
        // Not a save/load event — ack as ignored.
        write_save_state_ack(root, &ctx.request_id, true, &format!("Ignored save-state event {}.", ctx.event), &[])?;
        archive_event(root, path, "");
        info!("ignored save-state event {} ({})", ctx.event, ctx.request_id);
        return Ok(());
    };
    if ctx.save_id.is_empty() {
        anyhow::bail!("save-state event {} had no save id", ctx.request_id);
    }

    let save_ctx = SaveContext {
        event,
        save_id: ctx.save_id.clone(),
        save_name: ctx.save_name.clone(),
        save_file: ctx.save_file.clone(),
        save_fingerprint: ctx.save_fingerprint.clone(),
        request_id: ctx.request_id.clone(),
        location: String::new(),
        source: "fallout-new-vegas-native-savestate".into(),
    };
    let outcome = call_save_sync_event(config, client, &save_ctx, &[("nativeSaveStateEventId", ctx.request_id.clone())]).await?;

    let ok = outcome.status != "disabled";
    let name = if save_ctx.save_name.is_empty() { &save_ctx.save_id } else { &save_ctx.save_name };
    let message = if outcome.status == "snapshot_missing" {
        format!("No ST checkpoint for {name}; continuing.")
    } else {
        let cp_name = if outcome.checkpoint_save_name.is_empty() { name } else { &outcome.checkpoint_save_name };
        format!("Save sync {} for {cp_name}.", outcome.status)
    };
    write_save_state_ack(
        root,
        &ctx.request_id,
        ok,
        &message,
        &[
            format!("save_sync_event={event}"),
            format!("save_sync_status={}", outcome.status),
            format!("checkpoint_id={}", outcome.checkpoint_id),
        ],
    )?;
    archive_event(root, path, "");
    info!("save-sync native {} {}: {}", event, save_ctx.save_id, outcome.status);
    Ok(())
}

/// `parseNativeSaveStateEventText`: 5 fixed lines.
struct ParsedEvent {
    request_id: String,
    event: String,
    save_id: String,
    save_name: String,
    save_file: String,
    save_fingerprint: String,
}

fn parse_save_state_event(path: &Path, text: &str) -> ParsedEvent {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let lines: Vec<String> = split_crlf(text).iter().map(|s| sanitize_bridge_line(s)).collect();
    let line = |i: usize| lines.get(i).cloned().unwrap_or_default();
    let event_id = {
        let l = line(0);
        if l.is_empty() { file_stem(path) } else { l }
    };
    let save_file = line(2);
    let save_name = {
        let l = line(3);
        if l.is_empty() {
            Path::new(&save_file).file_name().and_then(|s| s.to_str()).unwrap_or("").to_string()
        } else {
            l
        }
    };
    let save_id = first_non_empty([save_file.clone(), save_name.clone(), event_id.clone()]);
    ParsedEvent {
        request_id: event_id,
        event: line(1),
        save_id,
        save_name: first_non_empty([save_name, save_file.clone()]),
        save_file,
        save_fingerprint: line(4),
    }
}

fn write_save_state_ack(
    root: &Path,
    event_id: &str,
    ok: bool,
    message: &str,
    extra: &[String],
) -> anyhow::Result<()> {
    let dir = root.join("control").join("acks");
    std::fs::create_dir_all(&dir)?;
    let mut lines = vec![event_id.to_string(), if ok { "ok".into() } else { "error".into() }, message.to_string()];
    lines.extend(extra.iter().cloned());
    let body = format!("{}\r\n", lines.iter().map(|l| sanitize_bridge_line(l)).collect::<Vec<_>>().join("\r\n"));
    std::fs::write(dir.join(format!("{}.txt", safe_file_id(event_id))), body)
        .with_context(|| "writing save-state ack")?;
    Ok(())
}

fn archive_event(root: &Path, path: &Path, suffix: &str) {
    let dir = root.join("processed").join("save-state-events");
    let _ = std::fs::create_dir_all(&dir);
    let ext = path.extension().and_then(|e| e.to_str()).map(|e| format!(".{e}")).unwrap_or_else(|| ".txt".into());
    let name = format!("{}-{}{suffix}{ext}", now_epoch_millis(), safe_file_id(&file_stem(path)));
    if std::fs::rename(path, dir.join(&name)).is_err() {
        let _ = std::fs::remove_file(path);
    }
}

fn event_dir(root: &Path) -> PathBuf {
    root.join("control").join("events")
}

fn file_stem(path: &Path) -> String {
    path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string()
}

fn value_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn first_non_empty<const N: usize>(values: [String; N]) -> String {
    values.into_iter().find(|s| !s.is_empty()).unwrap_or_default()
}
