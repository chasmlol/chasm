//! Connection status — chasm is a passive backend, so instead of launching the
//! game it just reports whether the in-game plugin is talking to it.
//!
//! The plugin writes `runtime_heartbeat.json` into the bridge root every ~100ms
//! while the game is running. We resolve that bridge root the same way the tracing
//! page resolves the traces dir (`resolve_trace_dir`'s parent — the `NVBridge` dir
//! that holds `traces/`).
//!
//! Two signals come out of that file:
//!   * its **mtime** — fresh while the game is actively running its script loop;
//!   * a **`pid`** field (newer plugins) — the game's own process id, since the
//!     plugin runs inside the game.
//!
//! Many games (FNV included) pause their script loop on focus loss, so the mtime
//! goes stale on tab-out even though the game is still very much alive. The PID
//! lets us tell "paused" (process alive, heartbeat stale) apart from "quit"
//! (process gone), so the connection only drops when the game is physically
//! closed. Freshness remains the fallback for plugins that predate the PID field.

use std::{path::PathBuf, sync::Arc, time::SystemTime};

use axum::{extract::State, Json};
use chasm_core::AppSettings;

use crate::{stack_lifecycle::Phase, trace_routes::resolve_trace_dir, AppState, WebResult};

/// How fresh the heartbeat file's mtime must be (seconds) for the plugin to count
/// as connected. The plugin rewrites it every ~100ms, so a few seconds of slack
/// tolerates scheduling jitter without latching "connected" after the game quits.
/// Shared with the stack lifecycle task so the rail and the start trigger agree.
pub(crate) const FRESH_SECS: u64 = 5;

/// The heartbeat file the in-game plugin writes into the bridge root.
const HEARTBEAT_FILE: &str = "runtime_heartbeat.json";

/// The bridge root (`…/NVBridge`) — the parent of the traces dir the helper
/// writes to. Resolved from settings exactly like the tracing page does, so the
/// two always agree on where the bridge writes.
fn bridge_root(settings: &AppSettings) -> PathBuf {
    let traces = resolve_trace_dir(settings);
    // `traces` is `<bridge_root>/traces`; its parent is the bridge root. Fall back
    // to the traces dir itself if it somehow has no parent.
    traces
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or(traces)
}

/// The full path to the plugin's heartbeat file (`<bridge_root>/runtime_heartbeat.json`).
/// Shared with the stack lifecycle task so both read the exact same signal.
pub(crate) fn heartbeat_path(settings: &AppSettings) -> PathBuf {
    bridge_root(settings).join(HEARTBEAT_FILE)
}

/// A snapshot of the plugin's heartbeat file.
pub(crate) struct Heartbeat {
    /// Seconds since the file was last written, or `None` when it's missing/unreadable
    /// (never seen). Fresh ⇒ the game is actively running its script loop right now.
    pub last_seen_secs: Option<f64>,
    /// The game's process id, as reported by the plugin (it runs inside the game).
    /// `None` for older plugins that don't emit a `pid` field — callers fall back to
    /// freshness in that case.
    pub pid: Option<u32>,
}

/// Read the heartbeat's mtime age and the PID it reports in one pass. Shared by the
/// status endpoint and the lifecycle task so they key off the exact same file.
pub(crate) fn read_heartbeat(settings: &AppSettings) -> Heartbeat {
    let path = heartbeat_path(settings);
    let last_seen_secs = std::fs::metadata(&path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|mtime| SystemTime::now().duration_since(mtime).ok())
        .map(|elapsed| elapsed.as_secs_f64());
    let pid = std::fs::read_to_string(&path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|v| v.get("pid").and_then(serde_json::Value::as_u64))
        .and_then(|p| u32::try_from(p).ok())
        .filter(|&p| p != 0);
    Heartbeat {
        last_seen_secs,
        pid,
    }
}

/// True when process `pid` is currently alive. This is what lets the connection
/// survive a paused/tabbed-out game: the heartbeat goes stale, but the game process
/// is still running, so we hold the connection until the process actually exits.
/// Best-effort — any failure reads as "not alive" so we fail safe toward teardown.
#[cfg(windows)]
pub(crate) fn pid_is_alive(pid: u32) -> bool {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let out = Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output();
    match out {
        // tasklist echoes a CSV row containing `"<pid>"` when the process exists,
        // otherwise an "INFO: No tasks…" line that won't contain it.
        Ok(o) => String::from_utf8_lossy(&o.stdout).contains(&format!("\"{pid}\"")),
        Err(_) => false,
    }
}

/// Non-Windows fallback (chasm ships on Windows; this just keeps other targets
/// compiling and is good enough for dev on Linux).
#[cfg(not(windows))]
pub(crate) fn pid_is_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

/// `GET /connection/status` — `{ connected, phase, last_seen_secs }`.
///
/// `connected` is true when the heartbeat is fresh (game actively writing) OR the
/// lifecycle is holding the stack up (`starting`/`connected`) — the latter covers a
/// paused/tabbed-out game whose heartbeat has gone stale but whose process is still
/// alive. The lifecycle task is what verifies the PID; the rail trusts its phase so
/// we never spawn a liveness check on every status poll. `last_seen_secs` is how
/// long ago the heartbeat was last written (null when never seen). `phase` is the
/// AI-stack lifecycle state (`disconnected` / `starting` / `connected` / `stopping`).
pub async fn connection_status(
    State(state): State<Arc<AppState>>,
) -> WebResult<Json<serde_json::Value>> {
    let settings = AppSettings::load(&state.config.settings_path);
    let Heartbeat {
        last_seen_secs, ..
    } = read_heartbeat(&settings);

    let fresh = last_seen_secs.is_some_and(|secs| secs <= FRESH_SECS as f64);
    let phase = state.lifecycle.phase();
    let phase_up = matches!(phase, Phase::Starting | Phase::Connected);
    let connected = fresh || phase_up;

    Ok(Json(serde_json::json!({
        "connected": connected,
        // The AI-stack lifecycle phase the rail keys its label off.
        "phase": phase.as_str(),
        // Null when the file is missing/unreadable (never seen); otherwise the
        // seconds since the last heartbeat write.
        "last_seen_secs": last_seen_secs,
    })))
}
