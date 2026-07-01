//! UI bridge domain — the bridge/connection *configuration* + live status.
//!
//! The React Bridge screen edits the plain launcher fields that describe how the
//! FNV helper is wired (helper config / script / node / cwd) plus the tracing
//! `trace_dir` override, and surfaces the live game connection status. It reuses
//! the SAME `AppSettings` read/save path the Interface screen uses
//! (`AppSettings::load` / `save`), so a change persists to the one settings file
//! both UIs read.
//!
//! IMPORTANT: this surfaces bridge *configuration* only. It MUST NOT drive the
//! live transport or AI-stack lifecycle — those are owned by the backend and
//! off-limits to the UI. The connection status is the read-only
//! `/connection/status` projection (same data the ConnectionPill shows). Stays
//! under `/api/ui/v1`.

use std::{collections::HashMap, sync::Arc};

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use chasm_core::AppSettings;

use crate::{
    connection::{read_heartbeat, FRESH_SECS},
    stack_lifecycle::Phase,
    AppState, WebResult,
};

/// The editable bridge config fields + the live connection status.
#[derive(Serialize)]
pub(crate) struct UiBridgeView {
    pub settings_path: String,
    pub config: BridgeConfig,
    pub connection: BridgeConnection,
}

/// The plain bridge/launcher config the screen edits. Every field is an
/// override; blank = "auto-detect / built-in default" (re-resolved on read).
#[derive(Serialize, Deserialize, Default)]
pub(crate) struct BridgeConfig {
    /// Path to the helper config JSON (`nvbridge.config.json`). Blank = built-in
    /// default. Also drives traces-dir discovery.
    pub helper_config: String,
    /// Path to the helper script `nvbridge-helper.mjs`. Blank = built-in default.
    pub helper_script: String,
    /// `node.exe` path for the (legacy) Node helper. Blank = default / PATH.
    pub helper_node: String,
    /// Working directory for the helper. Blank = the helper script's folder.
    pub helper_cwd: String,
    /// Override for the traces directory (`tracing.trace_dir`). Blank =
    /// auto-discover from the helper config.
    pub trace_dir: String,
}

/// The read-only connection projection (mirrors `GET /connection/status`).
#[derive(Serialize)]
pub(crate) struct BridgeConnection {
    pub connected: bool,
    pub phase: String,
    pub last_seen_secs: Option<f64>,
}

fn config_from_settings(settings: &AppSettings) -> BridgeConfig {
    BridgeConfig {
        helper_config: settings.launcher.helper_config.clone(),
        helper_script: settings.launcher.helper_script.clone(),
        helper_node: settings.launcher.helper_node.clone(),
        helper_cwd: settings.launcher.helper_cwd.clone(),
        trace_dir: settings.tracing.trace_dir.clone(),
    }
}

fn build_bridge_view(state: &AppState, settings: &AppSettings) -> UiBridgeView {
    UiBridgeView {
        settings_path: state.config.settings_path.display().to_string(),
        config: config_from_settings(settings),
        connection: connection_projection(state, settings),
    }
}

/// Builds the connection projection from the SAME inputs `/connection/status`
/// uses: the heartbeat freshness OR the lifecycle phase holding the stack up.
fn connection_projection(state: &AppState, settings: &AppSettings) -> BridgeConnection {
    let last_seen_secs = read_heartbeat(settings).last_seen_secs;
    let fresh = last_seen_secs.is_some_and(|secs| secs <= FRESH_SECS as f64);
    let phase = state.lifecycle.phase();
    let phase_up = matches!(phase, Phase::Starting | Phase::Connected);
    BridgeConnection {
        connected: fresh || phase_up,
        phase: phase.as_str().to_string(),
        last_seen_secs,
    }
}

/// `GET /api/ui/v1/settings/bridge` — the bridge config + connection. Read-only.
pub(crate) async fn get_bridge(State(state): State<Arc<AppState>>) -> Json<UiBridgeView> {
    let settings = AppSettings::load(&state.config.settings_path);
    Json(build_bridge_view(&state, &settings))
}

/// `POST /api/ui/v1/settings/bridge/save` — persist the edited bridge config,
/// then return the fresh view. Reuses `AppSettings::load`/`save` (the same path
/// the Interface screen uses); every field is trimmed so a whitespace-only entry
/// resets to auto-detect. Does NOT touch the transport or lifecycle.
pub(crate) async fn save_bridge(
    State(state): State<Arc<AppState>>,
    Json(form): Json<HashMap<String, String>>,
) -> WebResult<Json<UiBridgeView>> {
    let mut settings = AppSettings::load(&state.config.settings_path);
    if let Some(value) = form.get("helper_config") {
        settings.launcher.helper_config = value.trim().to_string();
    }
    if let Some(value) = form.get("helper_script") {
        settings.launcher.helper_script = value.trim().to_string();
    }
    if let Some(value) = form.get("helper_node") {
        settings.launcher.helper_node = value.trim().to_string();
    }
    if let Some(value) = form.get("helper_cwd") {
        settings.launcher.helper_cwd = value.trim().to_string();
    }
    if let Some(value) = form.get("trace_dir") {
        settings.tracing.trace_dir = value.trim().to_string();
    }
    settings.save(&state.config.settings_path)?;
    Ok(Json(build_bridge_view(&state, &settings)))
}
