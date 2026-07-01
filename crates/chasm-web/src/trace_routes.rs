//! The **Tracing** settings page + its endpoints.
//!
//! Reads the per-request JSONL trace files the FNV helper writes to
//! `<nativeBridgeRoot>/traces/req_<id>.jsonl`, parses each into spans via
//! [`chasm_core::parse_trace_jsonl`], and renders a Chrome-DevTools-style
//! waterfall. Trace files are read **read-only**; nothing here writes to them.
//!
//! - `GET /traces`        — recent traces (newest first by mtime).
//! - `GET /traces/:id`     — one parsed trace (stages + totals + summary).
//! - `GET /settings/tracing` is handled by the shared settings router in
//!   `lib.rs`, which calls [`build_tracing_view`] here.
//!
//! Trace-dir discovery is generic: the `tracing.trace_dir` setting wins when set;
//! otherwise the helper config JSON's `nativeBridgeRoots[0]` + `/traces` is used,
//! falling back to the fixed bridge rendezvous dir (`default_bridge_root()/traces`).

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use axum::{
    extract::{Path as AxPath, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use chasm_core::{
    axis_ticks, group_legend, parse_trace_jsonl, summarize_trace, waterfall_rows, AppSettings,
    LlmMetrics, SettingsNavItem, TraceListEntry, TraceMetric, TraceSpans, WaterfallRow,
};

use crate::{AppState, WebResult};

/// Process-wide store of the most recent LLM generation metrics, keyed by trace
/// id. The generate path records here (correlated via the
/// `X-Chasm-Trace-Id` request header the helper sends); `/traces/:id` reads
/// it to fold tokens/sec etc. into the summary. A bounded ring keeps memory flat.
///
/// A `OnceLock` so it is shared by all handlers without threading a field
/// through `AppState` (the generate path lives in a different module).
fn llm_metrics_store() -> &'static Mutex<LlmMetricsStore> {
    static STORE: OnceLock<Mutex<LlmMetricsStore>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(LlmMetricsStore::default()))
}

/// Bounded most-recent-wins map of trace-id -> captured LLM metrics.
#[derive(Default)]
struct LlmMetricsStore {
    /// (trace_id, metrics) in insertion order; oldest evicted past the cap.
    entries: Vec<(String, LlmMetrics)>,
}

/// Max distinct trace ids we retain LLM metrics for.
const LLM_METRICS_CAP: usize = 64;

/// Records the LLM metrics captured for `trace_id` (called from the generate
/// path). A blank id is ignored. Replaces any prior capture for the same id so
/// the latest generation of a request wins.
pub fn record_llm_metrics(trace_id: &str, metrics: LlmMetrics) {
    let id = trace_id.trim();
    if id.is_empty() || metrics.is_empty() {
        return;
    }
    if let Ok(mut store) = llm_metrics_store().lock() {
        store.entries.retain(|(key, _)| key != id);
        store.entries.push((id.to_string(), metrics));
        let len = store.entries.len();
        if len > LLM_METRICS_CAP {
            store.entries.drain(0..len - LLM_METRICS_CAP);
        }
    }
}

/// The most recent LLM metrics captured for `trace_id`, if any.
fn llm_metrics_for(trace_id: &str) -> Option<LlmMetrics> {
    let store = llm_metrics_store().lock().ok()?;
    store
        .entries
        .iter()
        .rev()
        .find(|(key, _)| key == trace_id)
        .map(|(_, m)| m.clone())
}

/// Resolves the effective traces directory: the `tracing.trace_dir` override
/// when set, else the helper config's `nativeBridgeRoots[0]` + `/traces`, else
/// the known fallback path. Pure path logic — the directory may not exist.
pub fn resolve_trace_dir(settings: &AppSettings) -> PathBuf {
    let override_dir = settings.tracing.trace_dir.trim();
    if !override_dir.is_empty() {
        return PathBuf::from(override_dir);
    }
    if let Some(dir) = trace_dir_from_helper_config(&settings.launcher.helper_config) {
        return dir;
    }
    chasm_core::default_bridge_root().join("traces")
}

/// Reads `nativeBridgeRoots[0]` from the helper config JSON at `config_path`
/// (blank → the built-in default) and appends `traces`. Returns `None` when the
/// config is missing/invalid or has no usable root.
fn trace_dir_from_helper_config(config_path: &str) -> Option<PathBuf> {
    let path = config_path.trim();
    if path.is_empty() {
        return None;
    }
    let text = std::fs::read_to_string(path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&text).ok()?;
    let root = config
        .get("nativeBridgeRoots")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|v| v.as_str())
        // Also accept a singular `nativeBridgeRoot` string, just in case.
        .or_else(|| config.get("nativeBridgeRoot").and_then(|v| v.as_str()))?
        .trim();
    if root.is_empty() {
        return None;
    }
    Some(Path::new(root).join("traces"))
}

/// Lists `req_*.jsonl` trace files in `dir`, newest first by mtime, capped at
/// `limit`. Each entry is parsed only enough for the listing (request id,
/// start time, total ms, stage count). Missing dir → empty list (not an error).
fn list_traces(dir: &Path, limit: usize) -> Vec<TraceListEntry> {
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_string_lossy().to_string();
            if !(name.starts_with("req_") && name.ends_with(".jsonl")) {
                return None;
            }
            let mtime = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            Some((mtime, path))
        })
        .collect();
    // Newest first.
    files.sort_by(|a, b| b.0.cmp(&a.0));
    files.truncate(limit);

    files
        .into_iter()
        .filter_map(|(_, path)| {
            let request_id = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let body = std::fs::read_to_string(&path).ok()?;
            let spans = parse_trace_jsonl(&request_id, &body);
            Some(TraceListEntry {
                request_id: spans.request_id,
                started_at: spans.started_at,
                total_ms: spans.total_ms,
                stage_count: spans.stage_count,
            })
        })
        .collect()
}

/// Reads + parses a single trace by id from `dir`. The id is sanitized to a bare
/// `req_*` filename (no path separators) so it can't escape the traces dir.
fn read_trace(dir: &Path, request_id: &str) -> Option<TraceSpans> {
    let safe = sanitize_trace_id(request_id)?;
    let path = dir.join(format!("{safe}.jsonl"));
    let body = std::fs::read_to_string(path).ok()?;
    Some(parse_trace_jsonl(&safe, &body))
}

/// Validates a trace id is a plain file stem (alphanumerics, `_`, `-`, `.`) with
/// no path separators, so `/traces/:id` can never read outside the traces dir.
fn sanitize_trace_id(id: &str) -> Option<String> {
    let trimmed = id.trim();
    if trimmed.is_empty() {
        return None;
    }
    let ok = trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'));
    if ok && !trimmed.contains("..") {
        Some(trimmed.to_string())
    } else {
        None
    }
}

/// The recent-trace limit for the listing + nav.
const TRACE_LIST_LIMIT: usize = 50;

// ---------------------------------------------------------------------------
// JSON endpoints
// ---------------------------------------------------------------------------

/// `GET /traces` — recent traces, newest first.
pub async fn list_traces_endpoint(
    State(state): State<Arc<AppState>>,
) -> WebResult<Json<serde_json::Value>> {
    let settings = AppSettings::load(&state.config.settings_path);
    let dir = resolve_trace_dir(&settings);
    let traces = list_traces(&dir, TRACE_LIST_LIMIT);
    Ok(Json(serde_json::json!({
        "traceDir": dir.display().to_string(),
        "traces": traces,
    })))
}

/// `GET /traces/:id` — one parsed trace (stages + totals + summary).
pub async fn get_trace_endpoint(
    State(state): State<Arc<AppState>>,
    AxPath(id): AxPath<String>,
) -> Response {
    let settings = AppSettings::load(&state.config.settings_path);
    let dir = resolve_trace_dir(&settings);
    let Some(spans) = read_trace(&dir, &id) else {
        return (
            StatusCode::NOT_FOUND,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::json!({ "error": "trace not found" }).to_string(),
        )
            .into_response();
    };
    let llm = llm_metrics_for(&spans.request_id);
    let summary = summarize_trace(&spans, llm.as_ref());
    Json(serde_json::json!({
        "requestId": spans.request_id,
        "startedAt": spans.started_at,
        "totalMs": spans.total_ms,
        "stageCount": spans.stage_count,
        "stages": spans.stages,
        "summary": summary,
        "llm": llm,
    }))
    .into_response()
}

// ---------------------------------------------------------------------------
// Settings-page view model (rendered by the `tracing` settings branch)
// ---------------------------------------------------------------------------

/// One row in the recent-traces sidebar of the Tracing page.
#[derive(Debug, Clone, Serialize)]
pub struct TraceListRowView {
    pub request_id: String,
    pub started_at: String,
    pub total_label: String,
    pub stage_count: usize,
    pub selected: bool,
}

/// The selected trace, fully rendered for the waterfall.
#[derive(Debug, Clone, Serialize)]
pub struct TraceDetailView {
    pub request_id: String,
    pub started_at: String,
    pub total_label: String,
    pub stage_count: usize,
    pub axis_ticks: Vec<AxisTickView>,
    pub rows: Vec<WaterfallRow>,
    pub metrics: Vec<TraceMetric>,
    pub legend: Vec<LegendView>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AxisTickView {
    pub left_pct: f64,
    pub label: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LegendView {
    pub group: String,
    pub label: String,
}

/// The whole Tracing settings page view.
#[derive(Debug, Clone, Serialize)]
pub struct TracingPageView {
    pub category: String,
    pub nav: Vec<SettingsNavItem>,
    pub nav_groups: Vec<chasm_core::SettingsNavGroup>,
    pub saved: bool,
    pub settings_path: String,
    /// Effective traces dir (resolved) + the raw override for the form input.
    pub trace_dir: String,
    pub trace_dir_override: String,
    /// `true` when the resolved dir exists on disk.
    pub trace_dir_exists: bool,
    pub traces: Vec<TraceListRowView>,
    /// `None` when there are no traces (empty state).
    pub detail: Option<TraceDetailView>,
}

/// Human label for a stage group, used in the waterfall legend.
fn group_label(group: &str) -> String {
    match group {
        "stt" => "Speech-to-text",
        "llm" => "LLM / generation",
        "tts" => "Text-to-speech",
        "audio" => "Audio playback",
        "anim" => "Animation / face",
        "hold" => "Conversation hold",
        "http" => "Helper HTTP",
        "helper" => "Helper",
        "request" => "Request I/O",
        _ => "Other",
    }
    .to_string()
}

/// Builds the Tracing settings page view: resolves the traces dir, lists recent
/// traces, and renders the selected (or newest) trace into a waterfall + summary.
/// `nav` is built by the caller (so the shared nav list stays in one place).
pub fn build_tracing_view(
    settings: &AppSettings,
    nav: Vec<SettingsNavItem>,
    saved: bool,
    settings_path: String,
    selected_id: Option<&str>,
) -> TracingPageView {
    let dir = resolve_trace_dir(settings);
    let trace_dir_exists = dir.is_dir();
    let entries = list_traces(&dir, TRACE_LIST_LIMIT);

    // Auto-select: the requested id (if present in the list), else the newest.
    let active_id = selected_id
        .filter(|id| entries.iter().any(|e| e.request_id == *id))
        .map(|id| id.to_string())
        .or_else(|| entries.first().map(|e| e.request_id.clone()));

    let traces = entries
        .iter()
        .map(|entry| TraceListRowView {
            request_id: entry.request_id.clone(),
            started_at: entry.started_at.clone(),
            total_label: chasm_core::format_ms(entry.total_ms),
            stage_count: entry.stage_count,
            selected: Some(&entry.request_id) == active_id.as_ref(),
        })
        .collect();

    let detail = active_id
        .as_deref()
        .and_then(|id| read_trace(&dir, id))
        .map(|spans| build_detail(&spans));

    TracingPageView {
        category: "tracing".to_string(),
        nav_groups: chasm_core::settings_nav_groups("tracing"),
        nav,
        saved,
        settings_path,
        trace_dir: dir.display().to_string(),
        trace_dir_override: settings.tracing.trace_dir.clone(),
        trace_dir_exists,
        traces,
        detail,
    }
}

/// Renders parsed spans into the waterfall detail view (geometry + summary).
fn build_detail(spans: &TraceSpans) -> TraceDetailView {
    let llm = llm_metrics_for(&spans.request_id);
    let summary = summarize_trace(spans, llm.as_ref());
    let rows: Vec<WaterfallRow> = waterfall_rows(spans);
    let axis_ticks = axis_ticks(spans.total_ms)
        .into_iter()
        .map(|(left_pct, label)| AxisTickView { left_pct, label })
        .collect();
    let legend = group_legend(spans)
        .into_iter()
        .map(|group| LegendView {
            label: group_label(&group),
            group,
        })
        .collect();
    TraceDetailView {
        request_id: spans.request_id.clone(),
        started_at: spans.started_at.clone(),
        total_label: chasm_core::format_ms(spans.total_ms),
        stage_count: spans.stage_count,
        axis_ticks,
        rows,
        metrics: summary.metrics,
        legend,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_dir_override_wins() {
        let mut settings = AppSettings::default();
        settings.tracing.trace_dir = r"D:\custom\traces".to_string();
        assert_eq!(
            resolve_trace_dir(&settings),
            PathBuf::from(r"D:\custom\traces")
        );
    }

    #[test]
    fn sanitize_blocks_path_traversal() {
        assert!(sanitize_trace_id("req_123_4").is_some());
        assert!(sanitize_trace_id("req-abc.def").is_some());
        assert!(sanitize_trace_id("../etc/passwd").is_none());
        assert!(sanitize_trace_id("a/b").is_none());
        assert!(sanitize_trace_id("a\\b").is_none());
        assert!(sanitize_trace_id("").is_none());
        assert!(sanitize_trace_id("req..2").is_none());
    }

    #[test]
    fn helper_config_discovery_appends_traces() {
        // Write a tiny helper config to a temp dir and confirm discovery.
        let tmp = std::env::temp_dir().join(format!("sb-trace-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg = tmp.join("nvbridge.config.json");
        std::fs::write(&cfg, r#"{"nativeBridgeRoots":["C:\\Games\\NVBridge"]}"#).unwrap();
        let got = trace_dir_from_helper_config(&cfg.display().to_string()).unwrap();
        assert_eq!(got, Path::new(r"C:\Games\NVBridge").join("traces"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn list_and_read_round_trip_from_temp_dir() {
        let tmp = std::env::temp_dir().join(format!("sb-trace-list-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(
            tmp.join("req_unit_1.jsonl"),
            "{\"request_id\":\"req_unit_1\",\"stage\":\"a\",\"elapsed_ms\":0.0}\n{\"request_id\":\"req_unit_1\",\"stage\":\"b\",\"elapsed_ms\":50.0}\n",
        )
        .unwrap();
        // A non-trace file is ignored.
        std::fs::write(tmp.join("notes.txt"), "ignore me").unwrap();

        let listed = list_traces(&tmp, 10);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].request_id, "req_unit_1");
        assert_eq!(listed[0].stage_count, 2);

        let spans = read_trace(&tmp, "req_unit_1").unwrap();
        assert_eq!(spans.stage_count, 2);
        // Traversal id reads nothing.
        assert!(read_trace(&tmp, "../secret").is_none());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn llm_metrics_store_round_trips_and_is_bounded() {
        record_llm_metrics(
            "req_store_1",
            LlmMetrics {
                predicted_per_second: Some(33.0),
                ..Default::default()
            },
        );
        let got = llm_metrics_for("req_store_1").unwrap();
        assert_eq!(got.predicted_per_second, Some(33.0));
        // Empty metrics / blank id are no-ops.
        record_llm_metrics("", LlmMetrics::default());
        record_llm_metrics("req_x", LlmMetrics::default());
        assert!(llm_metrics_for("req_x").is_none());
    }
}
