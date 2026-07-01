//! Per-request trace parsing for the **Tracing** settings page.
//!
//! The FNV helper writes one JSONL trace file per game request to
//! `<nativeBridgeRoot>/traces/req_<id>.jsonl`. Each line is one stage event:
//!
//! ```json
//! { "request_id": "req_6428890_12", "stage": "request_file_written",
//!   "at": "2026-06-26T12:31:13.064Z", "elapsed_ms": 16.0, ...stage fields }
//! ```
//!
//! `elapsed_ms` is milliseconds since the request started (the canonical offset
//! used to lay out the waterfall). Some sub-process events instead carry
//! `helper_elapsed_ms` (relative to the *helper*, not the request) — those are
//! kept as a field but never used as the timeline offset; such a stage inherits
//! the previous stage's offset so it still slots into the waterfall in order.
//!
//! This module is pure (no I/O beyond what the caller hands in as a string) and
//! depends only on `serde_json`, so it lives in `chasm-core` and is unit
//! tested against a small synthetic JSONL.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

/// A small fixed width (ms) given to the LAST stage, which has no following
/// stage to measure a duration against. Also the floor for any zero/negative
/// computed duration so every bar stays visible in the waterfall.
pub const TRACE_TAIL_WIDTH_MS: f64 = 1.0;

/// One parsed stage event, in trace order.
#[derive(Debug, Clone, Serialize)]
pub struct TraceStage {
    /// Zero-based index in trace order (stable id for the UI row).
    pub index: usize,
    /// Stage name (the `stage` field), e.g. `speech_recognition_done`.
    pub name: String,
    /// The `at` timestamp string, verbatim (ISO-8601), if present.
    pub at: String,
    /// Milliseconds since request start (the `elapsed_ms` field). When a stage
    /// only had `helper_elapsed_ms` (helper-relative, not comparable), this is
    /// the previous stage's offset so ordering is preserved.
    pub elapsed_ms: f64,
    /// Computed bar width: the next stage's `elapsed_ms` minus this one's,
    /// floored at [`TRACE_TAIL_WIDTH_MS`]. The last stage gets the tail width.
    pub duration_ms: f64,
    /// Coarse group derived from the stage name (drives the waterfall color).
    pub group: String,
    /// `true` when the event looks like an error/failure (highlight band).
    pub is_error: bool,
    /// All other fields on the line (everything except `request_id`/`stage`/
    /// `at`/`elapsed_ms`), rendered as `(key, value)` strings in the detail
    /// drawer. Sorted by key for stable output.
    pub fields: Vec<(String, String)>,
}

/// The fully parsed trace: ordered stages plus request-level totals.
#[derive(Debug, Clone, Serialize)]
pub struct TraceSpans {
    pub request_id: String,
    /// First stage's `at` timestamp (request start), if any.
    pub started_at: String,
    /// Largest `elapsed_ms` seen + the last stage's width — the wall span the
    /// waterfall is scaled against. Always > 0 (so divisions are safe).
    pub total_ms: f64,
    pub stage_count: usize,
    pub stages: Vec<TraceStage>,
}

/// A single labeled metric on the summary panel (label, value, optional unit).
#[derive(Debug, Clone, Serialize)]
pub struct TraceMetric {
    pub label: String,
    pub value: String,
    /// `true` for the headline metrics (LLM tokens/sec, total) so the UI can
    /// emphasize them.
    pub primary: bool,
}

impl TraceMetric {
    fn new(label: &str, value: impl Into<String>) -> Self {
        Self {
            label: label.to_string(),
            value: value.into(),
            primary: false,
        }
    }
    fn primary(label: &str, value: impl Into<String>) -> Self {
        Self {
            label: label.to_string(),
            value: value.into(),
            primary: true,
        }
    }
}

/// Derived metrics surfaced above the waterfall.
#[derive(Debug, Clone, Serialize, Default)]
pub struct TraceSummary {
    pub metrics: Vec<TraceMetric>,
}

/// LLM generation timing captured from llama.cpp's `usage`/`timings` and
/// correlated to a request by its trace id. Folded into the summary when the
/// `/traces/:id` detail is built. Mirrors the llama.cpp OpenAI-compat extras.
#[derive(Debug, Clone, Serialize, Default)]
pub struct LlmMetrics {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    /// Tokens/sec for generation (`timings.predicted_per_second`).
    pub predicted_per_second: Option<f64>,
    /// Tokens/sec for prompt eval (`timings.prompt_per_second`).
    pub prompt_per_second: Option<f64>,
    pub predicted_ms: Option<f64>,
    pub prompt_ms: Option<f64>,
}

impl LlmMetrics {
    /// Parses the llama.cpp `/v1/chat/completions` response body's `usage` +
    /// `timings` blocks. Returns `None` when neither carries any field we track,
    /// so callers only store/attach a meaningful capture.
    pub fn from_completion_response(body: &Value) -> Option<Self> {
        let usage = body.get("usage");
        let timings = body.get("timings");
        let u64f = |v: Option<&Value>, key: &str| -> Option<u64> {
            v.and_then(|o| o.get(key)).and_then(Value::as_u64)
        };
        let f64f = |v: Option<&Value>, key: &str| -> Option<f64> {
            v.and_then(|o| o.get(key)).and_then(Value::as_f64)
        };
        let metrics = LlmMetrics {
            prompt_tokens: u64f(usage, "prompt_tokens"),
            completion_tokens: u64f(usage, "completion_tokens"),
            predicted_per_second: f64f(timings, "predicted_per_second"),
            prompt_per_second: f64f(timings, "prompt_per_second"),
            predicted_ms: f64f(timings, "predicted_ms"),
            prompt_ms: f64f(timings, "prompt_ms"),
        };
        if metrics.is_empty() {
            None
        } else {
            Some(metrics)
        }
    }

    /// `true` when nothing useful was captured.
    pub fn is_empty(&self) -> bool {
        self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.predicted_per_second.is_none()
            && self.prompt_per_second.is_none()
            && self.predicted_ms.is_none()
            && self.prompt_ms.is_none()
    }
}

/// A lightweight listing entry for `GET /traces` (no per-stage detail).
#[derive(Debug, Clone, Serialize)]
pub struct TraceListEntry {
    pub request_id: String,
    pub started_at: String,
    pub total_ms: f64,
    pub stage_count: usize,
}

/// Fields that are part of the envelope and therefore NOT shown in the per-stage
/// detail drawer (they have dedicated columns).
const ENVELOPE_FIELDS: &[&str] = &["request_id", "stage", "at", "elapsed_ms"];

/// Coarse stage groups, matched by name prefix/substring (first match wins).
/// Each maps to a CSS class suffix used by the waterfall colors.
const STAGE_GROUPS: &[(&str, &str)] = &[
    ("speech_recognition", "stt"),
    ("headless_generate", "llm"),
    ("live_chat", "llm"),
    ("tts_", "tts"),
    ("voice_request", "tts"),
    ("audio_chunk", "audio"),
    ("audio_playback", "audio"),
    ("audio_queue", "audio"),
    ("directsound", "audio"),
    ("speech_animation", "anim"),
    ("speech_face", "anim"),
    ("speech_first_weights", "anim"),
    ("conversation_hold", "hold"),
    ("helper_http", "http"),
    ("helper_stream", "http"),
    ("helper_", "helper"),
    ("request_file", "request"),
    ("final_response", "request"),
    ("final_audio", "audio"),
    ("response_final", "request"),
];

/// Derives the coarse group for a stage name (drives the bar color).
fn stage_group(name: &str) -> String {
    for (needle, group) in STAGE_GROUPS {
        if name.starts_with(needle) || name.contains(needle) {
            return (*group).to_string();
        }
    }
    "other".to_string()
}

/// Heuristic error/failure detection for a stage event: a falsy `call_ok`/`ok`/
/// `response_ok`, a non-2xx `status`, or an error-ish name.
fn stage_is_error(name: &str, obj: &serde_json::Map<String, Value>) -> bool {
    let lname = name.to_lowercase();
    if lname.contains("error") || lname.contains("fail") || lname.contains("timeout") {
        return true;
    }
    for key in ["call_ok", "ok", "response_ok"] {
        if let Some(flag) = obj.get(key) {
            // Stored as bool or as a 1/0 number/string in these traces.
            let truthy = match flag {
                Value::Bool(b) => *b,
                Value::Number(n) => n.as_f64().map(|v| v != 0.0).unwrap_or(true),
                Value::String(s) => !(s == "0" || s.eq_ignore_ascii_case("false") || s.is_empty()),
                _ => true,
            };
            if !truthy {
                return true;
            }
        }
    }
    if let Some(status) = obj.get("status").and_then(Value::as_u64) {
        if !(200..400).contains(&status) {
            return true;
        }
    }
    false
}

/// Renders a JSON value to a compact display string for the detail drawer.
fn value_to_display(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        Value::Number(n) => {
            // The helper writes many integers as `16.000`; show them cleanly.
            if let Some(f) = n.as_f64() {
                if f.fract() == 0.0 && f.abs() < 1e15 {
                    return format!("{}", f as i64);
                }
                // Trim to a few decimals for fractional values.
                let s = format!("{f:.3}");
                s.trim_end_matches('0').trim_end_matches('.').to_string()
            } else {
                n.to_string()
            }
        }
        other => other.to_string(),
    }
}

/// Parses a `req_*.jsonl` trace file body into an ordered [`TraceSpans`].
///
/// - Lines that are blank or not a JSON object are skipped (robust to partial
///   writes / trailing newlines).
/// - The timeline offset is `elapsed_ms`; a stage with only `helper_elapsed_ms`
///   inherits the previous stage's offset (helper-relative ms are not on the
///   request clock).
/// - Each stage's `duration_ms` is the next stage's offset minus its own,
///   floored at [`TRACE_TAIL_WIDTH_MS`]; the last stage gets the tail width.
/// - `total_ms` is the max offset plus the last stage's width (always > 0).
pub fn parse_trace_jsonl(request_id: &str, body: &str) -> TraceSpans {
    // First pass: decode every valid object line, capturing name/at/offset/fields.
    struct Raw {
        name: String,
        at: String,
        offset: f64,
        group: String,
        is_error: bool,
        fields: Vec<(String, String)>,
    }
    let mut raws: Vec<Raw> = Vec::new();
    let mut last_offset = 0.0_f64;
    let mut resolved_request_id = request_id.to_string();

    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        if resolved_request_id.is_empty() {
            if let Some(rid) = obj.get("request_id").and_then(Value::as_str) {
                resolved_request_id = rid.to_string();
            }
        }
        let name = obj
            .get("stage")
            .and_then(Value::as_str)
            .unwrap_or("(unnamed)")
            .to_string();
        let at = obj
            .get("at")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        // Timeline offset: prefer the request-clock `elapsed_ms`. A
        // helper-relative-only event inherits the previous offset.
        let offset = obj
            .get("elapsed_ms")
            .and_then(Value::as_f64)
            .unwrap_or(last_offset)
            .max(0.0);
        last_offset = offset;

        let is_error = stage_is_error(&name, &obj);
        let group = stage_group(&name);
        let mut fields: Vec<(String, String)> = obj
            .iter()
            .filter(|(key, _)| !ENVELOPE_FIELDS.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value_to_display(value)))
            .collect();
        fields.sort_by(|a, b| a.0.cmp(&b.0));

        raws.push(Raw {
            name,
            at,
            offset,
            group,
            is_error,
            fields,
        });
    }

    // Second pass: compute per-stage durations from successive offsets.
    let max_offset = raws.iter().map(|r| r.offset).fold(0.0_f64, f64::max);
    let total_ms = (max_offset + TRACE_TAIL_WIDTH_MS).max(TRACE_TAIL_WIDTH_MS);
    let started_at = raws.first().map(|r| r.at.clone()).unwrap_or_default();
    let count = raws.len();

    let stages = raws
        .iter()
        .enumerate()
        .map(|(index, raw)| {
            let duration_ms = if index + 1 < count {
                (raws[index + 1].offset - raw.offset).max(TRACE_TAIL_WIDTH_MS)
            } else {
                TRACE_TAIL_WIDTH_MS
            };
            TraceStage {
                index,
                name: raw.name.clone(),
                at: raw.at.clone(),
                elapsed_ms: raw.offset,
                duration_ms,
                group: raw.group.clone(),
                is_error: raw.is_error,
                fields: raw.fields.clone(),
            }
        })
        .collect();

    TraceSpans {
        request_id: resolved_request_id,
        started_at,
        total_ms,
        stage_count: count,
        stages,
    }
}

/// Finds the first stage whose name matches `name` exactly.
fn find_stage<'a>(spans: &'a TraceSpans, name: &str) -> Option<&'a TraceStage> {
    spans.stages.iter().find(|stage| stage.name == name)
}

/// Reads a numeric field off a stage (helper writes numbers as floats/strings).
fn stage_num(stage: &TraceStage, key: &str) -> Option<f64> {
    stage
        .fields
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| v.parse::<f64>().ok())
}

/// Reads a string field off a stage.
fn stage_str<'a>(stage: &'a TraceStage, key: &str) -> Option<&'a str> {
    stage
        .fields
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// Formats a millisecond duration for display (`812 ms` or `2.31 s`).
pub fn format_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2} s", ms / 1000.0)
    } else {
        format!("{:.0} ms", ms)
    }
}

/// Builds the derived summary metrics from the parsed stages plus an optional
/// LLM-metrics capture (tokens/sec etc.). Pulls the key numbers the user asked
/// for out of the rich stage fields: STT duration/bytes/transcript length, TTS
/// timing, the LLM round-trip, retrieval/book activation, and speaker selection.
pub fn summarize_trace(spans: &TraceSpans, llm: Option<&LlmMetrics>) -> TraceSummary {
    let mut metrics: Vec<TraceMetric> = Vec::new();

    // --- Headline: total + LLM tokens/sec ----------------------------------
    metrics.push(TraceMetric::primary("Total", format_ms(spans.total_ms)));
    metrics.push(TraceMetric::new("Stages", spans.stage_count.to_string()));

    if let Some(llm) = llm {
        if let Some(tps) = llm.predicted_per_second {
            metrics.push(TraceMetric::primary(
                "LLM tokens/sec",
                format!("{tps:.1} tok/s"),
            ));
        }
        if let Some(pt) = llm.prompt_tokens {
            metrics.push(TraceMetric::new("Prompt tokens", pt.to_string()));
        }
        if let Some(ct) = llm.completion_tokens {
            metrics.push(TraceMetric::new("Completion tokens", ct.to_string()));
        }
        if let Some(pps) = llm.prompt_per_second {
            metrics.push(TraceMetric::new("Prompt eval", format!("{pps:.1} tok/s")));
        }
        if let Some(pms) = llm.predicted_ms {
            metrics.push(TraceMetric::new("Generation time", format_ms(pms)));
        }
    }

    // --- LLM round-trip (helper-observed) ----------------------------------
    // The `headless_generate_done` / `live_chat_*_done` markers + the matching
    // HTTP response carry the helper-side latency of the generate call.
    if let Some(done) = find_stage(spans, "headless_generate_done") {
        if let Some(streamed) = stage_str(done, "streamed") {
            metrics.push(TraceMetric::new("LLM streamed", streamed));
        }
    }

    // --- STT ---------------------------------------------------------------
    if let Some(done) = find_stage(spans, "speech_recognition_done") {
        if let Some(len) = stage_num(done, "text_length") {
            metrics.push(TraceMetric::new(
                "Transcript length",
                format!("{len:.0} chars"),
            ));
        }
    }
    if let Some(loaded) = find_stage(spans, "speech_recognition_audio_loaded") {
        if let Some(bytes) = stage_num(loaded, "audio_bytes") {
            metrics.push(TraceMetric::new("STT audio", format_bytes(bytes)));
        }
    }
    // STT call latency: the `/speech/recognize` HTTP response's duration_ms.
    if let Some(dur) = recognize_duration_ms(spans) {
        metrics.push(TraceMetric::new("STT request", format_ms(dur)));
    }

    // --- TTS ---------------------------------------------------------------
    if let Some(start) = find_stage(spans, "tts_stream_start") {
        if let Some(first) = find_stage(spans, "tts_first_audio_chunk_received") {
            let ttfa = (first.elapsed_ms - start.elapsed_ms).max(0.0);
            metrics.push(TraceMetric::new("TTS first audio", format_ms(ttfa)));
        }
    }
    let chunk_count = spans
        .stages
        .iter()
        .filter(|s| s.name == "tts_audio_file_written")
        .count();
    if chunk_count > 0 {
        metrics.push(TraceMetric::new(
            "TTS audio chunks",
            chunk_count.to_string(),
        ));
    }

    // --- Retrieval / books activation --------------------------------------
    // The request-written stage lists how many action/quest books were in scope.
    if let Some(req) = find_stage(spans, "request_file_written") {
        if let Some(n) = stage_num(req, "action_books") {
            if n > 0.0 {
                metrics.push(TraceMetric::new("Action books", format!("{n:.0}")));
            }
        }
        if let Some(n) = stage_num(req, "quest_books") {
            if n > 0.0 {
                metrics.push(TraceMetric::new("Quest books", format!("{n:.0}")));
            }
        }
    }

    // --- Speaker selection -------------------------------------------------
    // The presence/generate stages record the chosen speaker + selection mode.
    let speaker = spans
        .stages
        .iter()
        .find_map(|s| stage_str(s, "speaker_name").filter(|v| !v.is_empty()))
        .or_else(|| {
            spans
                .stages
                .iter()
                .find_map(|s| stage_str(s, "npc_name").filter(|v| !v.is_empty()))
        });
    if let Some(speaker) = speaker {
        metrics.push(TraceMetric::new("Speaker", speaker));
    }
    let mode = spans
        .stages
        .iter()
        .find_map(|s| stage_str(s, "speaker_selection_mode").filter(|v| !v.is_empty()));
    if let Some(mode) = mode {
        metrics.push(TraceMetric::new("Selection mode", mode));
    }
    let count = spans
        .stages
        .iter()
        .find_map(|s| stage_num(s, "speaker_count"));
    if let Some(count) = count {
        metrics.push(TraceMetric::new("Speakers", format!("{count:.0}")));
    }

    // --- Location ----------------------------------------------------------
    if let Some(req) = find_stage(spans, "request_file_written") {
        let major = stage_str(req, "location_major").unwrap_or("");
        let minor = stage_str(req, "location_minor").unwrap_or("");
        let loc = match (major.is_empty(), minor.is_empty()) {
            (false, false) => format!("{major} / {minor}"),
            (false, true) => major.to_string(),
            (true, false) => minor.to_string(),
            _ => String::new(),
        };
        if !loc.is_empty() {
            metrics.push(TraceMetric::new("Location", loc));
        }
    }

    TraceSummary { metrics }
}

/// The `/speech/recognize` HTTP response's `duration_ms`, found by walking the
/// `helper_http_*` stages and matching the recognize endpoint.
fn recognize_duration_ms(spans: &TraceSpans) -> Option<f64> {
    spans
        .stages
        .iter()
        .filter(|s| s.name == "helper_http_response_headers")
        .find(|s| stage_str(s, "endpoint") == Some("/speech/recognize"))
        .and_then(|s| stage_num(s, "duration_ms"))
}

/// Formats a byte count for display (`28.7 KB`).
pub fn format_bytes(bytes: f64) -> String {
    if bytes >= 1_048_576.0 {
        format!("{:.1} MB", bytes / 1_048_576.0)
    } else if bytes >= 1024.0 {
        format!("{:.1} KB", bytes / 1024.0)
    } else {
        format!("{bytes:.0} B")
    }
}

/// A bar geometry pre-computed for the waterfall template: left offset and width
/// as percentages of `total_ms`, plus a label. Built per stage so the template
/// stays declarative (no arithmetic in Askama).
#[derive(Debug, Clone, Serialize)]
pub struct WaterfallRow {
    pub index: usize,
    pub name: String,
    pub group: String,
    pub is_error: bool,
    /// Left offset as a percentage string, e.g. `"12.5"`.
    pub left_pct: f64,
    /// Width as a percentage string (min 0.4 so a 1ms bar is still visible).
    pub width_pct: f64,
    pub offset_label: String,
    pub duration_label: String,
    pub fields: Vec<(String, String)>,
    /// `true` when the bar sits in the right half, so the label renders to the
    /// LEFT of the bar to stay on-screen (Chrome-DevTools behaviour).
    pub label_left: bool,
}

/// Minimum rendered bar width (%) so a sub-millisecond stage stays clickable.
const MIN_BAR_PCT: f64 = 0.4;

/// Builds the waterfall rows (bar geometry) from parsed spans.
pub fn waterfall_rows(spans: &TraceSpans) -> Vec<WaterfallRow> {
    let total = spans.total_ms.max(TRACE_TAIL_WIDTH_MS);
    spans
        .stages
        .iter()
        .map(|stage| {
            // Cap the left offset so there is always room for a minimum-width
            // bar; then size the bar to fit the remaining space. This keeps
            // `left + width <= 100` even for the final, right-most stage.
            let left_pct = (stage.elapsed_ms / total * 100.0).clamp(0.0, 100.0 - MIN_BAR_PCT);
            let remaining = (100.0 - left_pct).max(MIN_BAR_PCT);
            let width_pct = ((stage.duration_ms / total) * 100.0)
                .max(MIN_BAR_PCT)
                .min(remaining);
            WaterfallRow {
                index: stage.index,
                name: stage.name.clone(),
                group: stage.group.clone(),
                is_error: stage.is_error,
                left_pct,
                width_pct,
                offset_label: format_ms(stage.elapsed_ms),
                duration_label: format_ms(stage.duration_ms),
                fields: stage.fields.clone(),
                label_left: left_pct > 55.0,
            }
        })
        .collect()
}

/// Builds the time-axis gridline labels (0/25/50/75/100% of `total_ms`).
pub fn axis_ticks(total_ms: f64) -> Vec<(f64, String)> {
    let total = total_ms.max(TRACE_TAIL_WIDTH_MS);
    [0.0, 0.25, 0.5, 0.75, 1.0]
        .iter()
        .map(|frac| (frac * 100.0, format_ms(total * frac)))
        .collect()
}

/// Deduplicated count of stage groups present (for the legend).
pub fn group_legend(spans: &TraceSpans) -> Vec<String> {
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    for stage in &spans.stages {
        seen.insert(stage.group.clone(), ());
    }
    seen.into_keys().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A small synthetic trace: 4 stages, the third helper-relative-only.
    const SAMPLE: &str = r#"
{"request_id":"req_test_1","stage":"request_file_written","at":"2026-06-26T12:31:13.064Z","elapsed_ms":0.0,"location_major":"Goodsprings","location_minor":"Saloon","action_books":2.0,"quest_books":1.0}
{"request_id":"req_test_1","stage":"speech_recognition_audio_loaded","at":"2026-06-26T12:31:13.174Z","elapsed_ms":100.0,"audio_bytes":29422}
{"request_id":"req_test_1","stage":"helper_http_response_headers","source":"helper","at":"2026-06-26T12:31:13.750Z","helper_elapsed_ms":643.2,"endpoint":"/speech/recognize","status":200,"duration_ms":561.9}
{"request_id":"req_test_1","stage":"speech_recognition_done","at":"2026-06-26T12:31:13.760Z","elapsed_ms":700.0,"text_length":42.0,"speaker_name":"Sunny Smiles","speaker_selection_mode":"fallback"}
"#;

    #[test]
    fn parses_offsets_and_durations() {
        let spans = parse_trace_jsonl("req_test_1", SAMPLE);
        assert_eq!(spans.request_id, "req_test_1");
        assert_eq!(spans.stage_count, 4);
        assert_eq!(spans.started_at, "2026-06-26T12:31:13.064Z");

        // Offsets: the helper-relative-only line (#2) inherits #1's 100ms offset.
        assert_eq!(spans.stages[0].elapsed_ms, 0.0);
        assert_eq!(spans.stages[1].elapsed_ms, 100.0);
        assert_eq!(spans.stages[2].elapsed_ms, 100.0);
        assert_eq!(spans.stages[3].elapsed_ms, 700.0);

        // Durations are successive-offset deltas; the inherited stage gets the
        // tail floor (next offset == its own offset → clamped up).
        assert_eq!(spans.stages[0].duration_ms, 100.0);
        assert_eq!(spans.stages[1].duration_ms, TRACE_TAIL_WIDTH_MS);
        assert_eq!(spans.stages[2].duration_ms, 600.0);
        // Last stage gets the fixed tail width.
        assert_eq!(spans.stages[3].duration_ms, TRACE_TAIL_WIDTH_MS);

        // Total = max offset + tail.
        assert_eq!(spans.total_ms, 700.0 + TRACE_TAIL_WIDTH_MS);
    }

    #[test]
    fn groups_and_errors_are_derived() {
        let spans = parse_trace_jsonl("req_test_1", SAMPLE);
        assert_eq!(spans.stages[0].group, "request");
        assert_eq!(spans.stages[1].group, "stt");
        assert_eq!(spans.stages[2].group, "http");
        assert_eq!(spans.stages[3].group, "stt");
        assert!(spans.stages.iter().all(|s| !s.is_error));

        // A non-2xx status flags an error.
        let err = parse_trace_jsonl(
            "x",
            r#"{"request_id":"x","stage":"helper_http_response_headers","elapsed_ms":1.0,"status":500}"#,
        );
        assert!(err.stages[0].is_error);

        // A falsy call_ok flags an error.
        let err2 = parse_trace_jsonl(
            "x",
            r#"{"request_id":"x","stage":"conversation_hold_look","elapsed_ms":1.0,"call_ok":false}"#,
        );
        assert!(err2.stages[0].is_error);
    }

    #[test]
    fn fields_exclude_envelope_and_are_sorted() {
        let spans = parse_trace_jsonl("req_test_1", SAMPLE);
        let first = &spans.stages[0];
        // request_id/stage/at/elapsed_ms are NOT in the detail fields.
        assert!(first
            .fields
            .iter()
            .all(|(k, _)| !["request_id", "stage", "at", "elapsed_ms"].contains(&k.as_str())));
        // Sorted by key.
        let keys: Vec<&str> = first.fields.iter().map(|(k, _)| k.as_str()).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
        // Float-integers render without a trailing `.0`.
        let action = first
            .fields
            .iter()
            .find(|(k, _)| k == "action_books")
            .unwrap();
        assert_eq!(action.1, "2");
    }

    #[test]
    fn skips_blank_and_malformed_lines() {
        let body =
            "\n  \nnot json\n{\"request_id\":\"x\",\"stage\":\"a\",\"elapsed_ms\":5.0}\n{bad}\n";
        let spans = parse_trace_jsonl("x", body);
        assert_eq!(spans.stage_count, 1);
        assert_eq!(spans.stages[0].name, "a");
    }

    #[test]
    fn empty_trace_is_safe() {
        let spans = parse_trace_jsonl("empty", "");
        assert_eq!(spans.stage_count, 0);
        assert_eq!(spans.total_ms, TRACE_TAIL_WIDTH_MS);
        assert!(spans.started_at.is_empty());
        // Waterfall + axis don't divide by zero.
        assert!(waterfall_rows(&spans).is_empty());
        assert_eq!(axis_ticks(spans.total_ms).len(), 5);
    }

    #[test]
    fn summary_extracts_key_metrics() {
        let spans = parse_trace_jsonl("req_test_1", SAMPLE);
        let llm = LlmMetrics {
            prompt_tokens: Some(1200),
            completion_tokens: Some(48),
            predicted_per_second: Some(37.5),
            prompt_per_second: Some(900.0),
            predicted_ms: Some(1280.0),
            prompt_ms: Some(1333.0),
        };
        let summary = summarize_trace(&spans, Some(&llm));
        let find = |label: &str| {
            summary
                .metrics
                .iter()
                .find(|m| m.label == label)
                .map(|m| m.value.clone())
        };
        // Headline metrics.
        assert_eq!(find("Total"), Some(format_ms(spans.total_ms)));
        assert_eq!(find("LLM tokens/sec"), Some("37.5 tok/s".to_string()));
        assert_eq!(find("Prompt tokens"), Some("1200".to_string()));
        assert_eq!(find("Completion tokens"), Some("48".to_string()));
        // STT metrics from stage fields.
        assert_eq!(find("Transcript length"), Some("42 chars".to_string()));
        assert_eq!(find("STT audio"), Some(format_bytes(29422.0)));
        assert_eq!(find("STT request"), Some(format_ms(561.9)));
        // Books + speaker.
        assert_eq!(find("Action books"), Some("2".to_string()));
        assert_eq!(find("Quest books"), Some("1".to_string()));
        assert_eq!(find("Speaker"), Some("Sunny Smiles".to_string()));
        assert_eq!(find("Selection mode"), Some("fallback".to_string()));
        assert_eq!(find("Location"), Some("Goodsprings / Saloon".to_string()));
        // The tokens/sec + total metrics are flagged primary.
        assert!(summary
            .metrics
            .iter()
            .any(|m| m.primary && m.label == "LLM tokens/sec"));
    }

    #[test]
    fn summary_without_llm_omits_token_metrics() {
        let spans = parse_trace_jsonl("req_test_1", SAMPLE);
        let summary = summarize_trace(&spans, None);
        assert!(summary.metrics.iter().all(|m| m.label != "LLM tokens/sec"));
        // Total is still present.
        assert!(summary.metrics.iter().any(|m| m.label == "Total"));
    }

    #[test]
    fn llm_metrics_parse_from_llama_response() {
        let body = json!({
            "choices": [],
            "usage": { "prompt_tokens": 1500, "completion_tokens": 64 },
            "timings": {
                "predicted_per_second": 41.2,
                "prompt_per_second": 1100.0,
                "predicted_ms": 1553.0,
                "prompt_ms": 1363.0
            }
        });
        let metrics = LlmMetrics::from_completion_response(&body).unwrap();
        assert_eq!(metrics.prompt_tokens, Some(1500));
        assert_eq!(metrics.completion_tokens, Some(64));
        assert_eq!(metrics.predicted_per_second, Some(41.2));
        assert_eq!(metrics.prompt_per_second, Some(1100.0));

        // A response with neither block yields None.
        let bare = json!({ "choices": [] });
        assert!(LlmMetrics::from_completion_response(&bare).is_none());
    }

    #[test]
    fn waterfall_geometry_is_bounded() {
        let spans = parse_trace_jsonl("req_test_1", SAMPLE);
        let rows = waterfall_rows(&spans);
        assert_eq!(rows.len(), 4);
        for row in &rows {
            assert!(row.left_pct >= 0.0 && row.left_pct <= 100.0);
            assert!(row.width_pct >= 0.0);
            assert!(row.left_pct + row.width_pct <= 100.01);
        }
        // The first row starts at 0% and the last sits to the right.
        assert_eq!(rows[0].left_pct, 0.0);
        assert!(rows[3].left_pct > rows[0].left_pct);
        // Late bars render their label on the left.
        assert!(rows[3].label_left);
    }
}
