//! Backend-side per-turn stage traces.
//!
//! Appends one JSONL stage event per pipeline step to
//! `<bridge_root>/traces/<request_id>.jsonl` — the same file/format the C++
//! plugin's tracer uses and chasm's **Settings → Tracing** waterfall reads
//! (`chasm-core::request_trace`). Until now only the mod could write these, so
//! when its tracing was off (the default) a turn left no timing breakdown at
//! all; the first-turn-delay hunt had to be reconstructed from session-file
//! timestamps. With these events every turn records where its time went
//! (STT → generate → first LLM delta → TTS per chunk → response), making a cold
//! first turn vs a warm second turn directly visible in the UI.
//!
//! Design:
//! * `elapsed_ms` is measured from the moment the bridge picked the request up
//!   ([`TurnTrace::new`]), which is within a watcher debounce (~15 ms) + poll of
//!   the request file being written — close enough to the request clock that
//!   stage durations are meaningful.
//! * Stage names reuse the prefixes `request_trace.rs` already groups/colors
//!   (`live_chat_*` → LLM, `tts_*` → TTS, `speech_recognition_*` → STT,
//!   `final_response_*` → request I/O), and `tts_stream_start` /
//!   `tts_first_audio_chunk_received` feed the existing "TTS first audio"
//!   summary metric.
//! * Best-effort by construction: every write error is swallowed; tracing can
//!   never fail or slow a turn beyond one small buffered append per stage.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{json, Value};

use crate::protocol::{now_iso8601_millis, safe_file_id};

/// Cap on trace files kept in the traces dir. When exceeded at turn start, the
/// oldest files are pruned so the dir can't grow unbounded across long installs.
const MAX_TRACE_FILES: usize = 400;
/// How many files a prune pass removes (down to a comfortable margin).
const PRUNE_TO: usize = 200;

/// Appends stage events for ONE turn. Cheap to construct; all methods are
/// infallible (errors are swallowed — tracing must never break a turn).
pub struct TurnTrace {
    /// Target file; `None` disables the tracer (blank id / unwritable dir).
    path: Option<PathBuf>,
    request_id: String,
    started: Instant,
}

impl TurnTrace {
    /// Creates the tracer for `request_id`, writing under `<root>/traces/`.
    /// A blank id (or an uncreatable dir) yields a disabled tracer.
    pub fn new(root: &Path, request_id: &str) -> Self {
        // `safe_file_id` maps a blank id to an epoch stamp; a request without an
        // id shouldn't produce an anonymous trace file at all, so gate on the
        // RAW id first.
        let path = if request_id.trim().is_empty() {
            None
        } else {
            let id = safe_file_id(request_id);
            let dir = root.join("traces");
            if std::fs::create_dir_all(&dir).is_ok() {
                prune_old_traces(&dir);
                Some(dir.join(format!("{id}.jsonl")))
            } else {
                None
            }
        };
        Self {
            path,
            request_id: request_id.to_string(),
            started: Instant::now(),
        }
    }

    /// Records a stage with no extra fields.
    pub fn stage(&self, name: &str) {
        self.stage_with(name, json!({}));
    }

    /// Records a stage with extra fields (merged beside the envelope fields).
    pub fn stage_with(&self, name: &str, fields: Value) {
        let Some(path) = &self.path else {
            return;
        };
        let mut obj = fields.as_object().cloned().unwrap_or_default();
        obj.insert("request_id".to_string(), json!(self.request_id));
        obj.insert("stage".to_string(), json!(name));
        obj.insert("at".to_string(), json!(now_iso8601_millis()));
        obj.insert(
            "elapsed_ms".to_string(),
            json!(self.started.elapsed().as_secs_f64() * 1000.0),
        );
        obj.insert("source".to_string(), json!("chasm-bridge"));
        let Ok(line) = serde_json::to_string(&Value::Object(obj)) else {
            return;
        };
        // Best-effort append; a failed write must never fail the turn.
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(file, "{line}");
        }
    }
}

/// Deletes the oldest trace files when the dir holds more than
/// [`MAX_TRACE_FILES`], keeping the newest [`PRUNE_TO`]. Best-effort.
fn prune_old_traces(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                return None;
            }
            let mtime = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            Some((mtime, path))
        })
        .collect();
    if files.len() <= MAX_TRACE_FILES {
        return;
    }
    // Oldest first; delete down to the margin.
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let excess = files.len().saturating_sub(PRUNE_TO);
    for (_, path) in files.into_iter().take(excess) {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sb-turntrace-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn writes_parseable_stage_lines_with_monotonic_offsets() {
        let root = temp_root("basic");
        let trace = TurnTrace::new(&root, "req_123_4");
        trace.stage_with("helper_turn_start", json!({ "npc_key": "easy_pete" }));
        trace.stage_with("live_chat_generate_start", json!({}));
        trace.stage_with("tts_stream_start", json!({ "chunk_chars": 42 }));
        trace.stage("final_response_written");

        let body = std::fs::read_to_string(root.join("traces").join("req_123_4.jsonl")).unwrap();
        let lines: Vec<Value> = body
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 4);
        // Envelope fields present on every line.
        for line in &lines {
            assert_eq!(line["request_id"], json!("req_123_4"));
            assert!(line["stage"].is_string());
            assert!(line["at"].is_string());
            assert!(line["elapsed_ms"].is_number());
            assert_eq!(line["source"], json!("chasm-bridge"));
        }
        // Extra fields survive beside the envelope.
        assert_eq!(lines[0]["npc_key"], json!("easy_pete"));
        assert_eq!(lines[2]["chunk_chars"], json!(42));
        // Offsets never go backwards (same clock, sequential writes).
        let offsets: Vec<f64> = lines
            .iter()
            .map(|line| line["elapsed_ms"].as_f64().unwrap())
            .collect();
        assert!(offsets.windows(2).all(|w| w[1] >= w[0]), "{offsets:?}");

        // The parser chasm's Tracing page uses reads it back with grouped stages.
        let spans = chasm_core::parse_trace_jsonl("req_123_4", &body);
        assert_eq!(spans.stage_count, 4);
        assert_eq!(spans.stages[1].group, "llm"); // live_chat_* → LLM group
        assert_eq!(spans.stages[2].group, "tts"); // tts_* → TTS group
        assert_eq!(spans.stages[3].group, "request"); // final_response_* → request I/O

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn blank_or_hostile_ids_disable_the_tracer() {
        let root = temp_root("ids");
        // Blank id → disabled, no traces dir contents.
        let trace = TurnTrace::new(&root, "");
        trace.stage("helper_turn_start");
        let entries: Vec<_> = std::fs::read_dir(root.join("traces"))
            .map(|it| it.flatten().collect())
            .unwrap_or_default();
        assert!(entries.is_empty());
        // A path-hostile id is sanitized into a plain filename.
        let trace = TurnTrace::new(&root, "req/../evil");
        trace.stage("helper_turn_start");
        let names: Vec<String> = std::fs::read_dir(root.join("traces"))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names.len(), 1);
        assert!(!names[0].contains('/') && !names[0].contains(".."));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn prune_keeps_the_newest_files() {
        let root = temp_root("prune");
        let dir = root.join("traces");
        std::fs::create_dir_all(&dir).unwrap();
        for index in 0..(MAX_TRACE_FILES + 10) {
            std::fs::write(dir.join(format!("req_{index}.jsonl")), "{}").unwrap();
        }
        prune_old_traces(&dir);
        let count = std::fs::read_dir(&dir).unwrap().flatten().count();
        assert!(
            count <= MAX_TRACE_FILES && count >= PRUNE_TO,
            "expected pruned count in [{PRUNE_TO}, {MAX_TRACE_FILES}], got {count}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
