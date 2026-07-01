//! Byte-for-byte port of the NVBridge native file protocol from the Node helper
//! (`Chasm/tools/fnv/nvbridge-helper.mjs`).
//!
//! The C++ NVSE plugin reads these files positionally and exactly, so every
//! formatter here must reproduce the Node output byte-for-byte. Parity is locked
//! down by the unit tests below (against real captured bytes) and by the
//! `--replay` harness in [`crate::replay`].
//!
//! Source references are to function names in the Node helper:
//! `sanitizeBridgeLine`, `safeFileId`, `parseNativeTextRequest`,
//! `buildNativeArchivedRequest`, `writeNativeResponse`, `writeNativeAudioChunk`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Primitive string helpers
// ---------------------------------------------------------------------------

/// `sanitizeBridgeLine`: collapse every CR/LF to a single space, then trim.
pub fn sanitize_bridge_line(value: &str) -> String {
    value.replace(['\r', '\n'], " ").trim().to_string()
}

/// `safeFileId`: keep `[A-Za-z0-9_-]`, map everything else to `_`, cap at 80.
pub fn safe_file_id(value: &str) -> String {
    let mapped: String = value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(80)
        .collect();
    if mapped.is_empty() {
        now_epoch_millis().to_string()
    } else {
        mapped
    }
}

/// Split like JS `String.split(/\r?\n/)`: break on `\n`, drop a trailing `\r`.
pub fn split_crlf(text: &str) -> Vec<&str> {
    text.split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect()
}

// ---------------------------------------------------------------------------
// Timestamps (`new Date().toISOString()` -> "YYYY-MM-DDTHH:MM:SS.sssZ")
// ---------------------------------------------------------------------------

pub fn now_epoch_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Current UTC time formatted exactly like JS `new Date().toISOString()`.
pub fn now_iso8601_millis() -> String {
    epoch_millis_to_iso8601(now_epoch_millis())
}

/// Format epoch milliseconds as `YYYY-MM-DDTHH:MM:SS.sssZ` (UTC, 3-digit millis).
pub fn epoch_millis_to_iso8601(ms: i64) -> String {
    let days = ms.div_euclid(86_400_000);
    let mut rem = ms.rem_euclid(86_400_000);
    let millis = rem % 1000;
    rem /= 1000;
    let sec = rem % 60;
    rem /= 60;
    let min = rem % 60;
    rem /= 60;
    let hour = rem;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}.{millis:03}Z")
}

/// Howard Hinnant's `civil_from_days`: days-since-Unix-epoch -> (year, month, day).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

// ---------------------------------------------------------------------------
// Audio-tag / caption stripping (`stripTtsAudioTagsForBridge`)
// ---------------------------------------------------------------------------

/// `stripTtsAudioTagsForBridge`, partial port for Section 1.
///
/// Ports the deterministic pieces: sanitize, SSML tag removal, whitespace
/// normalization. The `[bracket]` audio-tag word-list rule
/// (`shouldStripBridgeBracketAudioTag`) is deferred to Section 2 (captions),
/// where it is parity-tested against real TTS text. Section 1 never feeds tagged
/// text through here (the stub echoes plain player text), and the `--replay`
/// response check re-frames already-stripped golden lines, so this partial port
/// does not affect Section 1 parity.
pub fn strip_tts_audio_tags_for_bridge(value: &str) -> String {
    let sanitized = sanitize_bridge_line(value);
    let no_ssml = ssml_re().replace_all(&sanitized, "");
    // TODO(Section 2): port shouldStripBridgeBracketAudioTag word lists here.
    let no_space_before_punct = space_before_punct_re().replace_all(&no_ssml, "$p");
    let collapsed = multi_space_re().replace_all(&no_space_before_punct, " ");
    collapsed.trim().to_string()
}

// ---------------------------------------------------------------------------
// Request parsing (`parseNativeTextRequest` + metadata helpers)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeLocation {
    pub cell: String,
    pub worldspace: String,
    pub region: String,
    pub major: String,
    pub minor: String,
}

#[derive(Debug, Clone, Default)]
pub struct NativeRequest {
    pub request_id: String,
    pub npc_key: String,
    pub npc_name: String,
    pub want_tts: bool,
    pub player_text: String,
    pub location: NativeLocation,
    /// Coerced `key=value` metadata from line 10+ (mirrors the JS spread object).
    pub metadata: BTreeMap<String, Value>,
}

/// `parseNativeTextRequest`: fixed fields on lines 0-9, key=value metadata 10+.
pub fn parse_native_text_request(file_path: &Path, text: &str) -> NativeRequest {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let lines = split_crlf(text);
    let metadata = parse_native_metadata_from_lines(&lines, 10);
    let line = |i: usize| sanitize_bridge_line(lines.get(i).copied().unwrap_or(""));

    let request_id = {
        let raw = line(0);
        if raw.is_empty() {
            file_stem(file_path)
        } else {
            raw
        }
    };
    // Node: Number(sanitize(lines[3] || '1')) !== 0
    let want_tts_src = {
        let raw = lines.get(3).copied().unwrap_or("");
        sanitize_bridge_line(if raw.is_empty() { "1" } else { raw })
    };

    NativeRequest {
        request_id,
        npc_key: line(1),
        npc_name: line(2),
        want_tts: js_number(&want_tts_src) != Some(0.0),
        player_text: line(4),
        location: NativeLocation {
            cell: line(5),
            worldspace: line(6),
            region: line(7),
            major: line(8),
            minor: line(9),
        },
        metadata,
    }
}

fn file_stem(p: &Path) -> String {
    p.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

/// `parseNativeMetadataFromLines`: optional JSON object on the first line, then
/// `key=value` lines (coerced) from `start` onward, later keys overriding.
fn parse_native_metadata_from_lines(lines: &[&str], start: usize) -> BTreeMap<String, Value> {
    let mut metadata = parse_native_metadata(lines.get(start).copied().unwrap_or(""));
    for line in lines.iter().skip(start) {
        if let Some(caps) = metadata_kv_re().captures(line) {
            let key = caps.get(1).unwrap().as_str().to_string();
            let value = coerce_native_metadata_value(caps.get(2).unwrap().as_str());
            metadata.insert(key, value);
        }
    }
    metadata
}

/// `parseNativeMetadata`: parse a line as a JSON object, else `{}`.
fn parse_native_metadata(value: &str) -> BTreeMap<String, Value> {
    let text = sanitize_bridge_line(value);
    if text.is_empty() {
        return BTreeMap::new();
    }
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(map)) => map.into_iter().collect(),
        _ => BTreeMap::new(),
    }
}

/// `coerceNativeMetadataValue`: `true`/`false` -> bool, numeric -> number, else text.
fn coerce_native_metadata_value(value: &str) -> Value {
    let text = sanitize_bridge_line(value);
    match text.to_ascii_lowercase().as_str() {
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        _ => {}
    }
    if number_literal_re().is_match(&text) {
        // Integer literals -> integer JSON numbers (JS `Number("1")` is 1, not 1.0),
        // so re-serialization stays clean. Decimals fall back to f64.
        if !text.contains('.') {
            if let Ok(i) = text.parse::<i64>() {
                return Value::Number(i.into());
            }
        }
        if let Ok(n) = text.parse::<f64>() {
            if n.is_finite() {
                if let Some(num) = serde_json::Number::from_f64(n) {
                    return Value::Number(num);
                }
            }
        }
    }
    Value::String(text)
}

// ---------------------------------------------------------------------------
// Archived request (`buildNativeArchivedRequest`)
// ---------------------------------------------------------------------------

/// `buildNativeArchivedRequest`: 12 fields, `\r\n`-joined, trailing `\r\n`.
pub fn build_native_archived_request(req: &NativeRequest) -> String {
    let voice_request = req
        .metadata
        .get("voice_request")
        .map(js_truthy)
        .unwrap_or(false);
    let transcript = req.metadata.get("transcript");
    let transcript_line = match transcript {
        Some(v) if js_truthy(v) => format!("transcript={}", js_to_string(v)),
        _ => String::new(),
    };

    let fields = [
        req.request_id.clone(),
        req.npc_key.clone(),
        req.npc_name.clone(),
        if req.want_tts { "1".into() } else { "0".into() },
        req.player_text.clone(),
        req.location.cell.clone(),
        req.location.worldspace.clone(),
        req.location.region.clone(),
        req.location.major.clone(),
        req.location.minor.clone(),
        if voice_request {
            "voice_request=1".into()
        } else {
            String::new()
        },
        transcript_line,
    ];
    frame_lines(&fields)
}

// ---------------------------------------------------------------------------
// Response writer (`writeNativeResponse`)
// ---------------------------------------------------------------------------

/// Structured inputs to [`build_native_response`], mirroring the `writeNativeResponse`
/// line layout (fixed 0-8, then `extra_lines`, then the game_master triple).
pub struct ResponseFields<'a> {
    pub status: &'a str,
    pub request_id: &'a str,
    pub npc_key: &'a str,
    pub npc_name: &'a str,
    pub audio_filename: &'a str,
    pub text: &'a str,
    pub error: &'a str,
    pub timestamp: &'a str,
    pub player_text: &'a str,
    pub extra_lines: &'a [String],
    pub gm_action: &'a str,
    pub gm_confidence: &'a str,
    pub gm_should_trigger: bool,
}

/// `writeNativeResponse` body: build the exact response file bytes.
pub fn build_native_response(f: &ResponseFields) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(12 + f.extra_lines.len());
    lines.push(f.status.to_string());
    lines.push(f.request_id.to_string());
    lines.push(f.npc_key.to_string());
    lines.push(f.npc_name.to_string());
    lines.push(f.audio_filename.to_string());
    lines.push(strip_tts_audio_tags_for_bridge(f.text));
    lines.push(f.error.to_string());
    lines.push(f.timestamp.to_string());
    lines.push(f.player_text.to_string());
    lines.extend(f.extra_lines.iter().cloned());
    lines.push(f.gm_action.to_string());
    lines.push(f.gm_confidence.to_string());
    lines.push(if f.gm_should_trigger { "1".into() } else { "0".into() });
    frame_lines(&lines)
}

// ---------------------------------------------------------------------------
// Audio chunk writer (`writeNativeAudioChunk`) — format only, wired in Section 2
// ---------------------------------------------------------------------------

/// `writeNativeAudioChunk` body. Returns `(chunk_filename, file_contents)`.
#[allow(clippy::too_many_arguments)]
pub fn build_native_audio_chunk(
    request_id: &str,
    index: u32,
    npc_key: &str,
    npc_name: &str,
    filename: &str,
    text: &str,
    timestamp: &str,
    caption_max_chars: Option<i64>,
    extra_lines: &[String],
) -> (String, String) {
    let mut lines: Vec<String> = vec![
        request_id.to_string(),
        index.to_string(),
        npc_key.to_string(),
        npc_name.to_string(),
        filename.to_string(),
        strip_tts_audio_tags_for_bridge(text),
        timestamp.to_string(),
    ];
    lines.extend(extra_lines.iter().cloned());
    if let Some(max) = caption_max_chars {
        lines.push(format!("caption_max_chars={max}"));
    }
    let contents = frame_lines(&lines);
    let chunk_filename = format!("{}.{:04}.txt", safe_file_id(request_id), index);
    (chunk_filename, contents)
}

// ---------------------------------------------------------------------------
// Placeholder audio (Section 1 only)
// ---------------------------------------------------------------------------

/// A minimal 16-bit PCM mono WAV holding `millis` of silence at `sample_rate` Hz.
///
/// Section 1 has no TTS, but the C++ plugin only renders an NPC caption alongside
/// audio playback (`ConsumeReply` gates the subtitle on a played audio file). So
/// the stub emits this silent clip purely to make the echo caption visible and to
/// exercise the audio-file → plugin-playback path. Section 2 replaces it with real
/// cloned-voice TTS. The plugin resamples the source, so 44.1kHz/16-bit/mono is safe.
pub fn build_silence_wav(sample_rate: u32, millis: u32) -> Vec<u8> {
    let channels: u16 = 1;
    let bits: u16 = 16;
    let num_samples = sample_rate as u64 * millis as u64 / 1000;
    let data_bytes = (num_samples * (bits as u64 / 8) * channels as u64) as u32;
    let block_align = channels * (bits / 8);
    let byte_rate = sample_rate * block_align as u32;

    let mut out = Vec::with_capacity(44 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    out.resize(out.len() + data_bytes as usize, 0); // silent samples
    out
}

// ---------------------------------------------------------------------------
// Framing + JS value semantics
// ---------------------------------------------------------------------------

/// Sanitize each line and join with `\r\n`, plus a trailing `\r\n`. This is the
/// common tail of every Node writer (`[...].map(sanitizeBridgeLine).join('\r\n') + '\r\n'`).
pub fn frame_lines(lines: &[impl AsRef<str>]) -> String {
    let sanitized: Vec<String> = lines
        .iter()
        .map(|l| sanitize_bridge_line(l.as_ref()))
        .collect();
    format!("{}\r\n", sanitized.join("\r\n"))
}

/// JS `Number(s)`: trimmed empty -> 0, numeric -> value, otherwise NaN (`None`).
fn js_number(s: &str) -> Option<f64> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Some(0.0);
    }
    trimmed.parse::<f64>().ok()
}

/// JS truthiness of a JSON value.
fn js_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0 && !f.is_nan()).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// JS template-literal stringification (`${value}`) for the value types we emit.
fn js_to_string(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::String(s) => s.clone(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else if let Some(f) = n.as_f64() {
                if f.fract() == 0.0 && f.is_finite() {
                    format!("{}", f as i64)
                } else {
                    format!("{f}")
                }
            } else {
                n.to_string()
            }
        }
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Cached regexes
// ---------------------------------------------------------------------------

fn metadata_kv_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*([A-Za-z0-9_.-]+)\s*=\s*(.*?)\s*$").unwrap())
}

fn number_literal_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^-?\d+(?:\.\d+)?$").unwrap())
}

fn ssml_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)</?(?:speak|voice|audio|break|prosody|emphasis|say-as|sub|phoneme|mstts:express-as|amazon:emotion|p|s|mark|bookmark|lang|w)\b[^>]*>",
        )
        .unwrap()
    })
}

fn space_before_punct_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Named replacement group `$p` so adjacent chars never swallow the index.
    RE.get_or_init(|| Regex::new(r"\s+(?P<p>[,.!?;:])").unwrap())
}

fn multi_space_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[ \t]{2,}").unwrap())
}

// ---------------------------------------------------------------------------
// Tests — locked against real captured bytes
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Real archived request captured from processed/req_34859625_252.txt (169 bytes).
    const GOLDEN_ARCHIVE: &str = "req_34859625_252\r\ntodd\r\nTodd\r\n1\r\nMake Sony attack me.\r\nGSProspectorSaloonInterior\r\n\r\n\r\nGoodsprings\r\nProspector Saloon\r\nvoice_request=1\r\ntranscript=Make Sony attack me.\r\n";

    // Real response captured from outbox/req_codex_latency_160544275.txt (183 bytes).
    const GOLDEN_RESPONSE: &str = "1\r\nreq_codex_latency_160544275\r\neasy_pete\r\nEasy Pete\r\nnvbridge_req_codex_latency_160544275.wav\r\nBack so soon, wanderer?\r\n\r\n2026-06-24T15:05:45.466Z\r\nHello there again.\r\n0\r\nNONE\r\n\r\n0\r\n";

    #[test]
    fn sanitize_collapses_crlf_and_trims() {
        assert_eq!(sanitize_bridge_line("  a\r\nb \r"), "a  b");
        assert_eq!(sanitize_bridge_line("plain"), "plain");
    }

    #[test]
    fn safe_file_id_maps_and_caps() {
        assert_eq!(safe_file_id("req_live.txt"), "req_live_txt");
        assert_eq!(safe_file_id("a/b\\c:d"), "a_b_c_d");
        assert_eq!(safe_file_id(&"x".repeat(100)).len(), 80);
    }

    #[test]
    fn iso8601_matches_js_to_iso_string() {
        assert_eq!(epoch_millis_to_iso8601(0), "1970-01-01T00:00:00.000Z");
        // 1 day + 1h1m1s.
        assert_eq!(epoch_millis_to_iso8601(90_061_000), "1970-01-02T01:01:01.000Z");
        // The well-known "Unix billennium": 1e9 seconds after the epoch.
        assert_eq!(
            epoch_millis_to_iso8601(1_000_000_000_000),
            "2001-09-09T01:46:40.000Z"
        );
    }

    #[test]
    fn archive_round_trips_byte_for_byte() {
        let path = PathBuf::from("req_34859625_252.txt");
        let req = parse_native_text_request(&path, GOLDEN_ARCHIVE);
        // Spot-check parsed fields.
        assert_eq!(req.request_id, "req_34859625_252");
        assert_eq!(req.npc_key, "todd");
        assert_eq!(req.player_text, "Make Sony attack me.");
        assert!(req.want_tts);
        assert_eq!(req.location.cell, "GSProspectorSaloonInterior");
        assert_eq!(req.location.major, "Goodsprings");
        assert_eq!(
            req.metadata.get("transcript").and_then(|v| v.as_str()),
            Some("Make Sony attack me.")
        );
        // The whole point: re-serialization reproduces the original bytes.
        let rebuilt = build_native_archived_request(&req);
        assert_eq!(rebuilt, GOLDEN_ARCHIVE);
        assert_eq!(rebuilt.len(), 169);
    }

    #[test]
    fn response_framing_reproduces_golden() {
        // Parse the golden response into its lines (dropping the trailing-CRLF
        // artifact) and re-frame: must reproduce the bytes exactly.
        let mut lines: Vec<String> = split_crlf(GOLDEN_RESPONSE)
            .iter()
            .map(|s| s.to_string())
            .collect();
        if lines.last().map(|s| s.is_empty()).unwrap_or(false) {
            lines.pop();
        }
        let rebuilt = frame_lines(&lines);
        assert_eq!(rebuilt, GOLDEN_RESPONSE);
        assert_eq!(rebuilt.len(), 183);
    }

    #[test]
    fn build_response_matches_golden_layout() {
        // Reconstruct the golden response through the structured writer.
        let extra = vec!["0".to_string()];
        let fields = ResponseFields {
            status: "1",
            request_id: "req_codex_latency_160544275",
            npc_key: "easy_pete",
            npc_name: "Easy Pete",
            audio_filename: "nvbridge_req_codex_latency_160544275.wav",
            text: "Back so soon, wanderer?",
            error: "",
            timestamp: "2026-06-24T15:05:45.466Z",
            player_text: "Hello there again.",
            extra_lines: &extra,
            gm_action: "NONE",
            gm_confidence: "",
            gm_should_trigger: false,
        };
        assert_eq!(build_native_response(&fields), GOLDEN_RESPONSE);
    }

    #[test]
    fn stub_echo_response_has_expected_shape() {
        // The Section 1 stub: status 1, no audio (-1), empty game_master.
        let extra = vec!["-1".to_string()];
        let fields = ResponseFields {
            status: "1",
            request_id: "req_live",
            npc_key: "easy_pete",
            npc_name: "Easy Pete",
            audio_filename: "",
            text: "Rust bridge heard: hello there",
            error: "",
            timestamp: "1970-01-01T00:00:00.000Z",
            player_text: "hello there",
            extra_lines: &extra,
            gm_action: "",
            gm_confidence: "",
            gm_should_trigger: false,
        };
        let out = build_native_response(&fields);
        let lines: Vec<&str> = split_crlf(&out);
        assert_eq!(lines[0], "1");
        assert_eq!(lines[4], ""); // no audio file
        assert_eq!(lines[5], "Rust bridge heard: hello there");
        assert_eq!(lines[9], "-1"); // no-audio extra line
        assert_eq!(lines[12], "0"); // should_trigger
        assert!(out.ends_with("\r\n"));
    }

    #[test]
    fn metadata_kv_parsing_and_coercion() {
        let lines = vec!["", "", "", "", "", "", "", "", "", "", "voice_request=1", "flag=true", "name=Sunny"];
        let meta = parse_native_metadata_from_lines(&lines, 10);
        assert_eq!(meta.get("voice_request"), Some(&Value::from(1)));
        assert_eq!(meta.get("flag"), Some(&Value::Bool(true)));
        assert_eq!(meta.get("name"), Some(&Value::from("Sunny")));
    }

    #[test]
    fn silence_wav_is_well_formed() {
        let wav = build_silence_wav(44_100, 200);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        // 200ms @ 44100 Hz, 16-bit mono = 8820 samples * 2 bytes = 17640 data bytes.
        assert_eq!(wav.len(), 44 + 17_640);
        // RIFF chunk size = 36 + data bytes.
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 36 + 17_640);
        assert!(wav[44..].iter().all(|&b| b == 0)); // silence
    }

    #[test]
    fn audio_chunk_filename_and_caption_meta() {
        let (name, body) = build_native_audio_chunk(
            "req_live",
            3,
            "easy_pete",
            "Easy Pete",
            "chunk.wav",
            "Hello.",
            "1970-01-01T00:00:00.000Z",
            Some(80),
            &[],
        );
        assert_eq!(name, "req_live.0003.txt");
        assert!(body.contains("caption_max_chars=80\r\n"));
        assert!(body.starts_with("req_live\r\n3\r\n"));
    }
}
