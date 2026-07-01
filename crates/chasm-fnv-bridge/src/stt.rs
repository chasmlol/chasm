//! Voice input (STT) — port of the Node helper's native speech path
//! (`isNativeVoiceRequest`, `getNativeAudioPathCandidates`,
//! `waitForNativeSpeechAudio`, `isNativeSpeechAudioReady`,
//! `stabilizeNativeSpeechAudio`, `recognizeNativeSpeech` + the file-race retries).
//!
//! The game records push-to-talk audio into a `*.stt.wav` sidecar next to the
//! request file. We wait for that WAV to finish writing (size/mtime settle, with
//! Windows file-lock retries), move it to `processed/<id>.stt.wav` for a stable
//! read, recognize it via chasm, and hand the transcript to the turn path.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, UNIX_EPOCH};

use anyhow::{anyhow, Context};
use base64::Engine;
use serde_json::{json, Map, Value};

use crate::chasm::ChasmClient;
use crate::config::BridgeConfig;
use crate::protocol::{safe_file_id, sanitize_bridge_line, NativeRequest};
use crate::{native_inbox, native_processed};

const READY_TIMEOUT_MS: u128 = 2_000;
const SETTLE_MS: u64 = 35;
const FILE_RETRY_MS: u64 = 50;
const FILE_RETRIES: u32 = 8;

/// `isNativeVoiceRequest`: does the request signal voice input?
pub fn is_native_voice_request(request: &NativeRequest) -> bool {
    const TRUTHY_KEYS: [&str; 12] = [
        "voice_request", "voiceRequest", "audio", "audio_path", "audioPath", "stt_audio_path",
        "sttAudioPath", "stt_audio_file", "sttAudioFile", "audio_file", "audioFile", "audio_base64",
    ];
    if TRUTHY_KEYS
        .iter()
        .any(|k| request.metadata.get(*k).map(js_truthy).unwrap_or(false))
    {
        return true;
    }
    ["input_mode", "inputMode"]
        .iter()
        .any(|k| request.metadata.get(*k).and_then(Value::as_str) == Some("voice"))
}

/// True if any candidate `*.stt.wav` sidecar exists on disk.
pub fn has_audio_sidecar(root: &Path, request_path: &Path, request: &NativeRequest) -> bool {
    audio_path_candidates(root, request_path, request)
        .iter()
        .any(|c| c.is_file())
}

/// Wait for the speech WAV, recognize it via chasm, return the transcript.
pub async fn recognize_native_speech(
    config: &BridgeConfig,
    client: &dyn ChasmClient,
    root: &Path,
    request_path: &Path,
    request: &NativeRequest,
) -> anyhow::Result<String> {
    let audio_path = wait_for_native_speech_audio(root, request_path, request)
        .await
        .ok_or_else(|| anyhow!("Voice request did not include a readable WAV audio payload."))?;
    let stable = stabilize_native_speech_audio(root, &audio_path, request).await?;
    let bytes = retry_file(|| std::fs::read(&stable))
        .await
        .with_context(|| format!("reading speech audio {}", stable.display()))?;
    if bytes.is_empty() {
        anyhow::bail!("Voice request audio file was empty.");
    }
    let audio_b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let text = client.recognize(&recognize_body(config, request, &audio_b64)).await?;
    let text = sanitize_bridge_line(&text);
    if text.is_empty() {
        anyhow::bail!("Speech recognition returned empty player text.");
    }
    Ok(text)
}

/// `getNativeAudioPathCandidates`: the sidecar paths to probe.
fn audio_path_candidates(root: &Path, request_path: &Path, request: &NativeRequest) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if !out.contains(&p) {
            out.push(p);
        }
    };

    // Explicit paths carried in the request metadata (rare for native).
    for key in [
        "audio_path", "audioPath", "stt_audio_path", "sttAudioPath", "audio_file", "audioFile",
        "stt_audio_file", "sttAudioFile",
    ] {
        if let Some(value) = request.metadata.get(key).and_then(Value::as_str) {
            let value = value.trim();
            if !value.is_empty() {
                let candidate = if Path::new(value).is_absolute() {
                    PathBuf::from(value)
                } else {
                    native_inbox(root).join(value)
                };
                push(candidate);
            }
        }
    }

    let stem = request_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("req_live");
    let req_id = safe_file_id(&request.request_id);
    if let Some(dir) = request_path.parent() {
        push(dir.join(format!("{stem}.stt.wav")));
    }
    push(native_inbox(root).join(format!("{req_id}.stt.wav")));
    push(native_inbox(root).join(format!("{stem}.stt.wav")));
    push(native_inbox(root).join("req_live.stt.wav"));
    out
}

/// `waitForNativeSpeechAudio`: poll candidates until one is ready or the timeout.
async fn wait_for_native_speech_audio(
    root: &Path,
    request_path: &Path,
    request: &NativeRequest,
) -> Option<PathBuf> {
    let request_mtime = mtime_ms(request_path).unwrap_or(0);
    let candidates = audio_path_candidates(root, request_path, request);
    let started = Instant::now();
    while started.elapsed().as_millis() < READY_TIMEOUT_MS {
        for candidate in &candidates {
            if candidate.is_file() && is_native_speech_audio_ready(candidate, request_mtime).await {
                return Some(candidate.clone());
            }
        }
        tokio::time::sleep(Duration::from_millis(FILE_RETRY_MS)).await;
    }
    None
}

/// `isNativeSpeechAudioReady`: size>0, not stale vs the request, and settled
/// (size + mtime unchanged across a short interval — the game finished writing).
async fn is_native_speech_audio_ready(path: &Path, request_mtime_ms: u64) -> bool {
    let before = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return false,
    };
    if !before.is_file() || before.len() == 0 {
        return false;
    }
    let before_mtime = meta_mtime_ms(&before).unwrap_or(0);
    // Reject a stale sidecar left from an earlier request.
    if request_mtime_ms > 0 && before_mtime + 100 < request_mtime_ms {
        return false;
    }
    tokio::time::sleep(Duration::from_millis(SETTLE_MS)).await;
    match std::fs::metadata(path) {
        Ok(after) => {
            let after_mtime = meta_mtime_ms(&after).unwrap_or(0);
            after.is_file()
                && after.len() > 0
                && after.len() == before.len()
                && after_mtime.abs_diff(before_mtime) < 1
        }
        Err(_) => false,
    }
}

/// `stabilizeNativeSpeechAudio`: move an in-root sidecar to `processed/<id>.stt.wav`
/// for a stable read (and as its archive), with copy+remove fallback if locked.
async fn stabilize_native_speech_audio(
    root: &Path,
    audio_path: &Path,
    request: &NativeRequest,
) -> anyhow::Result<PathBuf> {
    let root_prefix = format!("{}{}", root.to_string_lossy().to_lowercase(), std::path::MAIN_SEPARATOR);
    let audio_lower = audio_path.to_string_lossy().to_lowercase();
    if !audio_lower.starts_with(&root_prefix) {
        return Ok(audio_path.to_path_buf());
    }
    let dest = native_processed(root).join(format!("{}.stt.wav", safe_file_id(&request.request_id)));
    if audio_lower == dest.to_string_lossy().to_lowercase() {
        return Ok(dest);
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if retry_file(|| std::fs::rename(audio_path, &dest)).await.is_err() {
        retry_file(|| std::fs::copy(audio_path, &dest))
            .await
            .with_context(|| format!("copying speech audio to {}", dest.display()))?;
        let _ = retry_file(|| std::fs::remove_file(audio_path)).await;
    }
    Ok(dest)
}

/// Recognize a WAV file directly against chasm — offline smoke test for `--stt-selftest`.
pub async fn recognize_wav_file(
    config: &BridgeConfig,
    client: &dyn ChasmClient,
    wav_path: &Path,
) -> anyhow::Result<String> {
    let bytes = std::fs::read(wav_path).with_context(|| format!("reading {}", wav_path.display()))?;
    if bytes.is_empty() {
        anyhow::bail!("wav file is empty");
    }
    let audio_b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let request = NativeRequest::default();
    let text = client.recognize(&recognize_body(config, &request, &audio_b64)).await?;
    Ok(sanitize_bridge_line(&text))
}

fn recognize_body(config: &BridgeConfig, request: &NativeRequest, audio_b64: &str) -> Value {
    let mut obj: Map<String, Value> = config
        .speech_recognition
        .as_object()
        .cloned()
        .unwrap_or_default();
    obj.insert("timeoutMs".into(), json!(config.speech_recognition_timeout_ms));

    // language: request override, else the configured STT language.
    let language = request
        .metadata
        .get("language")
        .or_else(|| request.metadata.get("stt_language"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            config
                .speech_recognition
                .get("language")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    if let Some(language) = language {
        obj.insert("language".into(), json!(language));
    }

    if let Some(prompt) = request
        .metadata
        .get("stt_prompt")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            config
                .speech_recognition
                .get("prompt")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
    {
        if !prompt.is_empty() {
            obj.insert("prompt".into(), json!(prompt));
        }
    }

    obj.insert(
        "audio".into(),
        json!({ "data": audio_b64, "encoding": "base64", "format": "wav", "mimeType": "audio/wav" }),
    );
    Value::Object(obj)
}

// ---------------------------------------------------------------------------
// File-race helpers
// ---------------------------------------------------------------------------

/// Retry a filesystem op on Windows sharing/lock violations (Node's EBUSY/EPERM).
async fn retry_file<T, F>(mut operation: F) -> std::io::Result<T>
where
    F: FnMut() -> std::io::Result<T>,
{
    let mut last: Option<std::io::Error> = None;
    for attempt in 0..FILE_RETRIES {
        match operation() {
            Ok(value) => return Ok(value),
            Err(error) => {
                if !is_retryable(&error) {
                    return Err(error);
                }
                last = Some(error);
                tokio::time::sleep(Duration::from_millis(FILE_RETRY_MS * (attempt as u64 + 1))).await;
            }
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("file retry exhausted")))
}

fn is_retryable(error: &std::io::Error) -> bool {
    matches!(error.kind(), std::io::ErrorKind::PermissionDenied)
        || matches!(error.raw_os_error(), Some(32) | Some(33)) // SHARING_VIOLATION / LOCK_VIOLATION
}

fn mtime_ms(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().and_then(|m| meta_mtime_ms(&m))
}

fn meta_mtime_ms(meta: &std::fs::Metadata) -> Option<u64> {
    meta.modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as u64)
}

fn js_truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0 && !f.is_nan()).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Null => false,
        Value::Array(_) | Value::Object(_) => true,
    }
}
