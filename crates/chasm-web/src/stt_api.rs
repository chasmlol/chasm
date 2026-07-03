//! Hosted-STT provider adapters: turn a WAV byte payload into a transcript.
//!
//! chasm's `/speech/recognize` handler decodes the in-game audio to WAV bytes and
//! (for the managed-local provider) posts them to the Parakeet server. This module
//! routes the SAME bytes to a hosted API when the STT provider isn't `"local"`:
//!
//!   * **OpenAI** / **Groq** — OpenAI multipart `POST {base}/audio/transcriptions`
//!     (`file`, `model`, optional `language`/`prompt`), `Authorization: Bearer`.
//!     Response `{ "text": "…" }`.
//!   * **Deepgram** — `POST {base}/listen?model=…` with the raw WAV body and
//!     `Authorization: Token <key>`. Transcript at
//!     `results.channels[0].alternatives[0].transcript`.
//!   * **AssemblyAI** — async: upload the bytes to `{base}/upload`, POST a
//!     `{base}/transcript` job, then poll `{base}/transcript/{id}` until
//!     `completed`/`error`. `authorization: <key>` (no `Bearer`).
//!
//! The response PARSERS are pure + unit-tested; the async HTTP calls surface a
//! readable `Err(String)` (bad key / rate limit / provider message) for the UI.

use std::time::{Duration, Instant};

use base64::{engine::general_purpose::STANDARD, Engine as _};
use reqwest::Client;
use serde_json::{json, Value};

use chasm_core::ResolvedApi;

use crate::llm_api::format_http_error;

/// Transcribes `wav` (a complete mono 16-bit WAV byte buffer) via the hosted STT
/// `provider`. `resolved` carries the base URL + key + model (already defaulted).
pub async fn transcribe(
    client: &Client,
    provider: &str,
    resolved: &ResolvedApi,
    wav: Vec<u8>,
    language: &str,
    prompt: &str,
    timeout: Duration,
) -> Result<String, String> {
    if resolved.api_key.is_empty() {
        return Err(format!("{provider}: no API key set (Settings → STT)."));
    }
    match provider {
        // OpenAI / Groq speak the OpenAI MULTIPART transcription shape
        // (`{base}/audio/transcriptions`, `file` upload).
        "openai" | "groq" => {
            transcribe_openai_compat(client, provider, resolved, wav, language, prompt, timeout).await
        }
        // OpenRouter's /audio/transcriptions is JSON with base64 `input_audio`
        // (NOT multipart) — a different shape from OpenAI's.
        "openrouter" => transcribe_openrouter(client, resolved, wav, language, timeout).await,
        "deepgram" => transcribe_deepgram(client, resolved, wav, language, timeout).await,
        "assemblyai" => transcribe_assemblyai(client, resolved, wav, language, timeout).await,
        other => Err(format!("Unknown STT provider '{other}'.")),
    }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible (OpenAI, Groq)
// ---------------------------------------------------------------------------

async fn transcribe_openai_compat(
    client: &Client,
    provider: &str,
    resolved: &ResolvedApi,
    wav: Vec<u8>,
    language: &str,
    prompt: &str,
    timeout: Duration,
) -> Result<String, String> {
    let file_part = reqwest::multipart::Part::bytes(wav)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| format!("{provider}: building audio part failed: {e}"))?;
    let mut form = reqwest::multipart::Form::new()
        .part("file", file_part)
        .text("model", resolved.model.clone())
        .text("response_format", "json");
    let language = language.trim();
    if !language.is_empty() {
        form = form.text("language", language.to_string());
    }
    let prompt = prompt.trim();
    if !prompt.is_empty() {
        form = form.text("prompt", prompt.to_string());
    }

    let url = format!("{}/audio/transcriptions", resolved.base_url);
    let resp = client
        .post(&url)
        .bearer_auth(&resolved.api_key)
        .timeout(timeout)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("{provider}: request failed: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format_http_error(provider, status.as_u16(), &body));
    }
    let value: Value = serde_json::from_str(&body)
        .map_err(|e| format!("{provider}: bad JSON response: {e}"))?;
    parse_openai_transcript(&value)
        .ok_or_else(|| format!("{provider}: response had no transcript text."))
}

/// Extracts the transcript from an OpenAI/Groq `/audio/transcriptions` response
/// (`{ "text": "…" }`, or the verbose_json variant with the same top-level key).
pub fn parse_openai_transcript(value: &Value) -> Option<String> {
    value
        .get("text")
        .and_then(Value::as_str)
        .map(|s| s.trim().to_string())
}

// ---------------------------------------------------------------------------
// OpenRouter (JSON base64 `input_audio` — NOT multipart)
// ---------------------------------------------------------------------------

/// OpenRouter transcription request body: `{ model, input_audio: { data, format },
/// language? }` where `data` is raw base64 (NOT a data URI). Pure + tested.
pub fn build_openrouter_stt_body(model: &str, wav_b64: &str, language: &str) -> Value {
    let mut body = json!({
        "model": model,
        "input_audio": { "data": wav_b64, "format": "wav" }
    });
    let language = language.trim();
    if !language.is_empty() {
        body["language"] = json!(language);
    }
    body
}

async fn transcribe_openrouter(
    client: &Client,
    resolved: &ResolvedApi,
    wav: Vec<u8>,
    language: &str,
    timeout: Duration,
) -> Result<String, String> {
    let wav_b64 = STANDARD.encode(&wav);
    let resp = client
        .post(format!("{}/audio/transcriptions", resolved.base_url))
        .bearer_auth(&resolved.api_key)
        .timeout(timeout)
        .json(&build_openrouter_stt_body(&resolved.model, &wav_b64, language))
        .send()
        .await
        .map_err(|e| format!("OpenRouter: request failed: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format_http_error("OpenRouter", status.as_u16(), &body));
    }
    let value: Value =
        serde_json::from_str(&body).map_err(|e| format!("OpenRouter: bad JSON response: {e}"))?;
    parse_openai_transcript(&value)
        .filter(|t| !t.is_empty())
        .ok_or_else(|| "OpenRouter: response had no transcript text.".to_string())
}

// ---------------------------------------------------------------------------
// Deepgram
// ---------------------------------------------------------------------------

async fn transcribe_deepgram(
    client: &Client,
    resolved: &ResolvedApi,
    wav: Vec<u8>,
    language: &str,
    timeout: Duration,
) -> Result<String, String> {
    let mut url = format!(
        "{}/listen?model={}&smart_format=true&punctuate=true",
        resolved.base_url, resolved.model
    );
    let language = language.trim();
    if !language.is_empty() {
        url.push_str(&format!("&language={language}"));
    }
    let resp = client
        .post(&url)
        .header("Authorization", format!("Token {}", resolved.api_key))
        .header("Content-Type", "audio/wav")
        .timeout(timeout)
        .body(wav)
        .send()
        .await
        .map_err(|e| format!("Deepgram: request failed: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format_http_error("Deepgram", status.as_u16(), &body));
    }
    let value: Value =
        serde_json::from_str(&body).map_err(|e| format!("Deepgram: bad JSON response: {e}"))?;
    parse_deepgram_transcript(&value)
        .ok_or_else(|| "Deepgram: response had no transcript text.".to_string())
}

/// Extracts the transcript from a Deepgram `/listen` response
/// (`results.channels[0].alternatives[0].transcript`).
pub fn parse_deepgram_transcript(value: &Value) -> Option<String> {
    value
        .get("results")?
        .get("channels")?
        .as_array()?
        .first()?
        .get("alternatives")?
        .as_array()?
        .first()?
        .get("transcript")?
        .as_str()
        .map(|s| s.trim().to_string())
}

// ---------------------------------------------------------------------------
// AssemblyAI (upload → create job → poll)
// ---------------------------------------------------------------------------

async fn transcribe_assemblyai(
    client: &Client,
    resolved: &ResolvedApi,
    wav: Vec<u8>,
    language: &str,
    timeout: Duration,
) -> Result<String, String> {
    let deadline = Instant::now() + timeout;
    // 1) Upload raw bytes.
    let upload = client
        .post(format!("{}/upload", resolved.base_url))
        .header("authorization", &resolved.api_key)
        .header("Content-Type", "application/octet-stream")
        .timeout(timeout)
        .body(wav)
        .send()
        .await
        .map_err(|e| format!("AssemblyAI: upload failed: {e}"))?;
    let status = upload.status();
    let body = upload.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format_http_error("AssemblyAI", status.as_u16(), &body));
    }
    let upload_url = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| v.get("upload_url").and_then(Value::as_str).map(str::to_string))
        .ok_or_else(|| "AssemblyAI: upload returned no upload_url.".to_string())?;

    // 2) Create the transcript job.
    let mut job_body = json!({ "audio_url": upload_url, "speech_model": resolved.model });
    let language = language.trim();
    if !language.is_empty() {
        job_body["language_code"] = json!(language);
    }
    let create = client
        .post(format!("{}/transcript", resolved.base_url))
        .header("authorization", &resolved.api_key)
        .json(&job_body)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| format!("AssemblyAI: create failed: {e}"))?;
    let status = create.status();
    let body = create.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format_http_error("AssemblyAI", status.as_u16(), &body));
    }
    let id = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| v.get("id").and_then(Value::as_str).map(str::to_string))
        .ok_or_else(|| "AssemblyAI: create returned no transcript id.".to_string())?;

    // 3) Poll until completed/error or the deadline passes.
    let poll_url = format!("{}/transcript/{}", resolved.base_url, id);
    loop {
        if Instant::now() >= deadline {
            return Err("AssemblyAI: timed out waiting for the transcript.".to_string());
        }
        tokio::time::sleep(Duration::from_millis(1200)).await;
        let poll = client
            .get(&poll_url)
            .header("authorization", &resolved.api_key)
            .timeout(Duration::from_secs(20))
            .send()
            .await
            .map_err(|e| format!("AssemblyAI: poll failed: {e}"))?;
        let status = poll.status();
        let body = poll.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(format_http_error("AssemblyAI", status.as_u16(), &body));
        }
        let value: Value = serde_json::from_str(&body)
            .map_err(|e| format!("AssemblyAI: bad JSON response: {e}"))?;
        match parse_assemblyai_poll(&value) {
            AssemblyPoll::Completed(text) => return Ok(text),
            AssemblyPoll::Error(msg) => return Err(format!("AssemblyAI: {msg}")),
            AssemblyPoll::Pending => continue,
        }
    }
}

/// The three terminal/non-terminal states of an AssemblyAI poll response.
#[derive(Debug, PartialEq, Eq)]
pub enum AssemblyPoll {
    Completed(String),
    Pending,
    Error(String),
}

/// Classifies an AssemblyAI `GET /transcript/{id}` body by its `status` field.
pub fn parse_assemblyai_poll(value: &Value) -> AssemblyPoll {
    match value.get("status").and_then(Value::as_str) {
        Some("completed") => AssemblyPoll::Completed(
            value
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string(),
        ),
        Some("error") => AssemblyPoll::Error(
            value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("transcription failed")
                .to_string(),
        ),
        _ => AssemblyPoll::Pending,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_transcript_reads_text() {
        let v = json!({ "text": "  hello world " });
        assert_eq!(parse_openai_transcript(&v).as_deref(), Some("hello world"));
    }

    #[test]
    fn openrouter_stt_body_is_json_base64_input_audio() {
        let b = build_openrouter_stt_body("openai/whisper-1", "QUJD", "en");
        assert_eq!(b["model"], "openai/whisper-1");
        assert_eq!(b["input_audio"]["data"], "QUJD");
        assert_eq!(b["input_audio"]["format"], "wav");
        assert_eq!(b["language"], "en");
        // language omitted when blank.
        let b2 = build_openrouter_stt_body("m", "QUJD", "");
        assert!(b2.get("language").is_none());
    }

    #[test]
    fn deepgram_transcript_digs_into_nested_shape() {
        let v = json!({
            "results": { "channels": [ { "alternatives": [ { "transcript": "open the door", "confidence": 0.98 } ] } ] }
        });
        assert_eq!(parse_deepgram_transcript(&v).as_deref(), Some("open the door"));
    }

    #[test]
    fn deepgram_missing_transcript_is_none() {
        let v = json!({ "results": { "channels": [] } });
        assert!(parse_deepgram_transcript(&v).is_none());
    }

    #[test]
    fn assemblyai_poll_states() {
        assert_eq!(
            parse_assemblyai_poll(&json!({ "status": "completed", "text": " done " })),
            AssemblyPoll::Completed("done".to_string())
        );
        assert_eq!(
            parse_assemblyai_poll(&json!({ "status": "queued" })),
            AssemblyPoll::Pending
        );
        assert_eq!(
            parse_assemblyai_poll(&json!({ "status": "error", "error": "bad audio" })),
            AssemblyPoll::Error("bad audio".to_string())
        );
    }
}
