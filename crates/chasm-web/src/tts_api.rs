//! Hosted-TTS provider adapters: turn a line of text into a WAV byte buffer that
//! chasm's synthesis path streams to the bridge as an `audio.chunk`.
//!
//! The managed-local engine (faster-qwen3-tts) streams raw PCM that chasm slices;
//! hosted providers instead return a whole clip per line, which we hand back as a
//! single WAV chunk (self-describing sample rate, no resampling needed):
//!
//!   * **ElevenLabs** — `POST {base}/text-to-speech/{voice}?output_format=pcm_24000`,
//!     `xi-api-key` header, body `{ text, model_id }`. Returns raw little-endian
//!     PCM16 @24 kHz mono, which we wrap into a WAV container here.
//!   * **OpenAI** — `POST {base}/audio/speech`, `Authorization: Bearer`, body
//!     `{ model, input, voice, response_format: "wav" }`. Returns a WAV directly.
//!   * **Cartesia** — `POST {base}/tts/bytes`, `X-API-Key` + `Cartesia-Version`,
//!     body selecting a WAV/pcm_s16le container. Returns a WAV directly.
//!
//! Request builders are pure + unit-tested; the async HTTP surfaces a readable
//! `Err(String)` for the UI.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use reqwest::Client;
use serde_json::{json, Value};

use chasm_core::ResolvedApi;

use crate::llm_api::format_http_error;

/// The `Cartesia-Version` date the request pins.
pub const CARTESIA_VERSION: &str = "2024-11-13";

/// ElevenLabs raw-PCM output sample rate (matches the `pcm_24000` output format).
pub const ELEVENLABS_PCM_RATE: u32 = 24_000;

/// Synthesizes `text` via the hosted TTS `provider`, returning a complete WAV
/// byte buffer. `resolved` carries base URL + key + model + voice (defaulted).
pub async fn synthesize(
    client: &Client,
    provider: &str,
    resolved: &ResolvedApi,
    text: &str,
) -> Result<Vec<u8>, String> {
    if resolved.api_key.is_empty() {
        return Err(format!("{provider}: no API key set (Settings → TTS)."));
    }
    match provider {
        "elevenlabs" => synthesize_elevenlabs(client, resolved, text).await,
        "cartesia" => synthesize_cartesia(client, resolved, text).await,
        "inworld" => synthesize_inworld(client, resolved, text).await,
        other => Err(format!("Unknown TTS provider '{other}'.")),
    }
}

// ---------------------------------------------------------------------------
// ElevenLabs
// ---------------------------------------------------------------------------

/// The ElevenLabs synthesis URL (PCM output so we control the container).
pub fn elevenlabs_url(base_url: &str, voice: &str) -> String {
    format!(
        "{}/text-to-speech/{}?output_format=pcm_24000",
        base_url.trim_end_matches('/'),
        voice
    )
}

/// The ElevenLabs request body.
pub fn build_elevenlabs_body(model: &str, text: &str) -> Value {
    json!({ "text": text, "model_id": model })
}

async fn synthesize_elevenlabs(
    client: &Client,
    resolved: &ResolvedApi,
    text: &str,
) -> Result<Vec<u8>, String> {
    if resolved.voice.is_empty() {
        return Err("ElevenLabs: no voice id set (Settings → TTS).".to_string());
    }
    let resp = client
        .post(elevenlabs_url(&resolved.base_url, &resolved.voice))
        .header("xi-api-key", &resolved.api_key)
        .header("Accept", "audio/pcm")
        .json(&build_elevenlabs_body(&resolved.model, text))
        .send()
        .await
        .map_err(|e| format!("ElevenLabs: request failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format_http_error("ElevenLabs", status.as_u16(), &body));
    }
    let pcm = resp
        .bytes()
        .await
        .map_err(|e| format!("ElevenLabs: reading audio failed: {e}"))?;
    Ok(pcm16_to_wav(&pcm, ELEVENLABS_PCM_RATE, 1))
}

// ---------------------------------------------------------------------------
// Inworld
// ---------------------------------------------------------------------------

/// Inworld TTS output sample rate (LINEAR16 WAV).
pub const INWORLD_SAMPLE_RATE: u32 = 24_000;

/// The Inworld `/tts/v1/voice` request body (LINEAR16 WAV output).
pub fn build_inworld_body(model: &str, voice: &str, text: &str) -> Value {
    json!({
        "text": text,
        "voiceId": voice,
        "modelId": model,
        "audio_config": { "audio_encoding": "LINEAR16", "sample_rate_hertz": INWORLD_SAMPLE_RATE }
    })
}

async fn synthesize_inworld(
    client: &Client,
    resolved: &ResolvedApi,
    text: &str,
) -> Result<Vec<u8>, String> {
    if resolved.voice.is_empty() {
        return Err("Inworld: no voice set (Settings → TTS).".to_string());
    }
    let resp = client
        .post(format!("{}/tts/v1/voice", resolved.base_url.trim_end_matches('/')))
        // Inworld runtime keys are Base64 and used with Basic auth.
        .header("Authorization", format!("Basic {}", resolved.api_key))
        .json(&build_inworld_body(&resolved.model, &resolved.voice, text))
        .send()
        .await
        .map_err(|e| format!("Inworld: request failed: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format_http_error("Inworld", status.as_u16(), &body));
    }
    let value: Value =
        serde_json::from_str(&body).map_err(|e| format!("Inworld: bad JSON response: {e}"))?;
    let b64 = value
        .get("audioContent")
        .and_then(Value::as_str)
        .ok_or_else(|| "Inworld: response had no audioContent.".to_string())?;
    STANDARD
        .decode(b64.as_bytes())
        .map_err(|e| format!("Inworld: audio decode failed: {e}"))
}

// ---------------------------------------------------------------------------
// Cartesia
// ---------------------------------------------------------------------------

/// The Cartesia `/tts/bytes` request body (WAV / pcm_s16le @24 kHz).
pub fn build_cartesia_body(model: &str, voice: &str, text: &str) -> Value {
    json!({
        "model_id": model,
        "transcript": text,
        "voice": { "mode": "id", "id": voice },
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 24_000
        }
    })
}

async fn synthesize_cartesia(
    client: &Client,
    resolved: &ResolvedApi,
    text: &str,
) -> Result<Vec<u8>, String> {
    if resolved.voice.is_empty() {
        return Err("Cartesia: no voice id set (Settings → TTS).".to_string());
    }
    let resp = client
        .post(format!("{}/tts/bytes", resolved.base_url))
        .header("X-API-Key", &resolved.api_key)
        .header("Cartesia-Version", CARTESIA_VERSION)
        .json(&build_cartesia_body(&resolved.model, &resolved.voice, text))
        .send()
        .await
        .map_err(|e| format!("Cartesia: request failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format_http_error("Cartesia", status.as_u16(), &body));
    }
    let wav = resp
        .bytes()
        .await
        .map_err(|e| format!("Cartesia: reading audio failed: {e}"))?;
    Ok(wav.to_vec())
}

// ---------------------------------------------------------------------------
// Voice cloning (clone a character's sample -> a provider voice id)
// ---------------------------------------------------------------------------

/// Clones the voice in `sample_wav` (a complete WAV byte buffer) into the hosted
/// TTS `provider`, returning the provider's new **voice id**. chasm stores that id
/// per character and passes it as the voice on subsequent synthesis. All hosted
/// TTS providers chasm offers support API cloning.
pub async fn clone_voice(
    client: &Client,
    provider: &str,
    resolved: &ResolvedApi,
    display_name: &str,
    sample_wav: Vec<u8>,
) -> Result<String, String> {
    if resolved.api_key.is_empty() {
        return Err(format!("{provider}: no API key set (Settings → TTS)."));
    }
    match provider {
        "elevenlabs" => clone_elevenlabs(client, resolved, display_name, sample_wav).await,
        "cartesia" => clone_cartesia(client, resolved, display_name, sample_wav).await,
        "inworld" => clone_inworld(client, resolved, display_name, sample_wav).await,
        other => Err(format!("Voice cloning is not supported for '{other}'.")),
    }
}

fn wav_part(sample_wav: Vec<u8>, field_hint: &str) -> Result<reqwest::multipart::Part, String> {
    reqwest::multipart::Part::bytes(sample_wav)
        .file_name("sample.wav")
        .mime_str("audio/wav")
        .map_err(|e| format!("{field_hint}: building audio part failed: {e}"))
}

/// ElevenLabs instant voice clone: multipart `POST {base}/voices/add`
/// (`name` + `files`), returns `{ "voice_id": "…" }`.
async fn clone_elevenlabs(
    client: &Client,
    resolved: &ResolvedApi,
    display_name: &str,
    sample_wav: Vec<u8>,
) -> Result<String, String> {
    let form = reqwest::multipart::Form::new()
        .text("name", display_name.to_string())
        .part("files", wav_part(sample_wav, "ElevenLabs")?);
    let resp = client
        .post(format!("{}/voices/add", resolved.base_url.trim_end_matches('/')))
        .header("xi-api-key", &resolved.api_key)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("ElevenLabs clone: request failed: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format_http_error("ElevenLabs clone", status.as_u16(), &body));
    }
    parse_clone_voice_id("elevenlabs", &body)
}

/// Cartesia voice clone: multipart `POST {base}/voices/clone` (`clip` + `name`),
/// returns `{ "id": "…" }`.
async fn clone_cartesia(
    client: &Client,
    resolved: &ResolvedApi,
    display_name: &str,
    sample_wav: Vec<u8>,
) -> Result<String, String> {
    let form = reqwest::multipart::Form::new()
        .text("name", display_name.to_string())
        .text("mode", "similarity")
        .text("language", "en")
        .part("clip", wav_part(sample_wav, "Cartesia")?);
    let resp = client
        .post(format!("{}/voices/clone", resolved.base_url.trim_end_matches('/')))
        .header("X-API-Key", &resolved.api_key)
        .header("Cartesia-Version", CARTESIA_VERSION)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("Cartesia clone: request failed: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format_http_error("Cartesia clone", status.as_u16(), &body));
    }
    parse_clone_voice_id("cartesia", &body)
}

/// The Inworld clone request body (base64 WAV sample).
pub fn build_inworld_clone_body(display_name: &str, sample_b64: &str) -> Value {
    json!({
        "displayName": display_name,
        "langCode": "en",
        "voiceSamples": [ { "audio": sample_b64 } ]
    })
}

/// Inworld zero-shot clone: `POST {base}/voices/v1/voices:clone` (base64 sample),
/// returns `{ "voiceId": "…" }` (or a `name` path we take the last segment of).
async fn clone_inworld(
    client: &Client,
    resolved: &ResolvedApi,
    display_name: &str,
    sample_wav: Vec<u8>,
) -> Result<String, String> {
    let sample_b64 = STANDARD.encode(&sample_wav);
    let resp = client
        .post(format!(
            "{}/voices/v1/voices:clone",
            resolved.base_url.trim_end_matches('/')
        ))
        .header("Authorization", format!("Basic {}", resolved.api_key))
        .json(&build_inworld_clone_body(display_name, &sample_b64))
        .send()
        .await
        .map_err(|e| format!("Inworld clone: request failed: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format_http_error("Inworld clone", status.as_u16(), &body));
    }
    parse_clone_voice_id("inworld", &body)
}

/// Extracts the created voice id from a clone response, tolerant of the three
/// providers' field names (`voice_id` / `id` / `voiceId`, or an Inworld `name`
/// path whose last `/`-segment is the id).
pub fn parse_clone_voice_id(provider: &str, body: &str) -> Result<String, String> {
    let value: Value = serde_json::from_str(body)
        .map_err(|e| format!("{provider} clone: bad JSON response: {e}"))?;
    let id = value
        .get("voice_id")
        .or_else(|| value.get("id"))
        .or_else(|| value.get("voiceId"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            value
                .get("name")
                .and_then(Value::as_str)
                .and_then(|n| n.rsplit('/').next())
                .map(str::to_string)
        });
    id.filter(|s| !s.is_empty())
        .ok_or_else(|| format!("{provider} clone: response had no voice id."))
}

// ---------------------------------------------------------------------------
// PCM → WAV
// ---------------------------------------------------------------------------

/// Wraps raw little-endian PCM16 samples in a minimal WAV (RIFF) container.
/// Self-contained so the hosted-TTS path doesn't depend on the streaming helper.
pub fn pcm16_to_wav(pcm: &[u8], sample_rate: u32, channels: u16) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample) / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_len = pcm.len() as u32;
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elevenlabs_url_and_body() {
        assert_eq!(
            elevenlabs_url("https://api.elevenlabs.io/v1/", "voiceX"),
            "https://api.elevenlabs.io/v1/text-to-speech/voiceX?output_format=pcm_24000"
        );
        let b = build_elevenlabs_body("eleven_turbo_v2_5", "hello");
        assert_eq!(b["text"], "hello");
        assert_eq!(b["model_id"], "eleven_turbo_v2_5");
    }

    #[test]
    fn inworld_body_and_clone_parse() {
        let b = build_inworld_body("inworld-tts-1", "Ashley", "hi");
        assert_eq!(b["voiceId"], "Ashley");
        assert_eq!(b["text"], "hi");
        assert_eq!(b["audio_config"]["audio_encoding"], "LINEAR16");
        let cb = build_inworld_clone_body("Easy Pete", "QUJD");
        assert_eq!(cb["displayName"], "Easy Pete");
        assert_eq!(cb["voiceSamples"][0]["audio"], "QUJD");
    }

    #[test]
    fn clone_voice_id_parses_all_provider_shapes() {
        assert_eq!(
            parse_clone_voice_id("elevenlabs", r#"{"voice_id":"abc123"}"#).unwrap(),
            "abc123"
        );
        assert_eq!(
            parse_clone_voice_id("cartesia", r#"{"id":"car_9","name":"Pete"}"#).unwrap(),
            "car_9"
        );
        assert_eq!(
            parse_clone_voice_id("inworld", r#"{"voiceId":"iw_42"}"#).unwrap(),
            "iw_42"
        );
        // Inworld resource-name fallback: take the last path segment.
        assert_eq!(
            parse_clone_voice_id("inworld", r#"{"name":"workspaces/x/voices/iw_77"}"#).unwrap(),
            "iw_77"
        );
        assert!(parse_clone_voice_id("elevenlabs", r#"{"error":"bad"}"#).is_err());
    }

    #[test]
    fn cartesia_body_selects_wav_container() {
        let b = build_cartesia_body("sonic-2", "vid", "hey");
        assert_eq!(b["voice"]["mode"], "id");
        assert_eq!(b["voice"]["id"], "vid");
        assert_eq!(b["output_format"]["container"], "wav");
        assert_eq!(b["output_format"]["sample_rate"], 24_000);
    }

    #[test]
    fn pcm_wrap_has_valid_riff_header_and_sizes() {
        let pcm = vec![0u8; 8];
        let wav = pcm16_to_wav(&pcm, 24_000, 1);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
        // RIFF size = 36 + data; data size = 8.
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 36 + 8);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 8);
        assert_eq!(wav.len(), 44 + 8);
        // sample rate at offset 24.
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
    }
}
