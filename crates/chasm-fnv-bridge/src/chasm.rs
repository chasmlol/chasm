//! HTTP client for chasm's `/api/headless/v1` surface — the "talk to chasm" half
//! of the bridge. Mirrors the Node helper's `apiFetch`/`api`/`streamApi`:
//! `Bearer` auth, per-capability base routing (`/speech*` → TTS/STT base), and
//! line-by-line NDJSON streaming.
//!
//! The `ChasmClient` trait is the seam between the bridge ("FNV glue") and chasm
//! ("the AI"). `HttpChasmClient` (this file) is the reqwest impl used by the
//! standalone bin; Section 7's in-process impl (in chasm-web) implements the same
//! trait, so the bridge can fold in and drop the localhost hop.

use std::pin::Pin;
use std::time::Duration;

use anyhow::{anyhow, Context};
use async_trait::async_trait;
use base64::Engine;
use futures_util::{Stream, StreamExt};
use reqwest::{Client, Method, StatusCode};
use serde_json::Value;

use crate::config::BridgeConfig;

/// One streamed TTS audio chunk from `/speech/synthesize/stream`.
#[derive(Debug, Clone, Default)]
pub struct AudioChunk {
    pub index: Option<i64>,
    pub audio: Vec<u8>,
    pub mime_type: String,
    pub text: Option<String>,
    pub caption_max_chars: Option<i64>,
}

/// A queued "play a song" job, handed to the client when the play-a-song action
/// fires during a turn. The client runs it FIRE-AND-FORGET (lyrics via the
/// character's prompt stack -> ACE-Step -> store -> deliver to the mod) so the turn
/// pipeline is never blocked; a failure is logged, never surfaced to the turn.
/// `bridge_roots` is where the delivery (`control/songs/<id>.json`) is written for
/// the mod to pick up — the same roots the bridge writes its other control files to.
#[derive(Debug, Clone)]
pub struct SongJob {
    pub request_id: String,
    pub live_chat_id: String,
    pub character_id: String,
    pub character_name: String,
    pub npc_key: String,
    pub npc_name: String,
    /// The player's triggering words (what the song should be about).
    pub user_message: String,
    /// Genre steer for the song: `""` for the default (sung, guitar) song, or
    /// `"rap"` for the rap variant. Selects the base style tags + tweaks the lyric
    /// prompt so the same pipeline produces a rap instead of a folk song.
    pub style_hint: String,
    pub bridge_roots: Vec<std::path::PathBuf>,
}

/// The seam between the FNV bridge and chasm. `HttpChasmClient` backs the
/// standalone bin (reqwest → :7341); an in-process impl backs the folded-in build.
/// `&dyn ChasmClient` is threaded through the bridge so either can be swapped in.
#[async_trait]
pub trait ChasmClient: Send + Sync {
    async fn live_chat_exists(&self, id: &str) -> anyhow::Result<bool>;
    async fn create_live_chat(&self, body: &Value) -> anyhow::Result<()>;
    async fn presence(&self, id: &str, body: &Value) -> anyhow::Result<()>;
    async fn recognize(&self, body: &Value) -> anyhow::Result<String>;
    async fn generate_headless(&self, body: &Value) -> anyhow::Result<Value>;
    async fn save_sync_event(&self, body: &Value) -> anyhow::Result<Value>;
    /// Streamed NPC turn — yields each NDJSON event. Boxed so the trait stays
    /// dyn-compatible (each impl's concrete stream type is private).
    fn generate_stream_events<'a>(
        &'a self,
        id: &str,
        body: &Value,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<Value>> + Send + 'a>>;
    /// Streamed TTS — invokes `on_chunk` per audio chunk; returns the count. A
    /// `&mut dyn FnMut` (not a generic param) keeps the trait dyn-compatible.
    async fn synthesize_stream(
        &self,
        body: &Value,
        on_chunk: &mut (dyn FnMut(AudioChunk) -> anyhow::Result<()> + Send),
    ) -> anyhow::Result<usize>;

    /// Kick off an async song-generation job (the play-a-song action). Fire-and-
    /// forget: returns immediately so the turn is never blocked; the client spawns
    /// the lyrics -> ACE-Step -> store -> deliver work and logs any failure.
    /// Default: a no-op — music generation runs only in the in-process build
    /// (chasm-web), not the standalone HTTP bin / test mocks.
    fn start_song_job(&self, _job: SongJob) {}
}

pub struct HttpChasmClient {
    client: Client,
    api_base: String,
    tts_api_base: String,
    stt_api_base: String,
    api_key: String,
}

impl HttpChasmClient {
    pub fn new(config: &BridgeConfig) -> anyhow::Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_millis(config.request_timeout_ms))
            .build()
            .context("building reqwest client")?;
        Ok(Self {
            client,
            api_base: config.api_base.clone(),
            tts_api_base: config.tts_api_base.clone(),
            stt_api_base: config.stt_api_base.clone(),
            api_key: config.api_key.clone(),
        })
    }

    fn base_for(&self, endpoint: &str) -> &str {
        if endpoint == "/speech/recognize" {
            if !self.stt_api_base.is_empty() {
                return &self.stt_api_base;
            }
        } else if endpoint.starts_with("/speech") && !self.tts_api_base.is_empty() {
            return &self.tts_api_base;
        }
        &self.api_base
    }

    fn request(&self, method: Method, endpoint: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_for(endpoint), endpoint);
        let mut builder = self.client.request(method, url);
        if !self.api_key.is_empty() {
            builder = builder.bearer_auth(&self.api_key);
        }
        builder
    }

    /// GET probe: `Ok(true)` exists (2xx), `Ok(false)` on 404, else `Err`.
    pub async fn live_chat_exists(&self, id: &str) -> anyhow::Result<bool> {
        let endpoint = format!("/live-chats/{}", encode(id));
        let resp = self
            .request(Method::GET, &endpoint)
            .send()
            .await
            .with_context(|| format!("GET {endpoint}"))?;
        if resp.status() == StatusCode::NOT_FOUND {
            return Ok(false);
        }
        ensure_ok(resp).await?;
        Ok(true)
    }

    pub async fn create_live_chat(&self, body: &Value) -> anyhow::Result<()> {
        self.post_json("/live-chats", body).await.map(|_| ())
    }

    pub async fn presence(&self, id: &str, body: &Value) -> anyhow::Result<()> {
        let endpoint = format!("/live-chats/{}/presence", encode(id));
        self.post_json(&endpoint, body).await.map(|_| ())
    }

    /// `POST /speech/recognize` (STT) → the transcript text. chasm errors on empty,
    /// so a non-empty `Ok` is a real transcript.
    pub async fn recognize(&self, body: &Value) -> anyhow::Result<String> {
        let result = self.post_json("/speech/recognize", body).await?;
        Ok(result
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string())
    }

    /// Streamed NPC turn: `POST /live-chats/:id/generate/stream`, yielding each
    /// NDJSON event (`live.start`/`speaker.start`/`speech.delta`/`live.completed`/
    /// `live.error`) as it arrives, so the caller can synthesize early segments
    /// while the rest of the line is still generating. The returned stream owns its
    /// request, so the caller may freely call other methods (e.g. `synthesize_stream`)
    /// on the same client while consuming it.
    pub fn generate_stream_events<'a>(
        &'a self,
        id: &str,
        body: &Value,
    ) -> impl Stream<Item = anyhow::Result<Value>> + 'a {
        let endpoint = format!("/live-chats/{}/generate/stream", encode(id));
        let builder = self.request(Method::POST, &endpoint).json(body);
        async_stream::try_stream! {
            let resp = builder.send().await.with_context(|| format!("POST {endpoint}"))?;
            let resp = ensure_ok(resp).await?;
            let mut bytes = resp.bytes_stream();
            let mut buf: Vec<u8> = Vec::new();
            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.with_context(|| format!("reading {endpoint} stream"))?;
                buf.extend_from_slice(&chunk);
                while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    if let Some(value) = parse_ndjson_line(&line)? {
                        yield value;
                    }
                }
            }
            if let Some(value) = parse_ndjson_line(&buf)? {
                yield value;
            }
        }
    }

    /// Buffered single-character turn: `POST /generate` (the admin/Todd path) → turn.
    pub async fn generate_headless(&self, body: &Value) -> anyhow::Result<Value> {
        self.post_json("/generate", body).await
    }

    /// `POST /save-sync/events`: checkpoint/restore on game save/load.
    pub async fn save_sync_event(&self, body: &Value) -> anyhow::Result<Value> {
        self.post_json("/save-sync/events", body).await
    }

    async fn post_json(&self, endpoint: &str, body: &Value) -> anyhow::Result<Value> {
        let resp = self
            .request(Method::POST, endpoint)
            .json(body)
            .send()
            .await
            .with_context(|| format!("POST {endpoint}"))?;
        let resp = ensure_ok(resp).await?;
        let text = resp.text().await.unwrap_or_default();
        Ok(if text.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&text).with_context(|| format!("parsing {endpoint} response"))?
        })
    }

    /// `POST /speech/synthesize/stream`, invoking `on_chunk` for each `audio.chunk`
    /// as it arrives. Returns the chunk count. A `speech.error` event aborts.
    pub async fn synthesize_stream<F>(
        &self,
        body: &Value,
        mut on_chunk: F,
    ) -> anyhow::Result<usize>
    where
        F: FnMut(AudioChunk) -> anyhow::Result<()>,
    {
        let mut count = 0usize;
        self.stream_ndjson("/speech/synthesize/stream", body, |event| {
            match event.get("type").and_then(Value::as_str) {
                Some("speech.error") => {
                    let msg = event
                        .pointer("/error/message")
                        .and_then(Value::as_str)
                        .unwrap_or("Chasm speech streaming failed");
                    return Err(anyhow!("{msg}"));
                }
                Some("audio.chunk") => {
                    let Some(b64) = event.pointer("/audio/data").and_then(Value::as_str) else {
                        return Ok(());
                    };
                    if b64.is_empty() {
                        return Ok(());
                    }
                    let audio = base64::engine::general_purpose::STANDARD
                        .decode(b64)
                        .context("decoding audio.chunk base64")?;
                    on_chunk(AudioChunk {
                        index: event.get("index").and_then(Value::as_i64),
                        audio,
                        mime_type: event
                            .get("mimeType")
                            .and_then(Value::as_str)
                            .unwrap_or("audio/wav")
                            .to_string(),
                        text: event.get("text").and_then(Value::as_str).map(str::to_string),
                        caption_max_chars: event.get("captionMaxChars").and_then(Value::as_i64),
                    })?;
                    count += 1;
                }
                _ => {}
            }
            Ok(())
        })
        .await?;
        Ok(count)
    }

    /// `POST` an NDJSON-streaming endpoint, calling `on_event` per parsed line.
    /// Buffers bytes and only decodes complete lines, so multi-byte UTF-8 split
    /// across network chunks is never corrupted.
    pub async fn stream_ndjson<F>(
        &self,
        endpoint: &str,
        body: &Value,
        mut on_event: F,
    ) -> anyhow::Result<()>
    where
        F: FnMut(Value) -> anyhow::Result<()>,
    {
        let resp = self
            .request(Method::POST, endpoint)
            .json(body)
            .send()
            .await
            .with_context(|| format!("POST {endpoint}"))?;
        let resp = ensure_ok(resp).await?;

        let mut stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            let bytes = chunk.with_context(|| format!("reading {endpoint} stream"))?;
            buf.extend_from_slice(&bytes);
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                emit_line(&line, &mut on_event)?;
            }
        }
        if !buf.is_empty() {
            emit_line(&buf, &mut on_event)?;
        }
        Ok(())
    }
}

/// `HttpChasmClient` satisfies the seam by delegating to its inherent methods
/// (inherent method resolution wins over the trait, so these don't recurse). The
/// only shape changes are the two streaming methods: the NDJSON stream is boxed
/// and the TTS callback becomes `&mut dyn FnMut` — both to keep `dyn` dispatch.
#[async_trait]
impl ChasmClient for HttpChasmClient {
    async fn live_chat_exists(&self, id: &str) -> anyhow::Result<bool> {
        self.live_chat_exists(id).await
    }
    async fn create_live_chat(&self, body: &Value) -> anyhow::Result<()> {
        self.create_live_chat(body).await
    }
    async fn presence(&self, id: &str, body: &Value) -> anyhow::Result<()> {
        self.presence(id, body).await
    }
    async fn recognize(&self, body: &Value) -> anyhow::Result<String> {
        self.recognize(body).await
    }
    async fn generate_headless(&self, body: &Value) -> anyhow::Result<Value> {
        self.generate_headless(body).await
    }
    async fn save_sync_event(&self, body: &Value) -> anyhow::Result<Value> {
        self.save_sync_event(body).await
    }
    fn generate_stream_events<'a>(
        &'a self,
        id: &str,
        body: &Value,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<Value>> + Send + 'a>> {
        Box::pin(self.generate_stream_events(id, body))
    }
    async fn synthesize_stream(
        &self,
        body: &Value,
        on_chunk: &mut (dyn FnMut(AudioChunk) -> anyhow::Result<()> + Send),
    ) -> anyhow::Result<usize> {
        self.synthesize_stream(body, &mut *on_chunk).await
    }
}

/// Parse one buffered NDJSON line (a complete line is valid UTF-8). `None` for a
/// blank line.
fn parse_ndjson_line(bytes: &[u8]) -> anyhow::Result<Option<Value>> {
    let line = String::from_utf8_lossy(bytes);
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(line).context("parsing NDJSON line")?))
}

fn emit_line<F>(bytes: &[u8], on_event: &mut F) -> anyhow::Result<()>
where
    F: FnMut(Value) -> anyhow::Result<()>,
{
    if let Some(event) = parse_ndjson_line(bytes)? {
        on_event(event)?;
    }
    Ok(())
}

/// Mirror the Node client's error surfacing: pull `body.error.message`, else the
/// raw body, else the status.
async fn ensure_ok(resp: reqwest::Response) -> anyhow::Result<reqwest::Response> {
    if resp.status().is_success() {
        return Ok(resp);
    }
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    let message = serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|v| {
            v.pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| {
            if text.trim().is_empty() {
                format!("HTTP {status}")
            } else {
                text.trim().to_string()
            }
        });
    Err(anyhow!("HTTP {status}: {message}"))
}

/// `encodeURIComponent`-equivalent for path segments (live-chat ids).
fn encode(s: &str) -> String {
    urlencoding::encode(s).into_owned()
}
