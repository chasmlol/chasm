//! In-process FNV bridge (Section 7 — the fold). The `chasm-fnv-bridge`
//! crate's loop runs as a tokio task INSIDE chasm, talking to chasm through an
//! [`InProcessChasmClient`] that calls the request handlers/cores directly — no
//! Node helper, no `127.0.0.1:7341` socket. The `/api/headless/v1` HTTP surface is
//! untouched (the handlers stay; this just reaches their shared cores in-process),
//! so the standalone bin + the C++ mod's HTTP contract still work.
//!
//! Gated by the `CHASM_FNV_BRIDGE` env var (off by default until proven);
//! when on, `router()` spawns [`spawn_in_process`] and the launcher skips the Node
//! helper.

use std::pin::Pin;
use std::sync::Arc;

use anyhow::anyhow;
use async_trait::async_trait;
use axum::{
    extract::{Path, State},
    http::HeaderMap,
    Json,
};
use base64::Engine;
use futures_util::{Stream, StreamExt};
use serde_json::Value;
use chasm_core::AppSettings;
use chasm_fnv_bridge::chasm::{AudioChunk, ChasmClient};

use crate::{AppState, WebError};

/// Whether the in-process bridge is enabled (`CHASM_FNV_BRIDGE=1|true|on`).
/// Read by `router()` (to spawn it) and the launcher (to skip the Node helper).
pub(crate) fn in_process_enabled() -> bool {
    std::env::var("CHASM_FNV_BRIDGE")
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        })
        .unwrap_or(false)
}

/// Load the bridge config (same `nvbridge.config.json` the Node helper/standalone
/// bin use) and run the bridge loop in-process until chasm shuts down. Best-effort:
/// a missing config or a fatal loop error is logged, not propagated (chasm keeps
/// serving HTTP regardless).
pub async fn spawn_in_process(state: Arc<AppState>) {
    let settings = AppSettings::load(&state.config.settings_path);
    // Blank helper config = the default for a fresh install. `load_config` then runs
    // on built-in defaults pointed at the fixed rendezvous dir, so a standalone chasm
    // with no `nvbridge.config.json` still connects — no developer path involved.
    let config_path = settings.launcher.helper_config.trim().to_string();
    let mut config = match chasm_fnv_bridge::load_config(std::path::Path::new(&config_path)) {
        Ok(config) => config,
        Err(error) => {
            // Present-but-malformed config: warn and fall back to defaults rather
            // than refusing to run the bridge at all.
            tracing::warn!("FNV bridge: {config_path} unreadable ({error}); using defaults");
            chasm_fnv_bridge::default_config()
        }
    };

    // With no explicit NPC→character map (the default for a fresh install), map
    // nearby in-game NPCs to the active profile's characters by name, so a turn
    // resolves instead of failing with "no mapped NPC".
    enrich_config_from_profile(&mut config, &state);

    // Music generation is in-process only; surface the live setting to the bridge
    // loop so the play-a-song action only starts a job when the user enabled it.
    config.music_enabled = settings.music.enabled;

    // Ensure every rendezvous root exists before the loop attaches its file watcher,
    // so the plugin's heartbeat/request writes land where chasm reads them. Both
    // sides compute this same absolute path (%LOCALAPPDATA%\chasm\bridge by default).
    for root in &config.native_bridge_roots {
        if let Err(error) = std::fs::create_dir_all(root) {
            tracing::warn!("FNV bridge: could not create {}: {error}", root.display());
        }
    }
    let rendezvous = config
        .native_bridge_roots
        .first()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    tracing::info!("FNV bridge: starting in-process (rendezvous at {rendezvous})");

    let client: Arc<dyn ChasmClient> = Arc::new(InProcessChasmClient::new(state));
    // `force = true`: steal a stale lock — chasm owns the bridge now, so any
    // leftover lock from a prior run (or a crashed standalone bin) is ours to take.
    if let Err(error) = chasm_fnv_bridge::run_with_client(config, true, client).await {
        tracing::error!("FNV bridge exited with error: {error}");
    }
}

/// Fills in the FNV-specific bridge config a fresh install lacks (it used to live in
/// a developer-only helper config), so a turn resolves without one:
///   * `group_id` ← `live_chat_id` when unset, so the bridge can create the group
///     Live Chat on first turn (chasm's `create_live_chat` builds the group from the
///     participants, so any non-empty id works — it doesn't need a group file).
///   * `npc_character_map` ← the ACTIVE profile's character cards when unset, so
///     nearby in-game NPCs resolve to profile characters. Keyed by the card id
///     (= card filename = NPC name); the bridge's slug matcher tolerates case/spacing.
/// Anything already set (a developer's helper config) is left untouched.
pub(crate) fn enrich_config_from_profile(
    config: &mut chasm_fnv_bridge::BridgeConfig,
    state: &AppState,
) {
    if config.group_id.trim().is_empty() {
        config.group_id = config.live_chat_id.clone();
    }

    if config.npc_character_map.is_empty() {
        match state.repository.list_character_cards() {
            Ok(cards) => {
                for card in cards {
                    config.npc_character_map.insert(
                        card.id.clone(),
                        serde_json::json!({ "characterId": card.id, "characterName": card.name }),
                    );
                }
                tracing::info!(
                    "FNV bridge: built NPC map from profile ({} character(s))",
                    config.npc_character_map.len()
                );
            }
            Err(error) => {
                tracing::warn!(
                    "FNV bridge: could not list profile characters for NPC map: {error}"
                );
            }
        }
    }
}

/// A [`ChasmClient`] that calls chasm's own handlers/cores on a shared
/// [`AppState`], so the folded-in bridge skips the localhost HTTP hop. Non-stream
/// ops call the axum handlers with constructed extractors; the two streaming ops
/// call the extracted `*_core` fns and parse their NDJSON lines back to values.
pub struct InProcessChasmClient {
    state: Arc<AppState>,
}

impl InProcessChasmClient {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

/// `WebError` wraps an `anyhow::Error`; unwrap it for the bridge's `anyhow` results.
fn unwrap_web(error: WebError) -> anyhow::Error {
    error.0
}

#[async_trait]
impl ChasmClient for InProcessChasmClient {
    async fn live_chat_exists(&self, id: &str) -> anyhow::Result<bool> {
        // A direct repository read — cheaper and clearer than calling the handler
        // and decoding a 404 from the error.
        Ok(self.state.repository.get_live_chat(id).is_ok())
    }

    async fn create_live_chat(&self, body: &Value) -> anyhow::Result<()> {
        crate::generate::create_live_chat(State(self.state.clone()), Json(body.clone()))
            .await
            .map(|_| ())
            .map_err(unwrap_web)
    }

    async fn presence(&self, id: &str, body: &Value) -> anyhow::Result<()> {
        crate::generate::update_presence(
            State(self.state.clone()),
            Path(id.to_string()),
            Json(body.clone()),
        )
        .await
        .map(|_| ())
        .map_err(unwrap_web)
    }

    async fn recognize(&self, body: &Value) -> anyhow::Result<String> {
        let Json(result) = crate::speech_recognize(State(self.state.clone()), Json(body.clone()))
            .await
            .map_err(unwrap_web)?;
        Ok(result
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string())
    }

    async fn generate_headless(&self, body: &Value) -> anyhow::Result<Value> {
        let Json(result) = crate::generate::generate_headless(
            State(self.state.clone()),
            HeaderMap::new(),
            Json(body.clone()),
        )
        .await
        .map_err(unwrap_web)?;
        Ok(result)
    }

    async fn save_sync_event(&self, body: &Value) -> anyhow::Result<Value> {
        let Json(result) =
            crate::save_sync::save_sync_event(State(self.state.clone()), Json(body.clone()))
                .await
                .map_err(unwrap_web)?;
        Ok(result)
    }

    async fn event_log_ingest(&self, body: &Value) -> anyhow::Result<Value> {
        let Json(result) =
            crate::event_log::ingest_events(State(self.state.clone()), Json(body.clone()))
                .await
                .map_err(unwrap_web)?;
        Ok(result)
    }

    fn generate_stream_events<'a>(
        &'a self,
        id: &str,
        body: &Value,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<Value>> + Send + 'a>> {
        // Capture owned inputs so the stream borrows nothing (it's effectively
        // 'static, satisfying the `'a` bound).
        let state = self.state.clone();
        let id = id.to_string();
        let body = body.clone();
        Box::pin(async_stream::try_stream! {
            let stream = crate::generate::generate_stream_core(state, id, body, None)
                .await
                .map_err(unwrap_web)?;
            futures_util::pin_mut!(stream);
            while let Some(line) = stream.next().await {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                yield serde_json::from_str::<Value>(trimmed)?;
            }
        })
    }

    fn start_song_job(&self, job: chasm_fnv_bridge::chasm::SongJob) {
        // Fire-and-forget: spawn the lyrics -> ACE-Step -> store -> deliver pipeline
        // so the turn is never blocked. Failures are logged inside the job.
        crate::music::spawn_song_job(self.state.clone(), job);
    }

    fn schedule_task(&self, spec: Value) {
        // Fire-and-forget: parse + persist the scheduled task synchronously (a
        // cheap file write), so the turn is never blocked. A failure is logged and
        // does not disturb the turn.
        if let Err(error) = crate::scheduler::schedule_from_spec(&self.state, &spec) {
            tracing::warn!("scheduler: failed to schedule task: {error}");
        }
    }

    async fn synthesize_stream(
        &self,
        body: &Value,
        on_chunk: &mut (dyn FnMut(AudioChunk) -> anyhow::Result<()> + Send),
    ) -> anyhow::Result<usize> {
        let stream = crate::speech_synthesize_stream_core(self.state.clone(), body.clone());
        futures_util::pin_mut!(stream);
        let mut count = 0usize;
        while let Some(line) = stream.next().await {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let event: Value = serde_json::from_str(trimmed)?;
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
                        continue;
                    };
                    if b64.is_empty() {
                        continue;
                    }
                    let audio = base64::engine::general_purpose::STANDARD
                        .decode(b64)
                        .map_err(|error| anyhow!("decoding audio.chunk base64: {error}"))?;
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
        }
        Ok(count)
    }
}
