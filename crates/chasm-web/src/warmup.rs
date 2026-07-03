//! Connection-time warm-up of the AI stack, so the FIRST in-game NPC line is as
//! fast as every later one.
//!
//! Why this exists: the stack is torn down when the game exits and respawned on
//! the next connect ([`crate::stack_lifecycle`]), so every game session starts
//! with cold runtimes even once their models are "loaded":
//!
//! * **TTS (faster-qwen3-tts)** loads its weights at spawn, but the first
//!   `/v1/audio/speech` request still pays CUDA-graph capture / kernel warm-up
//!   (multi-second — the synth client even has a retry loop for it). Turn 1 paid
//!   this; now a discarded warm-up utterance does.
//! * **LLM (koboldcpp)** keeps a per-slot KV cache (`cache_prompt: true`), so
//!   turns 2+ fast-forward over the unchanged prompt prefix while turn 1
//!   ingested the whole system prompt + history cold. Priming a 1-token
//!   generation with the REAL first-turn prompt prefix (same live chat, same
//!   first eligible speaker) pre-fills that cache.
//! * **Retrieval** (`chasm-embed`) lazily loads two ONNX models; the embedder was
//!   already warmed at connect, but the reranker's first inference wasn't.
//! * **STT (koboldcpp Whisper)** pays its first-decode warm-up on the first
//!   push-to-talk line; a short silent clip absorbs it.
//!
//! Design constraints (all deliberate):
//! * **Non-blocking**: runs as a spawned task; readiness signaling and turn
//!   intake never wait on it. A real turn arriving mid-warm-up is at worst
//!   queued behind the same work it would have had to do itself (1-token
//!   generation / the same CUDA-graph capture), and then REUSES it.
//! * **No user-visible output**: nothing is persisted to any session, no audio
//!   chunk is ever written to the bridge outbox — responses are discarded.
//! * **Idempotent per connect**: the lifecycle task fires it once per
//!   connect edge, and an in-flight guard makes overlapping calls (e.g. the
//!   manual `/api/stack/start` endpoint) a logged no-op.
//! * **Best-effort**: every step degrades to a logged skip; a machine without a
//!   TTS engine or LLM still connects exactly as before.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chasm_core::AppSettings;
use serde_json::json;

use crate::AppState;

/// How long we wait for a runtime's port to come up before skipping its warm-up.
/// koboldcpp takes ~10-20 s to load weights, the TTS server ~45 s on a slow disk;
/// these run in a background task so a generous ceiling costs nothing.
const READY_TIMEOUT: Duration = Duration::from_secs(240);
/// Poll interval while waiting for a runtime to come up.
const READY_POLL: Duration = Duration::from_secs(1);
/// Per-request deadline for the warm-up calls themselves. First-inference work
/// (prompt ingestion, CUDA graph capture) can legitimately take tens of seconds.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Timeouts for [`run_stack_warmup_with`], injectable so tests don't wait minutes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WarmupDeadlines {
    pub ready_timeout: Duration,
    pub ready_poll: Duration,
    pub request_timeout: Duration,
}

impl Default for WarmupDeadlines {
    fn default() -> Self {
        Self {
            ready_timeout: READY_TIMEOUT,
            ready_poll: READY_POLL,
            request_timeout: REQUEST_TIMEOUT,
        }
    }
}

/// Spawns the whole-stack warm-up (retrieval → LLM → STT → TTS) as a background
/// task and registers it with the lifecycle so a disconnect can abort it. The
/// warm-up permit is claimed HERE, before spawning: an overlapping call (double
/// Start click, connect edge racing the endpoint) is a logged no-op, and the
/// tracked handle is always the one actually running. Never awaited on a request
/// or readiness path.
pub(crate) fn spawn_stack_warmup(state: &Arc<AppState>) {
    let Some(permit) = state.lifecycle.try_begin_warmup() else {
        tracing::info!("warmup: already in flight; skipping duplicate run");
        return;
    };
    let task_state = Arc::clone(state);
    let handle = tokio::spawn(async move {
        run_stack_warmup_with(task_state, permit, WarmupDeadlines::default()).await;
    });
    state.lifecycle.track_warmup_task(handle);
}

/// The warm-up body, logging one summary line:
/// `warmup: embedder 412 ms, reranker 88 ms, llm 2314 ms (1893 prompt tok), …`.
/// Holds `permit` for the whole run; it releases on return, panic, or abort
/// (RAII), so the slot can never stay claimed by a run that didn't finish.
pub(crate) async fn run_stack_warmup_with(
    state: Arc<AppState>,
    permit: crate::stack_lifecycle::WarmupPermit,
    deadlines: WarmupDeadlines,
) {
    let _permit = permit;
    let started = Instant::now();
    let mut parts: Vec<String> = Vec::new();

    // Retrieval first (CPU-only): embedder + reranker ONNX sessions + the spawn
    // catalog vectors. Runs before the GPU warm-ups so it never contends with them.
    parts.extend(warm_retrieval(&state).await);
    // LLM before TTS: koboldcpp is typically up first (smaller load), and the two
    // GPU warm-ups run sequentially so they don't thrash each other.
    parts.push(warm_llm(&state, deadlines).await);
    parts.push(warm_stt(&state, deadlines).await);
    parts.push(warm_tts(&state, deadlines).await);

    tracing::info!(
        "warmup: {} — total {:.1} s",
        parts.join(", "),
        started.elapsed().as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// Retrieval (embedder + reranker + catalog vectors) — moved from launcher.rs
// ---------------------------------------------------------------------------

/// Loads the shared retriever (the expensive ONNX load), runs one embed AND one
/// rerank so both sessions have done a first inference, then pre-warms the spawn
/// catalog vectors (~8k records, disk-cached, misses only). No-op when retrieval
/// is disabled or the models aren't downloaded. Returns summary fragments.
async fn warm_retrieval(state: &Arc<AppState>) -> Vec<String> {
    let settings = AppSettings::load(&state.config.settings_path);
    if !settings.retrieval.enabled {
        return vec!["retrieval skipped (disabled)".to_string()];
    }

    // `retriever()` loads + memoizes the ONNX models on first use — blocking work,
    // so hop off the async runtime for the load + the first inferences.
    let load_state = Arc::clone(state);
    let outcome = tokio::task::spawn_blocking(move || {
        let load_started = Instant::now();
        let Some(retriever) = load_state.retriever().cloned() else {
            return None;
        };
        let load_ms = load_started.elapsed().as_millis();

        let embed_started = Instant::now();
        let embed_ok = retriever.embed("warm").is_ok();
        let embed_ms = embed_started.elapsed().as_millis();

        // First reranker inference (ONNX session warm-up). With the reranker
        // disabled `rerank` falls back to cosine — already covered by the embed.
        let rerank_started = Instant::now();
        let rerank_ok = retriever.rerank("warm", &["warm"]).is_ok();
        let rerank_ms = rerank_started.elapsed().as_millis();
        let reranker_warmed = rerank_ok && retriever.has_reranker();
        Some((retriever, load_ms, embed_ms, embed_ok, rerank_ms, reranker_warmed))
    })
    .await
    .ok()
    .flatten();

    let Some((retriever, load_ms, embed_ms, embed_ok, rerank_ms, reranker_warmed)) = outcome
    else {
        return vec!["retrieval skipped (models not loaded)".to_string()];
    };

    let mut parts = vec![format!(
        "embedder {} ms (load {} ms{})",
        embed_ms,
        load_ms,
        if embed_ok { "" } else { ", embed failed" }
    )];
    parts.push(if reranker_warmed {
        format!("reranker {rerank_ms} ms")
    } else {
        "reranker off (cosine fallback)".to_string()
    });

    // Pre-warm the spawn catalogs (~8k spawnable records) so the first catalog
    // search resolves instantly instead of embedding them on demand. Batched +
    // disk-cached, misses only, so this is a no-op once warmed. The embed text
    // MUST match `catalog_item_vector_text` in the prompt crate or the cache
    // misses.
    if let Some(cache) = state.embed_cache().cloned() {
        let texts: Vec<String> = state
            .repository
            .list_action_catalogs()
            .into_iter()
            .flat_map(|catalog| catalog.items)
            .filter(|item| !item.disable)
            .map(|item| {
                let vector_text = item.vectorizable_text.trim();
                if vector_text.is_empty() {
                    item.name.trim().to_string()
                } else {
                    vector_text.to_string()
                }
            })
            .filter(|text| !text.is_empty())
            .collect();
        if !texts.is_empty() {
            let total = texts.len();
            let warmed =
                tokio::task::spawn_blocking(move || cache.warm_batch(&retriever, &texts, 64))
                    .await
                    .unwrap_or(0);
            parts.push(format!("catalogs {warmed} new / {total} total"));
        }
    }
    parts
}

// ---------------------------------------------------------------------------
// LLM (koboldcpp) — prime the KV cache with the real first-turn prompt prefix
// ---------------------------------------------------------------------------

/// Waits for koboldcpp, then runs a 1-token generation over the SAME prompt
/// prefix the first real turn will send (`cache_prompt: true`), so turn 1
/// fast-forwards over a warm KV cache instead of ingesting the whole system
/// prompt + history cold. Falls back to a tiny generic prompt (which still warms
/// the first-request compute path) when no live chat exists yet.
async fn warm_llm(state: &Arc<AppState>, deadlines: WarmupDeadlines) -> String {
    let endpoint = state.config.llm_endpoint.clone();
    let probe = format!("{endpoint}/v1/models");
    if !wait_for_http(&probe, deadlines).await {
        return "llm skipped (endpoint not reachable)".to_string();
    }

    // The real first-turn prompt: same live chat the bridge drives, same
    // deterministic first speaker, same structured/text mode. Building it runs
    // prompt assembly + retrieval (blocking ONNX inference), so hop off-runtime.
    let plan_state = Arc::clone(state);
    let plan = tokio::task::spawn_blocking(move || {
        let (live_chat_id, structured) = bridge_turn_shape(&plan_state);
        crate::generate::warmup_chat_messages(&plan_state, &live_chat_id, structured)
    })
    .await
    .ok()
    .flatten();

    let (messages, speaker) = match plan {
        Some((messages, speaker)) => (messages, Some(speaker)),
        None => (
            vec![json!({ "role": "user", "content": "Hi" })],
            None,
        ),
    };

    let started = Instant::now();
    match crate::llm::warmup_completion(&endpoint, &messages, deadlines.request_timeout).await {
        Ok(metrics) => {
            let prompt_tokens = metrics
                .as_ref()
                .and_then(|m| m.prompt_tokens)
                .map(|n| format!(", {n} prompt tok"))
                .unwrap_or_default();
            let who = speaker
                .map(|name| format!(", speaker '{name}'"))
                .unwrap_or_else(|| ", generic prompt".to_string());
            format!(
                "llm {} ms (prefix ingested{prompt_tokens}{who})",
                started.elapsed().as_millis()
            )
        }
        Err(error) => {
            tracing::debug!("warmup: llm priming failed: {error}");
            format!("llm failed after {} ms", started.elapsed().as_millis())
        }
    }
}

/// The live-chat id + structured flag the in-game bridge will use for real turns,
/// resolved from the same bridge config `spawn_in_process` loads. Defaults match
/// a fresh install (`fnv-goodsprings`, structured on).
fn bridge_turn_shape(state: &Arc<AppState>) -> (String, bool) {
    let settings = AppSettings::load(&state.config.settings_path);
    let config_path = settings.launcher.helper_config.trim().to_string();
    let config = chasm_fnv_bridge::load_config(Path::new(&config_path))
        .unwrap_or_else(|_| chasm_fnv_bridge::default_config());
    (config.live_chat_id, config.enable_action_books)
}

// ---------------------------------------------------------------------------
// STT (koboldcpp Whisper) — absorb the first-decode warm-up with a silent clip
// ---------------------------------------------------------------------------

/// Transcribes ~1.3 s of silence so the first real push-to-talk line doesn't pay
/// the STT engine's first-decode warm-up. Targets the active provider's endpoint
/// (koboldcpp Whisper, or the dedicated Parakeet server when selected). Skipped
/// when the Whisper path is active but no Whisper model is selected.
async fn warm_stt(state: &Arc<AppState>, deadlines: WarmupDeadlines) -> String {
    let settings = AppSettings::load(&state.config.settings_path);
    let parakeet = crate::launcher::stt_uses_parakeet(&settings, &state.config);
    let model = if parakeet {
        // The Parakeet server ignores the model field; send its repo id.
        chasm_core::PARAKEET_HF_REPO.to_string()
    } else {
        chasm_core::stt_effective_model(&settings.stt)
    };
    if model.is_empty() {
        return "stt skipped (no model selected)".to_string();
    }
    let endpoint = crate::effective_stt_endpoint(&state.config, &settings);
    // koboldcpp serves STT on the LLM port; by the time we get here the LLM wait
    // already succeeded (or failed), so only probe briefly.
    let brief = WarmupDeadlines {
        ready_timeout: deadlines.ready_poll.max(Duration::from_secs(3)),
        ..deadlines
    };
    if !wait_for_http(&endpoint, brief).await {
        return "stt skipped (endpoint not reachable)".to_string();
    }

    let started = Instant::now();
    match prime_stt(&endpoint, &model, deadlines.request_timeout).await {
        Ok(()) => format!("stt {} ms", started.elapsed().as_millis()),
        Err(error) => {
            tracing::debug!("warmup: stt priming failed: {error}");
            format!("stt failed after {} ms", started.elapsed().as_millis())
        }
    }
}

/// POSTs a short silent WAV to the OpenAI-compatible transcription endpoint.
pub(crate) async fn prime_stt(
    endpoint: &str,
    model: &str,
    timeout: Duration,
) -> Result<(), String> {
    let wav = silence_wav(16_000, 1_300);
    let part = reqwest::multipart::Part::bytes(wav)
        .file_name("warmup.wav")
        .mime_str("audio/wav")
        .map_err(|error| error.to_string())?;
    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", model.to_string());
    let client = reqwest::Client::new();
    let response = client
        .post(endpoint)
        .timeout(timeout)
        .multipart(form)
        .send()
        .await
        .map_err(|error| format!("stt warmup request failed: {error}"))?;
    if response.status().is_success() {
        Ok(())
    } else {
        Err(format!("stt warmup returned {}", response.status()))
    }
}

/// A minimal silent 16-bit mono PCM WAV of `duration_ms` at `sample_rate`.
pub(crate) fn silence_wav(sample_rate: u32, duration_ms: u32) -> Vec<u8> {
    let samples = (u64::from(sample_rate) * u64::from(duration_ms) / 1000) as u32;
    let data_len = samples * 2;
    let byte_rate = sample_rate * 2;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.resize(44 + data_len as usize, 0);
    wav
}

// ---------------------------------------------------------------------------
// TTS (faster-qwen3-tts / PocketTTS) — trigger first-request CUDA graph capture
// ---------------------------------------------------------------------------

/// Waits for the TTS server (it loads its model before binding the port), then
/// synthesizes one short discarded utterance with a real cloned voice. This is
/// where the first in-game line used to lose the most time: the first
/// `/v1/audio/speech` request captures CUDA graphs / warms kernels.
async fn warm_tts(state: &Arc<AppState>, deadlines: WarmupDeadlines) -> String {
    let Some(voice) = first_voice_with_reference(&crate::active_voices_dir(&state.config)) else {
        return "tts skipped (no cloned voices)".to_string();
    };
    let endpoint = state.config.tts_endpoint.clone();
    let probe = format!("{endpoint}/health");
    if !wait_for_http(&probe, deadlines).await {
        return "tts skipped (endpoint not reachable)".to_string();
    }

    let started = Instant::now();
    match prime_tts(&endpoint, &voice, deadlines.request_timeout).await {
        Ok(bytes) => format!(
            "tts {} ms (voice '{voice}', {bytes} bytes discarded)",
            started.elapsed().as_millis()
        ),
        Err(error) => {
            tracing::debug!("warmup: tts priming failed: {error}");
            format!("tts failed after {} ms", started.elapsed().as_millis())
        }
    }
}

/// POSTs one short synthesis and drains (discards) the audio. Returns the byte
/// count so the log line proves a real generation happened.
pub(crate) async fn prime_tts(
    endpoint: &str,
    voice: &str,
    timeout: Duration,
) -> Result<usize, String> {
    let url = format!("{}/v1/audio/speech", endpoint.trim_end_matches('/'));
    let body = json!({
        "model": "qwen3-tts",
        "input": "Warm up.",
        "voice": voice,
        "response_format": "pcm",
    });
    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .timeout(timeout)
        .json(&body)
        .send()
        .await
        .map_err(|error| format!("tts warmup request failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("tts warmup returned {}", response.status()));
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("tts warmup body read failed: {error}"))?;
    Ok(bytes.len())
}

/// The first voice under the active profile's voices dir with a usable
/// `reference.wav` clip, in stable (sorted) order — mirrors the TTS server's own
/// `_first_dir_voice` fallback, so the warm-up voice always resolves server-side.
pub(crate) fn first_voice_with_reference(voices_dir: &Path) -> Option<String> {
    let mut names: Vec<String> = std::fs::read_dir(voices_dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            entry
                .path()
                .join("reference.wav")
                .is_file()
                .then_some(name)
        })
        .collect();
    names.sort();
    names.into_iter().next()
}

// ---------------------------------------------------------------------------
// Shared: wait for a local HTTP service to come up
// ---------------------------------------------------------------------------

/// Polls `url` (GET) until ANY http response arrives or `ready_timeout` elapses.
/// A non-2xx still means the server is up (e.g. GET on a POST-only route), so
/// only transport errors count as "not ready".
pub(crate) async fn wait_for_http(url: &str, deadlines: WarmupDeadlines) -> bool {
    let client = reqwest::Client::new();
    let deadline = Instant::now() + deadlines.ready_timeout;
    loop {
        match client
            .get(url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(_) => return true,
            Err(_) if Instant::now() >= deadline => return false,
            Err(_) => tokio::time::sleep(deadlines.ready_poll).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// A real [`AppState`] over throwaway temp dirs, with every runtime endpoint
    /// pointing at a dead port (nothing ever listens on 127.0.0.1:9/discard),
    /// retrieval disabled, no STT model, and no cloned voices — so every warm-up
    /// step resolves to a fast skip and no test ever loads a model or network.
    fn dead_endpoint_state(tag: &str) -> Arc<AppState> {
        let root = std::env::temp_dir().join(format!(
            "sb-warmup-state-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let settings_path = root.join("settings.json");
        let mut settings = AppSettings::load(&settings_path); // defaults (file absent)
        settings.retrieval.enabled = false;
        settings.save(&settings_path).unwrap();
        let config = chasm_core::AppConfig {
            bind_addr: "127.0.0.1:0".into(),
            data_root: root.join("data"),
            workspace_root: root.clone(),
            settings_path,
            engines_dir: root.join("engines"),
            profiles_dir: root.join("profiles"),
            voices_dir: root.join("voices"),
            llm_models_dir: root.join("models-llm"),
            stt_endpoint: "http://127.0.0.1:9/v1/audio/transcriptions".into(),
            parakeet_stt_endpoint: "http://127.0.0.1:9/v1/audio/transcriptions".into(),
            llm_endpoint: "http://127.0.0.1:9".into(),
            tts_endpoint: "http://127.0.0.1:9".into(),
        };
        Arc::new(AppState::new(config))
    }

    /// With nothing listening and nothing configured, a FULL warm-up run
    /// completes quickly (all steps degrade to logged skips) and — the RAII
    /// property the connect edge depends on — releases the permit on return, so
    /// the next connect can warm again. A run that wedged the slot shut (the old
    /// `end_warmup`-at-the-end shape did on panic) would fail the final claim.
    #[tokio::test]
    async fn full_run_completes_and_releases_the_permit_with_nothing_listening() {
        let state = dead_endpoint_state("release");
        let deadlines = WarmupDeadlines {
            ready_timeout: Duration::from_millis(200),
            ready_poll: Duration::from_millis(50),
            request_timeout: Duration::from_secs(1),
        };
        let permit = state.lifecycle.try_begin_warmup().expect("first claim");
        let started = Instant::now();
        run_stack_warmup_with(Arc::clone(&state), permit, deadlines).await;
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "every step must degrade to a fast skip, not hang"
        );
        assert!(
            state.lifecycle.try_begin_warmup().is_some(),
            "the permit must be released when the run returns"
        );
    }

    /// One warm-up per connect: while a run holds the permit, a duplicate spawn
    /// (double Start click / connect-edge race) is a no-op that spawns and tracks
    /// NOTHING. Once the permit is free, spawn really runs and is tracked, so the
    /// disconnect edge can abort it and the abort returns the slot.
    #[tokio::test]
    async fn duplicate_spawn_is_a_no_op_while_a_warmup_holds_the_permit() {
        let state = dead_endpoint_state("dup");
        let held = state.lifecycle.try_begin_warmup().expect("hold the slot");
        spawn_stack_warmup(&state);
        assert!(
            state.lifecycle.abort_warmup().is_none(),
            "a refused duplicate must not have been spawned or tracked"
        );
        drop(held);

        spawn_stack_warmup(&state);
        let handle = state
            .lifecycle
            .abort_warmup()
            .expect("the real run must be tracked for the disconnect edge");
        let _ = handle.await; // aborted (or already done) — either way…
        assert!(
            state.lifecycle.try_begin_warmup().is_some(),
            "…the permit must be back after the task is gone"
        );
    }

    #[test]
    fn silence_wav_has_valid_header_and_length() {
        let wav = silence_wav(16_000, 1_300);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        // 16k samples/s * 1.3 s * 2 bytes = 41_600 data bytes + 44 header.
        assert_eq!(wav.len(), 44 + 41_600);
        // Declared data length matches.
        let declared = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]);
        assert_eq!(declared as usize, 41_600);
        // All-silence payload.
        assert!(wav[44..].iter().all(|&b| b == 0));
    }

    #[test]
    fn first_voice_prefers_sorted_dir_with_reference() {
        let dir = std::env::temp_dir().join(format!("sb-warm-voice-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // "Zed" has a clip; "Alpha" exists but has no reference.wav; "Bea" has one.
        std::fs::create_dir_all(dir.join("Zed")).unwrap();
        std::fs::write(dir.join("Zed").join("reference.wav"), b"riff").unwrap();
        std::fs::create_dir_all(dir.join("Alpha")).unwrap();
        std::fs::create_dir_all(dir.join("Bea")).unwrap();
        std::fs::write(dir.join("Bea").join("reference.wav"), b"riff").unwrap();

        assert_eq!(
            first_voice_with_reference(&dir).as_deref(),
            Some("Bea"),
            "first sorted dir WITH a reference clip wins"
        );
        // Missing dir → None, never an error.
        assert!(first_voice_with_reference(&dir.join("nope")).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// End-to-end warm-up calls against stub HTTP servers: proves the LLM prime
    /// sends a minimal 1-token request, the TTS prime drains a real body, and the
    /// STT prime posts a parseable WAV — without any real model anywhere.
    #[tokio::test]
    async fn primes_hit_stub_servers_with_minimal_requests() {
        use axum::{extract::Request, routing::any, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};

        static LLM_HITS: AtomicUsize = AtomicUsize::new(0);
        static TTS_HITS: AtomicUsize = AtomicUsize::new(0);
        static STT_HITS: AtomicUsize = AtomicUsize::new(0);

        let app = Router::new()
            .route(
                "/v1/models",
                any(|| async { axum::Json(json!({ "data": [{ "id": "stub-model" }] })) }),
            )
            .route(
                "/v1/chat/completions",
                any(|request: Request| async move {
                    let bytes = axum::body::to_bytes(request.into_body(), 1 << 20)
                        .await
                        .unwrap();
                    let body: Value = serde_json::from_slice(&bytes).unwrap();
                    // The warm-up generation must be minimal + cache-priming.
                    assert_eq!(body["max_tokens"], json!(1));
                    assert_eq!(body["cache_prompt"], json!(true));
                    assert_eq!(body["stream"], json!(false));
                    LLM_HITS.fetch_add(1, Ordering::SeqCst);
                    axum::Json(json!({
                        "choices": [{ "message": { "role": "assistant", "content": "." } }],
                        "usage": { "prompt_tokens": 123, "completion_tokens": 1 },
                        "timings": { "prompt_ms": 45.0 }
                    }))
                }),
            )
            .route("/health", any(|| async { "ok" }))
            .route(
                "/v1/audio/speech",
                any(|request: Request| async move {
                    let bytes = axum::body::to_bytes(request.into_body(), 1 << 20)
                        .await
                        .unwrap();
                    let body: Value = serde_json::from_slice(&bytes).unwrap();
                    assert_eq!(body["response_format"], json!("pcm"));
                    assert_eq!(body["voice"], json!("Easy Pete"));
                    TTS_HITS.fetch_add(1, Ordering::SeqCst);
                    vec![0u8; 4096]
                }),
            )
            .route(
                "/v1/audio/transcriptions",
                any(|request: Request| async move {
                    // Multipart body must carry a WAV file part; a size check is
                    // enough to prove the silent clip arrived.
                    let bytes = axum::body::to_bytes(request.into_body(), 1 << 20)
                        .await
                        .unwrap();
                    assert!(bytes.len() > 41_000, "expected the silent wav payload");
                    STT_HITS.fetch_add(1, Ordering::SeqCst);
                    axum::Json(json!({ "text": "" }))
                }),
            );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base = format!("http://{addr}");

        let deadlines = WarmupDeadlines {
            ready_timeout: Duration::from_secs(5),
            ready_poll: Duration::from_millis(50),
            request_timeout: Duration::from_secs(5),
        };
        assert!(wait_for_http(&format!("{base}/v1/models"), deadlines).await);

        // LLM prime.
        let messages = vec![json!({ "role": "user", "content": "Hi" })];
        let metrics = crate::llm::warmup_completion(&base, &messages, deadlines.request_timeout)
            .await
            .expect("llm warmup ok");
        assert_eq!(metrics.and_then(|m| m.prompt_tokens), Some(123));
        assert_eq!(LLM_HITS.load(Ordering::SeqCst), 1);

        // TTS prime drains the body.
        let bytes = prime_tts(&base, "Easy Pete", deadlines.request_timeout)
            .await
            .expect("tts warmup ok");
        assert_eq!(bytes, 4096);
        assert_eq!(TTS_HITS.load(Ordering::SeqCst), 1);

        // STT prime posts the silent clip.
        prime_stt(
            &format!("{base}/v1/audio/transcriptions"),
            "whisper-stub",
            deadlines.request_timeout,
        )
        .await
        .expect("stt warmup ok");
        assert_eq!(STT_HITS.load(Ordering::SeqCst), 1);
    }

    /// With nothing listening, every wait times out fast and the function still
    /// completes (all steps skipped) — the warm-up can never wedge a connect.
    #[tokio::test]
    async fn wait_for_http_times_out_cleanly() {
        let deadlines = WarmupDeadlines {
            ready_timeout: Duration::from_millis(200),
            ready_poll: Duration::from_millis(50),
            request_timeout: Duration::from_secs(1),
        };
        // Port 9 (discard) is not listening on localhost.
        let started = Instant::now();
        assert!(!wait_for_http("http://127.0.0.1:9/health", deadlines).await);
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
