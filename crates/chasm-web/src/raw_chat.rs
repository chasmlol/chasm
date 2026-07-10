//! `POST /api/headless/v1/raw-chat` — zero-assembly generation for PURE-LORA
//! characters (persona + protocol baked into the weights; e.g. the Minecraft
//! companion "Digby").
//!
//! ADDITIVE next to the card/prompt pipeline: this endpoint refuses normal
//! prompt-bearing cards, and no existing path changes behavior. For a card
//! whose `data.extensions.chasm.pure_lora.model` names a GGUF:
//!
//!   * the prompt is the request's `messages[]` VERBATIM — no system prompt,
//!     no persona/scenario/lorebook/structured-output instructions, nothing.
//!     llama-server's `--jinja` applies the GGUF's own chat template, giving
//!     train/serve parity (the LoRA was trained with no system turn).
//!   * an optional raw GBNF `grammar` is forwarded to llama.cpp per request
//!     (the caller swaps grammars by state, e.g. goal active vs not).
//!   * the response is the model's text VERBATIM (the caller parses it).
//!   * MODEL ROUTING: the turn is served by the card's own GGUF. If llama-server
//!    (:5001) is currently serving something else, it is relaunched on the
//!     card's model first (same spec as the managed runtime, only `-m` differs).
//!
//! The caller owns history, windowing, and any steering (goal pins / lesson
//! injections ride inside message text). Sessions, TTS, STT, save-sync all
//! stay on their existing endpoints.

use std::sync::Arc;
use std::time::Duration;

use axum::{extract::State, Json};
use serde_json::{json, Value};
use chasm_core::AppSettings;

use crate::{AppState, WebError, WebResult};

/// llama-server base (the managed runtime) — mirrors DEFAULT_STACK_LLM_ADDR.
const LLAMA_BASE: &str = "http://127.0.0.1:5001";
/// How long to wait for a freshly-routed model to come up (large GGUF load).
const MODEL_SWAP_TIMEOUT_SECS: u64 = 120;

pub async fn raw_chat(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> WebResult<Json<Value>> {
    // --- resolve + validate the pure-LoRA card --------------------------------
    let character_id = body
        .get("characterId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| WebError::from(anyhow::anyhow!("characterId is required")))?;
    let card = state
        .repository
        .read_character_card(character_id)
        .ok()
        .flatten()
        .ok_or_else(|| WebError::from(anyhow::anyhow!("unknown character '{character_id}'")))?;
    let Some(model_file) = card.pure_lora_model.clone() else {
        return Err(WebError::from(anyhow::anyhow!(
            "'{character_id}' is not a pure-LoRA card (no extensions.chasm.pure_lora.model); \
             use the normal generate endpoints"
        )));
    };

    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| WebError::from(anyhow::anyhow!("messages[] is required")))?
        .clone();

    // Local provider only: routing a hosted API to a local GGUF is meaningless.
    let settings = AppSettings::load(&state.config.settings_path);
    let provider = chasm_core::normalize_llm_provider(&settings.llm.provider);
    if provider != chasm_core::PROVIDER_LOCAL {
        return Err(WebError::from(anyhow::anyhow!(
            "pure-LoRA raw-chat requires the local llama.cpp provider (current: {provider})"
        )));
    }

    // --- model routing ---------------------------------------------------------
    let gguf = state.config.llm_models_dir.join(&model_file);
    if !gguf.exists() {
        return Err(WebError::from(anyhow::anyhow!(
            "pure-LoRA model '{model_file}' not found in the LLM models dir"
        )));
    }
    ensure_model_served(&state, &gguf, &model_file).await?;

    // --- sampling + grammar ----------------------------------------------------
    let opts = body.get("generationOptions").cloned().unwrap_or(Value::Null);
    let mut sampling = crate::llm::Sampling::from_settings(&settings.llm.sampling);
    if let Some(t) = opts.get("temperature").and_then(Value::as_f64) {
        sampling.temperature = t;
    }
    if let Some(p) = opts.get("topP").and_then(Value::as_f64) {
        sampling.top_p = p;
    }
    if let Some(k) = opts.get("topK").and_then(Value::as_u64) {
        sampling.top_k = Some(k as u32);
    }
    if let Some(m) = opts.get("maxTokens").and_then(Value::as_i64) {
        sampling.max_tokens = Some(m);
    }
    sampling.grammar = body
        .get("grammar")
        .and_then(Value::as_str)
        .filter(|g| !g.trim().is_empty())
        .map(str::to_string);

    // --- generate: messages verbatim, no response_format, raw text back --------
    let target = crate::llm::LlmTarget::resolve(&settings, &state.config);
    let (raw, _metrics) =
        crate::llm::chat_completion_capturing_sampled(&target, &messages, None, sampling)
            .await
            .map_err(|e| WebError::from(anyhow::anyhow!("generation failed: {e}")))?;

    Ok(Json(json!({
        "characterId": character_id,
        "model": model_file,
        "text": raw,
    })))
}

/// Ensures llama-server is serving `gguf`; relaunches it on that model when it
/// is serving anything else (or is down). Waits for health after a relaunch.
async fn ensure_model_served(
    state: &Arc<AppState>,
    gguf: &std::path::Path,
    model_file: &str,
) -> WebResult<()> {
    if currently_served_model().await.is_some_and(|served| served.ends_with(model_file)) {
        return Ok(());
    }
    tracing::info!("raw-chat: routing llama-server to {model_file}");
    let swap_state = Arc::clone(state);
    let swap_gguf = gguf.to_path_buf();
    let spawned =
        tokio::task::spawn_blocking(move || crate::launcher::respawn_llm_with_gguf(&swap_state, &swap_gguf))
            .await
            .unwrap_or(false);
    if !spawned {
        return Err(WebError::from(anyhow::anyhow!(
            "could not relaunch llama-server on '{model_file}' (llama.cpp installed?)"
        )));
    }
    // Wait for the model to finish loading (health + models both report ready).
    let deadline = std::time::Instant::now() + Duration::from_secs(MODEL_SWAP_TIMEOUT_SECS);
    loop {
        if currently_served_model().await.is_some_and(|served| served.ends_with(model_file)) {
            return Ok(());
        }
        if std::time::Instant::now() > deadline {
            return Err(WebError::from(anyhow::anyhow!(
                "llama-server did not come up on '{model_file}' within {MODEL_SWAP_TIMEOUT_SECS}s"
            )));
        }
        tokio::time::sleep(Duration::from_millis(1500)).await;
    }
}

/// The model path llama-server reports on GET /v1/models, or None when down /
/// still loading.
async fn currently_served_model() -> Option<String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .ok()?;
    let resp = client.get(format!("{LLAMA_BASE}/v1/models")).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let value: Value = resp.json().await.ok()?;
    value
        .get("data")
        .and_then(Value::as_array)
        .and_then(|models| models.first())
        .and_then(|m| m.get("id"))
        .and_then(Value::as_str)
        .map(|s| s.replace('\\', "/"))
}
