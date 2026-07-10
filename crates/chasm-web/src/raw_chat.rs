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
//!     We render the messages into the model's REAL Gemma chat format
//!     (`<start_of_turn>user\n…<end_of_turn>\n … <start_of_turn>model\n`) and
//!     call llama-server's raw `/completion`, deliberately BYPASSING the GGUF's
//!     embedded chat template: a wrongly-baked template (`<|turn>`/`<|think|>`
//!     instead of Gemma's turn tokens) is the #1 "great in training, garbage at
//!     serve" cause, and this LoRA shipped with exactly that. Building the
//!     prompt ourselves guarantees train/serve parity + the spec's "no system
//!     turn, ever" rule.
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
    let base = crate::llm::Sampling::from_settings(&settings.llm.sampling);
    let temperature = opts
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(base.temperature);
    let top_p = opts.get("topP").and_then(Value::as_f64).unwrap_or(base.top_p);
    let top_k = opts
        .get("topK")
        .and_then(Value::as_u64)
        .or(base.top_k.map(u64::from))
        .unwrap_or(0);
    let n_predict = opts
        .get("maxTokens")
        .and_then(Value::as_i64)
        .or(base.max_tokens)
        .unwrap_or(512);
    let grammar = body
        .get("grammar")
        .and_then(Value::as_str)
        .filter(|g| !g.trim().is_empty());

    // --- generate: raw /completion on the Gemma-formatted prompt ---------------
    let prompt = gemma_prompt(&messages);
    let mut req = json!({
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "n_predict": n_predict,
        "cache_prompt": true,
        // Stop cleanly at the end of the model turn (grammar has no end token).
        "stop": ["<end_of_turn>", "<start_of_turn>"],
    });
    if top_k > 0 {
        req["top_k"] = json!(top_k);
    }
    if let Some(g) = grammar {
        req["grammar"] = json!(g);
    }
    if let Some(seed) = base.seed {
        req["seed"] = json!(seed);
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .map_err(|e| WebError::from(anyhow::anyhow!("http client: {e}")))?;
    let resp = client
        .post(format!("{LLAMA_BASE}/completion"))
        .json(&req)
        .send()
        .await
        .map_err(|e| WebError::from(anyhow::anyhow!("generation request failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        return Err(WebError::from(anyhow::anyhow!(
            "llama-server /completion {status}: {}",
            detail.chars().take(300).collect::<String>()
        )));
    }
    let out: Value = resp
        .json()
        .await
        .map_err(|e| WebError::from(anyhow::anyhow!("bad /completion response: {e}")))?;
    let raw = out
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();

    Ok(Json(json!({
        "characterId": character_id,
        "model": model_file,
        "text": raw,
    })))
}

/// Render chat messages into the model's REAL Gemma chat format. No system turn
/// (the spec forbids one; a `system` role, if ever present, is folded into the
/// content stream as a user turn rather than silently dropped). `<start_of_turn>`
/// / `<end_of_turn>` are the Gemma turn tokens the LoRA was trained on.
fn gemma_prompt(messages: &[Value]) -> String {
    let mut prompt = String::from("<bos>");
    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = msg.get("content").and_then(Value::as_str).unwrap_or("");
        let turn = if role == "assistant" { "model" } else { "user" };
        prompt.push_str("<start_of_turn>");
        prompt.push_str(turn);
        prompt.push('\n');
        prompt.push_str(content);
        prompt.push_str("<end_of_turn>\n");
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
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
