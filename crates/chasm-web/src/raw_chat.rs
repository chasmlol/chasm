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
//!     We render the messages into Gemma 4's turn format
//!     (`<|turn>user\n…<turn|>\n … <|turn>model\n`) and call llama-server's raw
//!     `/completion`, deliberately BYPASSING the GGUF's embedded jinja: served
//!     via `--jinja` + /v1/chat/completions, the 17KB Gemma 4 template can
//!     inject a system turn and/or a `<|think|>` preamble, and the LoRA's data
//!     had ZERO system turns (a hard spec rule) — that off-distribution prefix
//!     was the real "garbage at serve" bug. Building the prompt ourselves gives
//!     train/serve parity + the spec's "no system turn, ever".
//!
//!     Gemma 4 uses `<|turn>` / `<turn|>` (NOT Gemma 2/3's `<start_of_turn>` /
//!     `<end_of_turn>`), confirmed by the training run: `train.log` reports
//!     `Detected turn markers: '<|turn>user\n' / '<|turn>model\n'`, the
//!     label-mask audit shows every trained span ending in `<turn|>`, and the
//!     GGUF's EOS is `<turn|>` (export.log `EOS check ok: '<turn|>'`). So the
//!     stop token is `<turn|>` and the model turn is prompted with `<|turn>model`.
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
use chasm_st_compat::STJsonlChatMessage;

use crate::save_sync::now_iso;
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

    // --- generate: raw /completion on the Gemma-4-formatted prompt -------------
    let prompt = gemma4_prompt(&messages);
    let mut req = json!({
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "n_predict": n_predict,
        "cache_prompt": true,
        // Stop at the Gemma 4 end-of-turn token (also the model's EOS).
        "stop": ["<turn|>"],
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

    // Persist the turn to a live-chat so chasm's Chat UI shows it (and "clear
    // chat" clears it) — the same on-disk shape the FNV path writes. Best-effort:
    // a persistence failure never fails the generation. `historyCount` lets the
    // bridge detect a UI clear (count dropped) and reset its own context.
    let history_count = persist_turn(&state, &body, &messages, &raw);

    Ok(Json(json!({
        "characterId": character_id,
        "model": model_file,
        "text": raw,
        "historyCount": history_count,
    })))
}

/// Appends the new player/world message + Digby's full reply to the live-chat's
/// current segment, so the Chat screen renders the companion thread and
/// clear-history wipes it. Returns the segment's message count after the append
/// (None when not in session mode or the live-chat doesn't exist yet).
fn persist_turn(state: &Arc<AppState>, body: &Value, messages: &[Value], raw: &str) -> Option<usize> {
    let live_chat_id = body
        .get("liveChatId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())?;
    let participant_id = body
        .get("participantId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("npc:Digby");
    let player_name = body
        .get("playerName")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("Player");

    // The live-chat is created by the bridge on spawn (POST /live-chats); if it's
    // not there yet, skip persistence rather than inventing one here.
    let live_chat = state.repository.get_live_chat(live_chat_id).ok()?;
    let segment = live_chat
        .segments
        .iter()
        .find(|s| s.id == live_chat.current_segment_id)
        .or_else(|| live_chat.segments.first())?
        .clone();

    // The new incoming message is the last user turn the bridge sent.
    let new_message = messages
        .last()
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let now = now_iso();

    // Incoming line (player chat / world event / lesson) — is_user so the UI's
    // fallback visibility renders it; label it by its tag.
    let user_msg = STJsonlChatMessage {
        name: incoming_label(&new_message, player_name),
        is_user: true,
        is_system: false,
        send_date: Some(now.clone()),
        mes: new_message,
        extra: Value::Null,
        original_avatar: None,
    };
    if state.repository.append_segment_message(&segment, &user_msg).is_err() {
        return None;
    }

    // Digby's turn — persist the FULL raw output (say + do + goal + learn) so the
    // whole context is visible for debugging, plus a turn_actions chip for the
    // do:. Live metadata marks him as the speaker so it shows in his thread.
    let digby_name = live_chat
        .participants
        .get(participant_id)
        .map(|p| p.name.clone())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "Digby".to_string());
    let turn_actions = parse_do_chip(raw).map(|chip| vec![chip]).unwrap_or_default();
    let assistant_msg = STJsonlChatMessage {
        name: digby_name,
        is_user: false,
        is_system: false,
        send_date: Some(now),
        mes: raw.to_string(),
        // HeadlessLiveMetadata is camelCase — snake_case keys silently fail to
        // deserialize, which drops the message's visibility (it then falls to the
        // non-user fallback and never shows in the thread).
        extra: json!({
            "headless": { "metadata": { "live": {
                "speakerParticipantId": participant_id,
                "audibleTo": ["player"],
                "present": ["player", participant_id],
            } } },
            "chasm": { "turn_actions": turn_actions },
        }),
        original_avatar: None,
    };
    if state.repository.append_segment_message(&segment, &assistant_msg).is_err() {
        return None;
    }

    // Bump updated_at so chat_view (newest-first) shows THIS conversation, not a
    // stale FNV one that happens to sit elsewhere in the store.
    let lcid = live_chat_id.to_string();
    let now2 = now_iso();
    let _ = state.repository.update_store(|store| {
        if let Some(lc) = store.items.get_mut(&lcid) {
            lc.updated_at = Some(now2);
        }
    });

    state.repository.read_segment_messages(&segment).ok().map(|m| m.len())
}

/// A display name for an incoming line based on its tag: `[Name] …` → Name,
/// `[world] …` / `[lesson …]` → "World", else the player name.
fn incoming_label(content: &str, player_name: &str) -> String {
    let trimmed = content.trim_start();
    if trimmed.starts_with("[world]") || trimmed.starts_with("[lesson") {
        return "World".to_string();
    }
    if let Some(rest) = trimmed.strip_prefix('[') {
        if let Some(end) = rest.find(']') {
            let tag = &rest[..end];
            if !tag.is_empty() && !tag.eq_ignore_ascii_case("goal") {
                return tag.to_string();
            }
        }
    }
    player_name.to_string()
}

/// Parses the first `do: action{args}` line of the raw output into a
/// turn_actions chip (`{ id, parameters }`) for the Chat UI's green action pill.
fn parse_do_chip(raw: &str) -> Option<Value> {
    for line in raw.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("do:") {
            let call = rest.trim();
            let (action, args) = match call.find('{') {
                Some(idx) => (&call[..idx], &call[idx..]),
                None => (call, ""),
            };
            let action = action.trim();
            if action.is_empty() {
                return None;
            }
            return Some(json!({ "id": action, "parameters": args }));
        }
    }
    None
}

/// Render chat messages into Gemma 4's turn format — the exact format the LoRA
/// was trained on. No system turn (the spec forbids one; a `system` role, if
/// ever present, is folded into the stream as a user turn rather than silently
/// dropped). `<|turn>` / `<turn|>` are Gemma 4's turn tokens (NOT Gemma 2/3's
/// `<start_of_turn>` / `<end_of_turn>`).
fn gemma4_prompt(messages: &[Value]) -> String {
    let mut prompt = String::from("<bos>");
    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = msg.get("content").and_then(Value::as_str).unwrap_or("");
        let turn = if role == "assistant" { "model" } else { "user" };
        prompt.push_str("<|turn>");
        prompt.push_str(turn);
        prompt.push('\n');
        prompt.push_str(content);
        prompt.push_str("<turn|>\n");
    }
    prompt.push_str("<|turn>model\n");
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
