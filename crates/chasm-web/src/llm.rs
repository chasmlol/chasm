//! Minimal client for the local OpenAI-compatible LLM (llama.cpp) at
//! `{endpoint}/v1/chat/completions`. Mirrors how the FNV helper points its
//! `provider: 'custom'` / `custom_url: '{endpoint}/v1'` generation at llama.cpp.
//!
//! `chat_completion_stream` opens an SSE stream (`"stream": true`) and forwards
//! each content delta over a channel; `chat_completion_capturing_sampled` buffers
//! the full text for the non-streaming generation paths.

use futures_util::StreamExt as _;
use serde_json::{json, Value};
use tokio::sync::mpsc;

/// First model id advertised by `{endpoint}/v1/models`, when reachable. The
/// helper resolves the loaded model the same way before generating.
/// Shared HTTP client: one connection pool for every LLM call. A fresh
/// `Client::new()` per turn threw away the pooled localhost connection, adding
/// a TCP handshake to the hot path.
pub(crate) fn http_client() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new)
}

async fn first_model_id(client: &reqwest::Client, endpoint: &str) -> Option<String> {
    // The loaded model only changes when the managed runtime restarts (which
    // changes nothing about the id llama.cpp reports for the same GGUF, and a
    // model SWAP goes through settings + full restart anyway). Cache per
    // endpoint: this lookup used to be an extra HTTP round-trip on EVERY turn
    // before the completion request could even be sent.
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, String>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(map) = cache.lock() {
        if let Some(hit) = map.get(endpoint) {
            return Some(hit.clone());
        }
    }
    let url = format!("{endpoint}/v1/models");
    let response = client.get(url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let body: Value = response.json().await.ok()?;
    let id = body
        .get("data")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(|item| item.get("id"))
        .and_then(Value::as_str)
        .map(str::to_string)?;
    if let Ok(mut map) = cache.lock() {
        map.insert(endpoint.to_string(), id.clone());
    }
    Some(id)
}

/// The structured-output JSON schema (verbatim shape of SillyTavern's
/// `buildStructuredOutputResponseFormat`). Passed as `response_format` so
/// llama.cpp constrains sampling to valid JSON — the format is *enforced*, not
/// merely requested in the prompt.
pub fn structured_response_format() -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "chasm_structured_reply",
            "description": "A Chasm live/headless reply with spoken text and optional client actions.",
            "strict": true,
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "speech": { "type": "string", "description": "The assistant or NPC spoken response only." },
                    "stateUpdates": { "type": "object", "description": "External state updates for the client. Use an empty object when none are needed.", "additionalProperties": true },
                    "actions": { "type": "array", "description": "Actions the character chooses this turn: each is the action's alias string, or an object with the alias as \"id\" when it needs fields. Empty array when none.", "items": { "type": ["string", "object"] } }
                },
                "required": ["speech", "stateUpdates", "actions"]
            }
        }
    })
}

/// Optional per-request generation knobs (used by the speaker-selection LLM
/// call, which honors the custom-model temperature/max_tokens settings).
#[derive(Debug, Clone, Copy, Default)]
pub struct GenerationOptions {
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
}

/// The full set of llama.cpp sampling params for an NPC / admin turn, sourced
/// from the saved `LlmSamplingSettings` and forwarded verbatim into the
/// OpenAI-compatible request body. Built via [`Sampling::from_settings`] so the
/// "send only when meaningful" rules live in one place (e.g. `top_k`/`min_p`/
/// `n_ctx` are omitted at their off values to preserve prior default behaviour).
#[derive(Debug, Clone, Copy, Default)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<u32>,
    pub min_p: Option<f64>,
    pub repeat_penalty: f64,
    pub max_tokens: Option<i64>,
    pub n_ctx: Option<u32>,
    pub seed: Option<i64>,
}

/// Rounds an `f32` sampling value to 3 decimals as `f64`, so the `f32`→`f64`
/// cast doesn't surface noise like `0.699999988` in the request JSON / debug log
/// (llama.cpp would accept either, but the clean value is nicer to read + test).
fn round3(value: f32) -> f64 {
    ((value as f64) * 1000.0).round() / 1000.0
}

impl Sampling {
    /// Maps the saved (normalized) sampling settings to the request shape,
    /// applying the "omit at off-value" rules so an untouched config produces the
    /// exact same request as before this feature existed.
    pub fn from_settings(s: &chasm_core::LlmSamplingSettings) -> Self {
        let s = s.normalized();
        Self {
            temperature: round3(s.temperature),
            top_p: round3(s.top_p),
            top_k: (s.top_k > 0).then_some(s.top_k),
            min_p: (s.min_p > 0.0).then_some(round3(s.min_p)),
            repeat_penalty: round3(s.repeat_penalty),
            max_tokens: (s.max_tokens > 0).then_some(s.max_tokens as i64),
            n_ctx: (s.n_ctx > 0).then_some(s.n_ctx),
            seed: (s.seed >= 0).then_some(s.seed),
        }
    }

    /// Overlays an explicit per-request [`GenerationOptions`] (the admin
    /// `generationOptions` body field) on top of the saved sampling: a present
    /// `temperature` / `max_tokens` wins, everything else (top_p/top_k/min_p/…)
    /// stays from settings. Keeps the admin path's request-level overrides while
    /// still honouring the global sampling config.
    pub fn with_overrides(mut self, options: GenerationOptions) -> Self {
        if let Some(temperature) = options.temperature {
            self.temperature = temperature;
        }
        if let Some(max_tokens) = options.max_tokens {
            self.max_tokens = Some(max_tokens);
        }
        self
    }

    /// Writes every active sampling field onto an OpenAI-compatible request body.
    /// llama.cpp's server honours these top-level keys (`temperature`, `top_p`,
    /// `top_k`, `min_p`, `repeat_penalty`, `max_tokens`/`n_predict`, `seed`,
    /// `n_ctx`).
    fn apply(&self, body: &mut Value) {
        body["temperature"] = json!(self.temperature);
        body["top_p"] = json!(self.top_p);
        body["repeat_penalty"] = json!(self.repeat_penalty);
        if let Some(top_k) = self.top_k {
            body["top_k"] = json!(top_k);
        }
        if let Some(min_p) = self.min_p {
            body["min_p"] = json!(min_p);
        }
        if let Some(max_tokens) = self.max_tokens {
            body["max_tokens"] = json!(max_tokens);
            // llama.cpp accepts both; send n_predict too for older builds.
            body["n_predict"] = json!(max_tokens);
        }
        if let Some(n_ctx) = self.n_ctx {
            body["n_ctx"] = json!(n_ctx);
        }
        if let Some(seed) = self.seed {
            body["seed"] = json!(seed);
        }
    }
}

/// Builds the request body for a full NPC / admin generation turn, applying the
/// saved sampling settings on top of the base body. This is the path the live
/// chat + admin generation use, so user-set temperature/top_p/etc. take effect.
fn request_body_sampled(
    model: Option<&str>,
    messages: &[Value],
    stream: bool,
    response_format: Option<&Value>,
    sampling: Sampling,
) -> Value {
    let mut body = json!({
        "messages": messages,
        "stream": stream,
    });
    if let Some(model) = model {
        body["model"] = json!(model);
    }
    if let Some(format) = response_format {
        body["response_format"] = format.clone();
    }
    // Reuse llama.cpp's KV cache for the unchanged prompt PREFIX (system + char
    // card + action/quest books + lore) across turns, so we don't re-prefill that
    // large stable block every turn — only the changed suffix (new history +
    // gamestate + player message). With `parallel: 1` the same slot is reused
    // turn-to-turn, so the prefix lands a cache hit. No-op when the prefix
    // changes, so it never costs anything.
    body["cache_prompt"] = json!(true);
    sampling.apply(&mut body);
    // Prove the wiring: the exact sampling params on the outgoing llama.cpp
    // request (temperature/top_p/top_k/min_p/repeat_penalty/max_tokens/seed).
    tracing::debug!(
        target: "chasm::llm",
        temperature = body.get("temperature").and_then(serde_json::Value::as_f64),
        top_p = body.get("top_p").and_then(serde_json::Value::as_f64),
        top_k = body.get("top_k").and_then(serde_json::Value::as_u64),
        min_p = body.get("min_p").and_then(serde_json::Value::as_f64),
        repeat_penalty = body.get("repeat_penalty").and_then(serde_json::Value::as_f64),
        max_tokens = body.get("max_tokens").and_then(serde_json::Value::as_i64),
        n_ctx = body.get("n_ctx").and_then(serde_json::Value::as_u64),
        seed = body.get("seed").and_then(serde_json::Value::as_i64),
        "llama.cpp request sampling"
    );
    body
}

/// Streams a chat completion. Returns a receiver of incremental content deltas
/// (`Ok(String)`) terminated by channel close, or a single `Err(String)` for a
/// transport/decode error.
///
/// `trace_id` (the `X-Chasm-Trace-Id` of the originating game request, when
/// known) lets the stream capture llama.cpp's `usage`/`timings` from the final
/// SSE chunk — emitted because we set `stream_options.include_usage` — and record
/// them for the Tracing page's tokens/sec metric. Passing `None` skips capture.
pub async fn chat_completion_stream(
    endpoint: &str,
    messages: &[Value],
    response_format: Option<&Value>,
    trace_id: Option<&str>,
    sampling: Sampling,
) -> Result<mpsc::Receiver<Result<String, String>>, String> {
    let client = http_client().clone();
    let model = first_model_id(&client, endpoint).await;
    let url = format!("{endpoint}/v1/chat/completions");
    let mut body =
        request_body_sampled(model.as_deref(), messages, true, response_format, sampling);
    // Ask llama.cpp to include the final `usage`/`timings` chunk in the stream so
    // we can capture tokens/sec without a second request.
    body["stream_options"] = json!({ "include_usage": true });
    // Env-gated (CHASM_LLM_DUMP=1) dump of the EXACT request body, for offline
    // replay when hunting prompt-cache misses / latency.
    if std::env::var_os("CHASM_LLM_DUMP").is_some() {
        if let Some(dir) = std::env::var_os("TEMP") {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            let path = std::path::Path::new(&dir).join(format!("chasm-llm-body-{stamp}.json"));
            let _ = std::fs::write(path, serde_json::to_vec_pretty(&body).unwrap_or_default());
        }
    }
    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|error| format!("llama.cpp request failed: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("llama.cpp returned {status}: {text}"));
    }

    let trace_id = trace_id.map(str::to_string);
    let (tx, rx) = mpsc::channel::<Result<String, String>>(64);
    tokio::spawn(async move {
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        while let Some(chunk) = stream.next().await {
            let bytes = match chunk {
                Ok(bytes) => bytes,
                Err(error) => {
                    let _ = tx
                        .send(Err(format!("llama.cpp stream error: {error}")))
                        .await;
                    return;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&bytes));
            // SSE events are separated by blank lines; data lines start "data: ".
            while let Some(newline) = buffer.find('\n') {
                let line = buffer[..newline].trim().to_string();
                buffer.drain(..=newline);
                let Some(payload) = line.strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.trim();
                if payload.is_empty() {
                    continue;
                }
                if payload == "[DONE]" {
                    return;
                }
                // Capture generation metrics from any chunk that carries them
                // (llama.cpp puts `usage`/`timings` on the final chunk).
                if let Some(id) = trace_id.as_deref() {
                    if let Ok(value) = serde_json::from_str::<Value>(payload) {
                        if let Some(metrics) =
                            chasm_core::LlmMetrics::from_completion_response(&value)
                        {
                            crate::trace_routes::record_llm_metrics(id, metrics);
                        }
                    }
                }
                if let Some(delta) = parse_delta(payload) {
                    if !delta.is_empty() && tx.send(Ok(delta)).await.is_err() {
                        return; // receiver dropped
                    }
                }
            }
        }
    });

    Ok(rx)
}

/// Buffered chat completion with explicit generation options (temperature /
/// max_tokens). Used by the speaker-selection call so the custom-model
/// temperature/max_tokens settings are honored.
/// Buffered chat completion for a full NPC / admin turn, applying the saved
/// `Sampling` to the request body and returning `(content, metrics)`. The
/// buffered (non-stream) live + admin generation paths call this so user-set
/// sampling reaches the model.
pub async fn chat_completion_capturing_sampled(
    endpoint: &str,
    messages: &[Value],
    response_format: Option<&Value>,
    sampling: Sampling,
) -> Result<(String, Option<chasm_core::LlmMetrics>), String> {
    let client = http_client().clone();
    let model = first_model_id(&client, endpoint).await;
    let url = format!("{endpoint}/v1/chat/completions");
    let response = client
        .post(&url)
        .json(&request_body_sampled(
            model.as_deref(),
            messages,
            false,
            response_format,
            sampling,
        ))
        .send()
        .await
        .map_err(|error| format!("llama.cpp request failed: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("llama.cpp returned {status}: {text}"));
    }
    let body: Value = response
        .json()
        .await
        .map_err(|error| format!("llama.cpp response decode failed: {error}"))?;
    let content = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let metrics = chasm_core::LlmMetrics::from_completion_response(&body);
    Ok((content, metrics))
}

/// Builds the minimal KV-cache-priming request body used by the connect-time
/// warm-up: the caller's messages verbatim, ONE predicted token, greedy, non-
/// streaming, with `cache_prompt` on so koboldcpp keeps the ingested prefix in
/// its slot for the first real turn to fast-forward over.
fn warmup_request_body(model: Option<&str>, messages: &[Value]) -> Value {
    let mut body = json!({
        "messages": messages,
        "stream": false,
        "max_tokens": 1,
        "n_predict": 1,
        "temperature": 0.0,
        "cache_prompt": true,
    });
    if let Some(model) = model {
        body["model"] = json!(model);
    }
    body
}

/// One-token, discarded chat completion that pre-ingests `messages` into the
/// LLM server's prompt (KV) cache. Returns the server's usage/timings metrics
/// (prompt token count etc.) for the warm-up log line. `timeout` bounds the
/// whole request — a cold multi-thousand-token prefill can take tens of seconds.
pub async fn warmup_completion(
    endpoint: &str,
    messages: &[Value],
    timeout: std::time::Duration,
) -> Result<Option<chasm_core::LlmMetrics>, String> {
    let client = http_client().clone();
    let model = first_model_id(&client, endpoint).await;
    let url = format!("{endpoint}/v1/chat/completions");
    let response = client
        .post(&url)
        .timeout(timeout)
        .json(&warmup_request_body(model.as_deref(), messages))
        .send()
        .await
        .map_err(|error| format!("llm warmup request failed: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("llm warmup returned {status}: {text}"));
    }
    let body: Value = response
        .json()
        .await
        .map_err(|error| format!("llm warmup response decode failed: {error}"))?;
    Ok(chasm_core::LlmMetrics::from_completion_response(&body))
}

/// Extracts `choices[0].delta.content` from one SSE data payload.
fn parse_delta(payload: &str) -> Option<String> {
    let value: Value = serde_json::from_str(payload).ok()?;
    value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("delta"))
        .and_then(|delta| delta.get("content"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chasm_core::LlmSamplingSettings;

    #[test]
    fn default_sampling_omits_off_value_keys() {
        // Untouched config: only the always-on keys are present, matching prior
        // behaviour (no top_k / min_p / max_tokens / n_ctx / seed in the body).
        let sampling = Sampling::from_settings(&LlmSamplingSettings::default());
        let body = request_body_sampled(Some("m"), &[], false, None, sampling);
        assert_eq!(body["temperature"], json!(0.7));
        assert_eq!(body["top_p"], json!(1.0));
        assert_eq!(body["repeat_penalty"], json!(1.0));
        assert!(body.get("top_k").is_none());
        assert!(body.get("min_p").is_none());
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("n_ctx").is_none());
        assert!(body.get("seed").is_none());
    }

    #[test]
    fn set_sampling_reaches_request_body() {
        // A fully-tweaked config lands every param on the outgoing request.
        let settings = LlmSamplingSettings {
            temperature: 0.4,
            top_p: 0.9,
            top_k: 50,
            min_p: 0.05,
            repeat_penalty: 1.15,
            max_tokens: 256,
            n_ctx: 8192,
            seed: 42,
        };
        let body = request_body_sampled(
            Some("m"),
            &[],
            true,
            None,
            Sampling::from_settings(&settings),
        );
        assert_eq!(body["temperature"], json!(0.4));
        assert_eq!(body["top_p"], json!(0.9));
        assert_eq!(body["top_k"], json!(50));
        assert_eq!(body["min_p"], json!(0.05));
        assert_eq!(body["repeat_penalty"], json!(1.15));
        assert_eq!(body["max_tokens"], json!(256));
        assert_eq!(body["n_predict"], json!(256));
        assert_eq!(body["n_ctx"], json!(8192));
        assert_eq!(body["seed"], json!(42));
        assert_eq!(body["stream"], json!(true));
    }

    #[test]
    fn warmup_body_is_a_minimal_cache_priming_generation() {
        let messages = vec![json!({ "role": "system", "content": "You are Easy Pete." })];
        let body = warmup_request_body(Some("m"), &messages);
        // One greedy token, non-streaming, prefix kept in the server's KV cache.
        assert_eq!(body["max_tokens"], json!(1));
        assert_eq!(body["n_predict"], json!(1));
        assert_eq!(body["temperature"], json!(0.0));
        assert_eq!(body["stream"], json!(false));
        assert_eq!(body["cache_prompt"], json!(true));
        assert_eq!(body["messages"], json!(messages));
        assert_eq!(body["model"], json!("m"));
        // No model id resolved → the key is simply absent (server default).
        assert!(warmup_request_body(None, &messages).get("model").is_none());
    }

    #[test]
    fn admin_overrides_win_over_saved_sampling() {
        // The admin generationOptions temperature/max_tokens override settings,
        // but top_p/top_k stay from the saved config.
        let settings = LlmSamplingSettings {
            top_p: 0.8,
            top_k: 20,
            ..LlmSamplingSettings::default()
        };
        let sampling = Sampling::from_settings(&settings).with_overrides(GenerationOptions {
            temperature: Some(0.1),
            max_tokens: Some(64),
        });
        let body = request_body_sampled(Some("m"), &[], false, None, sampling);
        assert_eq!(body["temperature"], json!(0.1)); // overridden
        assert_eq!(body["max_tokens"], json!(64)); // overridden
        assert_eq!(body["top_p"], json!(0.8)); // from settings
        assert_eq!(body["top_k"], json!(20)); // from settings
    }
}
