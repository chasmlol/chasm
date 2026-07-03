//! Hosted-LLM request/response adapters.
//!
//! chasm speaks the OpenAI `/v1/chat/completions` shape natively (see [`crate::llm`]),
//! so the OpenAI, OpenRouter and generic OpenAI-compatible providers reuse the
//! existing body builder and only differ by base URL + `Authorization` header +
//! model id + a JSON-mode `response_format`. The two providers with their OWN
//! wire shapes get adapters here:
//!
//!   * **Anthropic** (Messages API, `POST /v1/messages`): system prompt is a
//!     top-level `system` field, history is `messages` of `{role, content}`, and
//!     `max_tokens` is REQUIRED. chasm needs a structured JSON reply, which
//!     Anthropic has no `response_format` for — so we prefill the assistant turn
//!     with `{` (a standard trick) and prepend it back when parsing, which
//!     reliably forces a single JSON object. The prompt itself already documents
//!     the `speech`/`stateUpdates`/`actions` contract (see `chasm-prompt`).
//!   * **Gemini** (`POST /v1beta/models/{model}:generateContent`): system prompt
//!     is `systemInstruction`, history is `contents` with roles `user`/`model`,
//!     and structured JSON is requested via `generationConfig.responseMimeType =
//!     "application/json"`.
//!
//! These builders are pure (`Value` in → `Value` out) and unit-tested against
//! representative payloads; the async HTTP + streaming lives in [`crate::llm`].

use serde_json::{json, Value};

/// The `anthropic-version` header every Messages request must carry.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Fallback `max_tokens` for Anthropic when the user left the sampling cap at 0
/// (Anthropic requires the field). Generous enough for a full NPC reply + actions.
pub const ANTHROPIC_DEFAULT_MAX_TOKENS: u32 = 2048;

/// Provider-neutral sampling subset forwarded to the hosted adapters. Built from
/// chasm's `llm::Sampling` at the call site so this module stays decoupled.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApiSampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u32,
    pub max_tokens: u32,
    pub seed: i64,
}

// ---------------------------------------------------------------------------
// Shared message helpers
// ---------------------------------------------------------------------------

/// Pulls the `content` of a message as a plain string (chasm only ever sends
/// string content, but be defensive about an array-of-parts shape too).
fn message_text(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Splits OpenAI-style messages into (concatenated system prompt, non-system
/// turns as `(role, text)` with consecutive same-role turns merged). Merging
/// keeps Anthropic/Gemini happy (both dislike two `user` turns in a row).
fn split_system_and_turns(messages: &[Value]) -> (String, Vec<(String, String)>) {
    let mut system_parts: Vec<String> = Vec::new();
    let mut turns: Vec<(String, String)> = Vec::new();
    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        let text = message_text(msg);
        if role == "system" {
            if !text.is_empty() {
                system_parts.push(text);
            }
            continue;
        }
        let role = if role == "assistant" { "assistant" } else { "user" };
        if let Some(last) = turns.last_mut() {
            if last.0 == role {
                last.1.push('\n');
                last.1.push_str(&text);
                continue;
            }
        }
        turns.push((role.to_string(), text));
    }
    (system_parts.join("\n\n"), turns)
}

// ---------------------------------------------------------------------------
// Anthropic
// ---------------------------------------------------------------------------

/// Builds the Anthropic `POST /v1/messages` body. When `structured`, appends an
/// assistant prefill of `"{"` so the model is forced to emit a single JSON object
/// (paired with [`parse_anthropic_reply`], which prepends the `{` back).
pub fn build_anthropic_body(
    model: &str,
    messages: &[Value],
    sampling: ApiSampling,
    structured: bool,
) -> Value {
    let (system, turns) = split_system_and_turns(messages);
    let mut msgs: Vec<Value> = turns
        .into_iter()
        .map(|(role, content)| json!({ "role": role, "content": content }))
        .collect();
    if structured {
        // Prefill: continue the assistant turn from an opening brace.
        msgs.push(json!({ "role": "assistant", "content": "{" }));
    }

    let max_tokens = if sampling.max_tokens > 0 {
        sampling.max_tokens
    } else {
        ANTHROPIC_DEFAULT_MAX_TOKENS
    };
    let mut body = json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": msgs,
    });
    if !system.is_empty() {
        body["system"] = json!(system);
    }
    // Anthropic temperature is 0..=1 (OpenAI is 0..=2) — clamp so a high local
    // setting doesn't 400 the request.
    if sampling.temperature > 0.0 {
        body["temperature"] = json!(sampling.temperature.clamp(0.0, 1.0));
    }
    if sampling.top_p > 0.0 && sampling.top_p < 1.0 {
        body["top_p"] = json!(sampling.top_p);
    }
    if sampling.top_k > 0 {
        body["top_k"] = json!(sampling.top_k);
    }
    body
}

/// Extracts the assistant text from an Anthropic Messages response, prepending the
/// prefilled `{` when the request was `structured`. Returns `Err` with a readable
/// message for an error body.
pub fn parse_anthropic_reply(resp: &Value, structured: bool) -> Result<String, String> {
    if resp.get("type").and_then(Value::as_str) == Some("error") {
        return Err(anthropic_error_message(resp));
    }
    let text = resp
        .get("content")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter(|p| p.get("type").and_then(Value::as_str) != Some("thinking"))
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();
    if text.is_empty() && !structured {
        return Err("Anthropic returned an empty response.".to_string());
    }
    Ok(if structured {
        // Prepend the prefilled opening brace the model continued from.
        let mut out = String::with_capacity(text.len() + 1);
        out.push('{');
        out.push_str(&text);
        out
    } else {
        text
    })
}

fn anthropic_error_message(resp: &Value) -> String {
    let msg = resp
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("unknown error");
    format!("Anthropic error: {msg}")
}

// ---------------------------------------------------------------------------
// Gemini
// ---------------------------------------------------------------------------

/// The Gemini generateContent path for `model` under `base_url`
/// (`{base}/models/{model}:generateContent`). The API key rides the
/// `x-goog-api-key` header, set at the call site.
pub fn gemini_generate_url(base_url: &str, model: &str) -> String {
    format!(
        "{}/models/{}:generateContent",
        base_url.trim_end_matches('/'),
        model
    )
}

/// Builds the Gemini `generateContent` body. `structured` requests a JSON mime so
/// the reply is a single JSON object (the prompt supplies the field contract).
pub fn build_gemini_body(messages: &[Value], sampling: ApiSampling, structured: bool) -> Value {
    let (system, turns) = split_system_and_turns(messages);
    let contents: Vec<Value> = turns
        .into_iter()
        .map(|(role, text)| {
            let role = if role == "assistant" { "model" } else { "user" };
            json!({ "role": role, "parts": [{ "text": text }] })
        })
        .collect();

    let mut generation_config = json!({});
    if sampling.temperature > 0.0 {
        generation_config["temperature"] = json!(sampling.temperature.clamp(0.0, 2.0));
    }
    if sampling.top_p > 0.0 && sampling.top_p < 1.0 {
        generation_config["topP"] = json!(sampling.top_p);
    }
    if sampling.top_k > 0 {
        generation_config["topK"] = json!(sampling.top_k);
    }
    if sampling.max_tokens > 0 {
        generation_config["maxOutputTokens"] = json!(sampling.max_tokens);
    }
    if structured {
        generation_config["responseMimeType"] = json!("application/json");
    }

    let mut body = json!({ "contents": contents });
    if !system.is_empty() {
        body["systemInstruction"] = json!({ "parts": [{ "text": system }] });
    }
    if generation_config.as_object().map(|o| !o.is_empty()).unwrap_or(false) {
        body["generationConfig"] = generation_config;
    }
    body
}

/// Extracts the concatenated text from a Gemini generateContent response, or an
/// `Err` with a readable message for an error / blocked body.
pub fn parse_gemini_reply(resp: &Value) -> Result<String, String> {
    if let Some(err) = resp.get("error") {
        let msg = err.get("message").and_then(Value::as_str).unwrap_or("unknown error");
        return Err(format!("Gemini error: {msg}"));
    }
    let candidate = resp
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|c| c.first());
    let Some(candidate) = candidate else {
        // A prompt blocked by safety filters returns no candidates.
        if let Some(reason) = resp
            .get("promptFeedback")
            .and_then(|f| f.get("blockReason"))
            .and_then(Value::as_str)
        {
            return Err(format!("Gemini blocked the prompt ({reason})."));
        }
        return Err("Gemini returned no candidates.".to_string());
    };
    let text = candidate
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();
    if text.is_empty() {
        return Err("Gemini returned an empty response.".to_string());
    }
    Ok(text)
}

// ---------------------------------------------------------------------------
// Shared error formatting for the OpenAI-compatible path
// ---------------------------------------------------------------------------

/// Turns a non-2xx HTTP status + raw body from any provider into a short, readable
/// one-liner for the UI (e.g. bad key / rate limit), extracting a message field
/// from the common `{error:{message}}` / `{message}` / `{error:"…"}` shapes.
pub fn format_http_error(provider: &str, status: u16, body: &str) -> String {
    let detail = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| v.get("error").and_then(Value::as_str).map(str::to_string))
                .or_else(|| v.get("message").and_then(Value::as_str).map(str::to_string))
        })
        .unwrap_or_else(|| {
            let trimmed = body.trim();
            if trimmed.is_empty() {
                "no response body".to_string()
            } else {
                trimmed.chars().take(300).collect()
            }
        });
    let hint = match status {
        401 | 403 => " (check your API key)",
        429 => " (rate limited — slow down or check your plan)",
        _ => "",
    };
    format!("{provider} API error {status}{hint}: {detail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_messages() -> Vec<Value> {
        vec![
            json!({ "role": "system", "content": "You are an NPC." }),
            json!({ "role": "user", "content": "Hi there." }),
            json!({ "role": "assistant", "content": "Hello." }),
            json!({ "role": "user", "content": "How are you?" }),
        ]
    }

    #[test]
    fn anthropic_body_extracts_system_and_prefills_when_structured() {
        let body = build_anthropic_body("claude-sonnet-5", &sample_messages(), ApiSampling { temperature: 0.7, max_tokens: 0, ..Default::default() }, true);
        assert_eq!(body["model"], "claude-sonnet-5");
        assert_eq!(body["system"], "You are an NPC.");
        assert_eq!(body["max_tokens"], ANTHROPIC_DEFAULT_MAX_TOKENS);
        let msgs = body["messages"].as_array().unwrap();
        // user, assistant, user, then the "{" prefill.
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs.last().unwrap()["role"], "assistant");
        assert_eq!(msgs.last().unwrap()["content"], "{");
        // temperature clamped into 0..=1.
        assert_eq!(body["temperature"], 0.7);
    }

    #[test]
    fn anthropic_body_clamps_temperature_to_one() {
        let body = build_anthropic_body("m", &sample_messages(), ApiSampling { temperature: 1.8, ..Default::default() }, false);
        assert_eq!(body["temperature"], 1.0);
        // no prefill when not structured
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.last().unwrap()["role"], "user");
    }

    #[test]
    fn anthropic_parse_prepends_brace_when_structured() {
        let resp = json!({ "content": [{ "type": "text", "text": "\"speech\":\"hi\",\"stateUpdates\":{},\"actions\":[]}" }] });
        let out = parse_anthropic_reply(&resp, true).unwrap();
        assert!(out.starts_with('{'));
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["speech"], "hi");
    }

    #[test]
    fn anthropic_parse_surfaces_error() {
        let resp = json!({ "type": "error", "error": { "type": "authentication_error", "message": "invalid x-api-key" } });
        let err = parse_anthropic_reply(&resp, true).unwrap_err();
        assert!(err.contains("invalid x-api-key"));
    }

    #[test]
    fn gemini_body_maps_roles_and_system() {
        let body = build_gemini_body(&sample_messages(), ApiSampling { temperature: 0.5, max_tokens: 256, ..Default::default() }, true);
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "You are an NPC.");
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(body["generationConfig"]["responseMimeType"], "application/json");
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 256);
    }

    #[test]
    fn gemini_parse_joins_parts() {
        let resp = json!({ "candidates": [{ "content": { "parts": [{ "text": "{\"speech\":" }, { "text": "\"hi\"}" }] } }] });
        let out = parse_gemini_reply(&resp).unwrap();
        assert_eq!(out, "{\"speech\":\"hi\"}");
    }

    #[test]
    fn gemini_parse_reports_block() {
        let resp = json!({ "promptFeedback": { "blockReason": "SAFETY" } });
        let err = parse_gemini_reply(&resp).unwrap_err();
        assert!(err.contains("SAFETY"));
    }

    #[test]
    fn gemini_url_builds() {
        assert_eq!(
            gemini_generate_url("https://generativelanguage.googleapis.com/v1beta/", "gemini-2.5-flash"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
    }

    #[test]
    fn http_error_extracts_message_and_hint() {
        let e = format_http_error("OpenAI", 401, r#"{"error":{"message":"Incorrect API key provided"}}"#);
        assert!(e.contains("401"));
        assert!(e.contains("check your API key"));
        assert!(e.contains("Incorrect API key"));
    }
}
