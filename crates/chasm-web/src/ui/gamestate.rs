//! UI gamestate domain — the Macros page backend.
//!
//! Two endpoints under `/api/ui/v1`:
//!
//!   * `GET  /gamestate`      — the LATEST recorded gamestate macro table (the
//!     `metadata.macros` map the mod sent on the most recent NPC turn, recorded
//!     by `finalize_turn` onto the persisted message's `extra.chasm.macros`),
//!     plus its timestamp, so the page can show what the mod is extracting.
//!   * `POST /gamestate/test` — the substitution proof: resolve a `{{macro}}`
//!     template against a macro table (an explicit override, else the latest
//!     recorded one), run ONE minimal system+user completion against the same
//!     local LLM the NPC turns use, and return the resolved prompt + the reply.
//!
//! Since the Globals rework, production macro substitution exists but is
//! scoped to ONE component: the GLOBAL scenario template (`generate.rs`
//! resolves it per turn; see `chasm_prompt::scenario`). Cards, lore, and
//! system prompts still never run macros; this page and the Globals preview
//! stay the free-form proof surfaces.
//!
//! Like the rest of `/api/ui/v1`, this is UI-only: it never touches the game
//! transport (`/api/game/*`) or the headless contract (`/api/headless/*`).

use std::sync::Arc;

use axum::{extract::State, Json};
use serde::Serialize;
use serde_json::{json, Value};
use chasm_core::AppSettings;

// The latest-recorded-macros snapshot + active-chat pick are shared with the
// generation path (the scenario fallback reads the same source this page shows).
use crate::generate::{active_live_chat, latest_chat_macros};
use crate::{AppState, WebError, WebResult};

/// Builds a `WebError` carrying `message` (rendered as the JSON error body).
fn web_err(message: impl Into<String>) -> WebError {
    WebError::from(anyhow::anyhow!(message.into()))
}

/// `GET /api/ui/v1/gamestate` response: the latest recorded macro table.
#[derive(Serialize)]
pub(crate) struct UiGamestateView {
    /// The live chat the table came from (`None` before any chat exists).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_chat_id: Option<String>,
    /// `send_date` of the turn that recorded the table (`None` when no turn has
    /// carried macros yet).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    /// The latest `{ key: value, … }` macro table; `{}` before the first turn.
    pub macros: Value,
}

/// `POST /api/ui/v1/gamestate/test` response: the substitution + generation
/// proof for one template.
#[derive(Serialize)]
pub(crate) struct UiGamestateTest {
    /// The template with every `{{macro}}` substituted (unknown → empty).
    pub resolved_prompt: String,
    /// The model's reply to `resolved_prompt` (system) + `user_message` (user).
    pub reply: String,
    /// The macro table the resolution used (override or latest recorded).
    pub macros: Value,
    /// Set when the table was empty (no recorded turn and no override), so the
    /// page can explain why every macro resolved to nothing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// `GET /api/ui/v1/gamestate` — the latest recorded macro table + timestamp.
/// Returns an empty table (never errors) before the first live chat / first
/// macros-bearing turn so the page can render its empty state.
pub(crate) async fn gamestate_view(
    State(state): State<Arc<AppState>>,
) -> WebResult<Json<UiGamestateView>> {
    let Some(live_chat) = active_live_chat(&state)? else {
        return Ok(Json(UiGamestateView {
            live_chat_id: None,
            updated_at: None,
            macros: json!({}),
        }));
    };

    let (updated_at, macros) = latest_chat_macros(&state, &live_chat);
    Ok(Json(UiGamestateView {
        live_chat_id: Some(live_chat.id.clone()),
        updated_at,
        macros,
    }))
}

/// `POST /api/ui/v1/gamestate/test` — resolve a template and run one minimal
/// generation with it.
///
/// Request: `{ "template": "…{{player_name}}…", "user_message": "…",
/// "macros": { … } }` — `macros` is an OPTIONAL override; when omitted (or not
/// a non-empty object) the latest recorded table is used. An empty table is NOT
/// an error: the template still resolves (every macro → empty) and the
/// generation still runs, with `note` explaining the situation.
pub(crate) async fn gamestate_test(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> WebResult<Json<UiGamestateTest>> {
    let template = body
        .get("template")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|template| !template.is_empty())
        .ok_or_else(|| web_err("gamestate test requires a non-empty 'template'"))?
        .to_string();
    let user_message = body
        .get("user_message")
        .or_else(|| body.get("userMessage"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|message| !message.is_empty())
        .unwrap_or("Hello.")
        .to_string();

    // Macro table: explicit override > latest recorded > empty.
    let override_macros = body
        .get("macros")
        .filter(|macros| macros.as_object().is_some_and(|map| !map.is_empty()));
    let (macros, note) = match override_macros {
        Some(value) => (chasm_prompt::macros_from_value(value), None),
        None => {
            let recorded = active_live_chat(&state)?
                .map(|live_chat| latest_chat_macros(&state, &live_chat).1)
                .unwrap_or_else(|| json!({}));
            let macros = chasm_prompt::macros_from_value(&recorded);
            let note = macros.is_empty().then(|| {
                "No macros recorded yet (and no override provided) — every {{macro}} resolved to empty. \
                 Talk to an NPC in-game to record a live table, or pass a 'macros' override."
                    .to_string()
            });
            (macros, note)
        }
    };

    let resolved_prompt = chasm_prompt::apply_macros(&template, &macros);

    // One minimal system+user completion against the same local LLM the NPC
    // turns use — no retrieval, no cards, no live-chat writes, no action books.
    // Settings load fresh per request, so model/sampling tweaks apply without a
    // restart.
    let gs_settings = AppSettings::load(&state.config.settings_path);
    let sampling = crate::llm::Sampling::from_settings(&gs_settings.llm.sampling);
    let target = crate::llm::LlmTarget::resolve(&gs_settings, &state.config);
    let messages = vec![
        json!({ "role": "system", "content": resolved_prompt }),
        json!({ "role": "user", "content": user_message }),
    ];
    // A generation failure (LLM not running, model still loading, …) must not
    // 500 away the substitution proof: return the resolved prompt with the
    // error in `note` so the page always shows what the macros resolved to.
    let (reply, note) =
        match crate::llm::chat_completion_capturing_sampled(&target, &messages, None, sampling)
            .await
        {
            Ok((reply, _metrics)) => (reply, note),
            Err(error) => (
                String::new(),
                Some(format!(
                    "Template resolved, but the test generation failed: {error}"
                )),
            ),
        };

    Ok(Json(UiGamestateTest {
        resolved_prompt,
        reply,
        macros: serde_json::to_value(&macros).unwrap_or_else(|_| json!({})),
        note,
    }))
}
