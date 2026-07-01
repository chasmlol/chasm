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
//! This is the ONLY place `chasm_prompt::apply_macros` runs this pass — the
//! production NPC prompt path (`assemble_prompt_with_retrieval_collect`, cards,
//! lore) is deliberately untouched; where macros eventually plug into real
//! prompts is a later decision proven safe by this harness first.
//!
//! Like the rest of `/api/ui/v1`, this is UI-only: it never touches the game
//! transport (`/api/game/*`) or the headless contract (`/api/headless/*`).

use std::sync::Arc;

use axum::{extract::State, Json};
use serde::Serialize;
use serde_json::{json, Value};
use chasm_core::AppSettings;
use chasm_st_compat::LiveChat;

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

/// The newest macros-bearing turn of one live chat: `(send_date, macros)`.
///
/// The live game path writes each NPC's history to its per-participant
/// PROJECTION session while older / group-mode chats use the shared segment
/// sessions (see `messages_for_participant` in chasm-st-compat) — and the two
/// overlap for turns visible to several NPCs. Scanning both and keeping the
/// newest `send_date` works for every layout: duplicated copies of a turn carry
/// the same macro table, so whichever copy wins the tie is correct.
fn latest_chat_macros(state: &AppState, live_chat: &LiveChat) -> (Option<String>, Value) {
    let mut session_ids: Vec<String> = live_chat
        .segments
        .iter()
        .map(|segment| segment.session_id.clone())
        .collect();
    if let Some(sessions) = live_chat.participant_sessions.as_object() {
        // Projection entries are `{ "sessionId": "…" }` objects (see
        // `participant_session_id`), but tolerate raw strings too.
        session_ids.extend(sessions.values().filter_map(|entry| {
            entry
                .get("sessionId")
                .and_then(Value::as_str)
                .or_else(|| entry.as_str())
                .filter(|id| !id.is_empty())
                .map(str::to_string)
        }));
    }
    session_ids.sort();
    session_ids.dedup();

    let mut best: Option<(String, Value)> = None;
    for session_id in &session_ids {
        // Unreadable / not-yet-created sessions are skipped rather than failing
        // the whole view — the page should always render.
        let Ok(messages) = state.repository.read_session_messages(session_id) else {
            continue;
        };
        for message in messages {
            let Some(macros) = message
                .extra
                .get("chasm")
                .and_then(|chasm| chasm.get("macros"))
                .filter(|macros| macros.as_object().is_some_and(|map| !map.is_empty()))
            else {
                continue;
            };
            // ISO-8601 timestamps compare lexicographically; `>=` keeps the
            // later-in-file copy on equal stamps.
            let send_date = message.send_date.clone().unwrap_or_default();
            if best
                .as_ref()
                .map_or(true, |(best_date, _)| send_date.as_str() >= best_date.as_str())
            {
                best = Some((send_date, macros.clone()));
            }
        }
    }

    match best {
        Some((send_date, macros)) => {
            let updated_at = (!send_date.is_empty()).then_some(send_date);
            (updated_at, macros)
        }
        None => (None, json!({})),
    }
}

/// The active live chat: most recently updated first, so a stale chat sitting
/// in the store never shadows the one the game is writing to.
fn active_live_chat(state: &AppState) -> WebResult<Option<LiveChat>> {
    let mut chats = state.repository.list_live_chats()?;
    chats.sort_by(|a, b| {
        b.updated_at
            .clone()
            .unwrap_or_default()
            .cmp(&a.updated_at.clone().unwrap_or_default())
    });
    Ok(chats.into_iter().next())
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
    let endpoint = state.config.llm_endpoint.clone();
    let sampling = crate::llm::Sampling::from_settings(
        &AppSettings::load(&state.config.settings_path).llm.sampling,
    );
    let messages = vec![
        json!({ "role": "system", "content": resolved_prompt }),
        json!({ "role": "user", "content": user_message }),
    ];
    // A generation failure (LLM not running, model still loading, …) must not
    // 500 away the substitution proof: return the resolved prompt with the
    // error in `note` so the page always shows what the macros resolved to.
    let (reply, note) =
        match crate::llm::chat_completion_capturing_sampled(&endpoint, &messages, None, sampling)
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
