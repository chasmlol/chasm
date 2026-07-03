//! UI globals domain — the Globals page backend (global scenario template).
//!
//! The GLOBAL scenario replaces the per-character card `scenario` field: one
//! app-wide `{{macro}}` template, stored in the Globals store
//! (`headless/globals.json`, profile-aware) and resolved per turn by the
//! generation path with the turn's gamestate macros + backend-computed macros
//! (`{{participants}}`). See `chasm_prompt::scenario` and `generate.rs`.
//!
//! Three endpoints under `/api/ui/v1`:
//!
//!   * `GET  /globals/scenario`         — the effective template (saved value,
//!     else the built-in default), whether it IS the default, and the default
//!     text (for the page's "reset to default" affordance).
//!   * `PUT  /globals/scenario`         — save the template. Saving text equal
//!     to the default clears the override (back to the pristine default state);
//!     saving an empty string disables the scenario component entirely.
//!   * `POST /globals/scenario/preview` — resolve a template (the draft in the
//!     editor, else the saved one) through the LATEST recorded gamestate macro
//!     table + computed macros, without running any generation. This is the
//!     page's live "what will the NPC actually see" panel.
//!
//! Like the rest of `/api/ui/v1`, this is UI-only: it never touches the game
//! transport (`/api/game/*`) or the headless contract (`/api/headless/*`).

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::{extract::State, Json};
use serde::Serialize;
use serde_json::{json, Value};

use crate::generate::{active_live_chat, latest_chat_macros};
use crate::{orchestrator, AppState, WebError, WebResult};

/// Builds a `WebError` carrying `message` (rendered as the JSON error body).
fn web_err(message: impl Into<String>) -> WebError {
    WebError::from(anyhow::anyhow!(message.into()))
}

/// `GET`/`PUT /api/ui/v1/globals/scenario` response.
#[derive(Serialize)]
pub(crate) struct UiGlobalsScenario {
    /// The EFFECTIVE template: the saved value when one exists (may be empty =
    /// scenario disabled), else the built-in default.
    pub template: String,
    /// True when no override is saved (the built-in default is in effect).
    pub is_default: bool,
    /// The built-in default template (for the reset affordance).
    pub default_template: String,
}

/// `POST /api/ui/v1/globals/scenario/preview` response.
#[derive(Serialize)]
pub(crate) struct UiGlobalsPreview {
    /// The template with every `{{macro}}` resolved (unknown → empty).
    pub resolved: String,
    /// The macro table the preview used: the latest recorded gamestate table
    /// plus the backend-computed macros (`participants`).
    pub macros: Value,
    /// `send_date` of the turn the recorded table came from, when any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    /// Set when something degrades the preview (no recorded macros yet, empty
    /// template) or needs a caveat (participants preview includes every NPC).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// The current scenario view (shared by GET and PUT responses).
fn scenario_view(state: &AppState) -> WebResult<UiGlobalsScenario> {
    let stored = state.repository.read_globals()?.scenario_template;
    let is_default = stored.is_none();
    Ok(UiGlobalsScenario {
        template: stored.unwrap_or_else(|| chasm_prompt::DEFAULT_SCENARIO_TEMPLATE.to_string()),
        is_default,
        default_template: chasm_prompt::DEFAULT_SCENARIO_TEMPLATE.to_string(),
    })
}

/// `GET /api/ui/v1/globals/scenario` — the effective global scenario template.
pub(crate) async fn get_scenario(
    State(state): State<Arc<AppState>>,
) -> WebResult<Json<UiGlobalsScenario>> {
    Ok(Json(scenario_view(&state)?))
}

/// `PUT /api/ui/v1/globals/scenario` — save the template.
///
/// Request: `{ "template": "…" }` (string, required; empty allowed = disable
/// the scenario component). Saving text identical to the built-in default
/// clears the override so the store stays pristine (`is_default` flips back).
pub(crate) async fn put_scenario(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> WebResult<Json<UiGlobalsScenario>> {
    let template = body
        .get("template")
        .and_then(Value::as_str)
        .ok_or_else(|| web_err("globals scenario save requires a string 'template'"))?
        .to_string();

    state.repository.update_globals(|globals| {
        globals.scenario_template = if template == chasm_prompt::DEFAULT_SCENARIO_TEMPLATE {
            None
        } else {
            Some(template.clone())
        };
    })?;
    Ok(Json(scenario_view(&state)?))
}

/// `POST /api/ui/v1/globals/scenario/preview` — resolve a template through the
/// latest recorded gamestate macros + computed macros. No generation runs.
///
/// Request: `{ "template": "…" }` — optional; when omitted the saved/effective
/// template is previewed. An empty recorded table is NOT an error: the
/// template still resolves (gamestate macros → empty) with `note` explaining.
pub(crate) async fn preview_scenario(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> WebResult<Json<UiGlobalsPreview>> {
    let template = match body.get("template").and_then(Value::as_str) {
        Some(template) => template.to_string(),
        None => scenario_view(&state)?.template,
    };

    // Latest recorded gamestate table (the same source the Gamestate page and
    // the generation fallback read), plus the present-NPC list for the
    // computed participants preview.
    let live_chat = active_live_chat(&state)?;
    let (updated_at, recorded) = live_chat
        .as_ref()
        .map(|chat| latest_chat_macros(&state, chat))
        .unwrap_or((None, json!({})));
    let mut macros: BTreeMap<String, String> = chasm_prompt::macros_from_value(&recorded);
    let recorded_empty = macros.is_empty();

    // Computed {{participants}}: the preview has no "current speaker" to
    // exclude, so it lists the player plus EVERY present NPC of the active
    // conversation (a real turn excludes the NPC being prompted).
    let npc_names: Vec<String> = live_chat
        .as_ref()
        .map(|chat| {
            orchestrator::compute_eligible(chat)
                .into_iter()
                .map(|participant| participant.name)
                .collect()
        })
        .unwrap_or_default();
    let player_name = macros.get("player_name").cloned().unwrap_or_default();
    macros.insert(
        "participants".to_string(),
        chasm_prompt::participants_macro(&player_name, &npc_names),
    );

    let resolved = chasm_prompt::apply_macros(&template, &macros);

    let mut notes: Vec<String> = Vec::new();
    if template.trim().is_empty() {
        notes.push(
            "Template is empty — the scenario component is omitted from NPC prompts entirely."
                .to_string(),
        );
    }
    if recorded_empty {
        notes.push(
            "No gamestate macros recorded yet — location/time placeholders resolved to empty. \
             Talk to an NPC in-game (with the bridge running) to record a live table."
                .to_string(),
        );
    }
    if !npc_names.is_empty() {
        notes.push(
            "Preview includes every present NPC in {{participants}}; a real turn excludes \
             the NPC being prompted."
                .to_string(),
        );
    }

    Ok(Json(UiGlobalsPreview {
        resolved,
        macros: serde_json::to_value(&macros).unwrap_or_else(|_| json!({})),
        updated_at,
        note: (!notes.is_empty()).then(|| notes.join(" ")),
    }))
}
