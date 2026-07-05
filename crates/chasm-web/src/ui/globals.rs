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

use crate::generate::{active_live_chat, latest_chat_macros, merge_scenario_variants};
use crate::{movement, orchestrator, AppState, WebError, WebResult};

/// Builds a `WebError` carrying `message` (rendered as the JSON error body).
fn web_err(message: impl Into<String>) -> WebError {
    WebError::from(anyhow::anyhow!(message.into()))
}

/// `GET`/`PUT /api/ui/v1/globals/scenario` response.
#[derive(Serialize)]
pub(crate) struct UiGlobalsScenario {
    /// The EFFECTIVE template: the saved value when one exists (may be empty =
    /// scenario disabled), else the built-in default. This is the DEFAULT
    /// variant of the dynamic-scenario system — the fallback wording when no
    /// situation variant below matches.
    pub template: String,
    /// True when no override is saved (the built-in default is in effect).
    pub is_default: bool,
    /// The built-in default template (for the reset affordance).
    pub default_template: String,
    /// The dynamic-scenario variants (stored config merged over the built-in
    /// catalog), in catalog order. Selection order is by `priority` (desc).
    pub variants: Vec<UiScenarioVariant>,
}

/// One dynamic-scenario variant as the UI sees it: the user-editable config
/// plus the FIXED catalog facts (label, condition, shipped defaults).
#[derive(Serialize)]
pub(crate) struct UiScenarioVariant {
    pub id: String,
    /// Display label ("Companion, sneaking"). Empty for unknown stored ids.
    pub label: String,
    /// Read-only description of the state that triggers this variant.
    pub condition_hint: String,
    pub enabled: bool,
    pub priority: i32,
    pub template: String,
    /// The shipped template (per-variant reset affordance).
    pub default_template: String,
    /// The shipped priority (reorder reset affordance).
    pub default_priority: i32,
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
    /// The variant the state-picker selection matched (only when the request
    /// carried a `state` object): its id and display label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_label: Option<String>,
}

/// The current scenario view (shared by GET and PUT responses).
fn scenario_view(state: &AppState) -> WebResult<UiGlobalsScenario> {
    let globals = state.repository.read_globals()?;
    let is_default = globals.scenario_template.is_none();
    let variants = merge_scenario_variants(globals.scenario_variants.as_deref())
        .into_iter()
        .map(|variant| {
            let def = chasm_prompt::variant_def(&variant.id);
            UiScenarioVariant {
                label: def.map(|d| d.label.to_string()).unwrap_or_default(),
                condition_hint: def.map(|d| d.condition_hint.to_string()).unwrap_or_default(),
                default_template: def.map(|d| d.default_template.to_string()).unwrap_or_default(),
                default_priority: def.map(|d| d.default_priority).unwrap_or_default(),
                id: variant.id,
                enabled: variant.enabled,
                priority: variant.priority,
                template: variant.template,
            }
        })
        .collect();
    Ok(UiGlobalsScenario {
        template: globals
            .scenario_template
            .unwrap_or_else(|| chasm_prompt::DEFAULT_SCENARIO_TEMPLATE.to_string()),
        is_default,
        default_template: chasm_prompt::DEFAULT_SCENARIO_TEMPLATE.to_string(),
        variants,
    })
}

/// `GET /api/ui/v1/globals/scenario` — the effective global scenario template.
pub(crate) async fn get_scenario(
    State(state): State<Arc<AppState>>,
) -> WebResult<Json<UiGlobalsScenario>> {
    Ok(Json(scenario_view(&state)?))
}

/// `PUT /api/ui/v1/globals/scenario` — save the template (and, optionally,
/// the dynamic-scenario variants).
///
/// Request: `{ "template": "…", "variants"?: [{ id, enabled, priority,
/// template }] }`. `template` is required (empty allowed = disable the
/// scenario component); saving text identical to the built-in default clears
/// the override so the store stays pristine (`is_default` flips back).
/// `variants` (when present) replaces the stored variant config wholesale; a
/// list identical to the shipped defaults clears the override the same way.
/// Omitting `variants` (an older client) leaves the stored variants untouched.
pub(crate) async fn put_scenario(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> WebResult<Json<UiGlobalsScenario>> {
    let template = body
        .get("template")
        .and_then(Value::as_str)
        .ok_or_else(|| web_err("globals scenario save requires a string 'template'"))?
        .to_string();

    let variants: Option<Vec<chasm_st_compat::ScenarioVariantConfig>> = match body.get("variants")
    {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            serde_json::from_value(value.clone())
                .map_err(|error| web_err(format!("invalid 'variants' payload: {error}")))?,
        ),
    };

    state.repository.update_globals(|globals| {
        globals.scenario_template = if template == chasm_prompt::DEFAULT_SCENARIO_TEMPLATE {
            None
        } else {
            Some(template.clone())
        };
        if let Some(variants) = &variants {
            globals.scenario_variants =
                if variants_equal_defaults(variants) { None } else { Some(variants.clone()) };
        }
    })?;
    Ok(Json(scenario_view(&state)?))
}

/// True when a submitted variant list is EXACTLY the shipped catalog defaults
/// (same ids, all enabled at default priority/template, nothing extra) — the
/// pristine state we store as `None`, mirroring the template behavior.
fn variants_equal_defaults(variants: &[chasm_st_compat::ScenarioVariantConfig]) -> bool {
    let defaults = chasm_prompt::default_variants();
    variants.len() == defaults.len()
        && variants.iter().all(|config| {
            defaults.iter().any(|default| {
                default.id == config.id
                    && config.enabled == default.enabled
                    && config.priority == default.priority
                    && config.template == default.template
            })
        })
}

/// `POST /api/ui/v1/globals/scenario/preview` — resolve a template through the
/// latest recorded gamestate macros + computed macros. No generation runs.
///
/// Request: `{ "template"?, "variants"?, "state"? }` — all optional.
/// * `template` — the default-variant draft; omitted → the saved/effective one.
/// * `state` — the STATE-PICKER: an object of gamestate flags (`teammate`,
///   `sneaking`, `player_sneaking`, `traveling`, …). When present, the preview
///   runs the SAME variant selection as a real turn against those flags
///   (using the `variants` drafts when given, else the stored config) and
///   resolves the winning template; `variant_id`/`variant_label` report the
///   match. Absent → the old template-only preview.
/// * `variants` — draft variant configs from the editor, `[{ id, enabled,
///   priority, template }]`, merged over the catalog like generation does.
///
/// An empty recorded table is NOT an error: the template still resolves
/// (gamestate macros → empty) with `note` explaining.
pub(crate) async fn preview_scenario(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> WebResult<Json<UiGlobalsPreview>> {
    let default_template = match body.get("template").and_then(Value::as_str) {
        Some(template) => template.to_string(),
        None => scenario_view(&state)?.template,
    };

    // State-picker selection (only when the request carries a `state` object).
    let picked_state = body.get("state").filter(|value| value.is_object());
    let (template, variant_id, variant_label, travel) = match picked_state {
        None => (default_template.clone(), None, None, None),
        Some(picked) => {
            let variants = match body.get("variants") {
                None | Some(Value::Null) => crate::generate::global_scenario_variants(&state),
                Some(value) => {
                    let configs: Vec<chasm_st_compat::ScenarioVariantConfig> =
                        serde_json::from_value(value.clone()).map_err(|error| {
                            web_err(format!("invalid 'variants' payload: {error}"))
                        })?;
                    merge_scenario_variants(Some(&configs))
                }
            };
            // The picker sends the flags as a flat object — the raw
            // `npc_state` spelling `NpcStateFlags` already accepts.
            let mut flags =
                chasm_prompt::NpcStateFlags::from_metadata(&json!({ "npc_state": picked }));
            flags.traveling = picked
                .get("traveling")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let selected =
                chasm_prompt::select_scenario(&variants, &default_template, &flags);
            let label = chasm_prompt::variant_def(selected.variant_id)
                .map(|def| def.label.to_string())
                .unwrap_or_else(|| "Default".to_string());
            // Travel macros for the preview: the newest live en-route journey
            // when one exists, else sample values so a "traveling" preview
            // still reads like a sentence.
            let travel = flags.traveling.then(|| {
                newest_live_travel(&state).unwrap_or(movement::ActiveTravel {
                    dest_name: "Prospector Saloon".to_string(),
                    arrive_total_hours: 15.0,
                })
            });
            (
                selected.template.to_string(),
                Some(selected.variant_id.to_string()),
                Some(label),
                travel,
            )
        }
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

    // Travel macros, exactly like a real turn: empty when not traveling.
    let (travel_dest, travel_arrival) = match &travel {
        Some(travel) => (
            travel.dest_name.clone(),
            movement::format_game_hour(travel.arrive_total_hours),
        ),
        None => (String::new(), String::new()),
    };
    macros.insert("travel_destination".to_string(), travel_dest);
    macros.insert("travel_arrival_time".to_string(), travel_arrival);

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
    if variant_id.is_some() && travel.is_some() && newest_live_travel(&state).is_none() {
        notes.push(
            "No live journey right now — travel macros use sample values in this preview."
                .to_string(),
        );
    }

    Ok(Json(UiGlobalsPreview {
        resolved,
        macros: serde_json::to_value(&macros).unwrap_or_else(|_| json!({})),
        updated_at,
        note: (!notes.is_empty()).then(|| notes.join(" ")),
        variant_id,
        variant_label,
    }))
}

/// The newest live EN-ROUTE journey (any NPC) — the state-picker preview's
/// source for realistic travel macros.
fn newest_live_travel(state: &AppState) -> Option<movement::ActiveTravel> {
    let store = movement::read_store(state);
    store
        .journeys
        .iter()
        .filter(|journey| journey.state == movement::JourneyState::EnRoute)
        .max_by_key(|journey| journey.created_at_ms)
        .map(|journey| movement::ActiveTravel {
            dest_name: journey.dest_name.clone(),
            arrive_total_hours: journey.arrive_total_hours,
        })
}
