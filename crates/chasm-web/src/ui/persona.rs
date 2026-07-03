//! UI persona domain — the Persona page backend.
//!
//! Two endpoints under `/api/ui/v1` (registered in `ui.rs`, mirroring the
//! gamestate domain):
//!
//!   * `GET  /persona`            — the stored persona view: the generated
//!     description + provenance, the character-data snapshot it used, and
//!     timestamps (empty state before the first capture).
//!   * `POST /persona/regenerate` — re-runs generation from the LAST received
//!     capture (the manual test hook), awaiting the result so the page can
//!     show the fresh description in one round-trip.
//!
//! Like the rest of `/api/ui/v1`, this is UI-only; the mod uploads captures on
//! the game transport (`POST /api/game/v1/persona`, see `crate::persona`).

use std::sync::Arc;

use std::fs;

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{persona, AppState, WebError, WebResult};

/// `GET /api/ui/v1/persona` response.
#[derive(Serialize)]
pub(crate) struct UiPersonaView {
    /// The generated third-person description (`None` before first generation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// When the description was generated (`None` before first generation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_at: Option<String>,
    /// When the underlying capture was taken in-game / received.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub captured_at: Option<String>,
    /// `"game_data"` (older records may carry `"vision"` / `"stats_only"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Human note on the generation outcome.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_note: Option<String>,
    /// Last generation error (kept alongside a previous good description).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_error: Option<String>,
    /// The exact prompt text sent to the LLM for the current description
    /// (records generated before prompt persistence existed have none).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// The character-data snapshot the description was generated from (else
    /// the latest received capture's snapshot; `{}` before any capture).
    pub stats: Value,
    /// True while a generation task is currently running.
    pub generating: bool,
    /// True when a capture exists (Regenerate is meaningful).
    pub has_capture: bool,
    /// The user's custom addition — a free-text paragraph appended to the
    /// persona at injection, persisted separately so it survives regeneration.
    /// Empty string when the user has never set one.
    pub custom_note: String,
}

/// Pulls a trimmed non-empty string field out of a JSON object.
fn field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

/// Builds the view from the on-disk store (persona.json + capture.json).
fn build_view(state: &AppState) -> UiPersonaView {
    let dir = persona::persona_dir(state);
    let stored = persona::read_json(&persona::persona_path(&dir));
    let capture = persona::read_json(&persona::capture_path(&dir));

    let stats = stored
        .as_ref()
        .and_then(|persona| persona.get("stats").filter(|stats| stats.is_object()).cloned())
        .or_else(|| {
            // Before the first generation, show the latest capture's snapshot
            // (same keys; the receive path stores display strings verbatim).
            capture.as_ref().map(|capture| {
                let mut map = serde_json::Map::new();
                for key in [
                    "player_name",
                    "level",
                    "special",
                    "skills",
                    "perks",
                    "equipped_weapon",
                    "equipped_apparel",
                    "location",
                    "sex",
                    "race",
                    "age_years",
                    "hair_color",
                    "hair_style",
                    "hair_length",
                    "eye_color",
                    "facial_hair",
                ] {
                    if let Some(value) = capture.get(key) {
                        if value.is_string() || value.is_number() {
                            map.insert(key.to_string(), value.clone());
                        }
                    }
                }
                Value::Object(map)
            })
        })
        .unwrap_or_else(|| json!({}));

    let stored_ref = stored.as_ref();
    UiPersonaView {
        description: stored_ref.and_then(|persona| field(persona, "description")),
        generated_at: stored_ref.and_then(|persona| field(persona, "generated_at")),
        captured_at: stored_ref
            .and_then(|persona| field(persona, "captured_at"))
            .or_else(|| {
                capture.as_ref().and_then(|capture| {
                    field(capture, "captured_at").or_else(|| field(capture, "received_at"))
                })
            }),
        source: stored_ref.and_then(|persona| field(persona, "source")),
        model_note: stored_ref.and_then(|persona| field(persona, "model_note")),
        generation_error: stored_ref.and_then(|persona| field(persona, "generation_error")),
        prompt: stored_ref.and_then(|persona| field(persona, "prompt")),
        stats,
        generating: persona::generation_in_flight(),
        has_capture: capture.is_some(),
        custom_note: persona::read_custom_note(&dir),
    }
}

/// `GET /api/ui/v1/persona` — the stored persona view (never errors; the empty
/// store renders the page's empty state).
pub(crate) async fn persona_view(
    State(state): State<Arc<AppState>>,
) -> WebResult<Json<UiPersonaView>> {
    Ok(Json(build_view(&state)))
}

/// `POST /api/ui/v1/persona/regenerate` — re-runs generation from the last
/// received capture and returns the refreshed view. Errors when no capture has
/// been received yet; a generation FAILURE is not an error (the view carries
/// `generation_error` and keeps the previous description).
pub(crate) async fn persona_regenerate(
    State(state): State<Arc<AppState>>,
) -> WebResult<Json<UiPersonaView>> {
    persona::generate_from_stored_capture(&state).await?;
    Ok(Json(build_view(&state)))
}

/// `POST /api/ui/v1/persona/custom` body — the user's custom addition.
#[derive(Deserialize)]
pub(crate) struct SetCustomNote {
    /// The free-text paragraph to append to the persona (stored trimmed; an
    /// empty/whitespace value clears the addition).
    pub note: String,
}

/// `POST /api/ui/v1/persona/custom` — persists the user's custom addition to
/// its own file in the persona store (SEPARATE from the generated description,
/// so it survives regeneration) and returns the refreshed view. Read per turn
/// by prompt assembly, so it applies on the next NPC turn with no restart.
pub(crate) async fn persona_set_custom(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SetCustomNote>,
) -> WebResult<Json<UiPersonaView>> {
    let dir = persona::persona_dir(&state);
    fs::create_dir_all(&dir).map_err(WebError::from)?;
    persona::write_custom_note(&dir, &body.note).map_err(WebError::from)?;
    Ok(Json(build_view(&state)))
}
