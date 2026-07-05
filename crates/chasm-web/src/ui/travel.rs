//! Travel UI API — the "Travel" page: the NPC movement/travel system's settings
//! plus a live view of every active journey (who is walking where, when they left,
//! when they arrive, how far along they are).
//!
//! Mirrors [`crate::movement`]: the journey store is per-playthrough and save-aware;
//! this is the projection + the settings round-trip + a cancel.
//!
//!   * `GET  /api/ui/v1/travel`               — clock + settings + journeys.
//!   * `POST /api/ui/v1/travel/settings`      — save the movement settings.
//!   * `POST /api/ui/v1/travel/:id/cancel`    — cancel an in-progress journey.

use std::sync::Arc;

use axum::{
    extract::{Path as AxPath, State},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::movement::{self, JourneyState};
use crate::scheduler;
use crate::{AppState, WebResult};

// ---------------------------------------------------------------------------
// View DTOs
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ClockView {
    day: u32,
    hour: f64,
    label: String,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SettingsView {
    enabled: bool,
    /// Metres of world distance per in-game hour.
    walk_speed: f32,
    offscreen_simulation: bool,
    waypoint_stride: f32,
}

impl From<&chasm_core::MovementSettings> for SettingsView {
    fn from(s: &chasm_core::MovementSettings) -> Self {
        SettingsView {
            enabled: s.enabled,
            walk_speed: s.walk_speed,
            offscreen_simulation: s.offscreen_simulation,
            waypoint_stride: s.waypoint_stride,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JourneyView {
    id: String,
    npc_name: String,
    character_name: String,
    dest_name: String,
    state: String,
    /// Straight-line route distance in metres (0 if the destination wasn't resolvable).
    distance_meters: f64,
    /// "Day 4, 8:00 AM" — when they left / will leave.
    depart_label: String,
    /// "Day 4, 10:00 AM" — when they arrive.
    arrive_label: String,
    /// 0..100 — fraction of the route covered right now.
    progress: u32,
    created_at_ms: i64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TravelView {
    clock: Option<ClockView>,
    settings: SettingsView,
    journeys: Vec<JourneyView>,
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

fn format_hour_label(hour: f64) -> String {
    let h = hour.floor() as i64;
    let m = ((hour - h as f64) * 60.0).round() as i64;
    let (h, m) = if m >= 60 { (h + 1, 0) } else { (h, m) };
    let hour24 = ((h % 24) + 24) % 24;
    let suffix = if hour24 < 12 { "AM" } else { "PM" };
    let mut h12 = hour24 % 12;
    if h12 == 0 {
        h12 = 12;
    }
    format!("{h12}:{m:02} {suffix}")
}

/// Turn an absolute in-game total-hour into "Day D, H:MM AM".
fn format_total_hours(total: f64) -> String {
    let total = total.max(0.0);
    let day = (total / 24.0).floor() as i64;
    let hour = total - (day as f64) * 24.0;
    format!("Day {day}, {}", format_hour_label(hour))
}

fn state_label(state: JourneyState) -> &'static str {
    match state {
        JourneyState::Waiting => "waiting",
        JourneyState::EnRoute => "en route",
        JourneyState::Arrived => "arrived",
        JourneyState::Cancelled => "cancelled",
        JourneyState::Failed => "failed",
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `GET /api/ui/v1/travel` — clock + movement settings + all journeys (newest first).
pub(crate) async fn list_travel(State(state): State<Arc<AppState>>) -> Json<TravelView> {
    let clock = scheduler::current_clock(&state);
    let now_total = clock.map(|c| c.total_hours());
    let clock_view = clock.map(|c| ClockView {
        day: c.day as u32,
        hour: c.hour,
        label: format_hour_label(c.hour),
    });

    let settings = chasm_core::AppSettings::load(&state.config.settings_path).movement;
    let store = movement::read_store(&state);

    let mut journeys: Vec<JourneyView> = store
        .journeys
        .iter()
        .map(|j| {
            let progress = now_total
                .map(|now| (j.progress(now) * 100.0).round() as u32)
                .unwrap_or(0);
            JourneyView {
                id: j.id.clone(),
                npc_name: j.npc_name.clone(),
                character_name: j.character_name.clone(),
                dest_name: j.dest_name.clone(),
                state: state_label(j.state).to_string(),
                distance_meters: j.distance_meters,
                depart_label: format_total_hours(j.depart_total_hours),
                arrive_label: format_total_hours(j.arrive_total_hours),
                progress,
                created_at_ms: j.created_at_ms,
            }
        })
        .collect();
    journeys.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));

    Json(TravelView {
        clock: clock_view,
        settings: (&settings).into(),
        journeys,
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MutationResult {
    ok: bool,
    error: String,
}

/// Body for the settings save (all fields optional → partial updates supported,
/// but the UI sends the full form).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SettingsBody {
    enabled: Option<bool>,
    walk_speed: Option<f32>,
    offscreen_simulation: Option<bool>,
    waypoint_stride: Option<f32>,
}

/// `POST /api/ui/v1/travel/settings` — persist the movement settings; returns the
/// fresh view so the UI reflects any normalization.
pub(crate) async fn save_settings(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SettingsBody>,
) -> WebResult<Json<SettingsView>> {
    let mut settings = chasm_core::AppSettings::load(&state.config.settings_path);
    if let Some(v) = body.enabled {
        settings.movement.enabled = v;
    }
    if let Some(v) = body.walk_speed {
        // Guard against a zero/negative speed (would make ETA infinite): clamp low.
        settings.movement.walk_speed = v.max(1.0);
    }
    if let Some(v) = body.offscreen_simulation {
        settings.movement.offscreen_simulation = v;
    }
    if let Some(v) = body.waypoint_stride {
        settings.movement.waypoint_stride = v.max(1.0);
    }
    settings.save(&state.config.settings_path)?;
    tracing::info!(
        "travel: saved settings (enabled={}, walk_speed={}, offscreen_sim={})",
        settings.movement.enabled,
        settings.movement.walk_speed,
        settings.movement.offscreen_simulation
    );
    Ok(Json((&settings.movement).into()))
}

/// `POST /api/ui/v1/travel/:id/cancel` — cancel an in-progress journey (a terminal
/// one is a no-op success). The NPC is left wherever they currently are.
pub(crate) async fn cancel_journey(
    State(state): State<Arc<AppState>>,
    AxPath(id): AxPath<String>,
) -> WebResult<Json<MutationResult>> {
    let mut store = movement::read_store(&state);
    let Some(journey) = store.journeys.iter_mut().find(|j| j.id == id) else {
        return Ok(Json(MutationResult { ok: false, error: "not_found".into() }));
    };
    if matches!(journey.state, JourneyState::Waiting | JourneyState::EnRoute) {
        journey.state = JourneyState::Cancelled;
        movement::write_store(&state, &store)?;
        tracing::info!("travel: cancelled journey {id}");
    }
    Ok(Json(MutationResult { ok: true, error: String::new() }))
}
