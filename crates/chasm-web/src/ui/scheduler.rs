//! Scheduler UI API — the read-only "Schedule" page: list the active/So-far NPC
//! scheduled tasks (owner, trigger, action, state) with a cancel button, plus the
//! current in-game day/time (handy for testing time-triggered tasks).
//!
//! Mirrors [`crate::scheduler`]: the store is per-playthrough and save-aware; this
//! is just the projection + the user's cancel + a test hook to raise a condition
//! flag (stands in for the `task/event-log` game-event stream until it lands).
//!
//!   * `GET  /api/ui/v1/scheduler`              — clock + tasks.
//!   * `POST /api/ui/v1/scheduler/:id/cancel`   — cancel a pending/active task.
//!   * `POST /api/ui/v1/scheduler/event`        — raise a condition flag on a task
//!     (e.g. `looted`), the minimal condition signal until the event stream lands.

use std::sync::Arc;

use axum::{
    extract::{Path as AxPath, State},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::movement::{self, JourneyState};
use crate::scheduler::{self, TaskState, Trigger};
use crate::{AppState, WebResult};

/// The in-game clock for the page header (null before a save loads / the plugin
/// reports it).
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ClockView {
    day: u32,
    hour: f64,
    /// "1:00 AM"-style label.
    label: String,
}

/// One task row.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TaskView {
    id: String,
    owner_npc_key: String,
    owner_name: String,
    character_name: String,
    action: String,
    summary: String,
    state: String,
    /// "In-game time" | "Condition" — the trigger category, for the table.
    trigger_kind: String,
    /// Human trigger detail ("Day 4, 1:00 AM" or "When Boone reaches the body").
    trigger_detail: String,
    /// Chain progress "2/4" for composite tasks, empty for one-shots.
    progress: String,
    last_error: String,
    created_at_ms: i64,
    /// "task" (a scheduler task) or "journey" (a travel journey) — the UI routes
    /// cancel to the right endpoint by this.
    kind: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SchedulerView {
    /// False when the profile declares no companion support (scheduler needs the
    /// companion command channel to fire). Advisory only — tasks still list.
    clock: Option<ClockView>,
    tasks: Vec<TaskView>,
}

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

/// Turn an absolute in-game total-hour into "Day D, H:MM AM" (for travel rows).
fn format_total_hours(total: f64) -> String {
    let total = total.max(0.0);
    let day = (total / 24.0).floor() as i64;
    let hour = total - (day as f64) * 24.0;
    format!("Day {day}, {}", format_hour_label(hour))
}

fn trigger_view(trigger: &Trigger) -> (String, String) {
    match trigger {
        Trigger::Time { day, hour } => (
            "In-game time".to_string(),
            format!("Day {day}, {}", format_hour_label(*hour)),
        ),
        Trigger::Condition { condition } => {
            let name = match serde_json::to_value(condition) {
                Ok(Value::Object(map)) => map
                    .get("condition")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                _ => String::new(),
            };
            // The common chain trigger is `immediate` (fires as soon as the prior
            // step is done); show it as a readable "when ready" rather than jargon.
            if name == "immediate" {
                ("Sequence".to_string(), "When the previous step is done".to_string())
            } else {
                ("Condition".to_string(), name.replace('_', " "))
            }
        }
    }
}

fn state_label(state: TaskState) -> &'static str {
    match state {
        TaskState::Pending => "pending",
        TaskState::Active => "active",
        TaskState::Done => "done",
        TaskState::Cancelled => "cancelled",
        TaskState::Failed => "failed",
    }
}

/// `GET /api/ui/v1/scheduler` — the current in-game clock + all scheduled tasks
/// (newest first).
pub(crate) async fn list_scheduler(State(state): State<Arc<AppState>>) -> Json<SchedulerView> {
    let clock = scheduler::current_clock(&state).map(|c| ClockView {
        day: c.day as u32,
        hour: c.hour,
        label: format_hour_label(c.hour),
    });
    let store = scheduler::read_store(&state);
    let mut tasks: Vec<TaskView> = store
        .tasks
        .iter()
        .map(|t| {
            let (trigger_kind, trigger_detail) = trigger_view(&t.trigger);
            let progress = if t.chain.is_empty() {
                String::new()
            } else {
                let done = t.chain.iter().filter(|s| s.done).count();
                format!("{}/{}", done, t.chain.len())
            };
            TaskView {
                id: t.id.clone(),
                owner_npc_key: t.owner_npc_key.clone(),
                owner_name: t.owner_name.clone(),
                character_name: t.character_name.clone(),
                action: t.action.clone(),
                summary: t.summary.clone(),
                state: state_label(t.state).to_string(),
                trigger_kind,
                trigger_detail,
                progress,
                last_error: t.last_error.clone(),
                created_at_ms: t.created_at_ms,
                kind: "task".to_string(),
            }
        })
        .collect();

    // Travel journeys share this board: a journey is a scheduled/active travel,
    // shown alongside time-triggered actions (the Travel page is settings-only).
    let now_total = scheduler::current_clock(&state).map(|c| c.total_hours());
    let jstore = movement::read_store(&state);
    for j in &jstore.journeys {
        let state_str = match j.state {
            JourneyState::Waiting => "pending",
            JourneyState::EnRoute => "active",
            JourneyState::Arrived => "done",
            JourneyState::Cancelled => "cancelled",
            JourneyState::Failed => "failed",
        };
        let progress = if matches!(j.state, JourneyState::EnRoute) {
            let pct = now_total.map(|n| (j.progress(n) * 100.0).round() as u32).unwrap_or(0);
            format!("{pct}%")
        } else {
            String::new()
        };
        tasks.push(TaskView {
            id: j.id.clone(),
            owner_npc_key: j.npc_key.clone(),
            owner_name: j.npc_name.clone(),
            character_name: j.character_name.clone(),
            action: "travel".to_string(),
            summary: format!("Travel to {}", j.dest_name),
            state: state_str.to_string(),
            trigger_kind: "Travel".to_string(),
            trigger_detail: format!(
                "Leaves {}, arrives {}",
                format_total_hours(j.depart_total_hours),
                format_total_hours(j.arrive_total_hours)
            ),
            progress,
            last_error: j.last_error.clone(),
            created_at_ms: j.created_at_ms,
            kind: "journey".to_string(),
        });
    }

    // Newest first.
    tasks.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));
    Json(SchedulerView { clock, tasks })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MutationResult {
    ok: bool,
    /// Empty on success, else why nothing changed (e.g. "not_found").
    error: String,
}

/// `POST /api/ui/v1/scheduler/:id/cancel` — cancel a pending/active task. A
/// terminal task (done/failed/already-cancelled) is a no-op success.
pub(crate) async fn cancel_task(
    State(state): State<Arc<AppState>>,
    AxPath(id): AxPath<String>,
) -> WebResult<Json<MutationResult>> {
    let mut store = scheduler::read_store(&state);
    let Some(task) = store.tasks.iter_mut().find(|t| t.id == id) else {
        return Ok(Json(MutationResult { ok: false, error: "not_found".into() }));
    };
    if matches!(task.state, TaskState::Pending | TaskState::Active) {
        task.state = TaskState::Cancelled;
        scheduler::write_store(&state, &store)?;
        tracing::info!("scheduler: cancelled task {id}");
    }
    Ok(Json(MutationResult { ok: true, error: String::new() }))
}

/// Body for the condition-flag test hook.
#[derive(Deserialize)]
pub(crate) struct EventBody {
    task_id: String,
    /// The flag to raise (e.g. "looted"), matching a chain step's `FlagSet`.
    flag: String,
    #[serde(default = "default_true")]
    value: bool,
}

fn default_true() -> bool {
    true
}

/// `POST /api/ui/v1/scheduler/event` — raise a condition flag on a task. This is
/// the MINIMAL condition signal (stands in for the `task/event-log` game-event
/// stream): the flag is stored on the task under `args._flags`, so the fetch
/// chain's `looted` step can advance in testing, and it rolls back with the save
/// exactly like the rest of the task.
pub(crate) async fn raise_event(
    State(state): State<Arc<AppState>>,
    Json(body): Json<EventBody>,
) -> WebResult<Json<MutationResult>> {
    let mut store = scheduler::read_store(&state);
    let Some(task) = store.tasks.iter_mut().find(|t| t.id == body.task_id) else {
        return Ok(Json(MutationResult { ok: false, error: "not_found".into() }));
    };
    let flags = task
        .args
        .entry("_flags".to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if let Some(map) = flags.as_object_mut() {
        map.insert(body.flag.clone(), Value::Bool(body.value));
    }
    scheduler::write_store(&state, &store)?;
    tracing::info!(
        "scheduler: raised flag '{}'={} on task {}",
        body.flag,
        body.value,
        body.task_id
    );
    Ok(Json(MutationResult { ok: true, error: String::new() }))
}
