//! NPC scheduler ("cronjob") — companions and the in-conversation NPC can
//! schedule an action to fire at a specific IN-GAME TIME or when a CONDITION is
//! met, without the model ever authoring a task spec.
//!
//! Design principle: the LLM picks ONE high-level action with 1-2 natural args
//! (a destination + a plain-English time, or a fetch target); chasm does the rest
//! — parsing the time, expanding a composite action into a conditional chain,
//! persisting the task, and firing the resulting game command when the trigger is
//! met. The model never sees steps, conditions, or day/hour numbers.
//!
//! Surfaces (all additive, scoped to companions + the conversing NPC):
//!   * `meet_player(where, when)` — a Time task: travel to the player at `when`.
//!   * `fetch_loot(target)`       — a Condition CHAIN: go to the target, loot it,
//!     return, hand it over (each step gated on the prior one completing).
//!   * `schedule(action, when)`   — the general escape-hatch primitive.
//!
//! Plumbing:
//!   * In-game clock: the NVSE plugin reports GameDaysPassed + GameHour in the
//!     runtime heartbeat (`<bridge>/runtime_heartbeat.json` `.game`) every ~100ms
//!     and in each turn's `metadata.macros`. [`current_clock`] reads the heartbeat
//!     so tasks fire even while the player is idle (not in a dialogue turn).
//!   * Store: [`SchedulerStore`] persists under the profile
//!     (`headless/scheduler.json`, [`chasm_core::ProfilePaths::scheduler_store`]),
//!     so it lives beside the save-sync snapshots and rolls back with the save.
//!   * Condition engine: a MINIMAL, self-contained predicate set evaluated over a
//!     [`WorldSnapshot`] (player/npc positions + a few event flags). This overlaps
//!     the in-flight `task/event-log` feature (a game-event stream is the natural
//!     substrate for conditions); it is kept deliberately separable so the human
//!     integrator can later back these predicates onto that stream. See the module
//!     `condition` section.
//!   * Tick: [`tick`] runs on a timer (spawned beside the in-process bridge),
//!     reads the clock, evaluates pending tasks, and fires due ones by writing a
//!     companion command file the plugin already polls. Fire-and-forget: a failed
//!     task logs and is marked, never hanging a turn.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::AppState;

/// Current schema version of the on-disk store; bumped if the shape changes.
const STORE_VERSION: u32 = 1;

/// How close (game units) an actor must be to a target position for an
/// "arrived"/"returned" condition to count as met. 256 units ≈ 4.6 m — a couple
/// of body-lengths, generous enough to survive pathing jitter.
const ARRIVE_RADIUS_UNITS: f64 = 256.0;

// ===========================================================================
// In-game clock
// ===========================================================================

/// The current in-game time, read from the plugin's runtime heartbeat. `day` is
/// the monotonic GameDaysPassed counter (increments ~1.0/in-game-day) and `hour`
/// is the 0..24 GameHour wall clock. Together they give a total ordering of
/// in-game time via [`Self::total_hours`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct GameClock {
    pub day: f64,
    pub hour: f64,
}

impl GameClock {
    /// Build a clock from the game's raw values. `days_passed` (GameDaysPassed) is
    /// a FRACTIONAL counter whose fraction already IS the time of day, so we floor
    /// it to the integer day number — otherwise `total_hours` (day*24 + hour) would
    /// double-count the time of day and every scheduled task would look overdue.
    pub fn from_game(days_passed: f64, hour: f64) -> Self {
        GameClock { day: days_passed.floor(), hour }
    }

    /// Absolute in-game hours since day 0 — the scalar tasks are ordered by.
    pub fn total_hours(&self) -> f64 {
        self.day * 24.0 + self.hour
    }
}

/// Reads the plugin's runtime heartbeat and returns the current in-game clock,
/// or `None` at the main menu / before a save loads (`clock_valid == false`) or
/// if the heartbeat is missing/unparseable. The heartbeat is the AUTHORITY for
/// idle firing — it is rewritten every ~100ms regardless of conversation.
pub fn current_clock(state: &AppState) -> Option<GameClock> {
    let path = bridge_root(state).join("runtime_heartbeat.json");
    let text = std::fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    let game = value.get("game")?;
    if !game.get("clock_valid").and_then(Value::as_bool).unwrap_or(false) {
        return None;
    }
    // The plugin emits these as JSON strings (stable across ostream flags); accept
    // a number too, for robustness against a future format tweak.
    let day = parse_num_field(game.get("days_passed"))?;
    let hour = parse_num_field(game.get("hour"))?;
    Some(GameClock::from_game(day, hour))
}

fn parse_num_field(value: Option<&Value>) -> Option<f64> {
    match value {
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

// ===========================================================================
// Task model
// ===========================================================================

/// What fires a task: an absolute in-game time, or a condition predicate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Trigger {
    /// Fires when the in-game clock reaches (or passes) `day`+`hour`.
    Time { day: u32, hour: f64 },
    /// Fires when `condition` evaluates true against the current world snapshot.
    Condition {
        #[serde(flatten)]
        condition: Condition,
    },
}

/// The lifecycle of a task. `pending` → `active` (chain in progress) → `done`,
/// or `cancelled` (user) / `failed` (a step couldn't be issued).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Pending,
    Active,
    Done,
    Cancelled,
    Failed,
}

impl TaskState {
    fn is_terminal(self) -> bool {
        matches!(self, TaskState::Done | TaskState::Cancelled | TaskState::Failed)
    }
}

/// One step of a composite chain (e.g. fetch_loot: travel → loot → return →
/// give). A step becomes eligible only once the previous step is `done`; it then
/// waits for its own `trigger` before issuing `command` and completing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChainStep {
    /// Short id for logs/UI (e.g. "travel", "loot", "return", "give").
    pub id: String,
    /// Human description shown in the UI ("Travel to the body").
    pub description: String,
    /// The trigger that advances PAST this step (met → issue command, mark done).
    pub trigger: Trigger,
    /// The game command to issue when this step fires (a companion op payload).
    /// `null` for a pure wait/marker step.
    pub command: Value,
    /// Extra real-time delay (ms) applied AFTER the trigger is met, from an
    /// `after:"30 seconds"` on the step. 0 = fire as soon as the trigger is met.
    #[serde(default)]
    pub delay_ms: u64,
    /// Epoch ms when the trigger first became satisfied (0 = not yet). Used with
    /// `delay_ms` to hold the step for the delay before issuing its command.
    #[serde(default)]
    pub armed_at_ms: i64,
    pub done: bool,
}

/// A scheduled task. One per model action; a composite action carries its chain.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SchedulerTask {
    pub id: String,
    /// The plugin npc key of the owner (companion / conversing NPC).
    pub owner_npc_key: String,
    /// Display name of the owner (for UI).
    pub owner_name: String,
    /// The chasm character card name of the owner (for UI / logs).
    pub character_name: String,
    /// The live-chat this was scheduled in (for context / future speak-back).
    pub live_chat_id: String,
    /// The action alias the model chose (`meet_player` / `fetch_loot` / …).
    pub action: String,
    /// The natural-language args the model gave, preserved verbatim (where/when/
    /// target/…). Surfaced in the UI so the user sees what was asked.
    pub args: Map<String, Value>,
    /// One-line human summary ("Meet you at Prospector Saloon at 1:00.").
    pub summary: String,
    /// The top-level trigger. For chains, this mirrors the FIRST pending step's
    /// trigger; `chain` holds the full sequence.
    pub trigger: Trigger,
    /// Composite steps (empty for a simple one-shot task like meet_player).
    #[serde(default)]
    pub chain: Vec<ChainStep>,
    pub state: TaskState,
    /// Last error (if `failed`), for the UI / logs.
    #[serde(default)]
    pub last_error: String,
    /// Epoch millis when created / fired (for UI ordering + display).
    pub created_at_ms: i64,
    #[serde(default)]
    pub fired_at_ms: i64,
    /// The in-game clock at creation (so relative times are anchored + shown).
    pub created_day: u32,
    pub created_hour: f64,
}

/// The persisted store: a flat list of tasks + a schema version.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SchedulerStore {
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub tasks: Vec<SchedulerTask>,
}

impl SchedulerStore {
    fn active_tasks(&self) -> impl Iterator<Item = &SchedulerTask> {
        self.tasks.iter().filter(|t| !t.state.is_terminal())
    }
}

// ===========================================================================
// Condition engine (minimal, self-contained, separable)
// ===========================================================================

/// A predicate over the current world. Deliberately small and evaluated against
/// a [`WorldSnapshot`] the plugin already gives us (positions) plus a few event
/// flags — NOT tied to any external event stream. The `task/event-log` feature's
/// game-event stream is the natural substrate for a richer version of these; this
/// is kept separable so the integrator can re-home the evaluation later without
/// touching callers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "condition", rename_all = "snake_case")]
pub enum Condition {
    /// The owner NPC is within [`ARRIVE_RADIUS_UNITS`] of a world position.
    NpcArrived { x: f64, y: f64, z: f64 },
    /// The owner NPC is within [`ARRIVE_RADIUS_UNITS`] of the player.
    NpcNearPlayer,
    /// The player has said something matching `phrase` since the task was created.
    /// Raised by [`note_player_message`] (an EVENT, not a timed poll): it sets the
    /// task flag [`player_said_flag`] when an incoming player line matches.
    PlayerSaid { phrase: String },
    /// A named actor (not the player) comes near the owner NPC. Cannot be evaluated
    /// from the current heartbeat (it only carries the player + owner positions), so
    /// it stays false until a plugin proximity signal raises its flag. Captured now
    /// so the intent persists and a later mod event can fire it.
    ActorNear { name: String },
    /// A named boolean flag has been set on the task (by a game event signal),
    /// e.g. "looted". Flags are stored in [`WorldSnapshot::flags`].
    FlagSet { flag: String },
    /// Always true — an immediate step (issues its command on the next tick).
    Immediate,
}

/// The task flag raised when the player says a [`Condition::PlayerSaid`] phrase.
/// Derived from the phrase so the event side and the predicate agree on the key.
pub fn player_said_flag(phrase: &str) -> String {
    format!("said:{}", phrase.trim().to_ascii_lowercase())
}

/// The "this travel step is DONE" predicate for a handed-off journey: the NPC has
/// reached the destination. "come to me" gates on nearness to the player; a named
/// place on arrival at its resolved position; an unresolved place doesn't block the
/// chain (fires the next step immediately, since we can't detect its arrival).
fn arrival_condition(journey: &crate::movement::Journey) -> Condition {
    if crate::movement::is_player_dest(&journey.dest_name) {
        Condition::NpcNearPlayer
    } else if let Some(p) = journey.dest_pos {
        Condition::NpcArrived { x: p.x, y: p.y, z: p.z }
    } else {
        Condition::Immediate
    }
}

/// The world facts a condition is evaluated against on a tick. Assembled from the
/// runtime heartbeat (player/last-npc positions) + the task's own event flags.
/// Keeping this a plain struct (not a live query) is what makes the condition
/// engine unit-testable and event-stream-agnostic.
#[derive(Debug, Clone, Default)]
pub struct WorldSnapshot {
    pub player: Option<(f64, f64, f64)>,
    pub npc: Option<(f64, f64, f64)>,
    /// Event flags raised on the owning task (e.g. "looted": true).
    pub flags: std::collections::BTreeMap<String, bool>,
}

impl Condition {
    /// Evaluate the predicate. Unknown/unavailable facts → `false` (never fire on
    /// missing data), except [`Condition::Immediate`] which is always true.
    pub fn is_met(&self, world: &WorldSnapshot) -> bool {
        match self {
            Condition::Immediate => true,
            Condition::FlagSet { flag } => world.flags.get(flag).copied().unwrap_or(false),
            Condition::PlayerSaid { phrase } => {
                world.flags.get(&player_said_flag(phrase)).copied().unwrap_or(false)
            }
            // Not evaluable from the heartbeat's player+owner positions; fires only
            // if a plugin proximity event has raised its flag.
            Condition::ActorNear { name } => {
                world.flags.get(&format!("near:{}", name.to_ascii_lowercase())).copied().unwrap_or(false)
            }
            Condition::NpcNearPlayer => match (world.player, world.npc) {
                (Some(p), Some(n)) => within(p, n, ARRIVE_RADIUS_UNITS),
                _ => false,
            },
            Condition::NpcArrived { x, y, z } => match world.npc {
                Some(n) => within((*x, *y, *z), n, ARRIVE_RADIUS_UNITS),
                None => false,
            },
        }
    }
}

fn within(a: (f64, f64, f64), b: (f64, f64, f64), radius: f64) -> bool {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    let dz = a.2 - b.2;
    (dx * dx + dy * dy + dz * dz).sqrt() <= radius
}

// ===========================================================================
// Store persistence (write-safe under the profile, like persona/relationships)
// ===========================================================================

fn store_path(state: &AppState) -> PathBuf {
    state.config.active_profile_paths().scheduler_store()
}

/// Read the store (an empty store if the file is absent/corrupt — a fresh
/// playthrough, or one restored to before any task existed).
pub fn read_store(state: &AppState) -> SchedulerStore {
    let path = store_path(state);
    match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
        Err(_) => SchedulerStore::default(),
    }
}

/// Persist the store atomically (tmp + rename), creating the parent dir.
pub fn write_store(state: &AppState, store: &SchedulerStore) -> anyhow::Result<()> {
    let path = store_path(state);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut out = store.clone();
    out.version = STORE_VERSION;
    let text = serde_json::to_string_pretty(&out)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text.as_bytes())?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

// ===========================================================================
// Save-aware rollback (sidecar keyed by the save-sync checkpoint id)
// ===========================================================================
//
// The scheduler store rolls back with the save exactly like chat history: on a
// save checkpoint we snapshot `scheduler.json` into a sidecar keyed by the
// checkpoint id; on load we restore it. This lives beside — and reuses — the
// existing save-sync checkpoint id (SHA256(gameId|saveId)), hooked at the
// save-sync event dispatch (see save_sync.rs). Kept as plain byte copies (not a
// re-serialize) so it is trivially correct and independent of the store shape.

/// `content_root/headless/scheduler.json` — the live store (same path
/// [`chasm_core::ProfilePaths::scheduler_store`] resolves for the active profile).
fn scheduler_store_path_at(content_root: &Path) -> PathBuf {
    content_root.join("headless").join("scheduler.json")
}

/// The per-checkpoint sidecar snapshot of the store.
fn scheduler_checkpoint_path(content_root: &Path, checkpoint_id: &str) -> PathBuf {
    content_root
        .join("headless")
        .join("save-sync")
        .join("scheduler-checkpoints")
        .join(format!("{checkpoint_id}.json"))
}

const EMPTY_STORE_JSON: &[u8] = b"{\"version\":1,\"tasks\":[]}";

/// Snapshot the scheduler store for a save checkpoint. If no store exists yet, an
/// EMPTY snapshot is written so a later restore to this checkpoint correctly
/// CLEARS any tasks scheduled after it (rollback of a discarded branch).
pub fn checkpoint_scheduler_store(content_root: &Path, checkpoint_id: &str) {
    let dst = scheduler_checkpoint_path(content_root, checkpoint_id);
    if let Some(dir) = dst.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match std::fs::read(scheduler_store_path_at(content_root)) {
        Ok(bytes) => {
            let _ = std::fs::write(&dst, bytes);
        }
        Err(_) => {
            let _ = std::fs::write(&dst, EMPTY_STORE_JSON);
        }
    }
    tracing::info!("scheduler: checkpointed store for {checkpoint_id}");
}

/// Restore the scheduler store from a checkpoint's sidecar on load. A missing
/// sidecar means the save predates any scheduled task, so the live store is
/// CLEARED (a task scheduled in the now-discarded branch vanishes).
pub fn restore_scheduler_store(content_root: &Path, checkpoint_id: &str) {
    let dst = scheduler_store_path_at(content_root);
    if let Some(dir) = dst.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match std::fs::read(scheduler_checkpoint_path(content_root, checkpoint_id)) {
        Ok(bytes) => {
            let _ = std::fs::write(&dst, bytes);
            tracing::info!("scheduler: restored store from checkpoint {checkpoint_id}");
        }
        Err(_) => {
            let _ = std::fs::write(&dst, EMPTY_STORE_JSON);
            tracing::info!("scheduler: cleared store (no sidecar for {checkpoint_id})");
        }
    }
}

// ===========================================================================
// Action-call parsing (function-call syntax)
// ===========================================================================
//
// The model emits each action as a FUNCTION CALL string — the shape it is most
// reliably trained on, which removes the ambiguity of free-form natural language
// (where does the place end, is that a time or a target, etc.):
//   * wave()                                  -> a gesture, now.
//   * wave(at="1:00AM")                       -> schedule the gesture for 1am.
//   * travel(to="Prospector Saloon")          -> go there now (fires next tick).
//   * travel(to="Prospector Saloon", at="3:00PM") -> go there at 3pm.
//   * attack(target="raider")                 -> aimed at someone.
// A call with `at=` or `to=` becomes a scheduled task; anything else fires
// immediately. `generate.rs normalize` calls [`parse_action_call`] and routes
// each into `structured.actions` (immediate) or `structured.scheduled`.

/// A parsed action CALL — the function-style form the model emits, e.g.
/// `travel(to="Prospector Saloon", at="3:00PM")`, `wave()`, `attack(target="raider")`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ActionCall {
    /// The action word (alias): "travel", "wave", "attack", …
    pub name: String,
    /// `to=` — a travel destination.
    pub to: Option<String>,
    /// `at=` — a clock time to run the action LATER instead of now.
    pub at: Option<String>,
    /// `target=` — who the action is aimed at.
    pub target: Option<String>,
}

impl ActionCall {
    /// True when the call is deferred (a time) or a travel (a destination) — i.e.
    /// it becomes a scheduled task rather than firing immediately.
    pub fn is_scheduled(&self) -> bool {
        self.at.is_some() || self.to.is_some()
    }
}

/// Parse a function-call action string: `name(key="value", …)`. Lenient about
/// quotes (`"`, `'`, or none) and spacing. Recognised keys: `to`/`dest`/
/// `destination`/`place`/`location`, `at`/`when`/`time`, `target`/`who`/`on`. A
/// single positional value maps to `to` for a travel action, else `target`.
/// Returns `None` when there are no parens (the caller then treats the whole
/// string as a bare action word, e.g. `wave`).
pub fn parse_action_call(raw: &str) -> Option<ActionCall> {
    let raw = raw.trim();
    let open = raw.find('(')?;
    let close = raw.rfind(')')?;
    if close < open {
        return None;
    }
    let name = raw[..open].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut call = ActionCall { name, ..Default::default() };
    for part in split_call_args(&raw[open + 1..close]) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((key, value)) = part.split_once('=') {
            let value = unquote(value);
            if value.is_empty() {
                continue;
            }
            match key.trim().to_ascii_lowercase().as_str() {
                "to" | "dest" | "destination" | "place" | "location" => call.to = Some(value),
                "at" | "when" | "time" => call.at = Some(value),
                "target" | "who" | "on" => call.target = Some(value),
                _ => {}
            }
        } else {
            // A positional value (no `key=`): a travel action reads it as the
            // destination, anything else as the target.
            let value = unquote(part);
            if value.is_empty() {
                continue;
            }
            if is_travel_verb(&call.name) && call.to.is_none() {
                call.to = Some(value);
            } else if call.target.is_none() {
                call.target = Some(value);
            }
        }
    }
    Some(call)
}

/// Split a call's argument list on top-level commas (commas inside quotes stay).
fn split_call_args(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for c in body.chars() {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                }
                cur.push(c);
            }
            None => {
                if c == '"' || c == '\'' {
                    quote = Some(c);
                    cur.push(c);
                } else if c == ',' {
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.push(c);
                }
            }
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// Strip a surrounding pair of matching quotes (`'` or `"`) and trim.
fn unquote(value: &str) -> String {
    let value = value.trim();
    let bytes = value.as_bytes();
    if value.len() >= 2
        && (bytes[0] == b'"' || bytes[0] == b'\'')
        && bytes[value.len() - 1] == bytes[0]
    {
        value[1..value.len() - 1].trim().to_string()
    } else {
        value.to_string()
    }
}

/// Verbs whose fire should bring the companion to the player (movement/hand-over
/// intents that have no first-class game action). Used as the firing fallback
/// when a step's verb didn't resolve to a native Action-Book action.
fn is_travel_verb(verb: &str) -> bool {
    let v = verb.to_ascii_lowercase();
    [
        "meet", "come", "go ", "go", "return", "bring", "give", "hand", "find",
        "rendezvous", "travel", "walk", "head", "arrive", "fetch", "get ", "deliver",
        "follow me",
    ]
    .iter()
    .any(|kw| v.contains(kw))
}

// ===========================================================================
// Scheduling (parsed phrase -> task)
// ===========================================================================

/// Turn a parsed, scheduled action into a persisted task. Called by the in-process
/// bridge client (`ChasmClient::schedule_task`) — fire-and-forget from the turn's
/// perspective. The `spec` (built in `run_turn`) is:
/// `{ owner_npc_key, owner_name, character_name, live_chat_id, raw, day, hour,
///    steps: [ { verb, action_id, when, command_body? } ] }`
/// where `command_body` is the pre-built native command file the bridge captured
/// at schedule time for steps whose verb resolved to a native Action-Book action.
pub fn schedule_from_spec(state: &AppState, spec: &Value) -> anyhow::Result<Option<SchedulerTask>> {
    let Some(steps_json) = spec.get("steps").and_then(Value::as_array) else {
        return Ok(None);
    };
    if steps_json.is_empty() {
        return Ok(None);
    }
    let owner_npc_key = str_field(spec, "owner_npc_key");
    let owner_name = first_nonempty(&[str_field(spec, "owner_name"), str_field(spec, "character_name")]);
    let raw = str_field(spec, "raw");
    // Anchor relative times to the in-game clock: the turn's macros if present,
    // else the live heartbeat, else day 0 hour 0 (still schedulable).
    let clock = match (parse_num_field(spec.get("day")), parse_num_field(spec.get("hour"))) {
        (Some(day), Some(hour)) => GameClock::from_game(day, hour),
        _ => current_clock(state).unwrap_or(GameClock { day: 0.0, hour: 0.0 }),
    };

    let mut chain: Vec<ChainStep> = Vec::new();
    for (i, st) in steps_json.iter().enumerate() {
        let verb = str_field(st, "verb");
        if verb.is_empty() {
            continue;
        }
        let when_text = st.get("when").and_then(Value::as_str).unwrap_or("").trim().to_string();
        let event = st.get("event").and_then(Value::as_str).unwrap_or("").trim().to_string();
        let after = st.get("after").and_then(Value::as_str).unwrap_or("").trim().to_string();
        let command_body = st.get("command_body").and_then(Value::as_str).unwrap_or("").to_string();
        let destination = st.get("destination").and_then(Value::as_str).unwrap_or("").trim().to_string();
        let delay_ms = parse_delay_ms(&after);

        // A step with a destination is a TRAVEL step. When the movement system is
        // enabled it owns travel end-to-end: it measures the distance, leaves early
        // and walks the NPC so they arrive on time (a live journey on the Travel
        // page) — so we hand it off here and do NOT add a scheduler chain step. When
        // movement is disabled, `start_journey` returns None and we fall through to
        // the legacy instant-at-trigger-time teleport step below.
        // EXCEPTION: an EVENT-gated travel ("run to X when attacked") can't be
        // pre-timed, so it stays a chain step and issues a travel command when the
        // event fires.
        if !destination.is_empty() && event.is_empty() {
            let arrive_at =
                parse_when(&when_text, clock).map(|(day, hour)| (day as f64) * 24.0 + hour);
            let who = crate::movement::Traveller {
                npc_key: owner_npc_key.clone(),
                npc_name: owner_name.clone(),
                character_name: str_field(spec, "character_name"),
                live_chat_id: str_field(spec, "live_chat_id"),
            };
            match crate::movement::start_journey(state, &who, &destination, arrive_at) {
                Ok(Some(journey)) => {
                    // Movement now walks the NPC. A travel step is only DONE when the
                    // NPC ARRIVES — so if more steps follow (e.g. "come to me THEN
                    // wave"), keep it in the chain as a gate whose completion is the
                    // arrival, and the next step triggers off that. A trailing travel
                    // (nothing after it) needs no gate: the journey/Travel page owns it.
                    let has_following = i + 1 < steps_json.len();
                    if has_following {
                        chain.push(ChainStep {
                            id: format!("{}_{}", i + 1, slugify(&verb)),
                            description: format!("Travel to {destination}"),
                            trigger: Trigger::Condition { condition: arrival_condition(&journey) },
                            command: Value::Null, // movement already walking; this only gates
                            delay_ms: 0,
                            armed_at_ms: 0,
                            done: false,
                        });
                    }
                    continue;
                }
                Ok(None) => {}           // disabled → legacy chain step below
                Err(error) => {
                    tracing::warn!("scheduler: movement handoff failed, using legacy travel: {error}");
                }
            }
        }

        // Trigger, in priority: an EVENT (`when` in the plan) classified to a
        // condition/time-of-day; else a clock time -> Time; else fire as soon as
        // the previous step is done (chain order = "then").
        let trigger = if !event.is_empty() {
            classify_event(&event, Some(clock))
        } else {
            match parse_when(&when_text, clock) {
                Some((day, hour)) => Trigger::Time { day, hour },
                None => Trigger::Condition { condition: Condition::Immediate },
            }
        };

        // Command, in priority order:
        //   * an explicit travel destination ("travel to <place>") -> travel_to,
        //     which the plugin resolves to a map marker (or the player for
        //     "me"/"you"/unknown);
        //   * a captured native Action-Book command (e.g. wave) -> replay verbatim;
        //   * a movement/hand-over verb with no destination -> rendezvous;
        //   * otherwise a recorded no-op step.
        let command = if !destination.is_empty() {
            companion_travel_command(&owner_npc_key, &owner_name, &destination)
        } else if !command_body.is_empty() {
            json!({ "op": "native_action", "body": command_body })
        } else if is_travel_verb(&verb) {
            companion_travel_command(&owner_npc_key, &owner_name, "")
        } else {
            Value::Null
        };

        let label = if destination.is_empty() {
            capitalize(&verb)
        } else {
            format!("{} to {}", capitalize(&verb), destination)
        };
        let mut description = if !event.is_empty() {
            format!("{label} when {event}")
        } else {
            match parse_when(&when_text, clock) {
                Some((_, hour)) => format!("{} at {}", label, format_hour(hour)),
                None => label,
            }
        };
        if !after.is_empty() {
            description.push_str(&format!(" (after {after})"));
        }
        chain.push(ChainStep {
            id: format!("{}_{}", i + 1, slugify(&verb)),
            description,
            trigger,
            command,
            delay_ms,
            armed_at_ms: 0,
            done: false,
        });
    }
    if chain.is_empty() {
        return Ok(None);
    }

    let summary = if raw.is_empty() {
        chain.iter().map(|s| s.description.clone()).collect::<Vec<_>>().join(", then ")
    } else {
        capitalize(raw.trim_end_matches('.'))
    };
    let now = epoch_millis();
    let mut args = Map::new();
    args.insert("raw".to_string(), json!(raw));
    let task = SchedulerTask {
        id: format!("task_{}_{}", now, rand_suffix()),
        owner_npc_key: owner_npc_key.clone(),
        owner_name: first_nonempty(&[str_field(spec, "owner_name"), str_field(spec, "character_name")]),
        character_name: str_field(spec, "character_name"),
        live_chat_id: str_field(spec, "live_chat_id"),
        action: chain.first().map(|_| first_word(&chain[0].description)).unwrap_or_default(),
        args,
        summary,
        trigger: chain[0].trigger.clone(),
        chain,
        state: TaskState::Pending,
        last_error: String::new(),
        created_at_ms: now,
        fired_at_ms: 0,
        created_day: clock.day as u32,
        created_hour: clock.hour,
    };

    persist_task(state, task)
}

/// Persist a freshly built task, de-duplicating: the same owner re-emitting the
/// same summary (across a multi-line turn, or a re-ask) refreshes the existing
/// non-terminal task rather than piling up duplicates.
fn persist_task(state: &AppState, task: SchedulerTask) -> anyhow::Result<Option<SchedulerTask>> {
    let mut store = read_store(state);
    if let Some(existing) = store.tasks.iter_mut().find(|t| {
        !t.state.is_terminal()
            && t.owner_npc_key == task.owner_npc_key
            && t.summary == task.summary
    }) {
        *existing = task.clone();
    } else {
        store.tasks.push(task.clone());
    }
    write_store(state, &store)?;
    tracing::info!(
        "scheduler: scheduled '{}' for {} ({}) — {} step(s)",
        task.summary,
        task.owner_name,
        task.owner_npc_key,
        task.chain.len()
    );
    Ok(Some(task))
}

/// Lowercase kebab of the first couple of words, for a stable step id.
fn slugify(text: &str) -> String {
    text.to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .split('_')
        .filter(|s| !s.is_empty())
        .take(2)
        .collect::<Vec<_>>()
        .join("_")
}

fn first_word(text: &str) -> String {
    text.split_whitespace().next().unwrap_or("").to_string()
}

// ===========================================================================
// Tick: evaluate + fire
// ===========================================================================

/// One scheduler tick: read the clock, evaluate every non-terminal task, and fire
/// any due triggers. Persists only if something changed. Never propagates a fire
/// failure — a failed task is marked `failed` and logged, the rest proceed.
pub fn tick(state: &AppState) {
    let clock = current_clock(state);
    let world_base = read_world_snapshot(state);
    let mut store = read_store(state);
    if store.active_tasks().count() == 0 {
        return;
    }
    let mut changed = false;
    // Index-based loop so a task can be mutated in place.
    for idx in 0..store.tasks.len() {
        if store.tasks[idx].state.is_terminal() {
            continue;
        }
        let task = store.tasks[idx].clone();
        match advance_task(state, &task, clock, &world_base) {
            Some(updated) => {
                store.tasks[idx] = updated;
                changed = true;
            }
            None => {}
        }
    }
    if changed {
        if let Err(error) = write_store(state, &store) {
            tracing::warn!("scheduler: failed to persist after tick: {error}");
        }
    }
}

/// EVENT hook: an incoming player line. For every non-terminal task whose current
/// pending step waits on [`Condition::PlayerSaid`], raise that phrase's flag when
/// the line contains it (case-insensitive) — the next tick then fires the step
/// (honouring any `after` delay). Cheap; called once per player turn from the
/// bridge before the NPC responds.
pub fn note_player_message(state: &AppState, message: &str) {
    let msg = message.to_ascii_lowercase();
    if msg.trim().is_empty() {
        return;
    }
    let mut store = read_store(state);
    let mut changed = false;
    for task in store.tasks.iter_mut() {
        if task.state.is_terminal() {
            continue;
        }
        let Some(step) = task.chain.iter().find(|s| !s.done) else {
            continue;
        };
        let Trigger::Condition { condition: Condition::PlayerSaid { phrase } } = &step.trigger else {
            continue;
        };
        let needle = phrase.trim().to_ascii_lowercase();
        if needle.is_empty() || !msg.contains(&needle) {
            continue;
        }
        let flag = player_said_flag(phrase);
        let flags = task
            .args
            .entry("_flags".to_string())
            .or_insert_with(|| json!({}));
        if let Some(obj) = flags.as_object_mut() {
            obj.insert(flag, json!(true));
            changed = true;
            tracing::info!("scheduler: player said '{}' -> armed task {}", phrase, task.id);
        }
    }
    if changed {
        if let Err(error) = write_store(state, &store) {
            tracing::warn!("scheduler: failed to persist player-said flags: {error}");
        }
    }
}

/// Evaluate one task against the clock/world. Every task is a chain of one or more
/// steps (a single scheduled action is a chain of one). Returns `Some(updated)` if
/// it changed (fired a step, completed, or failed), else `None`.
fn advance_task(
    state: &AppState,
    task: &SchedulerTask,
    clock: Option<GameClock>,
    world_base: &WorldSnapshot,
) -> Option<SchedulerTask> {
    advance_chain(state, task, clock, world_base)
}

/// Advance a task's chain: find the first not-done step and, if its trigger is met
/// (its "at" time reached, or immediately when it has none), issue its command and
/// mark it done; when all steps are done, complete the task. Only ONE step
/// advances per tick, so a just-issued step ("then" the next) has a beat to take
/// effect before the following one fires.
fn advance_chain(
    state: &AppState,
    task: &SchedulerTask,
    clock: Option<GameClock>,
    _world_base: &WorldSnapshot,
) -> Option<SchedulerTask> {
    let world = task_world(state, task);
    let Some(step_idx) = task.chain.iter().position(|s| !s.done) else {
        // Every step done → task complete.
        if task.state != TaskState::Done {
            let mut updated = task.clone();
            updated.state = TaskState::Done;
            return Some(updated);
        }
        return None;
    };
    let step = &task.chain[step_idx];
    if !trigger_met(&step.trigger, clock, &world) {
        // Not yet — mark active on first evaluation so the UI shows progress.
        if task.state == TaskState::Pending {
            let mut updated = task.clone();
            updated.state = TaskState::Active;
            return Some(updated);
        }
        return None;
    }
    // Trigger met. Honour an `after` delay: arm on first satisfaction, then hold
    // for `delay_ms` of real time before issuing (e.g. "wait 30 seconds then …").
    if step.delay_ms > 0 {
        let now = epoch_millis();
        if step.armed_at_ms == 0 {
            let mut updated = task.clone();
            updated.state = TaskState::Active;
            updated.chain[step_idx].armed_at_ms = now;
            return Some(updated);
        }
        if now < step.armed_at_ms + step.delay_ms as i64 {
            return None; // still waiting out the delay
        }
    }
    let mut updated = task.clone();
    updated.state = TaskState::Active;
    updated.fired_at_ms = epoch_millis();
    // Issue the step's command (if any); a null command is a pure wait/marker.
    if !step.command.is_null() {
        if let Err(error) = issue_command(state, &step.command) {
            updated.state = TaskState::Failed;
            updated.last_error = format!("step '{}': {error}", step.id);
            tracing::warn!("scheduler: chain task {} step {} failed: {error}", task.id, step.id);
            return Some(updated);
        }
    }
    updated.chain[step_idx].done = true;
    tracing::info!(
        "scheduler: chain {} advanced step '{}' for {}",
        task.id,
        step.id,
        task.owner_name
    );
    if updated.chain.iter().all(|s| s.done) {
        updated.state = TaskState::Done;
    }
    Some(updated)
}

/// Is a trigger satisfied right now?
fn trigger_met(trigger: &Trigger, clock: Option<GameClock>, world: &WorldSnapshot) -> bool {
    match trigger {
        Trigger::Time { day, hour } => match clock {
            Some(c) => c.total_hours() >= (*day as f64) * 24.0 + *hour,
            None => false,
        },
        Trigger::Condition { condition } => condition.is_met(world),
    }
}

/// The world snapshot a task is evaluated against, folding the task's own event
/// flags over the base positions.
fn task_world(state: &AppState, task: &SchedulerTask) -> WorldSnapshot {
    let mut world = read_world_snapshot(state);
    for step in &task.chain {
        // Chain steps whose trigger is a FlagSet expose their flag; the flag's
        // value is looked up from the store's task flags (raised via the event
        // endpoint). Read fresh from the persisted task each tick.
        let _ = step;
    }
    // Event flags live on the task itself in `args["_flags"]` (a small object we
    // set from the event endpoint), so a discarded save branch's flags roll back
    // with the task.
    if let Some(flags) = task.args.get("_flags").and_then(Value::as_object) {
        for (k, v) in flags {
            world.flags.insert(k.clone(), v.as_bool().unwrap_or(false));
        }
    }
    world
}

/// Read player + last-NPC positions from the runtime heartbeat for condition
/// evaluation. Positions absent → the relevant conditions simply stay false.
fn read_world_snapshot(state: &AppState) -> WorldSnapshot {
    let mut world = WorldSnapshot::default();
    let path = bridge_root(state).join("runtime_heartbeat.json");
    let Ok(text) = std::fs::read_to_string(path) else {
        return world;
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return world;
    };
    if let Some(p) = value.get("player") {
        if p.get("present").and_then(Value::as_bool).unwrap_or(false) {
            world.player = Some((
                p.get("pos_x").and_then(Value::as_f64).unwrap_or(0.0),
                p.get("pos_y").and_then(Value::as_f64).unwrap_or(0.0),
                p.get("pos_z").and_then(Value::as_f64).unwrap_or(0.0),
            ));
        }
    }
    if let Some(n) = value.get("last_npc") {
        if n.get("snapshot_valid").and_then(Value::as_bool).unwrap_or(false) {
            world.npc = Some((
                n.get("pos_x").and_then(Value::as_f64).unwrap_or(0.0),
                n.get("pos_y").and_then(Value::as_f64).unwrap_or(0.0),
                n.get("pos_z").and_then(Value::as_f64).unwrap_or(0.0),
            ));
        }
    }
    world
}

// ===========================================================================
// Firing: write the game command file the plugin polls
// ===========================================================================

const COMPANION_COMMAND_VERSION: &str = "CHASM_COMPANION_V1";

/// A `travel_to` command payload. The plugin moves the owner to `dest_name` (a
/// named map marker) or, for "me"/empty/unknown, the player. `npc_name` lets the
/// plugin resolve a NON-companion conversing NPC (e.g. Chet) by name when the
/// npc_key isn't a companion slot. Both name + dest are base64'd for the file.
fn companion_travel_command(npc_key: &str, npc_name: &str, dest_name: &str) -> Value {
    json!({
        "op": "travel_to",
        "npc_key": npc_key,
        "npc_name_base64": STANDARD.encode(npc_name.as_bytes()),
        "dest_name_base64": STANDARD.encode(dest_name.as_bytes()),
    })
}

/// Write the command a fired step issues to the bridge, atomically (tmp+rename):
///   * `op=native_action` → the pre-built native Action-Book command body, written
///     to `control/actions/` (the plugin replays it, resolving the companion actor
///     by npc_key). This is how "wave at 1am" actually waves.
///   * any other `op` (e.g. `travel_to`) → a `CHASM_COMPANION_V1` command under
///     `control/companions/` (the companions channel), for movement/hand-over.
///   * null / no op → a recorded no-op (a step with no game effect).
fn issue_command(state: &AppState, command: &Value) -> anyhow::Result<()> {
    let op = command.get("op").and_then(Value::as_str).unwrap_or("");
    if op.is_empty() {
        return Ok(()); // pure wait / no-op step
    }
    let root = bridge_root(state);

    // Native Action-Book command: write the captured body verbatim to control/actions.
    if op == "native_action" {
        let body = command.get("body").and_then(Value::as_str).unwrap_or("");
        if body.trim().is_empty() {
            return Ok(());
        }
        let dir = root.join("control").join("actions");
        std::fs::create_dir_all(&dir)?;
        let file_id = format!("sched_action_{}_{}", epoch_millis(), rand_suffix());
        let final_path = dir.join(format!("{file_id}.txt"));
        let tmp_path = dir.join(format!("{file_id}.tmp"));
        std::fs::write(&tmp_path, body.as_bytes())?;
        std::fs::rename(&tmp_path, &final_path)?;
        tracing::info!("scheduler: issued native_action command {file_id}");
        return Ok(());
    }

    issue_companion_command(state, command)
}

/// Write a `CHASM_COMPANION_V1` command (any `op` other than `native_action`) to
/// `control/companions/`, atomically (tmp+rename). Every non-`op` field of the
/// JSON object is rendered as a `key=value` line. Shared with the movement engine
/// (which emits `move_to_pos` / `travel_to` here).
pub(crate) fn issue_companion_command(state: &AppState, command: &Value) -> anyhow::Result<()> {
    let op = command.get("op").and_then(Value::as_str).unwrap_or("");
    if op.is_empty() {
        return Ok(());
    }
    let dir = bridge_root(state).join("control").join("companions");
    std::fs::create_dir_all(dir.join("acks"))?;
    let request_id = format!("sched_{}_{}", op, epoch_millis());
    let mut body = format!(
        "{COMPANION_COMMAND_VERSION}\r\nrequest_id={request_id}\r\nop={op}\r\n"
    );
    if let Some(map) = command.as_object() {
        for (key, value) in map {
            if key == "op" {
                continue;
            }
            let rendered = match value {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            body.push_str(&format!("{key}={rendered}\r\n"));
        }
    }
    let final_path = dir.join(format!("{request_id}.txt"));
    let tmp_path = dir.join(format!("{request_id}.tmp"));
    std::fs::write(&tmp_path, body.as_bytes())?;
    std::fs::rename(&tmp_path, &final_path)?;
    tracing::info!("scheduler: issued {op} command {request_id}");
    Ok(())
}

/// The rendezvous root the NVSE plugin uses — same resolution as the in-process
/// bridge fold + companions UI (helper config's first `nativeBridgeRoots`, else
/// the fixed default `%LOCALAPPDATA%\chasm\bridge`).
pub(crate) fn bridge_root(state: &AppState) -> PathBuf {
    let settings = chasm_core::AppSettings::load(&state.config.settings_path);
    let config_path = settings.launcher.helper_config.trim().to_string();
    if let Ok(config) = chasm_fnv_bridge::load_config(Path::new(&config_path)) {
        if let Some(root) = config.native_bridge_roots.first() {
            return root.clone();
        }
    }
    chasm_core::default_bridge_root()
}

// ===========================================================================
// Natural-language EVENT + delay classification (the `when` / `after` fields)
// ===========================================================================

/// Classify a natural-language `when` event into a trigger. Recognised kinds:
/// * "the player says X" / "I say X" -> [`Condition::PlayerSaid`] on phrase X.
/// * time of day ("dark"/"night", "morning"/"dawn", "noon", "midnight",
///   "dusk"/"evening") -> a [`Trigger::Time`] at the NEXT occurrence of that hour.
/// * "<name> comes near"/"approaches" -> [`Condition::NpcNearPlayer`] when it is
///   the player, else [`Condition::ActorNear`] (held until a plugin proximity
///   signal, since the heartbeat has no third-party positions).
/// Anything unrecognised -> a `FlagSet` that nothing raises yet, so the step holds
/// rather than firing at the wrong moment.
fn classify_event(event: &str, clock: Option<GameClock>) -> Trigger {
    let e = event.trim().to_ascii_lowercase();
    if let Some(phrase) = extract_said_phrase(&e) {
        return Trigger::Condition { condition: Condition::PlayerSaid { phrase } };
    }
    let hour = if e.contains("midnight") {
        Some(0.0)
    } else if e.contains("dawn") || e.contains("sunrise") || e.contains("morning") {
        Some(6.0)
    } else if e.contains("noon") || e.contains("midday") {
        Some(12.0)
    } else if e.contains("dusk") || e.contains("sunset") || e.contains("evening") {
        Some(19.0)
    } else if e.contains("dark") || e.contains("night") {
        Some(20.0)
    } else {
        None
    };
    if let (Some(h), Some(c)) = (hour, clock) {
        let day = if c.hour < h { c.day as u32 } else { c.day as u32 + 1 };
        return Trigger::Time { day, hour: h };
    }
    if e.contains("near") || e.contains("approach") || e.contains("comes close") || e.contains("gets close") {
        if e.contains("player") || e.contains("you ") || e.starts_with("you") || e.contains(" me") || e.starts_with("i ") {
            return Trigger::Condition { condition: Condition::NpcNearPlayer };
        }
        return Trigger::Condition { condition: Condition::ActorNear { name: proximity_actor_name(event) } };
    }
    Trigger::Condition { condition: Condition::FlagSet { flag: format!("event:{e}") } }
}

/// Pull the spoken phrase out of a "the player says X" style event (lowercased
/// input). Returns `None` when the event is not about the player speaking.
fn extract_said_phrase(e: &str) -> Option<String> {
    let markers = [" says ", " say ", " said ", "says ", "say "];
    let rest = markers.iter().find_map(|m| e.find(m).map(|i| &e[i + m.len()..]))?;
    let mut phrase = rest.trim();
    for lead in ["the word ", "word ", "the phrase ", "phrase ", "the words ", "words "] {
        if let Some(s) = phrase.strip_prefix(lead) {
            phrase = s.trim();
        }
    }
    let phrase = phrase
        .trim_matches(|c| c == '\'' || c == '"' || c == ' ')
        .trim_end_matches(|c: char| matches!(c, '.' | ',' | '!' | '?'))
        .trim_matches(|c| c == '\'' || c == '"');
    (!phrase.is_empty()).then(|| phrase.to_string())
}

/// The named actor in a "<name> comes near" event (original casing preserved).
fn proximity_actor_name(event: &str) -> String {
    let lower = event.to_ascii_lowercase();
    let cut = ["comes near", "gets near", "approaches", "comes close", "gets close", "is near", "near"]
        .iter()
        .filter_map(|m| lower.find(m))
        .min();
    let name = match cut {
        Some(i) => &event[..i],
        None => event,
    };
    name.trim()
        .trim_start_matches("if ")
        .trim_start_matches("when ")
        .trim_start_matches("once ")
        .trim()
        .trim_end_matches(',')
        .trim()
        .to_string()
}

/// Parse an `after:"30 seconds"` / "5 minutes" / "an hour" delay into milliseconds
/// of REAL time (the delay runs after the trigger is met). A bare number is
/// seconds. Unparseable -> 0 (no delay).
fn parse_delay_ms(after: &str) -> u64 {
    let a = after.trim().to_ascii_lowercase();
    if a.is_empty() {
        return 0;
    }
    let digits: String = a.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let n = digits.parse::<f64>().unwrap_or(if a.starts_with('a') { 1.0 } else { 0.0 });
    let n = if n <= 0.0 { 1.0 } else { n };
    let unit = if a.contains("hour") {
        3_600_000.0
    } else if a.contains("min") {
        60_000.0
    } else {
        1000.0 // seconds, or a bare number
    };
    (n * unit) as u64
}

// ===========================================================================
// Natural-language in-game time parsing
// ===========================================================================

/// Parse a plain-English time ("1am", "tonight", "in an hour", "noon",
/// "tomorrow morning", "13:30") into an ABSOLUTE in-game (day, hour), anchored to
/// `now`. Clock-times that have already passed today roll to the next day.
/// Returns `None` only when the text carries no time at all.
pub fn parse_when(text: &str, now: GameClock) -> Option<(u32, f64)> {
    let t = text.trim().to_lowercase();
    if t.is_empty() {
        return None;
    }

    // Relative: "in N hours/minutes", "in an hour".
    if let Some(rest) = t.strip_prefix("in ") {
        if let Some(hours) = parse_relative_hours(rest) {
            let total = now.total_hours() + hours;
            return Some(split_total_hours(total));
        }
    }

    // Named times of day.
    let mut day_offset: i64 = 0;
    let mut base = t.as_str();
    if let Some(rest) = t.strip_prefix("tomorrow") {
        day_offset = 1;
        base = rest.trim();
        if base.is_empty() {
            base = "morning";
        }
    } else if let Some(rest) = t.strip_prefix("tonight") {
        // tonight → this evening (or next if already past).
        let _ = rest;
        return Some(next_occurrence_of_hour(21.0, now));
    }

    let named = match base {
        "morning" | "this morning" => Some(8.0),
        "noon" | "midday" => Some(12.0),
        "afternoon" | "this afternoon" => Some(14.0),
        "evening" | "this evening" | "tonight" => Some(19.0),
        "night" | "midnight" => Some(0.0),
        "dawn" | "sunrise" => Some(6.0),
        "dusk" | "sunset" => Some(18.0),
        _ => None,
    };
    if let Some(hour) = named {
        return Some(apply_day_offset(next_occurrence_of_hour(hour, now), now, day_offset, hour));
    }

    // Clock time: "1am", "1 am", "1:30 pm", "13:00", "1pm".
    if let Some(hour) = parse_clock_time(base) {
        return Some(apply_day_offset(next_occurrence_of_hour(hour, now), now, day_offset, hour));
    }

    None
}

/// If an explicit day offset was given ("tomorrow"), honor it relative to today's
/// date rather than the "next occurrence" roll. Otherwise return the computed
/// next-occurrence pair.
fn apply_day_offset(
    next_occ: (u32, f64),
    now: GameClock,
    day_offset: i64,
    hour: f64,
) -> (u32, f64) {
    if day_offset == 0 {
        return next_occ;
    }
    let day = (now.day as i64 + day_offset).max(0) as u32;
    (day, hour)
}

/// The next day/hour at which the wall-clock reads `hour`, at or after `now`. If
/// `hour` is still ahead today, it's today; otherwise tomorrow.
pub fn next_occurrence_of_hour(hour: f64, now: GameClock) -> (u32, f64) {
    let today = now.day as u32;
    if hour > now.hour + f64::EPSILON {
        (today, hour)
    } else {
        (today + 1, hour)
    }
}

/// "an hour"/"1 hour"/"2 hours"/"30 minutes"/"90 mins" → hours as f64.
fn parse_relative_hours(rest: &str) -> Option<f64> {
    let rest = rest.trim();
    // A leading article ("an hour", "a minute") means quantity 1 — there is no
    // digit for split_num_unit to find, so peel it off first.
    let (n, unit): (f64, String) = if let Some(u) = rest
        .strip_prefix("an ")
        .or_else(|| rest.strip_prefix("a "))
    {
        (1.0, u.trim().to_string())
    } else {
        let (num_str, unit) = split_num_unit(rest)?;
        let n = if num_str.is_empty() { 1.0 } else { num_str.parse().ok()? };
        (n, unit)
    };
    if unit.starts_with("hour") || unit == "hr" || unit.starts_with("hrs") {
        Some(n)
    } else if unit.starts_with("min") {
        Some(n / 60.0)
    } else if unit.starts_with("day") {
        Some(n * 24.0)
    } else {
        None
    }
}

fn split_num_unit(s: &str) -> Option<(String, String)> {
    let s = s.trim();
    let mut split_at = 0;
    for (i, c) in s.char_indices() {
        if c.is_ascii_digit() || c == '.' {
            split_at = i + c.len_utf8();
        } else {
            break;
        }
    }
    let num = s[..split_at].trim().to_string();
    let unit = s[split_at..].trim().to_string();
    if unit.is_empty() {
        return None;
    }
    Some((num, unit))
}

/// "1am" / "1 am" / "1:30pm" / "13:00" / "9" → hour in [0,24). None if not a time.
fn parse_clock_time(s: &str) -> Option<f64> {
    let s = s.trim().replace(' ', "");
    let (body, meridiem) = if let Some(rest) = s.strip_suffix("am") {
        (rest, Some(false))
    } else if let Some(rest) = s.strip_suffix("pm") {
        (rest, Some(true))
    } else {
        (s.as_str(), None)
    };
    if body.is_empty() {
        return None;
    }
    let (h_str, m_str) = match body.split_once(':') {
        Some((h, m)) => (h, m),
        None => (body, "0"),
    };
    let mut hour: f64 = h_str.parse().ok()?;
    let minute: f64 = m_str.parse().ok()?;
    if !(0.0..=24.0).contains(&hour) || !(0.0..60.0).contains(&minute) {
        return None;
    }
    match meridiem {
        Some(true) => {
            // pm: 12pm stays 12, 1–11pm add 12.
            if hour < 12.0 {
                hour += 12.0;
            }
        }
        Some(false) => {
            // am: 12am → 0.
            if hour == 12.0 {
                hour = 0.0;
            }
        }
        None => {}
    }
    Some((hour + minute / 60.0).rem_euclid(24.0))
}

fn split_total_hours(total: f64) -> (u32, f64) {
    let day = (total / 24.0).floor().max(0.0) as u32;
    let hour = total - (day as f64) * 24.0;
    (day, hour)
}

// ===========================================================================
// Small helpers
// ===========================================================================

fn format_hour(hour: f64) -> String {
    let h = hour.floor() as i64;
    let m = ((hour - h as f64) * 60.0).round() as i64;
    let (h, m) = if m >= 60 { (h + 1, 0) } else { (h, m) };
    let hour24 = ((h % 24) + 24) % 24;
    let suffix = if hour24 < 12 { "AM" } else { "PM" };
    let mut h12 = hour24 % 12;
    if h12 == 0 {
        h12 = 12;
    }
    format!("{h12}:{m:02}{suffix}")
}

fn str_field(value: &Value, key: &str) -> String {
    value.get(key).and_then(Value::as_str).unwrap_or("").trim().to_string()
}

fn first_nonempty(values: &[String]) -> String {
    values.iter().find(|s| !s.trim().is_empty()).cloned().unwrap_or_default()
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

fn epoch_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn rand_suffix() -> String {
    let n: u32 = rand::random();
    format!("{:06x}", n & 0xFFFFFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(day: f64, hour: f64) -> GameClock {
        GameClock { day, hour }
    }

    // ---- time parsing ----

    #[test]
    fn parses_clock_time_am_pm() {
        assert_eq!(parse_clock_time("1am"), Some(1.0));
        assert_eq!(parse_clock_time("12am"), Some(0.0));
        assert_eq!(parse_clock_time("12pm"), Some(12.0));
        assert_eq!(parse_clock_time("1pm"), Some(13.0));
        assert_eq!(parse_clock_time("1:30pm"), Some(13.5));
        assert_eq!(parse_clock_time("13:00"), Some(13.0));
        assert_eq!(parse_clock_time("nope"), None);
    }

    #[test]
    fn one_am_rolls_to_next_day_when_already_past() {
        // It's day 3, 09:00. "1am" already happened today → tomorrow 01:00.
        let (day, hour) = parse_when("1am", clock(3.0, 9.0)).unwrap();
        assert_eq!(day, 4);
        assert!((hour - 1.0).abs() < 1e-9);
    }

    #[test]
    fn one_pm_is_today_when_ahead() {
        // Day 3, 09:00. "1pm" is later today.
        let (day, hour) = parse_when("1pm", clock(3.0, 9.0)).unwrap();
        assert_eq!(day, 3);
        assert!((hour - 13.0).abs() < 1e-9);
    }

    #[test]
    fn relative_in_an_hour_and_minutes() {
        let (day, hour) = parse_when("in an hour", clock(2.0, 23.5)).unwrap();
        // 23.5 + 1.0 = 24.5 → day 3, 00.5
        assert_eq!(day, 3);
        assert!((hour - 0.5).abs() < 1e-9);

        let (day, hour) = parse_when("in 90 minutes", clock(0.0, 0.0)).unwrap();
        assert_eq!(day, 0);
        assert!((hour - 1.5).abs() < 1e-9);
    }

    #[test]
    fn named_times_and_tomorrow() {
        let (_d, hour) = parse_when("noon", clock(1.0, 6.0)).unwrap();
        assert!((hour - 12.0).abs() < 1e-9);

        let (day, hour) = parse_when("tomorrow morning", clock(1.0, 6.0)).unwrap();
        assert_eq!(day, 2);
        assert!((hour - 8.0).abs() < 1e-9);

        // tonight → 21:00 today when still morning.
        let (day, hour) = parse_when("tonight", clock(1.0, 6.0)).unwrap();
        assert_eq!(day, 1);
        assert!((hour - 21.0).abs() < 1e-9);
    }

    #[test]
    fn empty_time_is_none() {
        assert_eq!(parse_when("", clock(0.0, 0.0)), None);
        assert_eq!(parse_when("somewhere", clock(0.0, 0.0)), None);
    }

    // ---- trigger evaluation ----

    #[test]
    fn time_trigger_fires_at_or_after() {
        let world = WorldSnapshot::default();
        let trig = Trigger::Time { day: 3, hour: 1.0 };
        assert!(!trigger_met(&trig, Some(clock(3.0, 0.5)), &world));
        assert!(trigger_met(&trig, Some(clock(3.0, 1.0)), &world));
        assert!(trigger_met(&trig, Some(clock(4.0, 0.0)), &world));
        // No clock → never fires (don't fire on missing data).
        assert!(!trigger_met(&trig, None, &world));
    }

    #[test]
    fn fractional_days_passed_does_not_fire_early() {
        // GameDaysPassed is a FRACTIONAL counter (its fraction IS the time of day).
        // from_game must floor it, or total_hours double-counts and a same-day task
        // looks overdue and fires the instant it's scheduled.
        let now = GameClock::from_game(10.4514, 10.833); // 10:50 AM, day 10
        assert_eq!(now.day, 10.0);
        assert!((now.total_hours() - (10.0 * 24.0 + 10.833)).abs() < 1e-6);

        // Scheduling 11:50 AM (an hour later) must land today and NOT fire yet.
        let (day, hour) = parse_when("11:50AM", now).unwrap();
        assert_eq!(day, 10);
        let trig = Trigger::Time { day, hour };
        assert!(!trigger_met(&trig, Some(now), &WorldSnapshot::default()));
        // Later (noon, past 11:50 AM) it fires.
        let later = GameClock::from_game(10.5, 12.0);
        assert!(trigger_met(&trig, Some(later), &WorldSnapshot::default()));
    }

    #[test]
    fn condition_near_player() {
        let mut world = WorldSnapshot::default();
        world.player = Some((0.0, 0.0, 0.0));
        world.npc = Some((100.0, 0.0, 0.0));
        assert!(Condition::NpcNearPlayer.is_met(&world)); // within 256
        world.npc = Some((1000.0, 0.0, 0.0));
        assert!(!Condition::NpcNearPlayer.is_met(&world));
        // Missing npc → false, never fire on missing data.
        world.npc = None;
        assert!(!Condition::NpcNearPlayer.is_met(&world));
    }

    #[test]
    fn flag_condition() {
        let mut world = WorldSnapshot::default();
        let cond = Condition::FlagSet { flag: "looted".into() };
        assert!(!cond.is_met(&world));
        world.flags.insert("looted".into(), true);
        assert!(cond.is_met(&world));
    }

    // ---- function-call action parsing (the core of the rework) ----

    #[test]
    fn bare_call_is_immediate() {
        // wave() -> not scheduled (no at/to), fires now.
        let c = parse_action_call("wave()").unwrap();
        assert_eq!(c.name, "wave");
        assert!(!c.is_scheduled());
        assert_eq!(c.to, None);
        assert_eq!(c.at, None);
    }

    #[test]
    fn bare_word_is_not_a_call() {
        // "wave" (no parens) parses as None -> caller treats it as a bare alias.
        assert!(parse_action_call("wave").is_none());
    }

    #[test]
    fn at_arg_schedules() {
        let c = parse_action_call(r#"wave(at="1:00AM")"#).unwrap();
        assert_eq!(c.name, "wave");
        assert!(c.is_scheduled());
        assert_eq!(c.at.as_deref(), Some("1:00AM"));
        assert_eq!(c.to, None);
    }

    #[test]
    fn travel_with_destination_and_time() {
        let c = parse_action_call(r#"travel(to="Prospector Saloon", at="3:00PM")"#).unwrap();
        assert_eq!(c.name, "travel");
        assert_eq!(c.to.as_deref(), Some("Prospector Saloon"));
        assert_eq!(c.at.as_deref(), Some("3:00PM"));
        assert!(c.is_scheduled());
    }

    #[test]
    fn travel_no_time_is_still_scheduled() {
        // A destination alone still routes through the scheduler (fires next tick).
        let c = parse_action_call(r#"travel(to="Novac")"#).unwrap();
        assert_eq!(c.to.as_deref(), Some("Novac"));
        assert_eq!(c.at, None);
        assert!(c.is_scheduled());
    }

    #[test]
    fn target_arg_and_single_quotes() {
        let c = parse_action_call("attack(target='Easy Pete')").unwrap();
        assert_eq!(c.name, "attack");
        assert_eq!(c.target.as_deref(), Some("Easy Pete"));
        assert!(!c.is_scheduled());
    }

    #[test]
    fn positional_value_maps_by_action() {
        // travel(x) -> destination; a non-travel action -> target.
        let c = parse_action_call(r#"travel("the saloon")"#).unwrap();
        assert_eq!(c.to.as_deref(), Some("the saloon"));
        let c = parse_action_call(r#"attack("raider")"#).unwrap();
        assert_eq!(c.target.as_deref(), Some("raider"));
    }

    #[test]
    fn comma_inside_a_quoted_value_is_kept() {
        let c = parse_action_call(r#"travel(to="Doc Mitchell, MD", at="5:00PM")"#).unwrap();
        assert_eq!(c.to.as_deref(), Some("Doc Mitchell, MD"));
        assert_eq!(c.at.as_deref(), Some("5:00PM"));
    }

    #[test]
    fn unquoted_values_are_lenient() {
        let c = parse_action_call("travel(to=Novac, at=3:00PM)").unwrap();
        assert_eq!(c.to.as_deref(), Some("Novac"));
        assert_eq!(c.at.as_deref(), Some("3:00PM"));
    }

    #[test]
    fn travel_verbs_detected() {
        assert!(is_travel_verb("travel"));
        assert!(is_travel_verb("come to me"));
        assert!(!is_travel_verb("wave"));
        assert!(!is_travel_verb("dance"));
    }

    // ---- store round-trip ----

    fn sample_task() -> SchedulerTask {
        SchedulerTask {
            id: "task_1".into(),
            owner_npc_key: "sunny".into(),
            owner_name: "Sunny Smiles".into(),
            character_name: "Sunny Smiles".into(),
            live_chat_id: "fnv".into(),
            action: "wave".into(),
            args: serde_json::from_value(json!({ "raw": "wave at 1am" })).unwrap(),
            summary: "Wave at 1am".into(),
            trigger: Trigger::Time { day: 4, hour: 1.0 },
            chain: vec![ChainStep {
                id: "1_wave".into(),
                description: "Wave at 1:00AM".into(),
                trigger: Trigger::Time { day: 4, hour: 1.0 },
                command: json!({ "op": "native_action", "body": "CMD" }),
                delay_ms: 0,
                armed_at_ms: 0,
                done: false,
            }],
            state: TaskState::Pending,
            last_error: String::new(),
            created_at_ms: 1000,
            fired_at_ms: 0,
            created_day: 3,
            created_hour: 9.0,
        }
    }

    #[test]
    fn store_serde_round_trip() {
        let task = sample_task();
        let store = SchedulerStore { version: STORE_VERSION, tasks: vec![task.clone()] };
        let text = serde_json::to_string_pretty(&store).unwrap();
        let back: SchedulerStore = serde_json::from_str(&text).unwrap();
        assert_eq!(back.tasks.len(), 1);
        assert_eq!(back.tasks[0], task);
        assert!(matches!(back.tasks[0].trigger, Trigger::Time { day: 4, .. }));
    }

    // ---- save-aware rollback (sidecar round-trip) ----

    fn write_store_at(content_root: &std::path::Path, json: &str) {
        let path = scheduler_store_path_at(content_root);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, json).unwrap();
    }

    fn read_store_at(content_root: &std::path::Path) -> String {
        std::fs::read_to_string(scheduler_store_path_at(content_root)).unwrap_or_default()
    }

    #[test]
    fn task_in_discarded_branch_vanishes_on_load() {
        let dir = std::env::temp_dir().join(format!("chasm_sched_test_{}", rand_suffix()));
        std::fs::create_dir_all(&dir).unwrap();

        // At save time there is ONE task. Checkpoint it.
        write_store_at(&dir, r#"{"version":1,"tasks":[{"id":"a"}]}"#);
        checkpoint_scheduler_store(&dir, "cp1");

        // The player keeps playing and schedules a SECOND task in this branch.
        write_store_at(&dir, r#"{"version":1,"tasks":[{"id":"a"},{"id":"b"}]}"#);

        // They load the earlier save → restore the checkpoint. The branch (task b)
        // is discarded; only task a survives.
        restore_scheduler_store(&dir, "cp1");
        let restored = read_store_at(&dir);
        assert!(restored.contains("\"a\""), "task a should survive: {restored}");
        assert!(!restored.contains("\"b\""), "task b (discarded branch) must vanish: {restored}");

        // Loading a save with NO checkpoint (predates the scheduler) clears the store.
        restore_scheduler_store(&dir, "nonexistent");
        let cleared = read_store_at(&dir);
        assert!(cleared.contains("\"tasks\":[]"), "no sidecar → cleared: {cleared}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- chain trigger sequencing ("then") ----

    /// A "loot then meet at 1am" chain: step 1 fires immediately, step 2 waits for
    /// its 1am time. Asserts each step's trigger fires only when due — the "then"
    /// ordering itself is enforced by advance_chain (one not-done step at a time).
    #[test]
    fn chain_step_triggers_gate_correctly() {
        let step1 = ChainStep {
            id: "1_loot".into(),
            description: "Loot the body".into(),
            trigger: Trigger::Condition { condition: Condition::Immediate },
            command: Value::Null,
            delay_ms: 0,
            armed_at_ms: 0,
            done: false,
        };
        let step2 = ChainStep {
            id: "2_meet".into(),
            description: "Meet me at 1:00AM".into(),
            trigger: Trigger::Time { day: 1, hour: 1.0 },
            command: json!({ "op": "travel_to", "npc_key": "boone" }),
            delay_ms: 0,
            armed_at_ms: 0,
            done: false,
        };

        // Step 1 (Immediate) fires right away.
        assert!(trigger_met(&step1.trigger, Some(clock(0.0, 12.0)), &WorldSnapshot::default()));

        // Step 2's time hasn't arrived yet (day 0) → waits.
        assert!(!trigger_met(&step2.trigger, Some(clock(0.0, 12.0)), &WorldSnapshot::default()));
        // Once the clock reaches day 1 / 1am → fires.
        assert!(trigger_met(&step2.trigger, Some(clock(1.0, 1.0)), &WorldSnapshot::default()));
    }

    #[test]
    fn classify_player_said_event() {
        let now = clock(2.0, 15.0);
        for phrasing in ["the player says hi", "when I say hi", "I say 'hi'", "you say the word hi"] {
            match classify_event(phrasing, Some(now)) {
                Trigger::Condition { condition: Condition::PlayerSaid { phrase } } => {
                    assert_eq!(phrase, "hi", "from: {phrasing}");
                }
                other => panic!("{phrasing} -> {other:?}"),
            }
        }
    }

    #[test]
    fn player_said_flag_fires_only_after_match() {
        let cond = Condition::PlayerSaid { phrase: "hi".into() };
        let mut world = WorldSnapshot::default();
        assert!(!cond.is_met(&world)); // nothing said yet
        world.flags.insert(player_said_flag("hi"), true);
        assert!(cond.is_met(&world)); // armed by note_player_message
    }

    #[test]
    fn classify_time_of_day_and_proximity() {
        let now = clock(3.0, 15.0); // 3pm
        // "it gets dark" -> next 8pm, still today.
        assert!(matches!(classify_event("it gets dark", Some(now)), Trigger::Time { day: 3, hour } if hour == 20.0));
        // "in the morning" at 3pm -> tomorrow 6am.
        assert!(matches!(classify_event("in the morning", Some(now)), Trigger::Time { day: 4, hour } if hour == 6.0));
        // Player coming near -> near-player condition.
        assert!(matches!(
            classify_event("you come near", Some(now)),
            Trigger::Condition { condition: Condition::NpcNearPlayer }
        ));
        // A named third party -> ActorNear, name preserved.
        match classify_event("Easy Pete comes near", Some(now)) {
            Trigger::Condition { condition: Condition::ActorNear { name } } => assert_eq!(name, "Easy Pete"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn delay_parsing() {
        assert_eq!(parse_delay_ms("30 seconds"), 30_000);
        assert_eq!(parse_delay_ms("5 minutes"), 300_000);
        assert_eq!(parse_delay_ms("an hour"), 3_600_000);
        assert_eq!(parse_delay_ms("10"), 10_000); // bare number = seconds
        assert_eq!(parse_delay_ms(""), 0);
    }
}
