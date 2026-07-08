//! Witness memory + event triggers — NPCs remember what they saw happen.
//!
//! A consumer of the game event-log stream ([`crate::event_log`]): every event
//! the plugin captures arrives with a capture-time `witnesses` list (the mapped
//! NPC keys that were within speaking range when it happened). This module fans
//! those events out into each witness's LIVE CHAT HISTORY as narration lines —
//! real persisted messages (`is_system` + `extra.chasm.witnessed`), so they ride
//! the append-only history (prompt-cache friendly) and roll back with saves like
//! any other chat line. The NPC does not respond immediately; on the player's
//! next message their history simply shows everything they witnessed.
//!
//! ## Bundling
//! One history line per raw event would be noise. Witnessed events accumulate
//! per NPC in a PENDING bundle and flush as ONE narration message when the
//! bundle goes quiet / gets old / gets big — and always BEFORE anything else is
//! generated for that NPC (a trigger reaction, or the player's next message —
//! see the `flush_for_live_chat` call in [`crate::generate`]), so ordering in
//! history is event-lines-then-message.
//!
//! ## Triggers
//! Event types enabled on the Triggers page ALSO fire an immediate reaction:
//! ONE witnessing NPC (the nearest — the plugin orders `witnesses` by distance)
//! speaks unprompted. The reaction is delivered as a `control/reactions/` queue
//! file the plugin polls when idle (the song-queue pattern): the plugin gates on
//! its own `awaitingReply` state and re-enters the NORMAL chat request path, so
//! TTS / lip-sync / captions all work and a reply already in flight is never
//! interrupted. Each trigger type carries its own % CHANCE (default 100) and
//! its own COOLDOWN (default 10s, per type — not global), plus an optional
//! GLOBAL cooldown across all triggers (off by default). Reactions never stack
//! (one queue file at a time).
//!
//! ## Guardrails
//! * `conversation` events never fan out (the dialogue IS the history).
//! * An event's own NPC subject never witnesses itself (`subjectNpcKey`).
//! * Bundled non-trigger events never generate turns.
//! * The event-log store / Events page / save-sync rollback are untouched —
//!   `witnesses` is read from the RAW batch and never persisted to the store.
//!
//! Pending bundles are in-memory (a chasm restart drops at most a few seconds
//! of not-yet-flushed lines); the durable record is the chat history itself.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::save_sync::{epoch_millis, now_iso};
use crate::{AppState, WebError, WebResult};
use chasm_st_compat::{LiveChat, LiveChatSegment, STJsonlChatMessage};

/// Flush a bundle once no new witnessed event arrived for this long.
const PENDING_QUIET_MS: i64 = 10_000;
/// Flush a bundle regardless of quiet once it is this old.
const PENDING_MAX_AGE_MS: i64 = 30_000;
/// Flush a bundle early once it holds this many events.
const PENDING_MAX_EVENTS: usize = 8;
/// Drop a bundle whose NPC never resolves to a chat participant (the player
/// never talked to them and they never joined a live chat).
const PENDING_ORPHAN_MAX_AGE_MS: i64 = 10 * 60_000;
/// Default per-trigger cooldown (each trigger TYPE has its own timer).
const DEFAULT_TRIGGER_COOLDOWN_SECS: u32 = 10;
/// Default global-cooldown length shown when the user first enables it.
const DEFAULT_GLOBAL_COOLDOWN_SECS: u32 = 30;

/// Every event type the plugin's GAME EVENT LOG section emits today (see
/// mod-source main.cpp `QueueGameEvent` call sites). The Triggers page shows
/// this list UNIONED with any type observed in the store, so future plugin
/// types appear automatically.
pub(crate) const STATIC_EVENT_TYPES: [&str; 23] = [
    "combat",
    "arrival",
    "death",
    "murder",
    "item",
    "theft",
    "pickpocket",
    "lockpick",
    "hacking",
    "shooting",
    "weapon",
    "sneak",
    "location",
    "trade",
    "repair",
    "injury",
    "rads",
    "day",
    "level",
    "karma",
    "companion",
    "quest",
    "conversation",
];

/// Types whose summaries describe something THE PLAYER did in an implied
/// second/first person ("Picked up …", "Stole …", "Arrived at …"): witnessed
/// narration re-anchors them onto the player's name. Other types ("Sunny
/// Smiles died", quest beats, world beats) already carry their subject.
const PLAYER_SUBJECT_TYPES: [&str; 16] = [
    "item",
    "location",
    "shooting",
    "combat",
    "level",
    "theft",
    "pickpocket",
    "lockpick",
    "hacking",
    "murder",
    "repair",
    "trade",
    "weapon",
    "sneak",
    "injury",
    "rads",
];

fn web_err(message: impl Into<String>) -> WebError {
    WebError::from(anyhow::anyhow!(message.into()))
}

// ---------------------------------------------------------------------------
// Trigger settings (profile-scoped, like the scheduler/movement stores)
// ---------------------------------------------------------------------------

/// One trigger type's reaction knobs. A type with no saved rule is
/// witness-memory only (reactions off).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub(crate) struct TriggerRule {
    /// Fire an immediate reaction when a witnessed event of this type lands.
    pub enabled: bool,
    /// Probability (0–100) that an eligible event actually fires the reaction.
    pub chance_percent: u32,
    /// Cooldown between reactions of THIS type (seconds) — per type, not global.
    pub cooldown_secs: u32,
    /// Sight-gate this type: an NPC the player is HIDDEN from (the engine's
    /// detection state, sent per event as `hiddenFrom`) does not witness it at
    /// all — no memory line, no reaction. Off for things anyone could hear
    /// (gunshots); applies independently of `enabled` (memory-only types can
    /// still be sight-gated).
    pub require_sight: bool,
}

impl Default for TriggerRule {
    fn default() -> Self {
        Self {
            enabled: true,
            chance_percent: 100,
            cooldown_secs: DEFAULT_TRIGGER_COOLDOWN_SECS,
            require_sight: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub(crate) struct TriggerSettings {
    pub version: u32,
    /// Master switch for the whole witness system (memory + triggers).
    pub enabled: bool,
    /// Restrict witnessing to companions (`companion:<slot>` keys) only.
    pub companions_only: bool,
    /// Optional cooldown shared across ALL trigger types (off by default).
    pub global_cooldown_enabled: bool,
    pub global_cooldown_secs: u32,
    /// Per-type reaction rules, keyed by event type. Absent type = memory only.
    pub triggers: std::collections::BTreeMap<String, TriggerRule>,
}

impl Default for TriggerSettings {
    fn default() -> Self {
        Self {
            version: 2,
            enabled: true,
            companions_only: false,
            global_cooldown_enabled: false,
            global_cooldown_secs: DEFAULT_GLOBAL_COOLDOWN_SECS,
            triggers: std::collections::BTreeMap::new(),
        }
    }
}

impl TriggerSettings {
    /// The effective rule for a type: the saved rule, or "reactions off".
    fn rule(&self, event_type: &str) -> Option<&TriggerRule> {
        self.triggers.get(event_type).filter(|rule| rule.enabled)
    }

    /// Whether this type is sight-gated — independent of the trigger switch,
    /// so a memory-only type can still require being seen.
    fn requires_sight(&self, event_type: &str) -> bool {
        self.triggers
            .get(event_type)
            .is_some_and(|rule| rule.require_sight)
    }
}

fn settings_path(data_root: &Path) -> PathBuf {
    data_root.join("headless").join("triggers.json")
}

/// Reads the trigger settings, migrating a v1 file in place. v1 stored
/// `triggers` as a plain array of the CHECKED types; v2 stores a per-type rule
/// map (enabled / chancePercent / cooldownSecs). A v1 checked type becomes an
/// enabled rule with the defaults (100%, 10s).
pub(crate) fn read_trigger_settings(data_root: &Path) -> TriggerSettings {
    let Some(raw) = fs::read_to_string(settings_path(data_root))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
    else {
        return TriggerSettings::default();
    };
    let mut raw = raw;
    if let Some(list) = raw.get("triggers").and_then(Value::as_array).cloned() {
        // v1 shape: convert the checked-type list into default-knob rules.
        let mut map = serde_json::Map::new();
        for entry in list {
            if let Some(event_type) = entry.as_str() {
                map.insert(
                    event_type.trim().to_lowercase(),
                    serde_json::to_value(TriggerRule::default()).unwrap_or_default(),
                );
            }
        }
        if let Some(obj) = raw.as_object_mut() {
            obj.insert("triggers".into(), Value::Object(map));
            obj.insert("version".into(), json!(2));
        }
    }
    serde_json::from_value(raw).unwrap_or_default()
}

fn write_trigger_settings(data_root: &Path, settings: &TriggerSettings) -> anyhow::Result<()> {
    let path = settings_path(data_root);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_vec_pretty(settings)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Pending bundles (in-memory, keyed by native NPC key)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PendingEvent {
    id: String,
    event_type: String,
    summary: String,
}

#[derive(Debug, Clone, Default)]
struct PendingBundle {
    events: Vec<PendingEvent>,
    first_ms: i64,
    last_ms: i64,
}

#[derive(Default)]
struct WitnessState {
    pending: HashMap<String, PendingBundle>,
    /// Last fired reaction per trigger TYPE (each type has its own cooldown).
    last_fire_by_type: HashMap<String, i64>,
    /// Last fired reaction of ANY type (for the optional global cooldown).
    last_fire_any_ms: i64,
}

fn store() -> &'static Mutex<WitnessState> {
    static STORE: OnceLock<Mutex<WitnessState>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(WitnessState::default()))
}

/// Drops every pending bundle. Called on a save-sync RESTORE: the pending
/// events belong to the just-abandoned timeline. (Already-flushed lines roll
/// back with the chat-history checkpoint itself.)
pub(crate) fn clear_pending() {
    if let Ok(mut state) = store().lock() {
        state.pending.clear();
    }
}

// ---------------------------------------------------------------------------
// Fan-out (called from event-log ingest, post-dedup)
// ---------------------------------------------------------------------------

/// Fans a batch of NEWLY-APPENDED raw events (see
/// [`crate::event_log::append_events_detailed`]) out into the witnesses'
/// pending bundles, then immediately flushes + fires a reaction for any event
/// whose type is trigger-checked. Best-effort throughout: a failure never
/// propagates to ingest.
pub(crate) fn fan_out_events(state: &Arc<AppState>, raw_events: &[Value]) {
    let data_root = state.config.active_profile_paths().content_root();
    let settings = read_trigger_settings(&data_root);
    if !settings.enabled {
        return;
    }

    let now = epoch_millis() as i64;
    let triggered: Vec<(String, String)> = {
        let Ok(mut wstate) = store().lock() else { return };
        bundle_incoming(&mut wstate, &settings, raw_events, now)
    };

    // Trigger path: gate (chance roll + per-type cooldown + optional global
    // cooldown), then flush the witness's whole bundle NOW (narration must
    // land in history before the reaction generates against it), then enqueue
    // the reaction for the plugin to pick up when idle. Cooldowns are recorded
    // only when a reaction file is actually written, so a gated/dropped
    // reaction never burns the timer.
    for (npc_key, event_type) in triggered {
        {
            let Ok(wstate) = store().lock() else { return };
            let roll = random_percent();
            if !reaction_gates_pass(&wstate, &settings, &event_type, now, roll) {
                tracing::info!(
                    "witness: {event_type} trigger for {npc_key} gated (chance/cooldown); memory only"
                );
                continue;
            }
        }
        match flush_npc_bundle(state, &npc_key, now) {
            Ok(Some(flushed)) => match enqueue_reaction(state, &npc_key, &flushed, now) {
                Ok(true) => {
                    if let Ok(mut wstate) = store().lock() {
                        record_reaction_fired(&mut wstate, &event_type, now);
                    }
                }
                Ok(false) => {} // dropped (one already queued) — timer untouched
                Err(error) => {
                    tracing::warn!("witness: reaction enqueue for {npc_key} failed: {error}");
                }
            },
            Ok(None) => {} // NPC not in any live chat (yet) — memory only.
            Err(error) => tracing::warn!("witness: trigger flush for {npc_key} failed: {error}"),
        }
    }
}

/// A uniform 0–99 roll for the per-trigger % chance, from a cheap xorshift of
/// the wall-clock nanos (no rand dependency; this gates game flavor, not
/// cryptography).
fn random_percent() -> u32 {
    let mut x = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 ^ (d.as_millis() as u64))
        .unwrap_or(0)
        | 1;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    (x % 100) as u32
}

/// The pure bundling core of [`fan_out_events`]: distributes raw events into
/// the witnesses' pending bundles and returns `(npc_key, event_type)` pairs
/// owed an immediate trigger reaction (one witness per triggering event — the
/// nearest that passed the filters; the plugin orders `witnesses`
/// nearest-first). Chance/cooldown gating happens later, in the caller.
fn bundle_incoming(
    wstate: &mut WitnessState,
    settings: &TriggerSettings,
    raw_events: &[Value],
    now: i64,
) -> Vec<(String, String)> {
    let mut triggered: Vec<(String, String)> = Vec::new();
    for raw in raw_events {
        let Some(obj) = raw.as_object() else { continue };
        let event_type = obj
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("world")
            .trim()
            .to_lowercase();
        // The dialogue already IS the history — never narrate it back.
        if event_type == "conversation" {
            continue;
        }
        let summary = obj
            .get("summary")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        let id = obj
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if summary.is_empty() || id.is_empty() {
            continue;
        }
        // The shared filter (subject, companions-only, sight gate…) — the SAME
        // list the Events page shows as `witnessedBy`.
        let witnesses = effective_witnesses(settings, obj);
        // Aggregate flushes whose FIRST event already fired an instant trigger
        // (the plugin marks them `noTrigger`) are memory-only.
        let no_trigger = obj
            .get("noTrigger")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let mut trigger_witness_pending = !no_trigger && settings.rule(&event_type).is_some();
        for witness in witnesses {
            let bundle = wstate.pending.entry(witness.clone()).or_default();
            // Idempotency belt-and-braces (ingest is already post-dedup).
            if bundle.events.iter().any(|e| e.id == id) {
                continue;
            }
            if bundle.events.is_empty() {
                bundle.first_ms = now;
            }
            bundle.last_ms = now;
            bundle.events.push(PendingEvent {
                id: id.clone(),
                event_type: event_type.clone(),
                summary: summary.clone(),
            });
            // ONE witnessing NPC reacts per triggering event.
            if trigger_witness_pending {
                trigger_witness_pending = false;
                let pair = (witness.clone(), event_type.clone());
                if !triggered.contains(&pair) {
                    triggered.push(pair);
                }
            }
        }
    }
    triggered
}

/// The EFFECTIVE witnesses of one raw event under `settings` — everyone in
/// range MINUS the event's own NPC subject, non-companions when the scope is
/// companions-only, and (for sight-gated types) anyone the player was hidden
/// from. Nearest-first order is preserved from the plugin. This is the single
/// source of truth shared by the fan-out AND the event-log's stored
/// `witnessedBy` annotation, so the Events page shows exactly who the memory
/// system credited.
pub(crate) fn effective_witnesses(
    settings: &TriggerSettings,
    obj: &serde_json::Map<String, Value>,
) -> Vec<String> {
    if !settings.enabled {
        return Vec::new();
    }
    let event_type = obj
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("world")
        .trim()
        .to_lowercase();
    // The dialogue already IS the history — conversations are never witnessed.
    if event_type == "conversation" {
        return Vec::new();
    }
    let subject_key = obj
        .get("subjectNpcKey")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    // Witnesses the player was HIDDEN from when it happened (the engine's
    // detection state, attached by the plugin). Only consulted for
    // sight-gated types.
    let hidden_from: Vec<&str> = obj
        .get("hiddenFrom")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|key| !key.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let sight_gated = settings.requires_sight(&event_type);
    obj.get("witnesses")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|key| !key.is_empty())
                .filter(|key| subject_key.is_empty() || *key != subject_key)
                .filter(|key| !settings.companions_only || key.starts_with("companion:"))
                .filter(|key| !(sight_gated && hidden_from.contains(key)))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Stamps the event's effective witnesses onto the raw value as `witnessedBy`
/// BEFORE it is stored, so the Events page can show who saw each event (an
/// empty list = it happened unobserved — e.g. the player was hidden). Only
/// touches events that actually carried a witness capture; older/foreign
/// events stay untouched.
pub(crate) fn annotate_witnessed_by(settings: &TriggerSettings, event: &mut Value) {
    let effective = match event.as_object() {
        Some(obj) if obj.contains_key("witnesses") => effective_witnesses(settings, obj),
        _ => return,
    };
    if let Some(map) = event.as_object_mut() {
        map.insert("witnessedBy".into(), json!(effective));
    }
}

/// Whether a pending bundle is ready to flush: quiet long enough, old enough,
/// or big enough.
fn bundle_due(bundle: &PendingBundle, now: i64) -> bool {
    !bundle.events.is_empty()
        && (now - bundle.last_ms >= PENDING_QUIET_MS
            || now - bundle.first_ms >= PENDING_MAX_AGE_MS
            || bundle.events.len() >= PENDING_MAX_EVENTS)
}

/// Whether a triggered `(event_type)` reaction may fire right now, given a
/// pre-rolled 0–99 `roll` (injected so tests are deterministic):
/// * `roll` must fall under the type's % chance,
/// * the type's OWN cooldown must have elapsed (per type, not global),
/// * and, when enabled, the GLOBAL cooldown across all types must have too.
/// Read-only — call [`record_reaction_fired`] once a reaction actually ships.
fn reaction_gates_pass(
    wstate: &WitnessState,
    settings: &TriggerSettings,
    event_type: &str,
    now: i64,
    roll: u32,
) -> bool {
    let Some(rule) = settings.rule(event_type) else {
        return false;
    };
    if roll >= rule.chance_percent.min(100) {
        return false;
    }
    // A zero timestamp means "never fired" — only a real prior fire cools down.
    let last_of_type = wstate.last_fire_by_type.get(event_type).copied().unwrap_or(0);
    if last_of_type > 0 && now - last_of_type < i64::from(rule.cooldown_secs) * 1000 {
        return false;
    }
    if settings.global_cooldown_enabled
        && wstate.last_fire_any_ms > 0
        && now - wstate.last_fire_any_ms < i64::from(settings.global_cooldown_secs) * 1000
    {
        return false;
    }
    true
}

fn record_reaction_fired(wstate: &mut WitnessState, event_type: &str, now: i64) {
    wstate.last_fire_by_type.insert(event_type.to_string(), now);
    wstate.last_fire_any_ms = now;
}

// ---------------------------------------------------------------------------
// Flushing pending bundles into chat history
// ---------------------------------------------------------------------------

/// What one flushed bundle produced (used to build the reaction request).
struct FlushedBundle {
    participant_name: String,
    narration: String,
    event_types: Vec<String>,
}

/// Periodic tick (spawned in `lib.rs` next to the scheduler tick): flushes
/// bundles that went quiet / aged out and drops orphans no live chat knows.
pub(crate) fn tick(state: &Arc<AppState>) {
    let now = epoch_millis() as i64;
    let due: Vec<String> = {
        let Ok(wstate) = store().lock() else { return };
        wstate
            .pending
            .iter()
            .filter(|(_, bundle)| bundle_due(bundle, now))
            .map(|(key, _)| key.clone())
            .collect()
    };
    for npc_key in due {
        match flush_npc_bundle(state, &npc_key, now) {
            Ok(Some(_)) | Ok(None) => {}
            Err(error) => tracing::warn!("witness: flush for {npc_key} failed: {error}"),
        }
    }
}

/// Flushes every pending bundle belonging to a participant of `live_chat`.
/// Called by the generate path BEFORE the player's message is persisted, so a
/// witnessed line always precedes the message that asks about it.
pub(crate) fn flush_for_live_chat(state: &Arc<AppState>, live_chat: &LiveChat) {
    let now = epoch_millis() as i64;
    let keys: Vec<String> = {
        let Ok(wstate) = store().lock() else { return };
        wstate
            .pending
            .keys()
            .filter(|key| resolve_participant(live_chat, key).is_some())
            .cloned()
            .collect()
    };
    for npc_key in keys {
        if let Err(error) = flush_npc_bundle(state, &npc_key, now) {
            tracing::warn!("witness: pre-turn flush for {npc_key} failed: {error}");
        }
    }
}

/// Takes `npc_key`'s pending bundle and appends it to their live-chat history
/// as ONE narration message. Returns `Ok(None)` when the NPC resolves to no
/// live-chat participant — the bundle is kept (or dropped once orphan-old).
fn flush_npc_bundle(
    state: &Arc<AppState>,
    npc_key: &str,
    now: i64,
) -> anyhow::Result<Option<FlushedBundle>> {
    // Resolve BEFORE taking the bundle so an unresolvable NPC keeps pending.
    let Some(live_chat) = crate::generate::active_live_chat(state)
        .map_err(|e| anyhow::anyhow!(e.0.to_string()))?
    else {
        return Ok(None);
    };
    let Some((participant_id, participant_name)) = resolve_participant(&live_chat, npc_key) else {
        // Never talked to this NPC: age the bundle out eventually.
        if let Ok(mut wstate) = store().lock() {
            if let Some(bundle) = wstate.pending.get(npc_key) {
                if now - bundle.first_ms >= PENDING_ORPHAN_MAX_AGE_MS {
                    wstate.pending.remove(npc_key);
                }
            }
        }
        return Ok(None);
    };

    let Some(bundle) = store()
        .lock()
        .ok()
        .and_then(|mut wstate| wstate.pending.remove(npc_key))
    else {
        return Ok(None);
    };
    if bundle.events.is_empty() {
        return Ok(None);
    }

    let segment = current_segment(&live_chat)
        .ok_or_else(|| anyhow::anyhow!("live chat has no current segment"))?;
    let player_name = player_display_name(state, &live_chat);

    let lines: Vec<String> = bundle
        .events
        .iter()
        .map(|event| format_narration(&player_name, &event.event_type, &event.summary))
        .collect();
    let narration = lines.join("\n");
    let event_ids: Vec<&str> = bundle.events.iter().map(|e| e.id.as_str()).collect();
    let mut event_types: Vec<String> = bundle.events.iter().map(|e| e.event_type.clone()).collect();
    event_types.dedup();

    write_narration_message(
        state,
        &live_chat,
        &segment,
        &participant_id,
        &narration,
        &event_ids,
        &event_types,
    )?;
    tracing::info!(
        "witness: {} narration line(s) flushed into {}'s history",
        bundle.events.len(),
        participant_name
    );

    Ok(Some(FlushedBundle {
        participant_name,
        narration,
        event_types,
    }))
}

/// Appends ONE narration message to a participant's history — a system line
/// inside the history flow (`is_system` renders/prompts it as narration),
/// audible only to that participant, marked `extra.chasm.witnessed` forever.
/// Shared by the bundle flush and the travel-arrival memory.
fn write_narration_message(
    state: &AppState,
    live_chat: &LiveChat,
    segment: &LiveChatSegment,
    participant_id: &str,
    narration: &str,
    event_ids: &[&str],
    event_types: &[String],
) -> anyhow::Result<()> {
    let message = STJsonlChatMessage {
        name: "Narrator".to_string(),
        is_user: false,
        is_system: true,
        send_date: Some(now_iso()),
        mes: narration.to_string(),
        extra: json!({
            "headless": {
                "characterId": Value::Null,
                "metadata": {
                    "live": {
                        "liveChatId": live_chat.id,
                        "segmentId": segment.id,
                        "speakerParticipantId": Value::Null,
                        "present": [participant_id],
                        "audibleTo": [participant_id],
                        "location": segment.location,
                        "strictVisibility": true,
                    }
                }
            },
            "chasm": {
                "witnessed": true,
                "event_ids": event_ids,
                "event_types": event_types,
            }
        }),
        original_avatar: None,
    };
    state
        .repository
        .append_segment_message(segment, &message)
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}

/// Resolves `npc_key` in the active live chat and appends one narration line
/// to their history. Returns `Ok(None)` when the NPC has no live-chat
/// participant to receive it (never spoken to).
fn append_narration_message(
    state: &AppState,
    npc_key: &str,
    narration: &str,
    event_ids: &[&str],
    event_types: &[String],
) -> anyhow::Result<Option<FlushedBundle>> {
    let Some(live_chat) =
        crate::generate::active_live_chat(state).map_err(|e| anyhow::anyhow!(e.0.to_string()))?
    else {
        return Ok(None);
    };
    let Some((participant_id, participant_name)) = resolve_participant(&live_chat, npc_key) else {
        return Ok(None);
    };
    let segment = current_segment(&live_chat)
        .ok_or_else(|| anyhow::anyhow!("live chat has no current segment"))?;
    write_narration_message(
        state,
        &live_chat,
        &segment,
        &participant_id,
        narration,
        event_ids,
        event_types,
    )?;
    Ok(Some(FlushedBundle {
        participant_name,
        narration: narration.to_string(),
        event_types: event_types.to_vec(),
    }))
}

/// The Events-page summary (third person) and the traveler's own memory line
/// (second person, per the design: "you have traveled to and met the player")
/// for one completed journey.
fn travel_texts(
    npc_name: &str,
    dest_name: &str,
    to_player: bool,
    player_name: &str,
) -> (String, String) {
    if to_player {
        (
            format!("{npc_name} traveled to meet {player_name}"),
            format!("You have traveled to and met {player_name}."),
        )
    } else {
        (
            format!("{npc_name} arrived at {dest_name}"),
            format!("You have traveled to {dest_name} and arrived."),
        )
    }
}

/// A journey from the movement engine completed: log an `arrival` event
/// (Events page), write the traveler's OWN memory line ("You have traveled
/// to…"), and — when `arrival` is trigger-enabled — make them speak up through
/// the normal reaction queue, gated by the usual chance/cooldowns. Best-effort
/// throughout; never fails the journey.
pub(crate) fn on_npc_travel_completed(
    state: &AppState,
    npc_key: &str,
    npc_name: &str,
    dest_name: &str,
    to_player: bool,
) {
    let data_root = state.config.active_profile_paths().content_root();
    let settings = read_trigger_settings(&data_root);
    let now_ms = epoch_millis() as i64;

    let player_name = crate::generate::active_live_chat(state)
        .ok()
        .flatten()
        .map(|live_chat| player_display_name(state, &live_chat))
        .unwrap_or_else(|| "the player".to_string());
    let (summary, narration) = travel_texts(npc_name, dest_name, to_player, &player_name);

    // The Events-page record (always — arriving is a real gameplay beat);
    // `witnessedBy` reflects whether the memory system credited the traveler.
    let event_id = format!("trv_{now_ms}");
    let witnessed: Vec<&str> = if settings.enabled { vec![npc_key] } else { Vec::new() };
    let event = json!({
        "id": event_id,
        "type": "arrival",
        "summary": summary,
        "witnessedBy": witnessed,
    });
    if let Err(error) = crate::event_log::append_events_detailed(&data_root, &[event]) {
        tracing::warn!("witness: arrival event append failed: {error}");
    }
    if !settings.enabled {
        return;
    }

    // The traveler's own memory: they know they made the trip.
    let event_types = vec!["arrival".to_string()];
    let flushed = match append_narration_message(
        state,
        npc_key,
        &narration,
        &[event_id.as_str()],
        &event_types,
    ) {
        Ok(Some(flushed)) => flushed,
        Ok(None) => return, // never spoken to — no history to remember in
        Err(error) => {
            tracing::warn!("witness: arrival narration for {npc_key} failed: {error}");
            return;
        }
    };
    tracing::info!("witness: arrival memory written for {npc_key} ({summary})");

    // Trigger: the traveler announces themselves ("Made it — you wanted me?").
    if settings.rule("arrival").is_none() {
        return;
    }
    {
        let Ok(wstate) = store().lock() else { return };
        let roll = random_percent();
        if !reaction_gates_pass(&wstate, &settings, "arrival", now_ms, roll) {
            tracing::info!("witness: arrival trigger for {npc_key} gated; memory only");
            return;
        }
    }
    match enqueue_reaction(state, npc_key, &flushed, now_ms) {
        Ok(true) => {
            if let Ok(mut wstate) = store().lock() {
                record_reaction_fired(&mut wstate, "arrival", now_ms);
            }
        }
        Ok(false) => {}
        Err(error) => tracing::warn!("witness: arrival reaction enqueue failed: {error}"),
    }
}

/// Maps a native NPC key to `(participant_id, display_name)` in one live chat.
/// The bridge keys participants `npc:<native_npc_key>` and records the native
/// key on presence metadata (`nativeNpcKey`) — check both.
fn resolve_participant(live_chat: &LiveChat, npc_key: &str) -> Option<(String, String)> {
    let direct_id = format!("npc:{npc_key}");
    for participant in live_chat.presence.values() {
        let matches = participant.participant_id == direct_id
            || participant
                .metadata
                .get("nativeNpcKey")
                .and_then(Value::as_str)
                .is_some_and(|key| key == npc_key);
        if matches {
            let name = if participant.name.trim().is_empty() {
                npc_key.to_string()
            } else {
                participant.name.clone()
            };
            return Some((participant.participant_id.clone(), name));
        }
    }
    None
}

fn current_segment(live_chat: &LiveChat) -> Option<LiveChatSegment> {
    live_chat
        .segments
        .iter()
        .find(|segment| segment.id == live_chat.current_segment_id)
        .or_else(|| live_chat.segments.last())
        .cloned()
}

/// The player's display name for narration, from the newest recorded gamestate
/// macro table (the same source the scenario + Gamestate page use).
fn player_display_name(state: &AppState, live_chat: &LiveChat) -> String {
    let (_, macros) = crate::generate::latest_chat_macros(state, live_chat);
    macros
        .get("player_name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("The player")
        .to_string()
}

/// One narration line for one witnessed event, in third person past tense.
/// The plugin's summaries write the player's deeds subject-less ("Picked up 5
/// items…", "Fought two Powder Gangers…") or as "You …" — both re-anchor onto
/// the player's name. Summaries with their own subject pass through verbatim.
fn format_narration(player_name: &str, event_type: &str, summary: &str) -> String {
    let summary = summary.trim();
    // Second person → third: possessives too ("from your Varmint Rifle").
    let third_person = |text: &str| text.replace(" your ", " their ");
    if let Some(rest) = summary.strip_prefix("You ") {
        // Past-tense verbs are person-invariant: "You picked" → "Alex picked".
        return format!("{player_name} {}", third_person(rest));
    }
    if summary
        .split_whitespace()
        .next()
        .is_some_and(|first| first.eq_ignore_ascii_case(player_name))
    {
        return summary.to_string();
    }
    if PLAYER_SUBJECT_TYPES.contains(&event_type) {
        let mut chars = summary.chars();
        if let Some(first) = chars.next() {
            let lowered: String = first.to_lowercase().chain(chars).collect();
            return format!("{player_name} {}", third_person(&lowered));
        }
    }
    summary.to_string()
}

// ---------------------------------------------------------------------------
// Trigger reactions (control/reactions queue, plugin-polled — song pattern)
// ---------------------------------------------------------------------------

/// Writes the reaction queue file the plugin polls when idle. LINE-BASED (the
/// plugin has no JSON parser), mirroring the song delivery format:
/// ```text
/// NVBRIDGE_REACTION_V1
/// <reactionId>
/// <npcKey>
/// <npcName>
/// <narration, single line>
/// <event types, comma-separated>
/// ```
/// Never stacks: one un-consumed reaction file at a time (chance and cooldowns
/// were already gated by the caller), so a burst of triggered events can never
/// cause a reaction storm. Returns whether a file was actually written.
fn enqueue_reaction(
    state: &AppState,
    npc_key: &str,
    flushed: &FlushedBundle,
    now: i64,
) -> anyhow::Result<bool> {
    let dir = crate::scheduler::bridge_root(state)
        .join("control")
        .join("reactions");
    fs::create_dir_all(&dir)?;
    // At most ONE queued reaction, and the NEWEST wins: an un-consumed file
    // means the game is busy (or the witness walked off) — the fresh event is
    // the one worth reacting to when the game frees up, so SUPERSEDE the old
    // file instead of dropping the new reaction. (Dropping here used to create
    // minute-long dead windows: one stuck file silenced every later trigger
    // until the plugin's staleness sweep cleared it.)
    for entry in fs::read_dir(&dir)?.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "txt") {
            tracing::info!(
                "witness: superseding queued reaction {} with a fresh one for {npc_key}",
                path.file_name().and_then(|n| n.to_str()).unwrap_or("?")
            );
            let _ = fs::remove_file(&path);
        }
    }

    let sanitize = |s: &str| s.replace(['\r', '\n'], " ");
    let id = format!("rx_{}", now);
    let content = format!(
        "NVBRIDGE_REACTION_V1\n{id}\n{key}\n{name}\n{narration}\n{types}\n",
        key = sanitize(npc_key),
        name = sanitize(&flushed.participant_name),
        narration = sanitize(&flushed.narration),
        types = flushed.event_types.join(","),
    );
    // Temp + rename so the plugin never reads a half-written file.
    let tmp = dir.join(format!("{id}.txt.tmp"));
    let final_path = dir.join(format!("{id}.txt"));
    fs::write(&tmp, content.as_bytes())?;
    fs::rename(&tmp, &final_path)?;
    tracing::info!("witness: reaction queued for {npc_key} ({})", final_path.display());
    Ok(true)
}

// ---------------------------------------------------------------------------
// Routes (Triggers page round-trip)
// ---------------------------------------------------------------------------

/// The catalog the Triggers page renders: every static plugin type UNIONED with
/// any type observed in the current event store (plus any saved rule for a
/// type we no longer know), each carrying its full reaction rule.
fn catalog_view(settings: &TriggerSettings, observed: &[String]) -> Value {
    let mut types: Vec<String> = STATIC_EVENT_TYPES.iter().map(|t| t.to_string()).collect();
    for t in observed.iter().chain(settings.triggers.keys()) {
        let t = t.trim().to_lowercase();
        if !t.is_empty() && !types.contains(&t) {
            types.push(t);
        }
    }
    let defaults = TriggerRule::default();
    let catalog: Vec<Value> = types
        .iter()
        .map(|t| {
            let rule = settings.triggers.get(t);
            json!({
                "type": t,
                "enabled": rule.map(|r| r.enabled).unwrap_or(false),
                "chancePercent": rule.map(|r| r.chance_percent).unwrap_or(defaults.chance_percent),
                "cooldownSecs": rule.map(|r| r.cooldown_secs).unwrap_or(defaults.cooldown_secs),
                "requireSight": rule.map(|r| r.require_sight).unwrap_or(false),
                // Known statically vs discovered from the store (future types).
                "dynamic": !STATIC_EVENT_TYPES.contains(&t.as_str()),
                // The dialogue is already the history: witness fan-out excludes
                // conversation events entirely, so it can't trigger either.
                "excluded": t == "conversation",
            })
        })
        .collect();
    json!({
        "enabled": settings.enabled,
        "companionsOnly": settings.companions_only,
        "globalCooldownEnabled": settings.global_cooldown_enabled,
        "globalCooldownSecs": settings.global_cooldown_secs,
        "catalog": catalog,
    })
}

/// Pushes the enabled trigger types to the plugin as a line-based control file
/// (`<bridge_root>/control/triggers.cfg`, the hotkeys.cfg pattern — the plugin
/// live-polls it). The plugin uses the list to make trigger-type events
/// INSTANT: flush the event batch immediately and fire the first event of an
/// aggregation window on its own (the rest of the window becomes a memory-only
/// `noTrigger` aggregate). Chance/cooldown/sight knobs stay chasm-side.
pub(crate) fn push_trigger_config(state: &AppState, settings: &TriggerSettings) {
    let dir = crate::scheduler::bridge_root(state).join("control");
    if let Err(error) = fs::create_dir_all(&dir) {
        tracing::warn!("witness: could not create {}: {error}", dir.display());
        return;
    }
    let mut content = String::from("NVBRIDGE_TRIGGERS_V1\n");
    if settings.enabled {
        for (event_type, rule) in &settings.triggers {
            if rule.enabled {
                content.push_str(event_type);
                content.push('\n');
            }
        }
    }
    let path = dir.join("triggers.cfg");
    if let Err(error) = fs::write(&path, content.as_bytes()) {
        tracing::warn!("witness: could not write {}: {error}", path.display());
    }
}

/// `GET /api/ui/v1/triggers` — settings + the full event-type catalog.
pub(crate) async fn triggers_view(
    State(state): State<Arc<AppState>>,
) -> WebResult<Json<Value>> {
    let data_root = state.config.active_profile_paths().content_root();
    let view = tokio::task::spawn_blocking(move || {
        let settings = read_trigger_settings(&data_root);
        let observed = crate::event_log::observed_event_types(&data_root);
        catalog_view(&settings, &observed)
    })
    .await
    .map_err(|e| web_err(e.to_string()))?;
    Ok(Json(view))
}

/// Applies a save body onto loaded settings. Body shape:
/// `{ enabled, companionsOnly, globalCooldownEnabled, globalCooldownSecs,
///    triggers: [{ type, enabled, chancePercent, cooldownSecs }, …] }`
/// (legacy plain-string entries are accepted as enabled-with-defaults).
fn apply_trigger_save(settings: &mut TriggerSettings, body: &Value) {
    if let Some(enabled) = body.get("enabled").and_then(Value::as_bool) {
        settings.enabled = enabled;
    }
    if let Some(companions_only) = body
        .get("companionsOnly")
        .or_else(|| body.get("companions_only"))
        .and_then(Value::as_bool)
    {
        settings.companions_only = companions_only;
    }
    if let Some(global_enabled) = body
        .get("globalCooldownEnabled")
        .or_else(|| body.get("global_cooldown_enabled"))
        .and_then(Value::as_bool)
    {
        settings.global_cooldown_enabled = global_enabled;
    }
    if let Some(global_secs) = body
        .get("globalCooldownSecs")
        .or_else(|| body.get("global_cooldown_secs"))
        .and_then(Value::as_u64)
    {
        settings.global_cooldown_secs = (global_secs as u32).min(86_400);
    }
    let Some(triggers) = body.get("triggers").and_then(Value::as_array) else {
        return;
    };
    settings.version = 2;
    settings.triggers.clear();
    for entry in triggers {
        // Legacy string entry = enabled with default knobs.
        if let Some(event_type) = entry.as_str() {
            let t = event_type.trim().to_lowercase();
            if !t.is_empty() && t != "conversation" {
                settings.triggers.insert(t, TriggerRule::default());
            }
            continue;
        }
        let Some(obj) = entry.as_object() else { continue };
        let event_type = obj
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_lowercase();
        if event_type.is_empty() || event_type == "conversation" {
            continue;
        }
        let defaults = TriggerRule::default();
        let rule = TriggerRule {
            enabled: obj.get("enabled").and_then(Value::as_bool).unwrap_or(true),
            chance_percent: obj
                .get("chancePercent")
                .or_else(|| obj.get("chance_percent"))
                .and_then(Value::as_u64)
                .map(|v| (v as u32).min(100))
                .unwrap_or(defaults.chance_percent),
            cooldown_secs: obj
                .get("cooldownSecs")
                .or_else(|| obj.get("cooldown_secs"))
                .and_then(Value::as_u64)
                .map(|v| (v as u32).min(86_400))
                .unwrap_or(defaults.cooldown_secs),
            require_sight: obj
                .get("requireSight")
                .or_else(|| obj.get("require_sight"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
        };
        // Persist the rule even when disabled, so a re-enable keeps the user's
        // tuned chance/cooldown instead of snapping back to the defaults.
        settings.triggers.insert(event_type, rule);
    }
}

/// `POST /api/ui/v1/triggers/save` — see [`apply_trigger_save`] for the body;
/// returns the refreshed view.
pub(crate) async fn triggers_save(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> WebResult<Json<Value>> {
    let data_root = state.config.active_profile_paths().content_root();
    let push_state = Arc::clone(&state);
    let view = tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let mut settings = read_trigger_settings(&data_root);
        apply_trigger_save(&mut settings, &body);
        write_trigger_settings(&data_root, &settings)?;
        // Keep the plugin's live copy of the enabled types in step (it makes
        // those events instant-flush / instant-first).
        push_trigger_config(&push_state, &settings);
        let observed = crate::event_log::observed_event_types(&data_root);
        Ok(catalog_view(&settings, &observed))
    })
    .await
    .map_err(|e| web_err(e.to_string()))?
    .map_err(WebError::from)?;
    Ok(Json(view))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narration_reanchors_player_subject_summaries() {
        assert_eq!(
            format_narration("The Courier", "item", "Picked up 5 items incl. Stimpak x3"),
            "The Courier picked up 5 items incl. Stimpak x3"
        );
        assert_eq!(
            format_narration("Alex", "combat", "Fought two Powder Gangers — killed 2"),
            "Alex fought two Powder Gangers — killed 2"
        );
        // "You …" prefixes swap straight to the player's name.
        assert_eq!(
            format_narration("Alex", "item", "You unequipped the Vault 21 jumpsuit"),
            "Alex unequipped the Vault 21 jumpsuit"
        );
        // Subject-bearing summaries pass through verbatim.
        assert_eq!(
            format_narration("Alex", "death", "Ringo died"),
            "Ringo died"
        );
        // Already-anchored summaries are not double-prefixed.
        assert_eq!(
            format_narration("Alex", "item", "Alex picked up a 9mm pistol"),
            "Alex picked up a 9mm pistol"
        );
        // Second-person possessives flip too.
        assert_eq!(
            format_narration("Alex", "shooting", "Fired 3 rounds from your Varmint Rifle"),
            "Alex fired 3 rounds from their Varmint Rifle"
        );
    }

    #[test]
    fn catalog_unions_static_saved_and_observed_types_with_rules() {
        let mut settings = TriggerSettings::default();
        settings.triggers.insert(
            "item".into(),
            TriggerRule {
                enabled: true,
                chance_percent: 35,
                cooldown_secs: 42,
                ..Default::default()
            },
        );
        settings
            .triggers
            .insert("custom_saved".into(), TriggerRule::default());
        let observed = vec!["weather".into(), "item".into()];
        let view = catalog_view(&settings, &observed);
        let catalog = view["catalog"].as_array().unwrap();
        let types: Vec<&str> = catalog.iter().map(|c| c["type"].as_str().unwrap()).collect();
        for t in STATIC_EVENT_TYPES {
            assert!(types.contains(&t), "missing static type {t}");
        }
        assert!(types.contains(&"weather"), "missing observed type");
        assert!(types.contains(&"custom_saved"), "missing saved type");
        let item = catalog.iter().find(|c| c["type"] == "item").unwrap();
        assert_eq!(item["enabled"], true);
        assert_eq!(item["chancePercent"], 35);
        assert_eq!(item["cooldownSecs"], 42);
        assert_eq!(item["dynamic"], false);
        // Types with no saved rule surface the defaults, reactions off.
        let theft = catalog.iter().find(|c| c["type"] == "theft").unwrap();
        assert_eq!(theft["enabled"], false);
        assert_eq!(theft["chancePercent"], 100);
        assert_eq!(theft["cooldownSecs"], 10);
        let weather = catalog.iter().find(|c| c["type"] == "weather").unwrap();
        assert_eq!(weather["dynamic"], true);
        let conversation = catalog.iter().find(|c| c["type"] == "conversation").unwrap();
        assert_eq!(conversation["excluded"], true);
    }

    fn raw_event(id: &str, event_type: &str, summary: &str, witnesses: &[&str]) -> Value {
        json!({
            "id": id,
            "type": event_type,
            "summary": summary,
            "witnesses": witnesses,
        })
    }

    fn settings_with_triggers(triggers: &[&str]) -> TriggerSettings {
        let mut settings = TriggerSettings::default();
        for t in triggers {
            settings.triggers.insert(t.to_string(), TriggerRule::default());
        }
        settings
    }

    #[test]
    fn bundling_fans_out_to_each_witness_and_dedups_by_event_id() {
        let mut wstate = WitnessState::default();
        let settings = settings_with_triggers(&[]);
        let batch = vec![raw_event("e1", "item", "Picked up 3 items", &["easy_pete", "sunny_smiles"])];
        bundle_incoming(&mut wstate, &settings, &batch, 1_000);
        assert_eq!(wstate.pending.len(), 2);
        assert_eq!(wstate.pending["easy_pete"].events.len(), 1);
        assert_eq!(wstate.pending["sunny_smiles"].events.len(), 1);

        // A redelivered event (same id) never double-inserts — idempotency at
        // the bundle level, on top of ingest's append-only dedup.
        bundle_incoming(&mut wstate, &settings, &batch, 2_000);
        assert_eq!(wstate.pending["easy_pete"].events.len(), 1);
    }

    #[test]
    fn bundling_excludes_conversations_and_the_events_own_subject() {
        let mut wstate = WitnessState::default();
        let settings = settings_with_triggers(&[]);
        bundle_incoming(
            &mut wstate,
            &settings,
            &[json!({
                "id": "e1",
                "type": "conversation",
                "summary": "Talked with Easy Pete",
                "witnesses": ["sunny_smiles"],
            })],
            1_000,
        );
        assert!(wstate.pending.is_empty(), "conversation events never fan out");

        // Self-witnessing: the event's NPC subject is filtered even when the
        // plugin left them in the witness list.
        bundle_incoming(
            &mut wstate,
            &settings,
            &[json!({
                "id": "e2",
                "type": "death",
                "summary": "Ringo died",
                "subjectNpcKey": "ringo",
                "witnesses": ["ringo", "easy_pete"],
            })],
            1_000,
        );
        assert!(!wstate.pending.contains_key("ringo"));
        assert_eq!(wstate.pending["easy_pete"].events.len(), 1);
    }

    #[test]
    fn bundling_respects_companions_only_scope() {
        let mut wstate = WitnessState::default();
        let settings = TriggerSettings {
            companions_only: true,
            ..Default::default()
        };
        bundle_incoming(
            &mut wstate,
            &settings,
            &[raw_event("e1", "item", "Picked up a thing", &["easy_pete", "companion:3"])],
            1_000,
        );
        assert!(!wstate.pending.contains_key("easy_pete"));
        assert!(wstate.pending.contains_key("companion:3"));
    }

    #[test]
    fn trigger_types_pick_exactly_one_witness_the_nearest() {
        let mut wstate = WitnessState::default();
        let settings = settings_with_triggers(&["item"]);
        let triggered = bundle_incoming(
            &mut wstate,
            &settings,
            &[
                raw_event("e1", "item", "Unequipped the jumpsuit", &["sunny_smiles", "easy_pete"]),
                raw_event("e2", "location", "Arrived at the saloon", &["sunny_smiles"]),
            ],
            1_000,
        );
        // Only the trigger-enabled type fires, and only its FIRST witness.
        assert_eq!(
            triggered,
            vec![("sunny_smiles".to_string(), "item".to_string())]
        );
        // Both witnesses still get the memory line.
        assert_eq!(wstate.pending["easy_pete"].events.len(), 1);
        assert_eq!(wstate.pending["sunny_smiles"].events.len(), 2);
    }

    #[test]
    fn travel_texts_read_naturally_for_both_destinations() {
        let (summary, narration) = travel_texts("Sunny Smiles", "me", true, "Alex");
        assert_eq!(summary, "Sunny Smiles traveled to meet Alex");
        assert_eq!(narration, "You have traveled to and met Alex.");

        let (summary, narration) =
            travel_texts("Easy Pete", "Prospector Saloon", false, "Alex");
        assert_eq!(summary, "Easy Pete arrived at Prospector Saloon");
        assert_eq!(narration, "You have traveled to Prospector Saloon and arrived.");
    }

    #[test]
    fn annotate_witnessed_by_stamps_the_effective_list() {
        let mut settings = TriggerSettings::default();
        settings.triggers.insert(
            "theft".into(),
            TriggerRule {
                require_sight: true,
                ..Default::default()
            },
        );
        // Sight-gated theft: the hidden witness is excluded from the stored list.
        let mut event = json!({
            "id": "e1",
            "type": "theft",
            "summary": "Stole a thing",
            "witnesses": ["easy_pete", "sunny_smiles"],
            "hiddenFrom": ["easy_pete"],
        });
        annotate_witnessed_by(&settings, &mut event);
        assert_eq!(event["witnessedBy"], json!(["sunny_smiles"]));

        // Everyone hidden -> an EMPTY list is stored (unseen, meaningfully so).
        let mut unseen = json!({
            "id": "e2",
            "type": "theft",
            "summary": "Stole a thing",
            "witnesses": ["easy_pete"],
            "hiddenFrom": ["easy_pete"],
        });
        annotate_witnessed_by(&settings, &mut unseen);
        assert_eq!(unseen["witnessedBy"], json!([]));

        // Events without a witness capture stay untouched (pre-feature saves).
        let mut foreign = json!({ "id": "e3", "type": "day", "summary": "A new day" });
        annotate_witnessed_by(&settings, &mut foreign);
        assert!(foreign.get("witnessedBy").is_none());

        // The subject never witnesses their own event.
        let mut death = json!({
            "id": "e4",
            "type": "death",
            "summary": "Ringo died",
            "subjectNpcKey": "ringo",
            "witnesses": ["ringo", "easy_pete"],
        });
        annotate_witnessed_by(&settings, &mut death);
        assert_eq!(death["witnessedBy"], json!(["easy_pete"]));
    }

    #[test]
    fn no_trigger_aggregates_are_memory_only() {
        // The plugin's instant-first windows: the first event fired alone (and
        // triggered); the follow-up aggregate arrives marked noTrigger and
        // must bundle WITHOUT firing a second reaction.
        let mut wstate = WitnessState::default();
        let settings = settings_with_triggers(&["item"]);
        let triggered = bundle_incoming(
            &mut wstate,
            &settings,
            &[json!({
                "id": "agg1",
                "type": "item",
                "summary": "Picked up 3 more items incl. Stimpak x2",
                "witnesses": ["easy_pete"],
                "noTrigger": true,
            })],
            1_000,
        );
        assert!(triggered.is_empty(), "noTrigger aggregate must not fire");
        assert_eq!(wstate.pending["easy_pete"].events.len(), 1, "memory still bundles");
    }

    #[test]
    fn disabled_rules_never_trigger() {
        let mut wstate = WitnessState::default();
        let mut settings = TriggerSettings::default();
        settings.triggers.insert(
            "item".into(),
            TriggerRule {
                enabled: false, // tuned but switched off — memory only
                chance_percent: 100,
                cooldown_secs: 0,
                ..Default::default()
            },
        );
        let triggered = bundle_incoming(
            &mut wstate,
            &settings,
            &[raw_event("e1", "item", "Picked up a thing", &["easy_pete"])],
            1_000,
        );
        assert!(triggered.is_empty());
        assert_eq!(wstate.pending["easy_pete"].events.len(), 1);
    }

    #[test]
    fn bundles_flush_on_quiet_age_or_size() {
        let mut bundle = PendingBundle {
            events: vec![PendingEvent {
                id: "e1".into(),
                event_type: "item".into(),
                summary: "Picked up a thing".into(),
            }],
            first_ms: 0,
            last_ms: 0,
        };
        // Fresh: not due.
        assert!(!bundle_due(&bundle, 1_000));
        // Quiet window elapsed: due.
        assert!(bundle_due(&bundle, PENDING_QUIET_MS));
        // Still active (recent last_ms) but old overall: due via max age.
        bundle.last_ms = PENDING_MAX_AGE_MS - 1_000;
        assert!(bundle_due(&bundle, PENDING_MAX_AGE_MS));
        // Size cap: due immediately.
        bundle.first_ms = PENDING_MAX_AGE_MS;
        bundle.last_ms = PENDING_MAX_AGE_MS;
        for i in 0..PENDING_MAX_EVENTS {
            bundle.events.push(PendingEvent {
                id: format!("e{i}"),
                event_type: "item".into(),
                summary: "x".into(),
            });
        }
        assert!(bundle_due(&bundle, PENDING_MAX_AGE_MS + 1));
        // Empty: never due.
        bundle.events.clear();
        assert!(!bundle_due(&bundle, i64::MAX));
    }

    #[test]
    fn sight_gated_types_skip_hidden_witnesses_entirely() {
        let mut wstate = WitnessState::default();
        let mut settings = TriggerSettings::default();
        settings.triggers.insert(
            "theft".into(),
            TriggerRule {
                enabled: true,
                require_sight: true,
                ..Default::default()
            },
        );
        let event = json!({
            "id": "e1",
            "type": "theft",
            "summary": "Stole a thing",
            "witnesses": ["easy_pete", "sunny_smiles"],
            "hiddenFrom": ["easy_pete"], // Pete can't see the player
        });
        let triggered = bundle_incoming(&mut wstate, &settings, &[event.clone()], 1_000);
        // Pete gets NOTHING (no memory, no trigger); Sunny gets both — and she
        // becomes the reacting witness even though Pete was listed nearer.
        assert!(!wstate.pending.contains_key("easy_pete"));
        assert_eq!(wstate.pending["sunny_smiles"].events.len(), 1);
        assert_eq!(
            triggered,
            vec![("sunny_smiles".to_string(), "theft".to_string())]
        );

        // The SAME event without the sight gate: Pete witnesses it after all.
        settings.triggers.get_mut("theft").unwrap().require_sight = false;
        let mut wstate2 = WitnessState::default();
        bundle_incoming(&mut wstate2, &settings, &[event], 1_000);
        assert_eq!(wstate2.pending["easy_pete"].events.len(), 1);
    }

    #[test]
    fn sight_gate_applies_to_memory_even_when_trigger_disabled() {
        let mut wstate = WitnessState::default();
        let mut settings = TriggerSettings::default();
        settings.triggers.insert(
            "item".into(),
            TriggerRule {
                enabled: false, // memory-only type…
                require_sight: true, // …but still sight-gated
                ..Default::default()
            },
        );
        let triggered = bundle_incoming(
            &mut wstate,
            &settings,
            &[json!({
                "id": "e1",
                "type": "item",
                "summary": "Picked up a thing",
                "witnesses": ["easy_pete"],
                "hiddenFrom": ["easy_pete"],
            })],
            1_000,
        );
        assert!(triggered.is_empty());
        assert!(wstate.pending.is_empty(), "hidden witness gets no memory line");
    }

    #[test]
    fn chance_percent_gates_the_roll() {
        let wstate = WitnessState::default();
        let mut settings = settings_with_triggers(&["item"]);
        // 100%: every roll (0–99) passes.
        assert!(reaction_gates_pass(&wstate, &settings, "item", 1_000, 0));
        assert!(reaction_gates_pass(&wstate, &settings, "item", 1_000, 99));
        // 0%: no roll passes.
        settings.triggers.get_mut("item").unwrap().chance_percent = 0;
        assert!(!reaction_gates_pass(&wstate, &settings, "item", 1_000, 0));
        // 30%: rolls under 30 pass, 30+ fail.
        settings.triggers.get_mut("item").unwrap().chance_percent = 30;
        assert!(reaction_gates_pass(&wstate, &settings, "item", 1_000, 29));
        assert!(!reaction_gates_pass(&wstate, &settings, "item", 1_000, 30));
        // Un-ruled / disabled types never pass at all.
        assert!(!reaction_gates_pass(&wstate, &settings, "theft", 1_000, 0));
    }

    #[test]
    fn cooldowns_are_per_type_not_shared() {
        let mut wstate = WitnessState::default();
        let settings = settings_with_triggers(&["item", "theft"]); // both 10s default
        record_reaction_fired(&mut wstate, "item", 100_000);
        // Same type inside its cooldown: gated.
        assert!(!reaction_gates_pass(&wstate, &settings, "item", 105_000, 0));
        // A DIFFERENT type is not gated by item's cooldown.
        assert!(reaction_gates_pass(&wstate, &settings, "theft", 105_000, 0));
        // Same type after its cooldown: allowed again.
        assert!(reaction_gates_pass(&wstate, &settings, "item", 110_000, 0));
    }

    #[test]
    fn global_cooldown_spans_all_types_when_enabled() {
        let mut wstate = WitnessState::default();
        let mut settings = settings_with_triggers(&["item", "theft"]);
        settings.global_cooldown_enabled = true;
        settings.global_cooldown_secs = 60;
        record_reaction_fired(&mut wstate, "item", 100_000);
        // Different type, its own cooldown clear — but the GLOBAL window gates it.
        assert!(!reaction_gates_pass(&wstate, &settings, "theft", 130_000, 0));
        // After the global window: allowed.
        assert!(reaction_gates_pass(&wstate, &settings, "theft", 160_000, 0));
        // Global off (default): the same moment would have been allowed.
        settings.global_cooldown_enabled = false;
        assert!(reaction_gates_pass(&wstate, &settings, "theft", 130_000, 0));
    }

    #[test]
    fn v1_settings_files_migrate_to_default_rules() {
        let dir = std::env::temp_dir().join(format!(
            "sb-witness-migrate-{}-{}",
            std::process::id(),
            epoch_millis()
        ));
        fs::create_dir_all(dir.join("headless")).unwrap();
        // A file written by the v1 feature: triggers as a plain string array.
        fs::write(
            dir.join("headless").join("triggers.json"),
            r#"{"version":1,"enabled":true,"companionsOnly":true,"triggers":["item","shooting"]}"#,
        )
        .unwrap();
        let settings = read_trigger_settings(&dir);
        assert_eq!(settings.version, 2);
        assert!(settings.enabled);
        assert!(settings.companions_only);
        assert!(!settings.global_cooldown_enabled, "global cooldown defaults off");
        let item = settings.triggers.get("item").expect("item rule migrated");
        assert!(item.enabled);
        assert_eq!(item.chance_percent, 100);
        assert_eq!(item.cooldown_secs, DEFAULT_TRIGGER_COOLDOWN_SECS);
        assert!(settings.triggers.contains_key("shooting"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_body_round_trips_rules_and_global_cooldown() {
        let mut settings = TriggerSettings::default();
        apply_trigger_save(
            &mut settings,
            &json!({
                "enabled": true,
                "companionsOnly": false,
                "globalCooldownEnabled": true,
                "globalCooldownSecs": 45,
                "triggers": [
                    { "type": "theft", "enabled": true, "chancePercent": 75, "cooldownSecs": 20, "requireSight": true },
                    { "type": "item", "enabled": false, "chancePercent": 50, "cooldownSecs": 5 },
                    { "type": "conversation", "enabled": true }, // always rejected
                ],
            }),
        );
        assert!(settings.global_cooldown_enabled);
        assert_eq!(settings.global_cooldown_secs, 45);
        let theft = settings.triggers.get("theft").unwrap();
        assert!(theft.enabled);
        assert_eq!(theft.chance_percent, 75);
        assert_eq!(theft.cooldown_secs, 20);
        assert!(theft.require_sight);
        assert!(settings.requires_sight("theft"));
        assert!(!settings.requires_sight("item"));
        // Disabled rules keep their tuned knobs for a later re-enable…
        let item = settings.triggers.get("item").unwrap();
        assert!(!item.enabled);
        assert_eq!(item.chance_percent, 50);
        // …but never trigger.
        assert!(settings.rule("item").is_none());
        assert!(!settings.triggers.contains_key("conversation"));
    }

    #[test]
    fn trigger_settings_round_trip_and_defaults() {
        let dir = std::env::temp_dir().join(format!(
            "sb-witness-settings-{}-{}",
            std::process::id(),
            epoch_millis()
        ));
        fs::create_dir_all(&dir).unwrap();
        // Missing file = defaults (enabled, no triggers).
        let fresh = read_trigger_settings(&dir);
        assert!(fresh.enabled);
        assert!(fresh.triggers.is_empty());

        let mut settings = TriggerSettings {
            enabled: false,
            companions_only: true,
            ..Default::default()
        };
        settings.triggers.insert(
            "item".into(),
            TriggerRule {
                enabled: true,
                chance_percent: 60,
                cooldown_secs: 25,
                ..Default::default()
            },
        );
        write_trigger_settings(&dir, &settings).unwrap();
        let read = read_trigger_settings(&dir);
        assert!(!read.enabled);
        assert!(read.companions_only);
        let item = read.triggers.get("item").unwrap();
        assert_eq!(item.chance_percent, 60);
        assert_eq!(item.cooldown_secs, 25);
        let _ = fs::remove_dir_all(&dir);
    }
}
