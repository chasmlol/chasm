//! Connection-driven AI stack lifecycle. chasm is a passive backend, so it keys
//! the local AI stack (koboldcpp for LLM + Whisper STT, plus the TTS server) off
//! the in-game plugin's heartbeat: when the game's bridge connects, chasm starts
//! the stack; when it leaves, chasm tears the whole stack down. Game opens →
//! stack up; game closes → stack down.
//!
//! A single background task (spawned at server startup, gated on the same
//! `CHASM_FNV_BRIDGE` bridge-mode flag) polls the heartbeat every
//! [`POLL_INTERVAL`] and edge-triggers a small [`Phase`] state machine, so it only
//! acts on transitions (never re-spawns a running stack, never busy-loops).
//!
//! Connect/disconnect are deliberately asymmetric. We START the stack on a *fresh*
//! heartbeat (the game is actively writing — a real session just began). Once up, we
//! keep the stack alive as long as the game *process* is alive, using the PID the
//! plugin reports: many games pause their script loop on focus loss, so the
//! heartbeat goes stale on tab-out even though the game is fine — checking the PID
//! means we only tear down when the game is physically closed. Plugins that predate
//! the PID field fall back to a stale-heartbeat grace window ([`STOP_GRACE_SECS`]).

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use chasm_core::{default_bridge_root, AppSettings};

use crate::{connection, launcher, AppState};

/// How often the lifecycle task samples the heartbeat. 2s keeps it responsive to
/// a game launch/quit without busy-looping.
const POLL_INTERVAL: Duration = Duration::from_secs(2);

/// The heartbeat must be at least this stale (seconds) before we tear the stack
/// down. Deliberately longer than the rail's connected-threshold (5s) so a brief
/// hiccup doesn't trigger a full stack teardown + reload.
const STOP_GRACE_SECS: f64 = 10.0;

/// Lifecycle phase, surfaced on `GET /connection/status` and the rail indicator.
/// `Starting` covers the ~12s (koboldcpp) to ~45s (TTS model load) warm-up after a
/// connect; `Stopping` is the brief teardown window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    Disconnected,
    Starting,
    Connected,
    Stopping,
}

impl Phase {
    /// The lowercase string the UI keys off (`disconnected` / `starting` /
    /// `connected` / `stopping`).
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Phase::Disconnected => "disconnected",
            Phase::Starting => "starting",
            Phase::Connected => "connected",
            Phase::Stopping => "stopping",
        }
    }
}

/// Shared lifecycle state. Held in `AppState` so the lifecycle task drives it and
/// `connection_status` reads it. A plain `Mutex` is fine — it's touched once every
/// couple of seconds and on each status poll, never under contention.
#[derive(Debug)]
pub(crate) struct StackLifecycle {
    phase: Mutex<Phase>,
}

impl Default for StackLifecycle {
    fn default() -> Self {
        Self {
            phase: Mutex::new(Phase::Disconnected),
        }
    }
}

impl StackLifecycle {
    /// The current phase (for `connection_status`).
    pub(crate) fn phase(&self) -> Phase {
        *self.phase.lock().expect("stack-lifecycle phase mutex")
    }

    fn set(&self, phase: Phase) {
        *self.phase.lock().expect("stack-lifecycle phase mutex") = phase;
    }
}

/// Imports any profile bundle(s) the connected mod staged into the shared bridge
/// folder (`<bridge_root>/chasm-profile/<id>/`) into chasm's `profiles/`. Called
/// once per connection on the fresh-connect edge (the transition into `Starting`),
/// so it never re-runs on the steady-state polls. Best-effort: every failure is a
/// logged outcome, never a panic.
///
/// On a first-time install where no profile is active yet, the freshly imported id
/// is set as the active profile and persisted, so `active_profile_paths()`
/// immediately resolves into the new content.
fn import_staged_profiles(state: &AppState) {
    let source_root = default_bridge_root().join("chasm-profile");
    let profiles_dir = &state.config.profiles_dir;
    let outcomes = chasm_core::profile_import::import_from_source_root(&source_root, profiles_dir);
    if outcomes.is_empty() {
        return;
    }

    let mut installed_id: Option<String> = None;
    for outcome in &outcomes {
        use chasm_core::ImportAction::*;
        match &outcome.action {
            Installed => {
                tracing::info!("profile '{}' imported", outcome.id);
                installed_id.get_or_insert_with(|| outcome.id.clone());
            }
            Updated => tracing::info!("profile '{}' updated", outcome.id),
            SkippedUpToDate => tracing::info!("profile '{}' already up-to-date", outcome.id),
            Rejected(reason) => {
                tracing::warn!("profile import rejected ('{}'): {reason}", outcome.id)
            }
        }
    }

    // If we just installed a profile and none is active yet, activate the new one so
    // the game "just works" on first connect. `active_profile_id` returns the first
    // listed profile when `profile` is blank/stale, so only set it when it does not
    // already resolve to a real profile.
    if let Some(new_id) = installed_id {
        let mut settings = AppSettings::load(&state.config.settings_path);
        let active = settings.active_profile_id(profiles_dir);
        if active.is_empty() || settings.profile.trim().is_empty() {
            settings.profile = new_id.clone();
            match settings.save(&state.config.settings_path) {
                Ok(()) => tracing::info!("activated imported profile '{new_id}'"),
                Err(e) => tracing::warn!("failed to persist active profile '{new_id}': {e}"),
            }
        }
    }
}

/// Background task: poll the heartbeat and edge-trigger the stack lifecycle.
///
/// - **Disconnected → Connected**: heartbeat went fresh → start the stack
///   ([`launcher::start_ai_stack`]). We flip to `Starting` while the runtimes boot
///   (the plugin retries during warm-up), then to `Connected`.
/// - **Connected → Disconnected**: the game process (heartbeat `pid`) has exited →
///   stop the stack ([`launcher::stop_ai_stack`]), flipping through `Stopping` then
///   `Disconnected`. A paused/tabbed-out game keeps its PID alive, so we hold the
///   stack up through stale heartbeats. Older plugins without a PID fall back to a
///   stale-heartbeat grace ([`STOP_GRACE_SECS`]).
///
/// `start_ai_stack` is idempotent (it skips any runtime already listening on its
/// port), and the edge-trigger means we only call start/stop on a transition, so
/// the task never double-spawns or busy-loops. Spawning/killing is blocking, so we
/// hop onto `spawn_blocking`.
pub(crate) async fn spawn_lifecycle(state: Arc<AppState>) {
    tracing::info!("AI stack lifecycle: watching the game heartbeat (poll {POLL_INTERVAL:?})");
    let mut ticker = tokio::time::interval(POLL_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        ticker.tick().await;

        let settings = AppSettings::load(&state.config.settings_path);
        let hb = connection::read_heartbeat(&settings);
        // Fresh: heartbeat written within the rail's connected window — a real
        // session is actively running right now.
        let fresh = hb
            .last_seen_secs
            .is_some_and(|secs| secs <= connection::FRESH_SECS as f64);

        match state.lifecycle.phase() {
            Phase::Disconnected if fresh => {
                tracing::info!("AI stack lifecycle: game connected — starting the AI stack");
                state.lifecycle.set(Phase::Starting);
                // A fresh connect: sync any profile bundle the connected mod staged
                // into the shared bridge folder, once per connection. Runs before the
                // stack starts so the imported content (and, on a first install, the
                // freshly-activated profile) is on disk when generation begins.
                import_staged_profiles(&state);
                let start_state = Arc::clone(&state);
                // Spawn (blocking) off-thread; flip to Connected when it returns.
                // The stack keeps warming after this returns — that's expected.
                let _ = tokio::task::spawn_blocking(move || {
                    launcher::start_ai_stack(&start_state);
                })
                .await;
                // Warm the in-process retriever (embedder + reranker) alongside the
                // external servers, so it comes up with the stack rather than at boot.
                let warm_state = Arc::clone(&state);
                tokio::spawn(async move {
                    launcher::warm_retrieval(&warm_state).await;
                });
                // Only advance to Connected if we're still meant to be up (the game
                // didn't quit mid-boot, which the next poll would catch anyway).
                if state.lifecycle.phase() == Phase::Starting {
                    state.lifecycle.set(Phase::Connected);
                }
            }
            Phase::Connected => {
                // Stay up while the game process is alive — a paused/tabbed-out game
                // stops writing the heartbeat but its PID lives on. Tear down only
                // when the process is physically gone. Plugins without a PID fall
                // back to the stale-heartbeat grace window.
                let should_stop = match hb.pid {
                    Some(pid) => !connection::pid_is_alive(pid),
                    None => hb
                        .last_seen_secs
                        .map(|secs| secs > STOP_GRACE_SECS)
                        .unwrap_or(true),
                };
                if should_stop {
                    tracing::info!(
                        "AI stack lifecycle: game disconnected — stopping the AI stack"
                    );
                    state.lifecycle.set(Phase::Stopping);
                    let stop_state = Arc::clone(&state);
                    let _ = tokio::task::spawn_blocking(move || {
                        launcher::stop_ai_stack(&stop_state);
                    })
                    .await;
                    state.lifecycle.set(Phase::Disconnected);
                }
            }
            // Already in the right steady state (or mid start/stop) — nothing to do.
            _ => {}
        }
    }
}
