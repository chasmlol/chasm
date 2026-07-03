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
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use chasm_core::{default_bridge_root, AppSettings};

use crate::{connection, launcher, warmup, AppState};

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
    /// Exclusivity flag for the connect-time stack warm-up ([`crate::warmup`]):
    /// `true` while a warm-up run is in flight, so the connect edge and the
    /// manual `/api/stack/start` endpoint can't run two at once (idempotent per
    /// connect — the edge itself only fires once per connect). `Arc` so the
    /// [`WarmupPermit`] can release it on Drop even when the task is aborted.
    warmup_in_flight: Arc<AtomicBool>,
    /// The in-flight warm-up task, so the DISCONNECT edge can abort it. Without
    /// this, a warm-up from a session that just ended would keep polling the
    /// (now killed) runtimes for up to its ready-timeout — and hold the permit
    /// across a quick game restart, silently starving the next session's warm-up.
    warmup_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl Default for StackLifecycle {
    fn default() -> Self {
        Self {
            phase: Mutex::new(Phase::Disconnected),
            warmup_in_flight: Arc::new(AtomicBool::new(false)),
            warmup_task: Mutex::new(None),
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

    /// Claims the warm-up slot. Returns `None` when a warm-up is already running
    /// (the caller skips instead of doubling the work). The returned permit
    /// releases the slot on Drop — including on panic and on task abort — so the
    /// slot can never be wedged shut by a warm-up that didn't finish cleanly.
    pub(crate) fn try_begin_warmup(&self) -> Option<WarmupPermit> {
        self.warmup_in_flight
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
            .then(|| WarmupPermit {
                flag: Arc::clone(&self.warmup_in_flight),
            })
    }

    /// Records the spawned warm-up task so [`Self::abort_warmup`] can cancel it.
    /// Only ever called with a task that holds the [`WarmupPermit`] (the permit
    /// is claimed BEFORE spawning), so the stored handle is always THE running
    /// warm-up, never a duplicate that skipped.
    pub(crate) fn track_warmup_task(&self, handle: tokio::task::JoinHandle<()>) {
        *self
            .warmup_task
            .lock()
            .expect("stack-lifecycle warmup-task mutex") = Some(handle);
    }

    /// Aborts the in-flight warm-up task (if any). Called on the disconnect edge:
    /// the runtimes it would warm are being torn down, and its permit must not
    /// leak into the next connect. Dropping the aborted future releases the
    /// permit via [`WarmupPermit::drop`]. Returns the handle so tests can await
    /// the abort deterministically; production callers ignore it.
    pub(crate) fn abort_warmup(&self) -> Option<tokio::task::JoinHandle<()>> {
        let handle = self
            .warmup_task
            .lock()
            .expect("stack-lifecycle warmup-task mutex")
            .take();
        if let Some(handle) = &handle {
            handle.abort();
        }
        handle
    }
}

/// RAII claim on the warm-up slot: exactly one exists at a time, and dropping it
/// (normal completion, panic, or task abort) reopens the slot. Handed out by
/// [`StackLifecycle::try_begin_warmup`] and held across the whole warm-up run.
#[derive(Debug)]
pub(crate) struct WarmupPermit {
    flag: Arc<AtomicBool>,
}

impl Drop for WarmupPermit {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::SeqCst);
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
                // Warm the WHOLE stack off the readiness path: retriever
                // (embedder + reranker), LLM KV-cache prefix, Whisper, and the
                // TTS first-inference — so the FIRST in-game line costs the same
                // as every later one. Fire-and-forget; a real turn arriving
                // mid-warm-up just reuses whatever finished (see `warmup.rs`).
                warmup::spawn_stack_warmup(&state);
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
                    // Cancel any still-running warm-up: the runtimes it polls are
                    // about to be killed, and its permit must not survive into the
                    // next connect (a quick game restart would otherwise skip its
                    // warm-up because a dead session's run still held the slot).
                    state.lifecycle.abort_warmup();
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The warm-up slot is exclusive while a permit is alive, and dropping the
    /// permit (normal completion or panic unwinding) reopens it.
    #[test]
    fn warmup_permit_is_exclusive_and_releases_on_drop() {
        let lifecycle = StackLifecycle::default();
        let permit = lifecycle.try_begin_warmup().expect("first claim wins");
        assert!(
            lifecycle.try_begin_warmup().is_none(),
            "a second claim while one is in flight must be refused"
        );
        drop(permit);
        assert!(
            lifecycle.try_begin_warmup().is_some(),
            "dropping the permit must reopen the slot"
        );
    }

    /// Aborting a tracked warm-up task releases its permit: a warm-up killed on
    /// the disconnect edge can never starve the NEXT session's warm-up (the bug
    /// class this guards against: quick game restart while a stale warm-up still
    /// polls dead endpoints for minutes, holding the slot the whole time).
    #[tokio::test]
    async fn aborting_the_tracked_warmup_task_releases_the_permit() {
        let lifecycle = StackLifecycle::default();
        let permit = lifecycle.try_begin_warmup().expect("claim");
        let handle = tokio::spawn(async move {
            let _held = permit; // released only when this future is dropped
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
        });
        lifecycle.track_warmup_task(handle);
        assert!(lifecycle.try_begin_warmup().is_none(), "slot busy while task runs");

        let handle = lifecycle.abort_warmup().expect("a task was tracked");
        let joined = handle.await;
        assert!(joined.is_err(), "the warm-up task must have been cancelled");
        assert!(
            lifecycle.try_begin_warmup().is_some(),
            "abort must drop the permit and reopen the slot"
        );
        // The slot is now empty; a second abort is a clean no-op.
        assert!(lifecycle.abort_warmup().is_none());
    }
}
