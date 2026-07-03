//! FNV → chasm bridge, Rust port.
//!
//! Section 1: skeleton + byte-parity file protocol (watcher → parse → respond →
//! archive), proven with a stub echo.
//! Section 2: the real regular-NPC turn — resolve the NPC, build presence +
//! gamestate, generate a reply via chasm, synthesize cloned-voice TTS into audio
//! chunks, and write the response the plugin plays + captions.

pub mod actions;
pub mod admin;
pub mod chasm;
pub mod config;
pub mod npc;
pub mod protocol;
pub mod replay;
pub mod saves;
pub mod sink;
pub mod stt;
pub mod trace;

use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use serde_json::{json, Value};
use tokio::sync::{mpsc, mpsc::UnboundedSender, Notify};
use tracing::{error, info, warn};

pub use config::{default_config, load_config, BridgeConfig};

use chasm::{ChasmClient, HttpChasmClient};
use npc::GeneratedLine;
use protocol::{
    build_native_archived_request, build_silence_wav, now_iso8601_millis,
    parse_native_text_request, safe_file_id, NativeRequest,
};
use sink::{FileSink, OutgoingResponse, TurnSink, WrittenChunk};

const WATCH_DEBOUNCE_MS: u64 = 15;
const GAME_UNITS_PER_METER: f64 = 70.0;

/// Run the bridge loop until Ctrl+C, talking to chasm over HTTP. `force` steals a
/// stale lock. The standalone-binary entry point.
pub async fn run(config: BridgeConfig, force: bool) -> anyhow::Result<()> {
    let client = HttpChasmClient::new(&config)?;
    run_with_client(config, force, Arc::new(client)).await
}

/// Run the bridge loop with an injected [`ChasmClient`]. The standalone bin passes
/// an [`HttpChasmClient`] (HTTP → :7341); the in-process fold (chasm-web) passes a
/// client that calls chasm's handlers directly, so there's no localhost hop. Stops
/// on Ctrl+C or when the spawning task is aborted.
pub async fn run_with_client(
    config: BridgeConfig,
    force: bool,
    client: Arc<dyn ChasmClient>,
) -> anyhow::Result<()> {
    let mut locks: Vec<PathBuf> = Vec::new();
    for root in &config.native_bridge_roots {
        ensure_native_root(root)?;
        locks.push(acquire_helper_lock(root, force)?);
    }
    let _guard = LockGuard(locks);

    let (tx, mut rx) = mpsc::unbounded_channel::<()>();

    let mut watchers: Vec<RecommendedWatcher> = Vec::new();
    let mut watch_count = 0usize;
    for root in &config.native_bridge_roots {
        for (dir, label) in [
            (native_inbox(root), "native inbox"),
            (native_event_dir(root), "save-state events"),
        ] {
            match make_watcher(&dir, tx.clone()) {
                Ok(w) => {
                    watchers.push(w);
                    watch_count += 1;
                }
                Err(e) => warn!(
                    "could not watch {} ({}): {e}; safety poll still covers it",
                    label,
                    dir.display()
                ),
            }
        }
    }

    let poll_ms = config.poll_ms.max(75);
    {
        let tx = tx.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_millis(poll_ms));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                if tx.send(()).is_err() {
                    break;
                }
            }
        });
    }

    for root in &config.native_bridge_roots {
        info!("watching bridge root {}", root.display());
    }
    info!(
        "chasm {} | live-chat {} | armed {watch_count} watcher(s), safety poll {poll_ms}ms. Waiting for in-game requests (Ctrl+C to stop).",
        config.api_base, config.live_chat_id
    );

    let _ = tx.send(());

    loop {
        tokio::select! {
            recv = rx.recv() => {
                if recv.is_none() { break; }
                tokio::time::sleep(Duration::from_millis(WATCH_DEBOUNCE_MS)).await;
                while rx.try_recv().is_ok() {}
                for root in &config.native_bridge_roots {
                    if let Err(e) = poll_native_root(&config, client.as_ref(), root).await {
                        error!("poll error for {}: {e}", root.display());
                    }
                }
            }
            _ = tokio::signal::ctrl_c() => {
                info!("shutting down");
                break;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Poll + dispatch
// ---------------------------------------------------------------------------

async fn poll_native_root(
    config: &BridgeConfig,
    client: &dyn ChasmClient,
    root: &Path,
) -> anyhow::Result<()> {
    // The plugin's save/load checkpoint events (control/events → control/acks).
    saves::process_save_state_events(config, client, root).await;

    let inbox = native_inbox(root);
    let entries = match std::fs::read_dir(&inbox) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.to_ascii_lowercase().ends_with(".txt") || name.starts_with("__") {
            continue;
        }
        if let Err(e) = process_native_request(config, client, root, &path).await {
            error!("request {} failed: {e}", name);
        }
    }
    Ok(())
}

async fn process_native_request(
    config: &BridgeConfig,
    client: &dyn ChasmClient,
    root: &Path,
    path: &Path,
) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(path)?;
    let request = parse_native_text_request(path, &text);

    // The default file transport: run_turn emits its reply/audio/action through this
    // sink, which reproduces the historical NVBridge file writes byte-for-byte.
    let sink = FileSink::new(root, path);

    // Supersede: cancel the in-flight turn the instant the game writes a newer
    // request (the player talked again), so the new line starts without waiting for
    // the old generate/TTS to finish.
    let cancel = Cancel::new();
    let monitor = spawn_supersede_monitor(path.to_path_buf(), request.request_id.clone(), cancel.clone());

    let outcome = tokio::select! {
        result = run_turn(config, client, &sink, root, path, &request) => result,
        _ = cancel.cancelled() => {
            warn!("request {} superseded by a newer request; cancelled mid-flight", request.request_id);
            monitor.abort();
            archive_native_request(root, path, &request)?;
            return Ok(());
        }
    };
    monitor.abort();

    if let Err(e) = &outcome {
        // Mirror the Node catch: surface a bridge error to the plugin (HUD).
        let resp = OutgoingResponse {
            status: "0".into(),
            error: e.to_string(),
            ..OutgoingResponse::base(&request)
        };
        let _ = sink.reply(&request, &resp);
    }
    archive_native_request(root, path, &request)?;
    outcome
}

/// A minimal cooperative cancellation flag (avoids a tokio-util dependency).
#[derive(Clone)]
struct Cancel {
    flag: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl Cancel {
    fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }
    fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }
    fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
    async fn cancelled(&self) {
        loop {
            if self.is_cancelled() {
                return;
            }
            let notified = self.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

/// Watch the request file; cancel when its request_id changes (a newer request).
fn spawn_supersede_monitor(
    path: PathBuf,
    request_id: String,
    cancel: Cancel,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if request_id.is_empty() {
            return;
        }
        loop {
            tokio::time::sleep(Duration::from_millis(120)).await;
            if cancel.is_cancelled() {
                return;
            }
            if let Ok(text) = std::fs::read_to_string(&path) {
                let current = parse_native_text_request(&path, &text).request_id;
                if !current.is_empty() && current != request_id {
                    cancel.cancel();
                    return;
                }
            }
        }
    })
}

/// Drive one NPC turn for a non-file transport (e.g. the HTTP `/api/game/v1/turn`
/// endpoint), emitting every output through `sink`. This is the public entry point
/// into [`run_turn`]'s orchestration; the file bridge keeps its own
/// [`process_native_request`] path (with archiving + the supersede monitor).
///
/// There is no inbound request file here, so a scratch dir stands in for `root`
/// and a synthetic, non-existent path for the request file. Those are only touched
/// by the on-disk push-to-talk STT sidecar path — callers that support voice should
/// transcribe up front and pass the transcript as `request.player_text`, so the
/// sidecar logic is never reached. A non-empty `player_text` guarantees that.
pub async fn run_turn_with_sink(
    config: &BridgeConfig,
    client: &dyn ChasmClient,
    sink: &dyn TurnSink,
    request: &NativeRequest,
) -> anyhow::Result<()> {
    // A scratch root keeps any incidental fs touch (none on the text path) off the
    // real bridge folders. The synthetic path never exists, so currency checks treat
    // the reply as current (there is no newer request file to supersede it).
    let root = stream_storage_dir().join("http-turn");
    let _ = std::fs::create_dir_all(&root);
    let path = native_inbox(&root).join(format!("{}.txt", safe_file_id(&request.request_id)));
    run_turn(config, client, sink, &root, &path, request).await
}

/// The regular-NPC turn: resolve, presence, generate, TTS, respond. Emits its
/// own success/early-return outputs through `sink`; the caller emits a generic
/// error + always archives. `root`/`path` are retained for the file-only paths
/// (save-sync relay, sidecar audio, archiving) that stay on files by design.
async fn run_turn(
    config: &BridgeConfig,
    client: &dyn ChasmClient,
    sink: &dyn TurnSink,
    root: &Path,
    path: &Path,
    request: &NativeRequest,
) -> anyhow::Result<()> {
    // Save-sync (save/load checkpoint) requests are handled before anything else.
    if saves::is_native_save_sync_request(request) {
        return process_save_sync_request(config, client, sink, request).await;
    }

    // Companions created mid-session live in the plugin's registry, not the
    // startup character map — merge them in per turn so their dialogue resolves
    // without a chasm restart (see npc::config_with_companions).
    let companion_config = npc::config_with_companions(config, root);
    let config = companion_config.as_ref().unwrap_or(config);

    let admin = admin::is_admin_request(request);
    let distance_meters = npc::native_distance_meters(request);

    // Per-turn stage trace (Settings → Tracing waterfall). Best-effort and
    // clock-local to this turn; see `trace.rs`. Makes first-vs-later-turn cost
    // (cold prefill, TTS first-inference, …) directly visible per request.
    let trace = trace::TurnTrace::new(root, &request.request_id);
    trace.stage_with(
        "helper_turn_start",
        serde_json::json!({
            "npc_key": request.npc_key,
            "want_tts": request.want_tts,
            "admin": admin,
        }),
    );

    // Distance gate — skipped for admin (Todd is heard from anywhere).
    if !admin && distance_meters.is_finite() && distance_meters > config.native_max_distance_meters {
        let resp = OutgoingResponse {
            status: "0".into(),
            error: format!(
                "Too far to speak. Distance {distance_meters:.1}m > {:.0}m.",
                config.native_max_distance_meters
            ),
            ..OutgoingResponse::base(request)
        };
        sink.reply(request, &resp)?;
        return Ok(());
    }

    // Resolve the player's words: typed text, or transcribe the push-to-talk WAV.
    let mut message = request.player_text.trim().to_string();
    if message.is_empty()
        && (stt::is_native_voice_request(request) || stt::has_audio_sidecar(root, path, request))
    {
        message = stt::recognize_native_speech(config, client, root, path, request).await?;
        trace.stage_with(
            "speech_recognition_done",
            serde_json::json!({ "text_length": message.trim().len() }),
        );
    }
    let message = message.trim().to_string();
    if message.is_empty() {
        write_placeholder(sink, request)?;
        trace.stage("final_response_written");
        return Ok(());
    }

    let location = npc::location_string(request);

    // Admin / Todd: god voice + force-any-NPC actions + spawns.
    if admin {
        return run_admin_turn(config, client, sink, request, &message, &location).await;
    }

    let distance_game_units = distance_meters * GAME_UNITS_PER_METER;
    let participants = npc::nearby_npc_candidates(config, request, distance_meters, distance_game_units);
    if participants.is_empty() {
        anyhow::bail!(
            "No mapped NPC within {} meters.",
            config.native_max_distance_meters
        );
    }
    let attention = npc::attention_target(request, &participants);
    let gamestate = npc::build_gamestate(config, request, &location);

    // Ensure the live chat exists, then set presence.
    if !client.live_chat_exists(&config.live_chat_id).await? {
        if config.group_id.is_empty() || config.group_id.contains("replace-with") {
            anyhow::bail!(
                "Live Chat '{}' does not exist and groupId is not configured.",
                config.live_chat_id
            );
        }
        client
            .create_live_chat(&ensure_body(config, &participants, &attention, &location))
            .await?;
    }
    client
        .presence(&config.live_chat_id, &presence_body(config, &participants, &attention))
        .await?;
    trace.stage("live_chat_ensure_done");

    // Stream the turn, synthesizing each speech.delta the moment it arrives so the
    // opener plays while the rest of the line is still generating (matches Node's
    // early-segment TTS). chasm pre-chunks the deltas (opener, then remainder); the
    // final `live.completed` turn supplies the caption text.
    let gen_body = generate_body(
        config, request, &message, &gamestate, &participants, distance_meters, distance_game_units,
        &location,
    );
    let mut written: Vec<WrittenChunk> = Vec::new();
    let mut next_index: u32 = 0;
    let mut turn: Option<Value> = None;
    let mut stream_speaker: Option<npc::ResolvedSpeaker> = None;
    trace.stage("live_chat_generate_start");
    let mut first_delta_seen = false;
    let mut first_audio_seen = false;
    {
        let mut events = std::pin::pin!(client.generate_stream_events(&config.live_chat_id, &gen_body));
        while let Some(event) = events.next().await {
            let event = event?;
            match event.get("type").and_then(|t| t.as_str()) {
                Some("speaker.start") => {
                    if let Some(speaker) = event.get("speaker") {
                        stream_speaker =
                            Some(npc::resolve_stream_speaker(config, &participants, request, speaker));
                    }
                }
                Some("speech.delta") => {
                    let segment = event
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !first_delta_seen {
                        first_delta_seen = true;
                        // First model output surfaced: everything before this is
                        // prompt assembly + retrieval + LLM prefill (+ decode of
                        // the opener chunk) — the cold-start hot spot.
                        trace.stage_with(
                            "live_chat_first_delta",
                            serde_json::json!({ "chars": segment.len() }),
                        );
                    }
                    if !segment.is_empty() && request.want_tts {
                        let speaker = stream_speaker
                            .clone()
                            .unwrap_or_else(|| npc::default_stream_speaker(config, request));
                        let line = GeneratedLine {
                            participant_id: speaker.participant_id,
                            native_npc_key: speaker.native_npc_key,
                            native_npc_name: speaker.native_npc_name,
                            character_name: speaker.character_name.clone(),
                            character_id: speaker.character_id,
                            text: segment.clone(),
                            turn: Value::Null,
                        };
                        let body = synth_body(config, &segment, &line.character_name, false);
                        let start = next_index;
                        let mut local: u32 = 0;
                        trace.stage_with(
                            "tts_stream_start",
                            serde_json::json!({ "chunk_chars": segment.len(), "start_index": start }),
                        );
                        client
                            .synthesize_stream(&body, &mut |chunk| {
                                if !first_audio_seen {
                                    first_audio_seen = true;
                                    // Feeds the existing "TTS first audio" summary
                                    // metric (tts_stream_start → this).
                                    trace.stage("tts_first_audio_chunk_received");
                                }
                                let index = start + chunk.index.map(|i| i.max(0) as u32).unwrap_or(local);
                                local += 1;
                                written.push(sink.audio_chunk(request, &line, &chunk, index, &[])?);
                                Ok(())
                            })
                            .await?;
                        next_index = written.iter().map(|w| w.index + 1).max().unwrap_or(next_index);
                        trace.stage_with(
                            "tts_synthesize_done",
                            serde_json::json!({ "chunks_written": written.len() }),
                        );
                    }
                }
                Some("live.error") => {
                    let error_message = event
                        .pointer("/error/message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("Live Chat streaming generation failed.");
                    anyhow::bail!("{error_message}");
                }
                Some("live.completed") => turn = event.get("turn").cloned(),
                _ => {}
            }
        }
    }
    trace.stage("live_chat_generate_done");

    let turn = turn
        .ok_or_else(|| anyhow::anyhow!("Live Chat streaming generation ended without a final turn."))?;
    let lines = npc::extract_lines(config, &participants, request, &turn);
    if lines.is_empty() {
        // The model chose silence — status 1, no audio.
        let resp = OutgoingResponse {
            status: "1".into(),
            player_text: message,
            extra_lines: vec!["-1".into()],
            ..OutgoingResponse::base(request)
        };
        sink.reply(request, &resp)?;
        sink.end_of_turn()?;
        trace.stage("final_response_written");
        info!("{}: live chat selected no NPC response", request.request_id);
        return Ok(());
    }

    // Native actions: classify each line, write a durable command file per triggered
    // action to every control/actions dir, and report a DISARMED game_master on the
    // response so the action fires exactly once (from the file, not the response).
    let line_gms: Vec<actions::GameMaster> = lines
        .iter()
        .map(|line| actions::get_native_game_master_action(config, &line.turn))
        .collect();
    let mut response_gm = line_gms
        .iter()
        .find(|g| g.should_trigger)
        .cloned()
        .unwrap_or_else(|| actions::get_native_game_master_action(config, &turn));
    let mut queued_native_action = false;
    for (index, (line, line_gm)) in lines.iter().zip(line_gms.iter()).enumerate() {
        if !line_gm.should_trigger {
            continue;
        }
        let line_request_id = if lines.len() > 1 {
            let suffix = first_non_empty([
                line.native_npc_key.clone(),
                line.character_id.clone(),
                (index + 1).to_string(),
            ]);
            format!("{}-{}", request.request_id, suffix)
        } else {
            request.request_id.clone()
        };
        let mut line_request = request.clone();
        line_request.request_id = line_request_id;
        let actor = actions::ActionActor {
            native_npc_key: line.native_npc_key.clone(),
            native_npc_name: line.native_npc_name.clone(),
            character_name: line.character_name.clone(),
            character_id: line.character_id.clone(),
        };
        if sink.action(
            config,
            &line_request,
            &actor,
            line_gm,
            "fallout-new-vegas-native-action",
        ) {
            queued_native_action = true;
            if !response_gm.should_trigger {
                response_gm = line_gm.clone();
            }
        }
    }

    let audio = written.first();
    let speaker = &lines[0];
    let resp = OutgoingResponse {
        status: "1".into(),
        request_id: request.request_id.clone(),
        npc_key: first_non_empty([speaker.native_npc_key.clone(), request.npc_key.clone()]),
        npc_name: first_non_empty([
            speaker.native_npc_name.clone(),
            speaker.character_name.clone(),
            request.npc_name.clone(),
        ]),
        audio_filename: audio.map(|w| w.filename.clone()).unwrap_or_default(),
        text: speaker.text.clone(),
        error: String::new(),
        player_text: message,
        extra_lines: vec![if audio.is_some() { "0".into() } else { "-1".into() }],
        gm_action: response_gm.action.clone(),
        gm_confidence: response_gm.confidence.clone(),
        gm_should_trigger: if queued_native_action { false } else { response_gm.should_trigger },
    };
    sink.reply(request, &resp)?;
    sink.end_of_turn()?;
    trace.stage_with(
        "final_response_written",
        serde_json::json!({ "audio_chunks": written.len(), "speaker_name": speaker.character_name }),
    );
    info!(
        "{}: {} -> {:?}",
        request.request_id,
        speaker.character_name,
        truncate(&speaker.text, 80)
    );
    Ok(())
}

/// Admin / Todd: god-voice reply, write the commanded action to a durable command
/// file (with the resolved actor), synthesize non-positional audio, write the response.
async fn run_admin_turn(
    config: &BridgeConfig,
    client: &dyn ChasmClient,
    sink: &dyn TurnSink,
    request: &NativeRequest,
    message: &str,
    location: &str,
) -> anyhow::Result<()> {
    let admin_turn = admin::generate_admin_turn(config, client, request, message, location).await?;
    let gm = admin_turn.game_master;

    // The acting NPC: Todd for world.* actions, else the named/crosshair NPC.
    let actor = admin::resolve_native_actor_for_admin(config, request, &gm).unwrap_or_else(|| {
        actions::ActionActor {
            native_npc_key: first_non_empty([request.npc_key.clone(), "todd".into()]),
            native_npc_name: first_non_empty([request.npc_name.clone(), config.admin_character_name.clone()]),
            character_name: first_non_empty([request.npc_name.clone(), config.admin_character_name.clone()]),
            character_id: config.admin_character_id.clone(),
        }
    });
    let queued = sink.action(
        config,
        request,
        &actor,
        &gm,
        "fallout-new-vegas-native-admin-action",
    );

    // Todd's voice is non-positional (2D, straight into the ear) → admin volume.
    let admin_meta = ["admin_voice=1".to_string(), "non_positional_audio=1".to_string()];
    let line = GeneratedLine {
        participant_id: "system:todd".into(),
        native_npc_key: "todd".into(),
        native_npc_name: config.admin_character_name.clone(),
        character_name: config.admin_character_name.clone(),
        character_id: config.admin_character_id.clone(),
        text: admin_turn.text.clone(),
        turn: Value::Null,
    };
    let mut written: Vec<WrittenChunk> = Vec::new();
    if request.want_tts {
        let body = synth_body(config, &admin_turn.text, &config.admin_character_name, true);
        let mut local = 0u32;
        client
            .synthesize_stream(&body, &mut |chunk| {
                let index = chunk.index.map(|i| i.max(0) as u32).unwrap_or(local);
                local += 1;
                written.push(sink.audio_chunk(request, &line, &chunk, index, &admin_meta)?);
                Ok(())
            })
            .await?;
    }

    let audio = written.first();
    let mut extra_lines = vec![
        if audio.is_some() { "0".to_string() } else { "-1".to_string() },
        "admin_voice=1".to_string(),
        "non_positional_audio=1".to_string(),
    ];
    if gm.should_trigger {
        extra_lines.push(format!("action_npc_key={}", actor.native_npc_key));
        extra_lines.push(format!(
            "action_npc_name={}",
            first_non_empty([actor.native_npc_name.clone(), actor.character_name.clone()])
        ));
    }
    let resp = OutgoingResponse {
        status: "1".into(),
        request_id: request.request_id.clone(),
        npc_key: "todd".into(),
        npc_name: config.admin_character_name.clone(),
        audio_filename: audio.map(|w| w.filename.clone()).unwrap_or_default(),
        text: admin_turn.text.clone(),
        error: String::new(),
        player_text: message.to_string(),
        extra_lines,
        gm_action: gm.action.clone(),
        gm_confidence: gm.confidence.clone(),
        gm_should_trigger: if queued { false } else { gm.should_trigger },
    };
    sink.reply(request, &resp)?;
    sink.end_of_turn()?;
    info!(
        "admin {}: {:?} (action {} queued={})",
        request.request_id,
        truncate(&admin_turn.text, 80),
        gm.action,
        queued
    );
    Ok(())
}

/// Inbox save-sync request: relay the save/load event to chasm and write the
/// checkpoint status back as the response.
async fn process_save_sync_request(
    config: &BridgeConfig,
    client: &dyn ChasmClient,
    sink: &dyn TurnSink,
    request: &NativeRequest,
) -> anyhow::Result<()> {
    let ctx = saves::save_context_from_request(request)?;
    let outcome = saves::call_save_sync_event(config, client, &ctx, &[]).await?;

    let missing = outcome.status == "snapshot_missing";
    let name = if !outcome.checkpoint_save_name.is_empty() {
        outcome.checkpoint_save_name.clone()
    } else if !ctx.save_name.is_empty() {
        ctx.save_name.clone()
    } else {
        ctx.save_id.clone()
    };
    let text = if ctx.event == "save" {
        format!("Save sync checkpoint {} for {name}.", outcome.status)
    } else {
        format!("Save sync {} for {name}.", outcome.status)
    };
    let resp = OutgoingResponse {
        status: if missing { "0".into() } else { "1".into() },
        request_id: request.request_id.clone(),
        npc_key: request.npc_key.clone(),
        npc_name: request.npc_name.clone(),
        audio_filename: String::new(),
        text,
        error: if missing {
            "No matching ST checkpoint exists for this game save.".into()
        } else {
            String::new()
        },
        player_text: request.player_text.clone(),
        extra_lines: vec![
            format!("save_sync_event={}", ctx.event),
            format!("save_sync_status={}", outcome.status),
            format!("checkpoint_id={}", outcome.checkpoint_id),
        ],
        gm_action: String::new(),
        gm_confidence: String::new(),
        gm_should_trigger: false,
    };
    sink.reply(request, &resp)?;
    sink.end_of_turn()?;
    info!("save-sync {} {}: {}", ctx.event, ctx.save_id, outcome.status);
    Ok(())
}

// ---------------------------------------------------------------------------
// chasm request bodies
// ---------------------------------------------------------------------------

fn player_participant(config: &BridgeConfig) -> Value {
    json!({
        "participantId": config.participant_id,
        "type": "user",
        "present": true,
        "audible": true,
        "name": "Player",
    })
}

fn participants_value(config: &BridgeConfig, participants: &[npc::NpcParticipant]) -> Value {
    let mut list = vec![player_participant(config)];
    list.extend(participants.iter().map(|p| p.to_presence_value()));
    Value::Array(list)
}

fn attention_strength(attention: &Option<String>) -> f64 {
    if attention.is_some() {
        0.35
    } else {
        0.0
    }
}

fn ensure_body(
    config: &BridgeConfig,
    participants: &[npc::NpcParticipant],
    attention: &Option<String>,
    location: &str,
) -> Value {
    json!({
        "id": config.live_chat_id,
        "groupId": config.group_id,
        "title": "Fallout New Vegas - Goodsprings",
        "location": if location.is_empty() { "Goodsprings" } else { location },
        "participants": participants_value(config, participants),
        "attentionTarget": attention,
        "attentionStrength": attention_strength(attention),
    })
}

fn presence_body(
    config: &BridgeConfig,
    participants: &[npc::NpcParticipant],
    attention: &Option<String>,
) -> Value {
    json!({
        "replace": true,
        "participants": participants_value(config, participants),
        "attentionTarget": attention,
        "attentionStrength": attention_strength(attention),
    })
}

#[allow(clippy::too_many_arguments)]
fn generate_body(
    config: &BridgeConfig,
    request: &NativeRequest,
    message: &str,
    gamestate: &Value,
    participants: &[npc::NpcParticipant],
    distance_meters: f64,
    distance_game_units: f64,
    location: &str,
) -> Value {
    let scopes = actions::action_book_scopes(config, &request.npc_key, location);
    json!({
        "message": message,
        // The game request's trace id, so the generate pipeline can correlate the
        // LLM usage/timings capture (tokens/sec, prompt_ms) to this request's
        // trace even on the in-process path where no HTTP header exists.
        "traceId": request.request_id,
        "participantId": config.participant_id,
        "responseFormat": if config.enable_action_books { "structured" } else { "text" },
        "enableActionBooks": config.enable_action_books,
        "enableQuestBooks": true,
        "includeActionBookBindings": config.enable_action_books,
        "includeQuestBookBindings": config.enable_action_books,
        "actionBookIds": config.action_book_ids,
        "actionBookScopes": scopes,
        "questBookScopes": scopes,
        "targetGame": config.action_book_target_game,
        "actionBookLimit": 12,
        "questBookLimit": 5,
        "extraContext": "",
        "gamestate": gamestate,
        "metadata": {
            "source": "fallout-new-vegas-native",
            "targetName": first_non_empty([request.npc_name.clone(), config.character_name.clone()]),
            "distanceGameUnits": distance_game_units,
            "distanceMeters": distance_meters,
            "nativeNpcKey": request.npc_key,
            "gamestate": gamestate,
            "nearbyNpcs": participants.iter().map(|p| p.metadata_value()).collect::<Vec<_>>(),
            // The mod's per-turn gamestate macro table (`metadata.macros`,
            // flat key→value). Forwarded verbatim so the generate path can
            // record it on the persisted turn (`extra.chasm.macros`) for the
            // Gamestate page — this body's `metadata` is otherwise built from
            // scratch, so without this line the mod's table would be dropped.
            "macros": request.metadata.get("macros").cloned().unwrap_or_else(|| json!({})),
        },
    })
}

fn synth_body(config: &BridgeConfig, text: &str, character_name: &str, non_positional: bool) -> Value {
    let mut obj = config
        .tts
        .as_object()
        .cloned()
        .unwrap_or_default();
    obj.insert("text".into(), json!(text));
    obj.insert("characterName".into(), json!(character_name));
    obj.insert("format".into(), json!("wav"));
    obj.insert("encoding".into(), json!("base64"));
    obj.insert("nonPositional".into(), json!(non_positional));
    Value::Object(obj)
}

// ---------------------------------------------------------------------------
// Response writing
// ---------------------------------------------------------------------------

/// Caption for a request that carried no usable text and no voice audio. Uses a
/// silent WAV so the plugin renders the caption (it only shows captions with audio).
fn write_placeholder(sink: &dyn TurnSink, request: &NativeRequest) -> anyhow::Result<()> {
    let audio_name = format!("nvbridge_{}.wav", safe_file_id(&request.request_id));
    let audio_dir = stream_storage_dir().join("audio");
    std::fs::create_dir_all(&audio_dir)?;
    std::fs::write(audio_dir.join(&audio_name), build_silence_wav(44_100, 250))?;
    let resp = OutgoingResponse {
        status: "1".into(),
        audio_filename: audio_name,
        text: "(didn't catch that)".into(),
        extra_lines: vec!["0".into()],
        ..OutgoingResponse::base(request)
    };
    sink.reply(request, &resp)?;
    sink.end_of_turn()?;
    Ok(())
}

fn first_non_empty<const N: usize>(values: [String; N]) -> String {
    values.into_iter().find(|s| !s.is_empty()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Request archiving
// ---------------------------------------------------------------------------

fn archive_native_request(root: &Path, path: &Path, request: &NativeRequest) -> anyhow::Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{e}"))
        .unwrap_or_else(|| ".txt".to_string());
    let processed = native_processed(root);
    std::fs::create_dir_all(&processed)?;
    let dest = processed.join(format!("{}{ext}", safe_file_id(&request.request_id)));
    std::fs::write(&dest, build_native_archived_request(request))?;

    let current = if path.exists() {
        std::fs::read_to_string(path)
            .ok()
            .map(|t| parse_native_text_request(path, &t))
    } else {
        None
    };
    match current {
        None => {
            let _ = std::fs::remove_file(path);
        }
        Some(c) if c.request_id == request.request_id => {
            let _ = std::fs::remove_file(path);
        }
        Some(_) => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Watcher
// ---------------------------------------------------------------------------

fn make_watcher(dir: &Path, tx: UnboundedSender<()>) -> notify::Result<RecommendedWatcher> {
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if res.is_ok() {
            let _ = tx.send(());
        }
    })?;
    watcher.watch(dir, RecursiveMode::NonRecursive)?;
    Ok(watcher)
}

// ---------------------------------------------------------------------------
// Directory layout
// ---------------------------------------------------------------------------

pub(crate) fn native_inbox(root: &Path) -> PathBuf {
    root.join("inbox")
}
pub(crate) fn native_outbox(root: &Path) -> PathBuf {
    root.join("outbox")
}
pub(crate) fn native_processed(root: &Path) -> PathBuf {
    root.join("processed")
}
fn native_event_dir(root: &Path) -> PathBuf {
    root.join("control").join("events")
}

pub(crate) fn stream_storage_dir() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("TEMP"))
        .or_else(|| std::env::var_os("TMP"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("NVBridgeStream")
}

fn ensure_native_root(root: &Path) -> anyhow::Result<()> {
    for dir in [
        native_inbox(root),
        native_outbox(root).join("chunks"),
        native_processed(root),
        root.join("control").join("actions"),
        native_event_dir(root),
        root.join("control").join("acks"),
    ] {
        std::fs::create_dir_all(&dir)
            .map_err(|e| anyhow::anyhow!("creating {}: {e}", dir.display()))?;
    }
    let stream = stream_storage_dir();
    let _ = std::fs::create_dir_all(stream.join("chunks"));
    let _ = std::fs::create_dir_all(stream.join("audio"));
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper lock
// ---------------------------------------------------------------------------

fn helper_lock_path(root: &Path) -> PathBuf {
    root.join("nvbridge-helper.lock")
}

fn acquire_helper_lock(root: &Path, force: bool) -> anyhow::Result<PathBuf> {
    let lock = helper_lock_path(root);
    if force && lock.exists() {
        warn!("--force: removing existing lock {}", lock.display());
        let _ = std::fs::remove_file(&lock);
    }
    let payload = json!({
        "pid": std::process::id(),
        "startedAt": now_iso8601_millis(),
        "root": root.to_string_lossy(),
    });
    match OpenOptions::new().write(true).create_new(true).open(&lock) {
        Ok(mut file) => {
            file.write_all(serde_json::to_string_pretty(&payload)?.as_bytes())?;
            Ok(lock)
        }
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {
            let existing = std::fs::read_to_string(&lock).unwrap_or_default();
            let existing = existing.replace(['\r', '\n'], " ");
            anyhow::bail!(
                "another NVBridge bridge holds the lock at {} ({}). Stop it first (e.g. the Node helper), or pass --force if the lock is stale.",
                lock.display(),
                existing.trim()
            )
        }
        Err(e) => Err(anyhow::anyhow!("opening lock {}: {e}", lock.display())),
    }
}

struct LockGuard(Vec<PathBuf>);

impl Drop for LockGuard {
    fn drop(&mut self) {
        let our_pid = std::process::id();
        for lock in &self.0 {
            let owned = std::fs::read_to_string(lock)
                .ok()
                .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                .and_then(|v| v.get("pid").and_then(|p| p.as_u64()))
                .map(|pid| pid == our_pid as u64)
                .unwrap_or(false);
            if owned {
                let _ = std::fs::remove_file(lock);
            }
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{head}…")
    }
}

/// Run a single turn against the live backend and print the outcome — an offline
/// smoke test of the whole chasm pipeline (no game needed). Used by `--turn-selftest`.
pub async fn turn_selftest(config: &BridgeConfig, npc_key: &str, message: &str) -> anyhow::Result<()> {
    let client = HttpChasmClient::new(config)?;
    let mut request = NativeRequest {
        request_id: format!("selftest_{}", now_iso8601_millis().replace([':', '.', '-'], "")),
        npc_key: npc_key.to_string(),
        npc_name: npc_key.to_string(),
        want_tts: true,
        player_text: message.to_string(),
        ..Default::default()
    };
    request.location.major = "Goodsprings".into();

    let distance_meters = 2.0;
    let dgu = distance_meters * GAME_UNITS_PER_METER;
    let participants = npc::nearby_npc_candidates(config, &request, distance_meters, dgu);
    println!("resolved {} participant(s):", participants.len());
    for p in &participants {
        println!("  participantId={} characterId={} name={}", p.participant_id, p.character_id, p.character_name);
    }
    if participants.is_empty() {
        anyhow::bail!("no participants resolved for npc_key '{npc_key}'");
    }
    let attention = npc::attention_target(&request, &participants);
    let gamestate = npc::build_gamestate(config, &request, "Goodsprings");

    if !client.live_chat_exists(&config.live_chat_id).await? {
        println!("creating live chat {}", config.live_chat_id);
        client.create_live_chat(&ensure_body(config, &participants, &attention, "Goodsprings")).await?;
    }
    client.presence(&config.live_chat_id, &presence_body(config, &participants, &attention)).await?;
    // Stream the turn (the early-segment TTS path) and report per-delta synthesis.
    let gen_body = generate_body(config, &request, message, &gamestate, &participants, distance_meters, dgu, "Goodsprings");
    let mut turn: Option<Value> = None;
    let mut stream_speaker: Option<npc::ResolvedSpeaker> = None;
    let mut total_chunks = 0usize;
    let mut total_bytes = 0usize;
    {
        let mut events = std::pin::pin!(client.generate_stream_events(&config.live_chat_id, &gen_body));
        while let Some(event) = events.next().await {
            let event = event?;
            match event.get("type").and_then(|t| t.as_str()) {
                Some("speaker.start") => {
                    if let Some(speaker) = event.get("speaker") {
                        stream_speaker =
                            Some(npc::resolve_stream_speaker(config, &participants, &request, speaker));
                    }
                }
                Some("speech.delta") => {
                    let segment = event.get("text").and_then(|t| t.as_str()).unwrap_or("").trim().to_string();
                    if !segment.is_empty() {
                        let name = stream_speaker
                            .as_ref()
                            .map(|s| s.character_name.clone())
                            .unwrap_or_else(|| config.character_name.clone());
                        let n = client
                            .synthesize_stream(&synth_body(config, &segment, &name, false), |chunk| {
                                total_chunks += 1;
                                total_bytes += chunk.audio.len();
                                Ok(())
                            })
                            .await?;
                        println!("  delta -> {n} TTS chunk(s): {}", truncate(&segment, 60));
                    }
                }
                Some("live.error") => {
                    let msg = event.pointer("/error/message").and_then(|m| m.as_str()).unwrap_or("live chat streaming failed");
                    anyhow::bail!("{msg}");
                }
                Some("live.completed") => turn = event.get("turn").cloned(),
                _ => {}
            }
        }
    }
    let turn = turn.ok_or_else(|| anyhow::anyhow!("stream ended without a final turn"))?;
    let lines = npc::extract_lines(config, &participants, &request, &turn);
    println!("generated {} line(s):", lines.len());
    for l in &lines {
        println!("  [{}] {}", l.character_name, l.text);
        let gm = actions::get_native_game_master_action(config, &l.turn);
        if gm.should_trigger {
            println!(
                "    -> ACTION {} (confidence {}, id={})",
                gm.action, gm.confidence, gm.action_id
            );
        }
    }
    let turn_gm = actions::get_native_game_master_action(config, &turn);
    println!(
        "turn action: {} (should_trigger={})",
        turn_gm.action, turn_gm.should_trigger
    );
    println!("TTS total: {total_chunks} chunk(s), {total_bytes} audio bytes");
    println!("turn-selftest OK");
    Ok(())
}

/// Recognize a WAV file against the live backend and print the transcript — an
/// offline smoke test of the STT path. Used by `--stt-selftest <wav>`.
pub async fn stt_selftest(config: &BridgeConfig, wav_path: &Path) -> anyhow::Result<()> {
    let client = HttpChasmClient::new(config)?;
    let bytes = std::fs::metadata(wav_path).map(|m| m.len()).unwrap_or(0);
    println!("recognizing {} ({} bytes)...", wav_path.display(), bytes);
    let text = stt::recognize_wav_file(config, &client, wav_path).await?;
    println!("transcript: {text:?}");
    println!("stt-selftest OK");
    Ok(())
}

/// Run one admin/Todd turn against the live backend and print the reply, the
/// classified action, the resolved actor, and a Todd-voice TTS check — `--admin-selftest`.
pub async fn admin_selftest(config: &BridgeConfig, message: &str) -> anyhow::Result<()> {
    let client = HttpChasmClient::new(config)?;
    let request = NativeRequest {
        request_id: format!("adminselftest_{}", protocol::now_epoch_millis()),
        npc_key: "todd".into(),
        npc_name: config.admin_character_name.clone(),
        want_tts: true,
        player_text: message.to_string(),
        ..Default::default()
    };
    println!("is_admin_request: {}", admin::is_admin_request(&request));
    let admin_turn = admin::generate_admin_turn(config, &client, &request, message, "Goodsprings").await?;
    println!("[{}] {}", config.admin_character_name, admin_turn.text);
    let gm = &admin_turn.game_master;
    println!(
        "action: {} (should_trigger={}, id={})",
        gm.action, gm.should_trigger, gm.action_id
    );
    if gm.should_trigger {
        match admin::resolve_native_actor_for_admin(config, &request, gm) {
            Some(a) => println!("resolved actor: key={} name={}", a.native_npc_key, a.native_npc_name),
            None => println!("resolved actor: <none — falls back to Todd>"),
        }
    }
    let body = synth_body(config, &admin_turn.text, &config.admin_character_name, true);
    let mut chunks = 0usize;
    let mut bytes = 0usize;
    client
        .synthesize_stream(&body, |c| {
            chunks += 1;
            bytes += c.audio.len();
            Ok(())
        })
        .await?;
    println!("Todd voice TTS: {chunks} chunk(s), {bytes} bytes");
    println!("admin-selftest OK");
    Ok(())
}

#[cfg(test)]
mod turn_body_tests {
    use super::*;

    /// The generate body must carry the game request's id as `traceId`, so the
    /// generate pipeline can correlate LLM usage/timings to this request's trace
    /// on the in-process path (no HTTP headers there).
    #[test]
    fn generate_body_carries_the_request_trace_id() {
        let config = default_config();
        let request = NativeRequest {
            request_id: "req_45109828_1".to_string(),
            npc_key: "easy_pete".to_string(),
            ..Default::default()
        };
        let body = generate_body(
            &config,
            &request,
            "Hi there, Pete.",
            &json!({}),
            &[],
            2.0,
            140.0,
            "Goodsprings",
        );
        assert_eq!(body["traceId"], json!("req_45109828_1"));
        assert_eq!(body["message"], json!("Hi there, Pete."));
    }
}
