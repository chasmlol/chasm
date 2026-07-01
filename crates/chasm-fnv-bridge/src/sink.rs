//! `TurnSink` — the output seam for a single NPC turn.
//!
//! [`run_turn`](crate::run_turn) produces three kinds of output: the NPC reply, zero
//! or more streamed TTS audio chunks, and at most one game_master action. Historically
//! it emitted all three by writing files under the NVBridge folder (the reply via
//! `write_outgoing`, each chunk via `write_audio_chunk`, the action via
//! [`actions::write_native_game_master_command`](crate::actions::write_native_game_master_command)).
//!
//! This trait lets the same orchestration drive a different transport without
//! touching `run_turn`'s logic. [`FileSink`] is the default impl and reproduces
//! TODAY's file writes byte-for-byte (the bodies were moved here verbatim), so the
//! file bridge is unchanged. A streaming HTTP transport (in `chasm-web`)
//! provides its own impl that taps the same three outputs.
//!
//! Scope note: only these three *outputs* are abstracted. The inbound request scan,
//! save-sync relay, request archiving, and the durable control/actions queue all
//! stay on files, untouched.

use std::path::{Path, PathBuf};

use tracing::warn;

use crate::actions::{self, ActionActor, GameMaster};
use crate::chasm::AudioChunk;
use crate::config::BridgeConfig;
use crate::npc::GeneratedLine;
use crate::protocol::{
    build_native_audio_chunk, build_native_response, now_iso8601_millis, parse_native_text_request,
    safe_file_id, ResponseFields,
};

/// The structured NPC reply for one turn — the fields `build_native_response` lays
/// out. `run_turn` builds these; the sink decides how to emit them (a response file,
/// an HTTP event, …).
pub struct OutgoingResponse {
    pub status: String,
    pub request_id: String,
    pub npc_key: String,
    pub npc_name: String,
    pub audio_filename: String,
    pub text: String,
    pub error: String,
    pub player_text: String,
    pub extra_lines: Vec<String>,
    pub gm_action: String,
    pub gm_confidence: String,
    pub gm_should_trigger: bool,
}

impl OutgoingResponse {
    /// Defaults carrying the request's identity, for `..base(request)` updates.
    pub fn base(request: &crate::protocol::NativeRequest) -> Self {
        Self {
            status: "0".into(),
            request_id: request.request_id.clone(),
            npc_key: request.npc_key.clone(),
            npc_name: request.npc_name.clone(),
            audio_filename: String::new(),
            text: String::new(),
            error: String::new(),
            player_text: request.player_text.clone(),
            extra_lines: Vec::new(),
            gm_action: String::new(),
            gm_confidence: String::new(),
            gm_should_trigger: false,
        }
    }
}

/// One emitted audio chunk's identity, returned to `run_turn` so it can name the
/// reply's `audio_filename` (the first chunk's file) and advance the chunk index.
#[derive(Clone)]
pub struct WrittenChunk {
    pub filename: String,
    pub index: u32,
}

/// The audio WAV filename for a request's chunk (chunk 0 has no index suffix). Used
/// by both the file sink (the on-disk name) and any sink that needs a stable name.
pub fn audio_filename(request_id: &str, index: u32) -> String {
    if index == 0 {
        format!("nvbridge_{}.wav", safe_file_id(request_id))
    } else {
        format!("nvbridge_{}.{:04}.wav", safe_file_id(request_id), index)
    }
}

/// The output seam for one NPC turn: the reply, each audio chunk, the game_master
/// action, and an end-of-turn marker. [`FileSink`] writes files (the default);
/// other impls stream the same outputs over a different transport.
///
/// Methods take `&self` (sinks hold per-request immutable context, e.g. the file
/// paths) so the streaming TTS callback in `run_turn` can call `audio_chunk` while
/// the reply is still pending.
pub trait TurnSink: Send + Sync {
    /// Emit the NPC reply. Returns whether it was emitted (`FileSink` skips a stale
    /// response when a newer request has superseded this one).
    fn reply(&self, request: &crate::protocol::NativeRequest, resp: &OutgoingResponse)
        -> anyhow::Result<bool>;

    /// Emit one TTS audio chunk. `line` carries the speaker identity + caption text,
    /// `chunk` the audio bytes + per-chunk caption, `index` the gapless playback
    /// order, `extra_meta` extra `key=value` caption lines (e.g. admin voice flags).
    fn audio_chunk(
        &self,
        request: &crate::protocol::NativeRequest,
        line: &GeneratedLine,
        chunk: &AudioChunk,
        index: u32,
        extra_meta: &[String],
    ) -> anyhow::Result<WrittenChunk>;

    /// Emit the turn's game_master action (the durable command-file queue). Returns
    /// whether anything was queued, so `run_turn` can disarm the response's
    /// `gm_should_trigger` (the action fires from the queue, not the reply).
    fn action(
        &self,
        config: &BridgeConfig,
        request: &crate::protocol::NativeRequest,
        actor: &ActionActor,
        gm: &GameMaster,
        source: &str,
    ) -> bool;

    /// Signal the turn is complete. `FileSink` has nothing to flush (each output is
    /// written as it is produced); a streaming sink emits a terminal event here.
    fn end_of_turn(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// The default [`TurnSink`]: writes the same NVBridge files the bridge always has.
/// Holds the active bridge `root` and the inbound request `path` — exactly what the
/// response/audio writers need. The bodies below were moved verbatim from
/// `lib.rs::{write_outgoing, write_audio_chunk}` so the on-disk bytes are identical.
pub struct FileSink {
    root: PathBuf,
    path: PathBuf,
}

impl FileSink {
    pub fn new(root: &Path, path: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            path: path.to_path_buf(),
        }
    }
}

impl TurnSink for FileSink {
    fn reply(
        &self,
        request: &crate::protocol::NativeRequest,
        resp: &OutgoingResponse,
    ) -> anyhow::Result<bool> {
        // Skip a stale response if a newer request has superseded this one.
        if !is_request_file_still_current(&self.path, request) {
            warn!(
                "skipped stale response for {}; a newer request is active",
                request.request_id
            );
            return Ok(false);
        }
        let timestamp = now_iso8601_millis();
        let fields = ResponseFields {
            status: &resp.status,
            request_id: &resp.request_id,
            npc_key: &resp.npc_key,
            npc_name: &resp.npc_name,
            audio_filename: &resp.audio_filename,
            text: &resp.text,
            error: &resp.error,
            timestamp: &timestamp,
            player_text: &resp.player_text,
            extra_lines: &resp.extra_lines,
            gm_action: &resp.gm_action,
            gm_confidence: &resp.gm_confidence,
            gm_should_trigger: resp.gm_should_trigger,
        };
        let outbox = crate::native_outbox(&self.root);
        std::fs::create_dir_all(&outbox)?;
        let out_path = outbox.join(self.path.file_name().unwrap());
        std::fs::write(&out_path, build_native_response(&fields))?;
        Ok(true)
    }

    fn audio_chunk(
        &self,
        request: &crate::protocol::NativeRequest,
        line: &GeneratedLine,
        chunk: &AudioChunk,
        index: u32,
        extra_meta: &[String],
    ) -> anyhow::Result<WrittenChunk> {
        // audio + chunk dirs are root-independent (stream storage dir).
        let wav_name = audio_filename(&request.request_id, index);
        let audio_dir = crate::stream_storage_dir().join("audio");
        std::fs::create_dir_all(&audio_dir)?;
        std::fs::write(audio_dir.join(&wav_name), &chunk.audio)?;

        let raw_caption = chunk.text.clone().unwrap_or_else(|| line.text.clone());
        let timestamp = now_iso8601_millis();
        let (chunk_file, contents) = build_native_audio_chunk(
            &request.request_id,
            index,
            &line.native_npc_key,
            &line.native_npc_name,
            &wav_name,
            &raw_caption,
            &timestamp,
            chunk.caption_max_chars,
            extra_meta,
        );
        let chunk_dir = crate::stream_storage_dir().join("chunks");
        std::fs::create_dir_all(&chunk_dir)?;
        std::fs::write(chunk_dir.join(&chunk_file), contents)?;

        Ok(WrittenChunk {
            filename: wav_name,
            index,
        })
    }

    fn action(
        &self,
        config: &BridgeConfig,
        request: &crate::protocol::NativeRequest,
        actor: &ActionActor,
        gm: &GameMaster,
        source: &str,
    ) -> bool {
        actions::write_native_game_master_command(config, request, actor, gm, source)
    }
}

/// Whether the inbound request file still carries `request`'s id (no newer request
/// has replaced it). An empty id or an unreadable file is treated as still-current,
/// matching the historical behavior.
fn is_request_file_still_current(path: &Path, request: &crate::protocol::NativeRequest) -> bool {
    if request.request_id.is_empty() {
        return true;
    }
    if !path.exists() {
        return false;
    }
    match std::fs::read_to_string(path) {
        Ok(text) => parse_native_text_request(path, &text).request_id == request.request_id,
        Err(_) => true,
    }
}
