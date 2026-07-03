use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use chasm_core::{
    format_message_timestamp, ActionView, InjectedEntryView, InjectedView, LiveChatView,
    MessageView, ParticipantView,
};
use thiserror::Error;

mod action_books;
mod lorebooks;
mod sources;
pub use action_books::*;
pub use lorebooks::*;
pub use sources::*;

#[derive(Debug, Error)]
pub enum CompatError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("JSON error at {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("session id is invalid")]
    InvalidSessionId,
    #[error("Live Chat '{0}' was not found")]
    LiveChatNotFound(String),
    #[error("Action Book '{0}' was not found")]
    ActionBookNotFound(String),
}

pub type Result<T> = std::result::Result<T, CompatError>;

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct LiveChatStore {
    #[serde(default)]
    pub items: BTreeMap<String, LiveChat>,
}

/// The Globals store (`headless/globals.json`): app-wide prompt building blocks
/// that belong to no single character or book. Today that is the global
/// scenario template; future globals land as new fields here (unknown keys
/// already survive a read→write round-trip via `extra`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct GlobalsStore {
    /// The global scenario prompt template ({{macro}} placeholders allowed).
    /// Replaces the per-character card `scenario` field in prompt assembly.
    /// Semantics: `None` = never saved → callers fall back to the built-in
    /// default template; `Some("")` = explicitly cleared → the scenario
    /// component is omitted from prompts entirely.
    #[serde(
        default,
        rename = "scenarioTemplate",
        skip_serializing_if = "Option::is_none"
    )]
    pub scenario_template: Option<String>,
    /// Forward-compat: any other keys in `globals.json` are preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct LiveChat {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub group_id: String,
    #[serde(default)]
    pub current_segment_id: String,
    #[serde(default)]
    pub active_participant_ids: Vec<String>,
    #[serde(default)]
    pub settings: Value,
    #[serde(default)]
    pub participants: BTreeMap<String, LiveChatParticipant>,
    #[serde(default)]
    pub presence: BTreeMap<String, LiveChatParticipant>,
    #[serde(default)]
    pub participant_sessions: Value,
    #[serde(default)]
    pub segments: Vec<LiveChatSegment>,
    #[serde(default)]
    pub events: Vec<Value>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct LiveChatParticipant {
    #[serde(default)]
    pub participant_id: String,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub character_id: Option<String>,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub present: Option<bool>,
    #[serde(default)]
    pub audible: Option<bool>,
    #[serde(default)]
    pub distance: Option<f64>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(default)]
    pub segment_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct LiveChatSegment {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub location: String,
    #[serde(default)]
    pub chat_id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct STJsonlChatMessage {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub is_user: bool,
    #[serde(default)]
    pub is_system: bool,
    #[serde(default)]
    pub send_date: Option<String>,
    #[serde(default)]
    pub mes: String,
    #[serde(default)]
    pub extra: Value,
    #[serde(default)]
    pub original_avatar: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct HeadlessLiveMetadata {
    #[serde(default)]
    pub live_chat_id: Option<String>,
    #[serde(default)]
    pub segment_id: Option<String>,
    #[serde(default)]
    pub speaker_participant_id: Option<String>,
    #[serde(default)]
    pub present: Vec<String>,
    #[serde(default)]
    pub audible_to: Vec<String>,
    #[serde(default)]
    pub location: Option<String>,
    #[serde(default)]
    pub strict_visibility: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct CharacterCardSummary {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionIdPayload {
    mode: String,
    character_id: Option<String>,
    group_id: Option<String>,
    chat_id: String,
}

/// Optional per-profile resolution inputs for a [`LiveChatRepository`]. When
/// present, content paths (characters, worlds, action/quest books, chats, the
/// live-chats store) resolve under the *active* profile folder (read from
/// `settings_path` on each call so a profile switch takes effect live), with a
/// fallback to the legacy `data_root`/`voices_dir` locations. When absent, every
/// path is the legacy `data_root` location — exactly the pre-profile behavior.
#[derive(Debug, Clone)]
struct ProfileResolution {
    profiles_dir: PathBuf,
    settings_path: PathBuf,
    voices_dir: PathBuf,
    embed_cache_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct LiveChatRepository {
    data_root: PathBuf,
    profile: Option<ProfileResolution>,
}

impl LiveChatRepository {
    /// Legacy constructor: all content resolves under `data_root` (no profile
    /// scoping). Retained for callers/tests that don't need profiles.
    pub fn new(data_root: impl Into<PathBuf>) -> Self {
        Self {
            data_root: data_root.into(),
            profile: None,
        }
    }

    /// Profile-aware constructor: content resolves under the active profile
    /// (resolved per call from `settings_path`) with a legacy `data_root`
    /// fallback. `voices_dir`/`embed_cache_dir` are the legacy bases used by the
    /// resolver for those two content kinds.
    pub fn with_profiles(
        data_root: impl Into<PathBuf>,
        profiles_dir: impl Into<PathBuf>,
        settings_path: impl Into<PathBuf>,
        voices_dir: impl Into<PathBuf>,
        embed_cache_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            data_root: data_root.into(),
            profile: Some(ProfileResolution {
                profiles_dir: profiles_dir.into(),
                settings_path: settings_path.into(),
                voices_dir: voices_dir.into(),
                embed_cache_dir: embed_cache_dir.into(),
            }),
        }
    }

    /// The legacy data root. Stays the literal `data_root` (used for content that
    /// is intentionally global, e.g. world-state, and as the resolver fallback).
    pub fn data_root(&self) -> &Path {
        &self.data_root
    }

    /// Builds a [`ProfilePaths`] for the active profile (resolved per call), or a
    /// legacy resolver (empty id → all paths fall back to `data_root`) when this
    /// repository was constructed without profile inputs.
    fn paths(&self) -> chasm_core::ProfilePaths {
        match self.profile.as_ref() {
            Some(p) => {
                let settings = chasm_core::AppSettings::load(&p.settings_path);
                let id = settings.active_profile_id(&p.profiles_dir);
                chasm_core::ProfilePaths::new(
                    &p.profiles_dir,
                    &id,
                    &self.data_root,
                    &p.voices_dir,
                    &p.embed_cache_dir,
                )
            }
            None => chasm_core::ProfilePaths::new(
                Path::new(""),
                "",
                &self.data_root,
                &self.data_root,
                &self.data_root,
            ),
        }
    }

    pub fn read_store(&self) -> Result<LiveChatStore> {
        let path = self.store_path();
        if !path.exists() {
            return Ok(LiveChatStore::default());
        }
        read_json_file(&path)
    }

    /// Path to the headless live-chats store, resolved under the active profile
    /// (`profiles/<id>/headless/live-chats.json`) with a legacy fallback.
    pub fn store_path(&self) -> PathBuf {
        self.paths().live_chats_store()
    }

    /// Persists the live-chats store to disk (`headless/live-chats.json`),
    /// pretty-printed to mirror the Node `writeLiveChatStore` output.
    pub fn write_store(&self, store: &LiveChatStore) -> Result<()> {
        let path = self.store_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| CompatError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let text = serde_json::to_string_pretty(store).map_err(|source| CompatError::Json {
            path: path.clone(),
            source,
        })?;
        fs::write(&path, text).map_err(|source| CompatError::Io {
            path: path.clone(),
            source,
        })
    }

    /// Reads, mutates, and writes the store in one shot. `mutate` receives the
    /// loaded store and may return a value carried back to the caller.
    pub fn update_store<T>(&self, mutate: impl FnOnce(&mut LiveChatStore) -> T) -> Result<T> {
        let mut store = self.read_store()?;
        let out = mutate(&mut store);
        self.write_store(&store)?;
        Ok(out)
    }

    /// Path to the Globals store, resolved under the active profile
    /// (`profiles/<id>/headless/globals.json`) with a legacy fallback — the
    /// same per-subpath rule as [`Self::store_path`].
    pub fn globals_store_path(&self) -> PathBuf {
        self.paths().globals_store()
    }

    /// Reads the Globals store (`headless/globals.json`). A missing file is the
    /// pristine default (every global unset), not an error.
    pub fn read_globals(&self) -> Result<GlobalsStore> {
        let path = self.globals_store_path();
        if !path.exists() {
            return Ok(GlobalsStore::default());
        }
        read_json_file(&path)
    }

    /// Persists the Globals store, pretty-printed like the live-chats store.
    pub fn write_globals(&self, store: &GlobalsStore) -> Result<()> {
        let path = self.globals_store_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| CompatError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let text = serde_json::to_string_pretty(store).map_err(|source| CompatError::Json {
            path: path.clone(),
            source,
        })?;
        fs::write(&path, text).map_err(|source| CompatError::Io {
            path: path.clone(),
            source,
        })
    }

    /// Reads, mutates, and writes the Globals store in one shot.
    pub fn update_globals<T>(&self, mutate: impl FnOnce(&mut GlobalsStore) -> T) -> Result<T> {
        let mut store = self.read_globals()?;
        let out = mutate(&mut store);
        self.write_globals(&store)?;
        Ok(out)
    }

    /// Appends one chat message line to the JSONL session file backing
    /// `segment` (creating the file/dirs if missing). Mirrors the Node
    /// `appendMessage` write path: each message is one JSON object per line.
    pub fn append_segment_message(
        &self,
        segment: &LiveChatSegment,
        message: &STJsonlChatMessage,
    ) -> Result<()> {
        let path = self.session_file_path(&segment.session_id)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| CompatError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let mut line = serde_json::to_string(message).map_err(|source| CompatError::Json {
            path: path.clone(),
            source,
        })?;
        line.push('\n');

        // ST writes one JSON object per line, but the last line of an existing
        // session file is not guaranteed to end in a newline. Prepend one when
        // the file already has bytes that don't end in `\n`, so we never
        // concatenate two objects onto a single (now-unparseable) line.
        let needs_leading_newline = match fs::metadata(&path) {
            Ok(meta) if meta.len() > 0 => !ends_with_newline(&path)?,
            _ => false,
        };

        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| CompatError::Io {
                path: path.clone(),
                source,
            })?;
        if needs_leading_newline {
            file.write_all(b"\n").map_err(|source| CompatError::Io {
                path: path.clone(),
                source,
            })?;
        }
        file.write_all(line.as_bytes())
            .map_err(|source| CompatError::Io {
                path: path.clone(),
                source,
            })
    }

    pub fn list_live_chats(&self) -> Result<Vec<LiveChat>> {
        Ok(self.read_store()?.items.into_values().collect())
    }

    pub fn get_live_chat(&self, id: &str) -> Result<LiveChat> {
        self.read_store()?
            .items
            .remove(id)
            .ok_or_else(|| CompatError::LiveChatNotFound(id.to_string()))
    }

    pub fn read_segment_messages(
        &self,
        segment: &LiveChatSegment,
    ) -> Result<Vec<STJsonlChatMessage>> {
        let path = self.session_file_path(&segment.session_id)?;
        read_jsonl_messages(&path)
    }

    /// Reads the raw JSONL messages of one session file by its session id.
    /// Same read as [`read_segment_messages`](Self::read_segment_messages) but
    /// usable for the per-participant PROJECTION sessions recorded in
    /// `live_chat.participant_sessions` (which have no `LiveChatSegment`) —
    /// and, unlike it, a session whose file does not exist yet yields an empty
    /// list instead of an IO error (projections are created lazily).
    pub fn read_session_messages(&self, session_id: &str) -> Result<Vec<STJsonlChatMessage>> {
        let path = self.session_file_path(session_id)?;
        if !path.exists() {
            return Ok(Vec::new());
        }
        read_jsonl_messages(&path)
    }

    pub fn messages_for_participant(
        &self,
        live_chat: &LiveChat,
        participant_id: &str,
    ) -> Result<Vec<MessageView>> {
        // A participant's conversation can be persisted in either of two places:
        //   * the participant's OWN PROJECTION session — a per-NPC `single`-mode
        //     file recorded in `live_chat.participant_sessions[id]`, the live
        //     game path's per-NPC view of every turn that NPC saw, and
        //   * the shared SEGMENT session files (`live_chat.segments`), the
        //     global group stream used by older / group-mode chats.
        // The two overlap heavily (a turn visible to an NPC is written to both),
        // and their copies drift (regenerated timestamps), so merging + deduping
        // is fragile. Instead we treat the participant projection as the
        // authoritative per-NPC source: read it when it holds any visible
        // messages, and fall back to the shared segments only when there is no
        // projection (or it is empty). Reading ONLY the segments was the bug —
        // the live path leaves the group segment empty and keeps each NPC's
        // history in its projection, so segment-only scanning returned no
        // threads for real, populated chats.
        let projection_path = participant_session_id(live_chat, participant_id)
            .map(|session_id| self.session_file_path(&session_id))
            .transpose()?;

        if let Some(path) = projection_path.as_ref().filter(|path| path.exists()) {
            let messages = self.collect_visible_messages(path, participant_id)?;
            if !messages.is_empty() {
                return Ok(messages);
            }
        }

        // No (usable) projection — fall back to the shared segment stream.
        let mut messages = Vec::new();
        for segment in &live_chat.segments {
            let path = self.session_file_path(&segment.session_id)?;
            if !path.exists() {
                continue;
            }
            messages.extend(self.collect_visible_messages(&path, participant_id)?);
        }
        // Re-key ids sequentially across segments so they stay unique/ordered.
        for (index, message) in messages.iter_mut().enumerate() {
            message.id = format!("m_{index}");
        }
        Ok(messages)
    }

    /// Reads one session file and projects every message visible to
    /// `participant_id` (the speaker, an `audibleTo` recipient, a player-visible
    /// line, or a globally-visible one) into a [`MessageView`], with ids keyed
    /// sequentially within the file.
    fn collect_visible_messages(
        &self,
        path: &Path,
        participant_id: &str,
    ) -> Result<Vec<MessageView>> {
        let mut messages = Vec::new();
        for message in read_jsonl_messages(path)? {
            let index = messages.len();
            if let Some(live) = extract_live_metadata(&message) {
                if let Some(reason) = visible_reason(&message, &live, participant_id) {
                    messages.push(to_message_view(index, &message, &live, reason));
                }
            } else if is_public_or_player_visible_without_live_metadata(&message) {
                messages.push(to_fallback_message_view(index, &message));
            }
        }
        Ok(messages)
    }

    /// Removes every message in a participant's conversation from the live
    /// chat's segment files — messages where the participant is the recorded
    /// live speaker or appears in `audibleTo`. Operates on raw JSON lines so no
    /// message fields are lost on rewrite, and keeps lines without live metadata
    /// (e.g. the chat header). Returns the number of messages removed.
    pub fn clear_participant_history(
        &self,
        live_chat: &LiveChat,
        participant_id: &str,
    ) -> Result<usize> {
        let mut removed = 0usize;
        for segment in &live_chat.segments {
            let path = self.session_file_path(&segment.session_id)?;
            if !path.exists() {
                continue;
            }
            let content = fs::read_to_string(&path).map_err(|source| CompatError::Io {
                path: path.clone(),
                source,
            })?;
            let (out, removed_here) = strip_participant_from_jsonl(&content, participant_id);
            // Only rewrite when something actually matched, so an unrelated
            // character's clear never re-normalizes another segment's file.
            if removed_here == 0 {
                continue;
            }
            removed += removed_here;
            fs::write(&path, out).map_err(|source| CompatError::Io {
                path: path.clone(),
                source,
            })?;
        }
        Ok(removed)
    }

    pub fn live_chat_view(
        &self,
        live_chat: &LiveChat,
        selected_participant_id: Option<&str>,
    ) -> Result<LiveChatView> {
        let mut participants = merged_participants(live_chat);
        let mut counts = BTreeMap::<String, usize>::new();
        for participant in &participants {
            counts.insert(
                participant.id.clone(),
                self.messages_for_participant(live_chat, &participant.id)?
                    .len(),
            );
        }
        for participant in &mut participants {
            participant.message_count = *counts.get(&participant.id).unwrap_or(&0);
            participant.selected =
                selected_participant_id.is_some_and(|selected| selected == participant.id);
        }

        Ok(LiveChatView {
            id: live_chat.id.clone(),
            title: if live_chat.title.is_empty() {
                live_chat.id.clone()
            } else {
                live_chat.title.clone()
            },
            participants,
            selected_participant_id: selected_participant_id.map(str::to_string),
        })
    }

    pub fn list_character_cards(&self) -> Result<Vec<CharacterCardSummary>> {
        let characters_dir = self.paths().characters_dir();
        if !characters_dir.exists() {
            return Ok(Vec::new());
        }

        let mut cards = Vec::new();
        for entry in fs::read_dir(&characters_dir).map_err(|source| CompatError::Io {
            path: characters_dir.clone(),
            source,
        })? {
            let entry = entry.map_err(|source| CompatError::Io {
                path: characters_dir.clone(),
                source,
            })?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            cards.push(CharacterCardSummary {
                id: stem.to_string(),
                name: stem.to_string(),
                path,
            });
        }
        cards.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        Ok(cards)
    }

    fn session_file_path(&self, session_id: &str) -> Result<PathBuf> {
        let payload = decode_session_id_typed(session_id)?;
        let sanitized = sanitize_file_name(&format!("{}.jsonl", payload.chat_id));
        let paths = self.paths();
        match payload.mode.as_str() {
            "single" => {
                let character_id = sanitize_path_segment(
                    payload
                        .character_id
                        .as_deref()
                        .ok_or(CompatError::InvalidSessionId)?,
                );
                Ok(paths.chats_dir().join(character_id).join(sanitized))
            }
            "group" => Ok(paths.group_chats_dir().join(sanitized)),
            _ => Err(CompatError::InvalidSessionId),
        }
    }
}

pub fn decode_session_id(session_id: &str) -> Result<Value> {
    let bytes = URL_SAFE_NO_PAD
        .decode(session_id)
        .map_err(|_| CompatError::InvalidSessionId)?;
    let payload: Value =
        serde_json::from_slice(&bytes).map_err(|_| CompatError::InvalidSessionId)?;
    if !payload
        .get("chatId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.is_empty())
    {
        return Err(CompatError::InvalidSessionId);
    }
    Ok(payload)
}

fn decode_session_id_typed(session_id: &str) -> Result<SessionIdPayload> {
    serde_json::from_value(decode_session_id(session_id)?)
        .map_err(|_| CompatError::InvalidSessionId)
}

/// The session id of a participant's projection chat (the per-NPC `single`-mode
/// file mirroring that participant's visible turns), read from the live chat's
/// `participantSessions` map. Returns `None` when there is no projection for the
/// participant or it lacks a non-empty `sessionId`.
fn participant_session_id(live_chat: &LiveChat, participant_id: &str) -> Option<String> {
    live_chat
        .participant_sessions
        .get(participant_id)?
        .get("sessionId")
        .and_then(Value::as_str)
        .map(str::to_string)
        .filter(|id| !id.is_empty())
}

/// Returns whether the last byte of `path` is a newline. Reads only the final
/// byte via a seek, so it is cheap even for large session files.
fn ends_with_newline(path: &Path) -> Result<bool> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = fs::File::open(path).map_err(|source| CompatError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let len = file
        .seek(SeekFrom::End(0))
        .map_err(|source| CompatError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    if len == 0 {
        return Ok(true);
    }
    file.seek(SeekFrom::End(-1))
        .map_err(|source| CompatError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let mut last = [0u8; 1];
    file.read_exact(&mut last)
        .map_err(|source| CompatError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(last[0] == b'\n')
}

pub(crate) fn read_json_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).map_err(|source| CompatError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| CompatError::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn read_jsonl_messages(path: &Path) -> Result<Vec<STJsonlChatMessage>> {
    let file = fs::File::open(path).map_err(|source| CompatError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut messages = Vec::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|source| CompatError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).map_err(|source| CompatError::Json {
            path: path.to_path_buf(),
            source,
        })?;
        if line_number == 0 && value.get("chat_metadata").is_some() {
            continue;
        }
        messages.push(
            serde_json::from_value(value).map_err(|source| CompatError::Json {
                path: path.to_path_buf(),
                source,
            })?,
        );
    }
    Ok(messages)
}

fn extract_live_metadata(message: &STJsonlChatMessage) -> Option<HeadlessLiveMetadata> {
    let value = message
        .extra
        .get("headless")?
        .get("metadata")?
        .get("live")?
        .clone();
    serde_json::from_value(value).ok()
}

/// Reads the per-message `extra.chasm` blob the generation path writes:
/// `{ "injected": { "lore"|"quests"|"actions": [{source,id,title,reason}…] },
/// "turn_actions": [{id,alias,target,params,reason}…] }`. Returns the parsed
/// injected groups (when present and non-empty) plus the turn's chosen actions.
/// Every field is optional and best-effort: a missing/old/garbled blob yields
/// `(None, [])` so pre-feature messages render as "no data recorded".
fn extract_chasm_metadata(
    message: &STJsonlChatMessage,
) -> (Option<InjectedView>, Vec<ActionView>) {
    let Some(blob) = message.extra.get("chasm") else {
        return (None, Vec::new());
    };

    let injected = blob.get("injected").map(|injected| InjectedView {
        lore: parse_injected_entries(injected.get("lore")),
        quests: parse_injected_entries(injected.get("quests")),
        actions: parse_injected_entries(injected.get("actions")),
        activated_actions: Vec::new(),
    });
    // Drop an all-empty injected object so the view treats it the same as
    // "nothing was recorded for this group" rather than three empty headings.
    let injected = injected.filter(|view| !view.is_empty());

    let turn_actions = blob
        .get("turn_actions")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(parse_action_view).collect())
        .unwrap_or_default();

    (injected, turn_actions)
}

/// Parses an array of injected-entry objects, skipping anything that isn't an
/// object. Each field defaults to empty so a partial record still renders.
fn parse_injected_entries(value: Option<&Value>) -> Vec<InjectedEntryView> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let object = item.as_object()?;
                    Some(InjectedEntryView {
                        source: json_str(object.get("source")),
                        id: json_str(object.get("id")),
                        title: json_str(object.get("title")),
                        reason: json_str(object.get("reason")),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Parses one persisted action object into an [`ActionView`]. `parameters` is
/// re-serialized to a compact JSON string for display (an empty object becomes
/// `{}`, which the template hides). Returns `None` for non-object entries.
fn parse_action_view(value: &Value) -> Option<ActionView> {
    let object = value.as_object()?;
    let params = object
        .get("parameters")
        .filter(|params| !matches!(params, Value::Null))
        .map(|params| serde_json::to_string(params).unwrap_or_default())
        .unwrap_or_default();
    Some(ActionView {
        id: json_str(object.get("id")),
        alias: json_str(object.get("alias")),
        target: json_str(object.get("target")),
        params,
        reason: json_str(object.get("reason")),
    })
}

/// A trimmed owned string from an optional JSON string value (empty otherwise).
fn json_str(value: Option<&Value>) -> String {
    value
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string()
}

pub fn is_live_message_visible_to_participant(
    message: &STJsonlChatMessage,
    participant_id: &str,
) -> bool {
    extract_live_metadata(message)
        .and_then(|live| visible_reason(message, &live, participant_id))
        .is_some()
}

fn visible_reason(
    message: &STJsonlChatMessage,
    live: &HeadlessLiveMetadata,
    participant_id: &str,
) -> Option<String> {
    if live
        .speaker_participant_id
        .as_deref()
        .is_some_and(|speaker| speaker == participant_id)
    {
        return Some("speaker".to_string());
    }
    if live.audible_to.iter().any(|id| id == participant_id) {
        return Some("audible".to_string());
    }
    if message.is_user && live.audible_to.is_empty() {
        return Some("player-visible".to_string());
    }
    if live.strict_visibility == Some(false) {
        return Some("global".to_string());
    }

    // TODO: Mirror any future ST metadata aliases for public/global visibility once
    // they are formalized in src/headless/live-chat-utils.js.
    None
}

fn is_public_or_player_visible_without_live_metadata(message: &STJsonlChatMessage) -> bool {
    message.is_system
        || message.is_user
            && message
                .extra
                .get("headless")
                .and_then(|value| value.get("metadata"))
                .is_none()
}

/// Whether a raw chat-message JSON value belongs to a participant's
/// conversation: the participant is the recorded live speaker, OR the line was
/// addressed to them privately (they are audible and no other NPC is audible).
/// A line the participant merely overheard in a shared scene does NOT match, so
/// clearing one NPC never wipes a bystander NPC's lines. Lines without live
/// metadata (e.g. the chat header) never match.
fn message_belongs_to_participant(message: &Value, participant_id: &str) -> bool {
    let Some(live) = message
        .get("extra")
        .and_then(|extra| extra.get("headless"))
        .and_then(|headless| headless.get("metadata"))
        .and_then(|metadata| metadata.get("live"))
    else {
        return false;
    };
    // The participant's own spoken lines always belong to their history.
    if live.get("speakerParticipantId").and_then(Value::as_str) == Some(participant_id) {
        return true;
    }
    // Otherwise a line only belongs to this participant if it was addressed to
    // THEM privately: they were audible AND no OTHER NPC was audible. This stops
    // "clear <NPC>" from wiping a bystander NPC's lines just because the cleared
    // NPC happened to overhear them in a shared scene (e.g. Goodsprings, where
    // every nearby NPC is listed in `audibleTo`). Player lines directed at this
    // NPC alone (audibleTo = {npc, player}) are still cleared.
    let Some(audible) = live.get("audibleTo").and_then(Value::as_array) else {
        return false;
    };
    let participant_audible = audible
        .iter()
        .any(|value| value.as_str() == Some(participant_id));
    if !participant_audible {
        return false;
    }
    let other_npc_audible = audible.iter().any(|value| {
        value
            .as_str()
            .is_some_and(|id| id != participant_id && id.starts_with("npc:"))
    });
    !other_npc_audible
}

/// Removes every JSONL line that belongs to `participant_id` (the recorded live
/// speaker, or listed in `audibleTo`) from a raw chat-session file body. Lines
/// without live metadata (e.g. the chat header) are preserved. Returns the
/// rewritten body — newline-terminated when non-empty — and the number of
/// messages removed.
///
/// Shared by the live participant-clear path and the save-sync checkpoint scrub
/// so a cleared conversation is removed by EXACTLY the same rule from both the
/// active chat file and any checkpoint snapshot (otherwise a game load restores
/// what the clear took out).
pub fn strip_participant_from_jsonl(content: &str, participant_id: &str) -> (String, usize) {
    let mut kept: Vec<&str> = Vec::new();
    let mut removed = 0usize;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let belongs = serde_json::from_str::<Value>(trimmed)
            .ok()
            .is_some_and(|value| message_belongs_to_participant(&value, participant_id));
        if belongs {
            removed += 1;
        } else {
            kept.push(trimmed);
        }
    }
    let mut out = kept.join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    (out, removed)
}

fn to_message_view(
    index: usize,
    message: &STJsonlChatMessage,
    live: &HeadlessLiveMetadata,
    visible_reason: String,
) -> MessageView {
    let (injected, turn_actions) = extract_chasm_metadata(message);
    MessageView {
        id: format!("m_{index}"),
        role: role_for_message(message).to_string(),
        speaker_participant_id: live.speaker_participant_id.clone(),
        speaker_name: if message.name.is_empty() {
            live.speaker_participant_id
                .clone()
                .unwrap_or_else(|| "Unknown".to_string())
        } else {
            message.name.clone()
        },
        speaker_initial: initial_for(if message.name.is_empty() {
            live.speaker_participant_id.as_deref().unwrap_or("Unknown")
        } else {
            &message.name
        }),
        content: message.mes.clone(),
        created_at: message.send_date.clone(),
        created_at_label: message
            .send_date
            .as_deref()
            .map(format_message_timestamp)
            .unwrap_or_default(),
        segment_id: live.segment_id.clone(),
        location: live.location.clone(),
        audible_to: live.audible_to.clone(),
        visible_reason,
        injected,
        turn_actions,
    }
}

fn to_fallback_message_view(index: usize, message: &STJsonlChatMessage) -> MessageView {
    let (injected, turn_actions) = extract_chasm_metadata(message);
    MessageView {
        id: format!("m_{index}"),
        role: role_for_message(message).to_string(),
        speaker_participant_id: None,
        speaker_name: if message.name.is_empty() {
            "Unknown".to_string()
        } else {
            message.name.clone()
        },
        speaker_initial: initial_for(if message.name.is_empty() {
            "Unknown"
        } else {
            &message.name
        }),
        content: message.mes.clone(),
        created_at: message.send_date.clone(),
        created_at_label: message
            .send_date
            .as_deref()
            .map(format_message_timestamp)
            .unwrap_or_default(),
        segment_id: None,
        location: None,
        audible_to: Vec::new(),
        visible_reason: "fallback".to_string(),
        injected,
        turn_actions,
    }
}

fn role_for_message(message: &STJsonlChatMessage) -> &'static str {
    if message.is_system {
        "system"
    } else if message.is_user {
        "player"
    } else {
        "npc"
    }
}

fn merged_participants(live_chat: &LiveChat) -> Vec<ParticipantView> {
    let mut ids = BTreeSet::new();
    ids.extend(live_chat.participants.keys().cloned());
    ids.extend(live_chat.presence.keys().cloned());
    ids.extend(live_chat.active_participant_ids.iter().cloned());

    let mut participants: Vec<_> = ids
        .into_iter()
        .map(|id| {
            let base = live_chat
                .presence
                .get(&id)
                .or_else(|| live_chat.participants.get(&id));
            let name = base
                .map(|participant| participant.name.clone())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| id.clone());
            let distance = base.and_then(|participant| participant.distance);
            ParticipantView {
                id: id.clone(),
                initial: initial_for(&name),
                name,
                kind: base
                    .map(|participant| participant.kind.clone())
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| "unknown".to_string()),
                character_id: base.and_then(|participant| participant.character_id.clone()),
                present: base
                    .and_then(|participant| participant.present)
                    .unwrap_or(false),
                audible: base
                    .and_then(|participant| participant.audible)
                    .unwrap_or(false),
                distance,
                distance_label: distance
                    .map(|value| format!("{value:.1}m"))
                    .unwrap_or_default(),
                message_count: 0,
                selected: false,
            }
        })
        .collect();

    participants.sort_by(|a, b| {
        b.present
            .cmp(&a.present)
            .then_with(|| b.audible.cmp(&a.audible))
            .then_with(|| a.kind.cmp(&b.kind))
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    participants
}

fn sanitize_path_segment(value: &str) -> String {
    value
        .chars()
        .filter(|ch| !matches!(ch, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'))
        .collect::<String>()
}

fn sanitize_file_name(value: &str) -> String {
    sanitize_path_segment(value)
        .trim_end_matches('.')
        .to_string()
}

fn initial_for(value: &str) -> String {
    value
        .chars()
        .find(|ch| ch.is_alphanumeric())
        .map(|ch| ch.to_uppercase().collect::<String>())
        .unwrap_or_else(|| "?".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_participant_removes_only_their_lines() {
        // Header (no live metadata) is kept; Sunny's speaker line and a player
        // line audible to Sunny are removed; Chet's line (not audible to Sunny)
        // stays. This is the exact rule shared by clear + the checkpoint scrub.
        let content = concat!(
            "{\"user_name\":\"Player\",\"chat_metadata\":{}}\n",
            "{\"is_user\":true,\"mes\":\"hi sunny\",\"extra\":{\"headless\":{\"metadata\":{\"live\":{\"audibleTo\":[\"player\",\"npc:sunny\"]}}}}}\n",
            "{\"name\":\"Sunny\",\"mes\":\"Howdy.\",\"extra\":{\"headless\":{\"metadata\":{\"live\":{\"speakerParticipantId\":\"npc:sunny\",\"audibleTo\":[\"player\",\"npc:sunny\"]}}}}}\n",
            "{\"name\":\"Chet\",\"mes\":\"Hey.\",\"extra\":{\"headless\":{\"metadata\":{\"live\":{\"speakerParticipantId\":\"npc:chet\",\"audibleTo\":[\"player\",\"npc:chet\"]}}}}}\n",
        );
        let (out, removed) = strip_participant_from_jsonl(content, "npc:sunny");
        assert_eq!(removed, 2);
        assert!(out.contains("chat_metadata"), "header kept");
        assert!(out.contains("Chet"), "unrelated NPC kept");
        assert!(!out.contains("Howdy"), "Sunny's line removed");
        assert!(
            !out.contains("hi sunny"),
            "player line audible to Sunny removed"
        );
        assert!(out.ends_with('\n'));
        // Idempotent: a second pass removes nothing.
        let (_, again) = strip_participant_from_jsonl(&out, "npc:sunny");
        assert_eq!(again, 0);
    }

    #[test]
    fn clearing_one_npc_keeps_a_bystander_npcs_lines() {
        // Goodsprings-style shared scene: every nearby NPC is listed in
        // `audibleTo`. Clearing Easy Pete must remove ONLY Pete's own lines and
        // player lines addressed to Pete alone — never Sunny's lines that Pete
        // merely overheard (the bug that wiped Sunny's history along with Pete's).
        let content = concat!(
            "{\"user_name\":\"Player\",\"chat_metadata\":{}}\n",
            // Player greets Pete privately (only Pete audible) -> cleared.
            "{\"is_user\":true,\"mes\":\"hi pete\",\"extra\":{\"headless\":{\"metadata\":{\"live\":{\"audibleTo\":[\"player\",\"npc:easy_pete\"]}}}}}\n",
            // Pete's own reply -> cleared.
            "{\"name\":\"Pete\",\"mes\":\"Howdy stranger.\",\"extra\":{\"headless\":{\"metadata\":{\"live\":{\"speakerParticipantId\":\"npc:easy_pete\",\"audibleTo\":[\"player\",\"npc:easy_pete\"]}}}}}\n",
            // Sunny speaks while Pete is also audible -> MUST stay.
            "{\"name\":\"Sunny\",\"mes\":\"Watch the geckos.\",\"extra\":{\"headless\":{\"metadata\":{\"live\":{\"speakerParticipantId\":\"npc:sunny\",\"audibleTo\":[\"player\",\"npc:sunny\",\"npc:easy_pete\"]}}}}}\n",
            // Player addresses the room (both NPCs audible) -> shared, MUST stay.
            "{\"is_user\":true,\"mes\":\"hey everyone\",\"extra\":{\"headless\":{\"metadata\":{\"live\":{\"audibleTo\":[\"player\",\"npc:sunny\",\"npc:easy_pete\"]}}}}}\n",
        );
        let (out, removed) = strip_participant_from_jsonl(content, "npc:easy_pete");
        assert_eq!(removed, 2, "only Pete's own line + the Pete-only player line");
        assert!(!out.contains("hi pete"), "player line to Pete alone removed");
        assert!(!out.contains("Howdy stranger"), "Pete's own line removed");
        assert!(out.contains("Watch the geckos"), "Sunny's overheard line kept");
        assert!(out.contains("hey everyone"), "shared room line kept");
    }

    #[test]
    fn globals_store_round_trips_and_defaults_when_missing() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let root = std::env::temp_dir().join(format!("chasm-test-globals-{nanos}"));
        std::fs::create_dir_all(&root).unwrap();
        let repo = LiveChatRepository::new(&root);

        // Missing file -> pristine defaults (scenario unset), not an error.
        let store = repo.read_globals().unwrap();
        assert!(store.scenario_template.is_none());

        // Save a template; it survives a re-read, and unknown keys placed in
        // the file by future versions survive an update round-trip.
        repo.update_globals(|globals| {
            globals.scenario_template = Some("It is {{time_of_day}}.".to_string());
            globals
                .extra
                .insert("futureKey".to_string(), serde_json::json!({ "keep": true }));
        })
        .unwrap();
        let store = repo.read_globals().unwrap();
        assert_eq!(
            store.scenario_template.as_deref(),
            Some("It is {{time_of_day}}.")
        );
        assert_eq!(store.extra["futureKey"]["keep"], true);
        // Written under headless/globals.json (mirrors live-chats.json).
        assert!(root.join("headless").join("globals.json").exists());

        // Explicitly cleared (`Some("")`) is distinct from never-saved (`None`).
        repo.update_globals(|globals| globals.scenario_template = Some(String::new()))
            .unwrap();
        assert_eq!(repo.read_globals().unwrap().scenario_template.as_deref(), Some(""));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn decodes_st_base64url_session_id() {
        let value = decode_session_id("eyJtb2RlIjoiZ3JvdXAiLCJncm91cElkIjoiZm52LWdvb2RzcHJpbmdzIiwiY2hhdElkIjoiR29vZHNwcmluZ3MifQ").unwrap();
        assert_eq!(value["mode"], "group");
        assert_eq!(value["chatId"], "Goodsprings");
    }

    #[test]
    fn filters_by_speaker_or_audible_to() {
        let message = STJsonlChatMessage {
            name: "Sunny Smiles".to_string(),
            mes: "Howdy.".to_string(),
            extra: serde_json::json!({
                "headless": {
                    "metadata": {
                        "live": {
                            "liveChatId": "fnv-goodsprings",
                            "segmentId": "Goodsprings",
                            "speakerParticipantId": "npc:sunny_smiles",
                            "audibleTo": ["npc:sunny_smiles", "player", "npc:easy_pete"],
                            "strictVisibility": true
                        }
                    }
                }
            }),
            ..Default::default()
        };

        assert!(is_live_message_visible_to_participant(
            &message,
            "npc:sunny_smiles"
        ));
        assert!(is_live_message_visible_to_participant(
            &message,
            "npc:easy_pete"
        ));
        assert!(!is_live_message_visible_to_participant(
            &message,
            "npc:cheyenne"
        ));
    }

    #[test]
    fn message_view_parses_chasm_injection_blob() {
        let live = HeadlessLiveMetadata {
            speaker_participant_id: Some("npc:sunny_smiles".to_string()),
            ..Default::default()
        };
        let message = STJsonlChatMessage {
            name: "Sunny Smiles".to_string(),
            mes: "Right behind you.".to_string(),
            extra: serde_json::json!({
                "headless": { "metadata": { "live": {} } },
                "chasm": {
                    "injected": {
                        "lore": [
                            { "source": "lore", "id": "Goodsprings", "title": "Goodsprings", "reason": "keyword" }
                        ],
                        "quests": [],
                        "actions": [
                            { "source": "action", "id": "movement.follow_target", "title": "Follow target", "reason": "vector" }
                        ]
                    },
                    "turn_actions": [
                        { "id": "movement.follow_target", "alias": "follow", "target": "player", "parameters": { "speed": 1 }, "reason": "Asked to follow." }
                    ]
                }
            }),
            ..Default::default()
        };

        let view = to_message_view(0, &message, &live, "speaker".to_string());
        let injected = view.injected.expect("injected present");
        assert_eq!(injected.lore.len(), 1);
        assert_eq!(injected.lore[0].title, "Goodsprings");
        assert_eq!(injected.lore[0].reason, "keyword");
        assert!(injected.quests.is_empty());
        assert_eq!(injected.actions.len(), 1);
        assert_eq!(injected.actions[0].id, "movement.follow_target");
        assert_eq!(injected.actions[0].reason, "vector");
        assert_eq!(view.turn_actions.len(), 1);
        assert_eq!(view.turn_actions[0].alias, "follow");
        assert_eq!(view.turn_actions[0].target, "player");
        assert_eq!(view.turn_actions[0].params, "{\"speed\":1}");
    }

    #[test]
    fn message_view_without_blob_has_no_injection() {
        // An old/player message with no `extra.chasm` => None + empty actions.
        let message = STJsonlChatMessage {
            name: "Player".to_string(),
            is_user: true,
            mes: "Hey, follow me.".to_string(),
            extra: serde_json::json!({ "headless": { "metadata": { "live": {} } } }),
            ..Default::default()
        };
        let view = to_fallback_message_view(3, &message);
        assert!(view.injected.is_none());
        assert!(view.turn_actions.is_empty());
    }
}
