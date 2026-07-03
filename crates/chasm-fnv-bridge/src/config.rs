//! Parse the same `nvbridge.config.json` the Node helper reads. Section 1 needed
//! only the native roots + poll interval; Section 2 adds the chasm API surface
//! (base/auth/live-chat ids), NPC mapping, distances, and TTS overrides.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Map, Value};

/// The fixed rendezvous dir both chasm and the in-game plugin meet at:
/// `%LOCALAPPDATA%\chasm\bridge` (override `CHASM_BRIDGE_ROOT`). Kept identical to
/// `chasm_core::default_bridge_root()` and the C++ plugin's `DefaultBridgeDir`
/// so a separately-installed chasm and an MO2-managed plugin always meet here —
/// the path is outside MO2's virtual filesystem, so the plugin's writes land on the
/// real disk where chasm reads them.
fn default_bridge_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("CHASM_BRIDGE_ROOT") {
        let path = PathBuf::from(dir);
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("APPDATA"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("chasm")
        .join("bridge")
}

const DEFAULT_POLL_MS: u64 = 750;
const DEFAULT_API_BASE: &str = "http://127.0.0.1:8000/api/headless/v1";
const DEFAULT_NATIVE_MAX_DISTANCE_METERS: f64 = 10.0;
const DEFAULT_GAMESTATE_RADIUS_METERS: f64 = 30.0;
const DEFAULT_ACTION_BOOK_TARGET_GAME: &str = "fallout-new-vegas";
const DEFAULT_API_REQUEST_TIMEOUT_MS: u64 = 180_000;
const DEFAULT_API_STT_TIMEOUT_MS: u64 = 45_000;

#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub native_bridge_roots: Vec<PathBuf>,
    pub poll_ms: u64,

    // chasm API surface
    pub api_base: String,
    /// Empty = use `api_base`. Mirrors the Node per-capability routing.
    pub tts_api_base: String,
    pub stt_api_base: String,
    pub api_key: String,
    pub request_timeout_ms: u64,

    // live-chat identity
    pub live_chat_id: String,
    pub group_id: String,
    pub participant_id: String,
    pub character_id: String,
    pub character_name: String,

    // NPC resolution + gamestate
    pub npc_character_map: Map<String, Value>,
    pub native_max_distance_meters: f64,
    pub gamestate_radius_meters: f64,

    // generation knobs
    pub enable_action_books: bool,
    pub action_book_target_game: String,
    pub action_book_ids: Vec<String>,
    pub native_action_confidence: f64,
    pub model: String,

    // admin / Todd (god voice + spawns)
    pub admin_character_id: String,
    pub admin_character_name: String,
    pub admin_action_book_limit: u64,
    pub admin_session_id: String,

    /// `ttsOverrides` object spread into every `/speech/synthesize*` body.
    pub tts: Value,
    /// `speechRecognition` object spread into every `/speech/recognize` body.
    pub speech_recognition: Value,
    pub speech_recognition_timeout_ms: u64,

    /// Whether music generation (the play-a-song action) is enabled. Set from
    /// `AppSettings.music.enabled` by the in-process bridge; the standalone HTTP
    /// bin leaves it `false` (music runs only in-process). When false, the song
    /// job is never started even if the action fires.
    pub music_enabled: bool,
}

impl BridgeConfig {
    /// The base URL for an endpoint, honoring the STT/TTS overrides like Node's
    /// `apiFetch`: exactly `/speech/recognize` → STT base; other `/speech*` → TTS
    /// base; everything else → `api_base`. Each falls back to `api_base`.
    pub fn base_for(&self, endpoint: &str) -> &str {
        if endpoint == "/speech/recognize" {
            if !self.stt_api_base.is_empty() {
                return &self.stt_api_base;
            }
        } else if endpoint.starts_with("/speech") && !self.tts_api_base.is_empty() {
            return &self.tts_api_base;
        }
        &self.api_base
    }
}

#[derive(Debug, Default, Deserialize)]
struct RawConfig {
    #[serde(default, rename = "nativeBridgeRoots")]
    native_bridge_roots: Vec<String>,
    #[serde(default, rename = "pollMs")]
    poll_ms: Option<u64>,
    #[serde(default, rename = "apiBase")]
    api_base: Option<String>,
    #[serde(default, rename = "ttsApiBase")]
    tts_api_base: Option<String>,
    #[serde(default, rename = "sttApiBase")]
    stt_api_base: Option<String>,
    #[serde(default, rename = "apiKey")]
    api_key: Option<String>,
    #[serde(default, rename = "requestTimeoutMs")]
    request_timeout_ms: Option<u64>,
    #[serde(default, rename = "liveChatId")]
    live_chat_id: Option<String>,
    #[serde(default, rename = "groupId")]
    group_id: Option<String>,
    #[serde(default, rename = "participantId")]
    participant_id: Option<String>,
    #[serde(default, rename = "characterId")]
    character_id: Option<String>,
    #[serde(default, rename = "characterName")]
    character_name: Option<String>,
    // Node merges characterMap + npcCharacters + npcCharacterMap (later wins).
    #[serde(default, rename = "characterMap")]
    character_map: Option<Map<String, Value>>,
    #[serde(default, rename = "npcCharacters")]
    npc_characters: Option<Map<String, Value>>,
    #[serde(default, rename = "npcCharacterMap")]
    npc_character_map: Option<Map<String, Value>>,
    #[serde(default, rename = "nativeMaxDistanceMeters")]
    native_max_distance_meters: Option<f64>,
    #[serde(default, rename = "gameStateRadiusMeters", alias = "gamestateRadiusMeters")]
    gamestate_radius_meters: Option<f64>,
    #[serde(default, rename = "enableActionBooks")]
    enable_action_books: Option<bool>,
    #[serde(default, rename = "actionBookTargetGame")]
    action_book_target_game: Option<String>,
    #[serde(default, rename = "actionBookIds")]
    action_book_ids: Option<Vec<String>>,
    #[serde(default, rename = "nativeActionConfidence")]
    native_action_confidence: Option<f64>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default, rename = "adminCharacterId")]
    admin_character_id: Option<String>,
    #[serde(default, rename = "adminCharacterName")]
    admin_character_name: Option<String>,
    #[serde(default, rename = "adminActionBookLimit")]
    admin_action_book_limit: Option<u64>,
    #[serde(default, rename = "adminSessionId")]
    admin_session_id: Option<String>,
    #[serde(default, rename = "ttsOverrides")]
    tts_overrides: Option<Value>,
    #[serde(default, rename = "speechRecognition", alias = "stt")]
    speech_recognition: Option<Value>,
    #[serde(default, rename = "speechRecognitionTimeoutMs", alias = "sttTimeoutMs")]
    speech_recognition_timeout_ms: Option<u64>,
}

fn trim_trailing_slashes(s: &str) -> String {
    s.trim_end_matches('/').to_string()
}

pub fn load_config(config_path: &Path) -> anyhow::Result<BridgeConfig> {
    let raw: RawConfig = if config_path.as_os_str().is_empty() || !config_path.exists() {
        // No helper config (the default for a fresh install): run on built-in
        // defaults, pointed at the fixed bridge rendezvous dir. We never bail, so a
        // standalone chasm with no nvbridge.config.json still connects to the game.
        RawConfig::default()
    } else {
        let text = std::fs::read_to_string(config_path)
            .map_err(|e| anyhow::anyhow!("reading config {}: {e}", config_path.display()))?;
        serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing config {}: {e}", config_path.display()))?
    };
    Ok(build_config(raw))
}

/// A fully-defaulted bridge config (no helper file), pointed at
/// [`default_bridge_root`]. Used when there is no `nvbridge.config.json`.
pub fn default_config() -> BridgeConfig {
    build_config(RawConfig::default())
}

/// Builds a [`BridgeConfig`] from a (possibly all-default) [`RawConfig`], applying
/// every default and the fixed-rendezvous fallback. Shared by [`load_config`] and
/// [`default_config`] so file-backed and default configs resolve identically.
fn build_config(raw: RawConfig) -> BridgeConfig {
    let mut roots: Vec<PathBuf> = raw
        .native_bridge_roots
        .iter()
        .filter(|r| !r.trim().is_empty())
        .map(PathBuf::from)
        .collect();
    if roots.is_empty() {
        roots.push(default_bridge_root());
    }
    // Drop configured roots whose parent is missing (dead/typo'd paths), but never
    // end up empty — fall back to the fixed rendezvous dir so the bridge always runs.
    roots.retain(|r| r.parent().map(|p| p.exists()).unwrap_or(false));
    if roots.is_empty() {
        roots.push(default_bridge_root());
    }

    // Merge the three character-map aliases, later keys winning (matches Node).
    let mut npc_character_map = Map::new();
    for source in [raw.character_map, raw.npc_characters, raw.npc_character_map]
        .into_iter()
        .flatten()
    {
        for (k, v) in source {
            npc_character_map.insert(k, v);
        }
    }

    BridgeConfig {
        native_bridge_roots: roots,
        poll_ms: raw.poll_ms.unwrap_or(DEFAULT_POLL_MS).max(75),
        api_base: trim_trailing_slashes(&raw.api_base.unwrap_or_else(|| DEFAULT_API_BASE.into())),
        tts_api_base: trim_trailing_slashes(&raw.tts_api_base.unwrap_or_default()),
        stt_api_base: trim_trailing_slashes(&raw.stt_api_base.unwrap_or_default()),
        api_key: raw.api_key.unwrap_or_default(),
        request_timeout_ms: raw.request_timeout_ms.unwrap_or(DEFAULT_API_REQUEST_TIMEOUT_MS),
        live_chat_id: raw.live_chat_id.unwrap_or_else(|| "fnv-goodsprings".into()),
        group_id: raw.group_id.unwrap_or_default(),
        participant_id: raw.participant_id.unwrap_or_else(|| "player".into()),
        character_id: raw.character_id.unwrap_or_else(|| "Easy Pete".into()),
        character_name: raw.character_name.unwrap_or_else(|| "Easy Pete".into()),
        npc_character_map,
        native_max_distance_meters: raw
            .native_max_distance_meters
            .unwrap_or(DEFAULT_NATIVE_MAX_DISTANCE_METERS),
        gamestate_radius_meters: raw
            .gamestate_radius_meters
            .unwrap_or(DEFAULT_GAMESTATE_RADIUS_METERS),
        // Node defaults enableActionBooks to TRUE → structured turns.
        enable_action_books: raw.enable_action_books.unwrap_or(true),
        action_book_target_game: raw
            .action_book_target_game
            .unwrap_or_else(|| DEFAULT_ACTION_BOOK_TARGET_GAME.into()),
        action_book_ids: raw
            .action_book_ids
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec!["Fallout New Vegas Action Book".to_string()]),
        native_action_confidence: raw.native_action_confidence.unwrap_or(0.65),
        model: raw.model.unwrap_or_default(),
        admin_character_id: raw.admin_character_id.unwrap_or_else(|| "Todd".into()),
        admin_character_name: raw.admin_character_name.unwrap_or_else(|| "Todd".into()),
        admin_action_book_limit: raw.admin_action_book_limit.filter(|v| *v > 0).unwrap_or(12),
        admin_session_id: raw.admin_session_id.unwrap_or_default(),
        tts: raw.tts_overrides.unwrap_or_else(|| Value::Object(Map::new())),
        speech_recognition: raw
            .speech_recognition
            .unwrap_or_else(|| Value::Object(Map::new())),
        speech_recognition_timeout_ms: raw
            .speech_recognition_timeout_ms
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_API_STT_TIMEOUT_MS),
        // Off unless the in-process bridge turns it on from AppSettings.music.enabled
        // (the standalone bin has no music engine).
        music_enabled: false,
    }
}

#[cfg(test)]
pub(crate) fn load_test_config() -> BridgeConfig {
    BridgeConfig {
        native_bridge_roots: Vec::new(),
        poll_ms: 100,
        api_base: DEFAULT_API_BASE.into(),
        tts_api_base: String::new(),
        stt_api_base: String::new(),
        api_key: String::new(),
        request_timeout_ms: DEFAULT_API_REQUEST_TIMEOUT_MS,
        live_chat_id: "fnv-goodsprings".into(),
        group_id: "fnv-goodsprings".into(),
        participant_id: "player".into(),
        character_id: "Easy Pete".into(),
        character_name: "Easy Pete".into(),
        npc_character_map: Map::new(),
        native_max_distance_meters: DEFAULT_NATIVE_MAX_DISTANCE_METERS,
        gamestate_radius_meters: DEFAULT_GAMESTATE_RADIUS_METERS,
        enable_action_books: true,
        action_book_target_game: DEFAULT_ACTION_BOOK_TARGET_GAME.into(),
        action_book_ids: vec!["Fallout New Vegas Action Book".into()],
        native_action_confidence: 0.65,
        model: String::new(),
        admin_character_id: "Todd".into(),
        admin_character_name: "Todd".into(),
        admin_action_book_limit: 12,
        admin_session_id: String::new(),
        tts: Value::Object(Map::new()),
        speech_recognition: Value::Object(Map::new()),
        speech_recognition_timeout_ms: DEFAULT_API_STT_TIMEOUT_MS,
        music_enabled: false,
    }
}
