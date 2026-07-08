use std::{
    env, fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

mod game_launcher;
pub mod hotkeys;
pub mod profile_import;
mod profiles;
mod providers;
mod request_trace;
mod settings;
mod system_info;
pub use game_launcher::*;
pub use providers::*;
pub use profile_import::{
    import_bundle, import_from_source_root, ImportAction, ImportOutcome, ALLOWLIST, DENYLIST,
};
pub use profiles::*;
pub use request_trace::*;
pub use settings::*;
pub use system_info::*;

pub const DEFAULT_BIND_ADDR: &str = "127.0.0.1:7341";

/// Legacy default STT endpoint on the LLM port. No managed engine serves this
/// any more (the managed local STT is the dedicated Parakeet server on :5003 —
/// see [`DEFAULT_PARAKEET_STT_ENDPOINT`]); kept only as the `stt_endpoint`
/// fallback value and overridable via `CHASM_STT_ENDPOINT`.
pub const DEFAULT_STT_ENDPOINT: &str = "http://127.0.0.1:5001/v1/audio/transcriptions";

/// Default local LLM endpoint — the managed llama.cpp `llama-server`
/// (OpenAI-compatible). The chat endpoint is `{llm_endpoint}/v1/chat/completions`.
/// STT moved to the dedicated Parakeet service and TTS to faster-qwen3-tts (see
/// [`DEFAULT_TTS_ENDPOINT`]). Env-overridable via `CHASM_LLM_ENDPOINT`.
pub const DEFAULT_LLM_ENDPOINT: &str = "http://127.0.0.1:5001";

/// Default endpoint of the dedicated faster-qwen3-tts streaming TTS service
/// (OpenAI-compatible; the speech endpoint is `{tts_endpoint}/v1/audio/speech`).
/// Separate from the LLM runtime so TTS can stream frame-level at low latency
/// while llama.cpp handles the LLM. Env-overridable via `CHASM_TTS_ENDPOINT`.
pub const DEFAULT_TTS_ENDPOINT: &str = "http://127.0.0.1:5002";

/// Default endpoint of the dedicated Parakeet STT service (OpenAI-compatible
/// `/v1/audio/transcriptions`), used when the STT provider is `parakeet`. Its
/// own process + port so voice input never queues behind an LLM generation.
/// Env-overridable via `CHASM_PARAKEET_STT_ENDPOINT`.
pub const DEFAULT_PARAKEET_STT_ENDPOINT: &str =
    "http://127.0.0.1:5003/v1/audio/transcriptions";

/// Default endpoint of the ACE-Step music-generation service (its own process +
/// port so a multi-second song render never queues behind an LLM/STT/TTS call).
/// POST lyrics + style tags + duration -> WAV. Env-overridable via
/// `CHASM_ACESTEP_ENDPOINT`.
pub const DEFAULT_ACESTEP_ENDPOINT: &str = "http://127.0.0.1:5004/v1/music";

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub bind_addr: String,
    pub data_root: PathBuf,
    pub workspace_root: PathBuf,
    pub settings_path: PathBuf,
    pub engines_dir: PathBuf,
    pub profiles_dir: PathBuf,
    pub voices_dir: PathBuf,
    /// Directory holding downloaded local LLM GGUFs. Env-overridable via
    /// `CHASM_LLM_MODELS_DIR`, else `<data_root>/models/llm`.
    pub llm_models_dir: PathBuf,
    /// Legacy OpenAI-compatible STT transcription endpoint on the LLM port. No
    /// managed engine serves it now (managed STT is Parakeet on :5003); kept as a
    /// fallback / override seam (`CHASM_STT_ENDPOINT`).
    pub stt_endpoint: String,
    /// Local OpenAI-compatible Parakeet transcription endpoint — the managed-local
    /// STT provider. Defaults to
    /// [`DEFAULT_PARAKEET_STT_ENDPOINT`]; override via `CHASM_PARAKEET_STT_ENDPOINT`.
    pub parakeet_stt_endpoint: String,
    /// Base URL of the local OpenAI-compatible LLM (llama.cpp). The chat
    /// endpoint is `{llm_endpoint}/v1/chat/completions`.
    pub llm_endpoint: String,
    /// Base URL of the dedicated faster-qwen3-tts streaming TTS service. The
    /// speech endpoint is `{tts_endpoint}/v1/audio/speech`. Defaults to
    /// [`DEFAULT_TTS_ENDPOINT`]; override via `CHASM_TTS_ENDPOINT`.
    pub tts_endpoint: String,
}

impl AppConfig {
    pub fn from_env() -> Self {
        let workspace_root = discover_workspace_root();
        let data_root = env::var_os("CHASM_DATA_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace_root.join("data").join("default-user"));
        let settings_path = env::var_os("CHASM_SETTINGS")
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace_root.join("chasm.settings.json"));
        migrate_legacy_settings(&settings_path);
        let engines_dir = env::var_os("CHASM_ENGINES_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace_root.join("engines"));
        let profiles_dir = env::var_os("CHASM_PROFILES_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace_root.join("profiles"));
        let voices_dir = env::var_os("CHASM_VOICES_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace_root.join("voices"));
        let llm_models_dir = env::var_os("CHASM_LLM_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| data_root.join("models").join("llm"));
        let bind_addr =
            env::var("CHASM_BIND_ADDR").unwrap_or_else(|_| DEFAULT_BIND_ADDR.to_string());
        let stt_endpoint = env::var("CHASM_STT_ENDPOINT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_STT_ENDPOINT.to_string());
        let parakeet_stt_endpoint = env::var("CHASM_PARAKEET_STT_ENDPOINT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_PARAKEET_STT_ENDPOINT.to_string());
        let llm_endpoint = env::var("CHASM_LLM_ENDPOINT")
            .ok()
            .map(|value| value.trim_end_matches('/').to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_LLM_ENDPOINT.to_string());
        let tts_endpoint = env::var("CHASM_TTS_ENDPOINT")
            .ok()
            .map(|value| value.trim_end_matches('/').to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_TTS_ENDPOINT.to_string());

        Self {
            bind_addr,
            data_root,
            workspace_root,
            settings_path,
            engines_dir,
            profiles_dir,
            voices_dir,
            llm_models_dir,
            stt_endpoint,
            parakeet_stt_endpoint,
            llm_endpoint,
            tts_endpoint,
        }
    }

    /// The legacy embed-cache directory (before per-profile scoping):
    /// `CHASM_EMBED_DIR` when set, else `{data_root}/embed-cache`. Used as
    /// the fallback base for [`ProfilePaths::embed_cache_dir`].
    pub fn legacy_embed_cache_dir(&self) -> PathBuf {
        env::var_os("CHASM_EMBED_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| self.data_root.join("embed-cache"))
    }

    /// Builds a [`ProfilePaths`] resolver for an explicit profile `id` (empty =
    /// legacy/global), wiring in this config's legacy bases.
    pub fn profile_paths(&self, profile_id: &str) -> ProfilePaths {
        ProfilePaths::new(
            &self.profiles_dir,
            profile_id,
            &self.data_root,
            &self.voices_dir,
            &self.legacy_embed_cache_dir(),
        )
    }

    /// Builds a [`ProfilePaths`] resolver for the active profile resolved from
    /// the persisted settings at [`Self::settings_path`]. Reads settings each
    /// call so a profile switch takes effect without restarting.
    pub fn active_profile_paths(&self) -> ProfilePaths {
        let settings = AppSettings::load(&self.settings_path);
        let id = settings.active_profile_id(&self.profiles_dir);
        self.profile_paths(&id)
    }
}

/// One-time settings migration from the pre-rename `sillybridge.settings.json`
/// to `chasm.settings.json`. If the new file does not yet exist but a legacy
/// file sits beside it, copy the legacy contents across so an existing user
/// keeps their settings. Best-effort: any IO error is ignored (the app then
/// just starts from defaults, exactly as it would for a fresh install).
fn migrate_legacy_settings(settings_path: &Path) {
    if settings_path.exists() {
        return;
    }
    let Some(file_name) = settings_path.file_name().and_then(|n| n.to_str()) else {
        return;
    };
    // Only auto-migrate for the default filename; a custom CHASM_SETTINGS path
    // is taken at face value.
    if file_name != "chasm.settings.json" {
        return;
    }
    let legacy = settings_path.with_file_name("sillybridge.settings.json");
    if legacy.is_file() {
        let _ = fs::copy(&legacy, settings_path);
    }
}

/// Finds the workspace root: the directory holding the top-level `Cargo.toml`
/// and `static/`. Honors a `CHASM_ROOT` override, otherwise walks up from
/// the current directory, and finally falls back to the compile-time crate
/// location so `cargo run` works from anywhere in the tree.
///
/// The result is always run through [`strip_verbatim_prefix`]: the packaged
/// desktop shell sets `CHASM_ROOT` from Tauri's `resource_dir()`, which on
/// Windows is an extended-length `\\?\C:\…` verbatim path. `Path::exists()`
/// accepts that form, but tools we spawn against workspace-derived paths do
/// not — `powershell -File \\?\C:\…\install-engine.ps1`, for one, exits
/// without running the script (which stranded TTS/engine installs on an
/// eternal "Installing…"). Everything downstream (engines dir, bundled
/// scripts, launcher exe paths) derives from this, so normalizing here fixes
/// every spawn at once.
fn discover_workspace_root() -> PathBuf {
    let root = if let Some(root) = env::var_os("CHASM_ROOT") {
        PathBuf::from(root)
    } else {
        let start = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        start
            .ancestors()
            .find(|c| c.join("Cargo.toml").is_file() && c.join("static").is_dir())
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| {
                // crates/chasm-core -> crates -> workspace root.
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .and_then(std::path::Path::parent)
                    .map(PathBuf::from)
                    .unwrap_or(start)
            })
    };
    strip_verbatim_prefix(root)
}

/// Strips a Windows `\\?\` extended-length (verbatim) prefix so the path is the
/// plain `C:\…` form that `CreateProcess`/`powershell -File`/`cmd` accept.
/// `\\?\C:\x` → `C:\x`; `\\?\UNC\server\share` → `\\server\share`. Non-verbatim
/// paths (and all non-Windows paths) are returned unchanged.
pub fn strip_verbatim_prefix(path: PathBuf) -> PathBuf {
    let Some(s) = path.to_str() else {
        return path;
    };
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        // Only strip for a real drive path (`C:\…`); leave anything exotic as-is.
        let bytes = rest.as_bytes();
        if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            return PathBuf::from(rest);
        }
    }
    path
}

/// chasm's fixed, machine-independent per-user home: `%LOCALAPPDATA%\chasm`
/// (override with `CHASM_HOME`). Used for the bridge rendezvous directory and the
/// last-ditch managed model directories, so a separately-installed chasm and an
/// MO2-managed game plugin agree on the same real paths without any configuration.
/// Falls back to `%APPDATA%` then the system temp dir on the (rare) machine with no
/// `LOCALAPPDATA`, so it never panics.
pub fn chasm_home() -> PathBuf {
    if let Some(dir) = env::var_os("CHASM_HOME") {
        let path = PathBuf::from(dir);
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    env::var_os("LOCALAPPDATA")
        .or_else(|| env::var_os("APPDATA"))
        .map(PathBuf::from)
        .unwrap_or_else(env::temp_dir)
        .join("chasm")
}

/// The fixed rendezvous directory both chasm and the in-game plugin meet at:
/// `%LOCALAPPDATA%\chasm\bridge` (override with `CHASM_BRIDGE_ROOT`). This is the
/// linchpin that makes a fresh install connect: the plugin (running under Mod
/// Organizer 2) and a separately-installed chasm both compute this identical
/// absolute path, and it sits *outside* MO2's virtual filesystem — so the plugin's
/// heartbeat/trace/turn files land on the real disk exactly where chasm reads them.
/// Holds `runtime_heartbeat.json`, `traces/`, `control/`, and (in file transport)
/// the turn request/reply/chunk files.
pub fn default_bridge_root() -> PathBuf {
    if let Some(dir) = env::var_os("CHASM_BRIDGE_ROOT") {
        let path = PathBuf::from(dir);
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    chasm_home().join("bridge")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantView {
    pub id: String,
    pub name: String,
    pub initial: String,
    pub kind: String,
    pub character_id: Option<String>,
    pub present: bool,
    pub audible: bool,
    pub distance: Option<f64>,
    pub distance_label: String,
    pub message_count: usize,
    pub selected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveChatView {
    pub id: String,
    pub title: String,
    pub participants: Vec<ParticipantView>,
    pub selected_participant_id: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageView {
    pub id: String,
    pub role: String,
    pub speaker_participant_id: Option<String>,
    pub speaker_name: String,
    pub speaker_initial: String,
    pub content: String,
    pub created_at: Option<String>,
    pub created_at_label: String,
    pub segment_id: Option<String>,
    pub location: Option<String>,
    pub audible_to: Vec<String>,
    pub visible_reason: String,
    /// Lore/quest/action entries injected into the prompt that produced THIS
    /// message, recorded at generation time (read from `extra.chasm`).
    /// `None` for the player's own messages and any message persisted before this
    /// feature existed (rendered as "no data recorded").
    pub injected: Option<InjectedView>,
    /// The structured actions the NPC chose this turn — the "actions executed"
    /// for v1. Empty for player turns and pre-feature messages.
    pub turn_actions: Vec<ActionView>,
    /// Whether this NPC turn was generated while the NPC was in combat (read from
    /// `extra.chasm.in_combat`). `false` for player turns and pre-feature messages.
    #[serde(default)]
    pub in_combat: bool,
    /// Display names of who the NPC was fighting this turn (from
    /// `extra.chasm.combat_with`). Empty unless `in_combat`.
    #[serde(default)]
    pub combat_with: Vec<String>,
    /// True for an interstitial speech FRAGMENT the NPC said mid-loop (before a
    /// tool result), persisted so the chat shows it where he said it rather than
    /// dumped after the world beats by `finalize_turn`. The turn's context strip
    /// (lore / quests / executed actions) rides the canonical turn message, so a
    /// fragment records no context and must NOT show the "no turn context
    /// recorded" note. `false` for full turns and pre-feature messages.
    #[serde(default)]
    pub interstitial: bool,
    /// `true` for witnessed-event narration lines the event-log fan-out inserts
    /// into an NPC's history (from `extra.chasm.witnessed`), so the chat UI can
    /// render them as dim narration instead of spoken dialogue.
    #[serde(default)]
    pub witnessed: bool,
    /// True for an EPHEMERAL world line — a point-in-time READ (a room search, an
    /// inventory scan, a `find_action` listing) persisted so it shows in the chat
    /// UI, but filtered out of the model's prompt on every LATER turn: by then it
    /// is stale (the item he "is holding" may already be gone). Only real action
    /// beats persist across turns. `false` for everything else.
    #[serde(default)]
    pub ephemeral: bool,
}

/// One lore/quest/action entry that was injected into a single turn's prompt,
/// with just enough identity to display: which source it came from, a stable id,
/// a human title, and why it activated (constant / keyword / vector).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectedEntryView {
    /// `lore`, `quest`, or `action`.
    pub source: String,
    /// Stable identity for the entry (lore: comment/index; quest: quest id;
    /// action: action id). Display-only; not guaranteed globally unique.
    pub id: String,
    /// Human label (lore comment, quest name/title, action alias/title).
    pub title: String,
    /// Activation reason: `constant`, `keyword`, or `vector`.
    pub reason: String,
}

/// The trusted execution spec for one activated action, relayed to the FNV helper
/// (via the turn's `metadata.activatedActions`) so it can build the native command
/// for non-native actions. Serialized camelCase to match the helper's reader
/// (`normalizeActivatedActionId` reads `actionId`; it reads `binding`/`execution`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivatedActionView {
    pub action_id: String,
    pub alias: String,
    /// The action's `binding` object (e.g. `{ "engine": "fallout-new-vegas:xnvse" }`).
    pub binding: serde_json::Value,
    /// The action's `execution` object (trusted GECK `script` + `arguments`).
    pub execution: serde_json::Value,
    /// True when the action needs a target name (player or NPC).
    pub requires_target: bool,
    /// Resolved scoped-catalog candidates (e.g. spawnable entities matched to the
    /// player's request). The helper resolves the chosen entity/item to its FormID
    /// from `items[].metadata.formId` (`findScopedCatalogItem`). Empty for actions
    /// without a scoped catalog.
    #[serde(default)]
    pub scoped_catalogs: Vec<ScopedCatalogView>,
}

/// One scoped catalog's resolved candidate items, relayed to the helper so it can
/// map the action's chosen `parameter_name` value to a catalog item (and its
/// FormID). Serialized camelCase to match the helper's `catalog.catalogId` /
/// `catalog.items` reader.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopedCatalogView {
    pub catalog_id: String,
    pub parameter_name: String,
    pub items: Vec<CatalogItemView>,
}

/// One catalog candidate. The helper matches by `id` / `name` / `aliases` /
/// `metadata.editorId` / `metadata.fullName` (`getCatalogCandidateLookupKeys`) and
/// reads `metadata.formId` for the spawn (`resolveTrustedExecutionArgument`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogItemView {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    /// Raw catalog metadata (carries `formId`, `editorId`, `fullName`, …).
    pub metadata: serde_json::Value,
}

/// The grouped set of injected entries for one message, split by source so the
/// panel can render lore / quests / actions under their own headings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InjectedView {
    pub lore: Vec<InjectedEntryView>,
    pub quests: Vec<InjectedEntryView>,
    pub actions: Vec<InjectedEntryView>,
    /// Full trusted specs for the activated actions (script/binding), relayed to
    /// the helper out-of-band. Kept off the panel JSON (it carries GECK scripts).
    #[serde(default, skip_serializing)]
    pub activated_actions: Vec<ActivatedActionView>,
}

impl InjectedView {
    /// True when no entry of any source was injected — the panel shows a
    /// "nothing injected this turn" note rather than three empty groups.
    pub fn is_empty(&self) -> bool {
        self.lore.is_empty() && self.quests.is_empty() && self.actions.is_empty()
    }
}

/// One structured action the NPC chose this turn, flattened for display: the
/// canonical id, the (best-effort) alias, an optional target, a compact JSON
/// rendering of any parameters, and the model's stated reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionView {
    pub id: String,
    pub alias: String,
    pub target: String,
    /// Parameters serialized to a compact JSON string (empty / `{}` omitted by
    /// the template), so the view stays a plain struct with no nested `Value`.
    pub params: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveChatPage {
    pub live_chat: LiveChatView,
    pub selected_participant: Option<ParticipantView>,
    pub messages: Vec<MessageView>,
    pub data_root: String,
}

/// One labeled piece of the assembled prompt, in the order it is sent to the model.
///
/// Mirrors the components that `src/headless/generation.js` builds: the ordered
/// `systemParts`, then the chat history messages, then the pending user turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptComponentView {
    /// 1-based position in the full send order.
    pub order: usize,
    /// Coarse grouping for the UI: `system`, `history`, or `input`.
    pub group: String,
    /// Stable identifier, e.g. `system_prompt`, `lore`, `history_3`.
    pub key: String,
    /// Human label, e.g. `System prompt`, `Activated lore`, or a speaker name.
    pub label: String,
    /// Chat-completion role this maps to: `system`, `user`, or `assistant`.
    pub role: String,
    /// Whether the part is actually present: `included`, `empty`,
    /// `generation-time` (supplied per request), or `unavailable`.
    pub status: String,
    /// Explanation for non-included or approximated parts.
    pub note: String,
    pub content: String,
    pub char_count: usize,
}

/// The full prompt for a live chat + participant, broken into ordered components.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptAssemblyView {
    pub participant_id: String,
    pub participant_name: String,
    pub character_id: Option<String>,
    pub character_found: bool,
    pub system_char_count: usize,
    pub history_count: usize,
    pub total_char_count: usize,
    pub components: Vec<PromptComponentView>,
    /// Top-level caveats about parity gaps (e.g. vector activation).
    pub notes: Vec<String>,
}

/// Formats a chat message's raw ISO-8601 `send_date` into a human-friendly
/// label like `Jun 20, 2026 · 9:28 PM` (date + 12-hour time).
///
/// Parses the fixed form SillyTavern emits (`YYYY-MM-DDTHH:MM:SS(.sss)?Z`),
/// also tolerating a missing trailing `Z` and/or fractional seconds. The value
/// is treated as-is (no timezone conversion). On any parse failure the original
/// string is returned unchanged, so a real value is never replaced with an
/// error or a blank. This is display-only and dependency-free.
pub fn format_message_timestamp(raw: &str) -> String {
    parse_iso_timestamp(raw).unwrap_or_else(|| raw.to_string())
}

const MONTH_ABBREVS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

fn parse_iso_timestamp(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    // Split on 'T' (or a space) into the date and time halves.
    let (date_part, time_part) = trimmed
        .split_once('T')
        .or_else(|| trimmed.split_once(' '))?;

    // Date: YYYY-MM-DD.
    let mut date_iter = date_part.split('-');
    let year: i32 = date_iter.next()?.parse().ok()?;
    let month: u32 = date_iter.next()?.parse().ok()?;
    let day: u32 = date_iter.next()?.parse().ok()?;
    if date_iter.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    // Time: HH:MM:SS, optionally followed by ".sss" and/or a trailing "Z".
    let time_core = time_part
        .strip_suffix('Z')
        .or_else(|| time_part.strip_suffix('z'))
        .unwrap_or(time_part);
    let time_core = time_core.split('.').next().unwrap_or(time_core);
    let mut time_iter = time_core.split(':');
    let hour24: u32 = time_iter.next()?.parse().ok()?;
    let minute: u32 = time_iter.next()?.parse().ok()?;
    // Seconds are optional in the readable label but validated if present.
    if let Some(sec) = time_iter.next() {
        let _sec: u32 = sec.parse().ok()?;
    }
    if hour24 > 23 || minute > 59 {
        return None;
    }

    let month_name = MONTH_ABBREVS[(month - 1) as usize];
    let (hour12, meridiem) = match hour24 {
        0 => (12, "AM"),
        1..=11 => (hour24, "AM"),
        12 => (12, "PM"),
        _ => (hour24 - 12, "PM"),
    };

    Some(format!(
        "{month_name} {day}, {year} \u{b7} {hour12}:{minute:02} {meridiem}"
    ))
}

#[cfg(test)]
mod timestamp_tests {
    use super::format_message_timestamp;

    #[test]
    fn formats_full_iso_with_fractional_seconds() {
        assert_eq!(
            format_message_timestamp("2026-06-20T21:28:27.700Z"),
            "Jun 20, 2026 \u{b7} 9:28 PM"
        );
    }

    #[test]
    fn formats_morning_and_midnight_and_noon() {
        assert_eq!(
            format_message_timestamp("2026-06-20T09:05:00.000Z"),
            "Jun 20, 2026 \u{b7} 9:05 AM"
        );
        assert_eq!(
            format_message_timestamp("2026-01-01T00:00:00Z"),
            "Jan 1, 2026 \u{b7} 12:00 AM"
        );
        assert_eq!(
            format_message_timestamp("2026-12-31T12:00:00Z"),
            "Dec 31, 2026 \u{b7} 12:00 PM"
        );
    }

    #[test]
    fn tolerates_missing_z_and_fraction() {
        assert_eq!(
            format_message_timestamp("2026-06-20T21:28:27"),
            "Jun 20, 2026 \u{b7} 9:28 PM"
        );
    }

    #[test]
    fn falls_back_on_garbage_or_empty() {
        assert_eq!(format_message_timestamp(""), "");
        assert_eq!(format_message_timestamp("not a date"), "not a date");
        assert_eq!(
            format_message_timestamp("2026-13-99T99:99:99Z"),
            "2026-13-99T99:99:99Z"
        );
    }
}
