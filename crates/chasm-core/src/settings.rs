//! App settings (LLM / TTS / STT) plus the TTS reference data mirrored from the
//! SillyTavern TTS extension (`public/scripts/extensions/tts`): the provider
//! list, the audio-tag profile options, and the per-provider audio-tag prompts.
//!
//! Settings persist to a small JSON file; nothing here is wired to an actual
//! TTS/LLM/STT engine yet.

use std::{collections::HashMap, fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::system_info::{recommended_index, GpuFit, SystemInfo};

// ---------------------------------------------------------------------------
// Persisted settings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    /// Active game profile id (e.g. `fallout-new-vegas`). Empty = no profile.
    pub profile: String,
    pub llm: LlmSettings,
    pub tts: TtsSettings,
    pub stt: SttSettings,
    pub retrieval: RetrievalSettings,
    /// Game launcher (Mod Organizer 2 + headless FNV launch) overrides.
    pub launcher: LauncherSettings,
    /// Per-request tracing (the Tracing settings page).
    pub tracing: TracingSettings,
    /// UI appearance (the Interface settings page). Emitted as a dynamic
    /// `/theme.css` stylesheet, so every field is genuinely wired to CSS.
    pub interface: InterfaceSettings,
    /// Player-persona generation (the mod's stealth capture → vision/stats LLM
    /// description shown on the Persona page and injected into NPC prompts).
    pub persona: PersonaSettings,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            profile: "fallout-new-vegas".to_string(),
            llm: LlmSettings::default(),
            tts: TtsSettings::default(),
            stt: SttSettings::default(),
            retrieval: RetrievalSettings::default(),
            launcher: LauncherSettings::default(),
            tracing: TracingSettings::default(),
            interface: InterfaceSettings::default(),
            persona: PersonaSettings::default(),
        }
    }
}

/// Default hard cap on the stored persona description, in characters. The
/// generation prompt demands a single ~100-word paragraph (~700 chars); this
/// is a safety net above that, not the primary length control.
pub const PERSONA_MAX_CHARS_DEFAULT: u32 = 1400;

/// Player-persona generation settings. The FNV mod uploads a stealth capture
/// (front screenshot + stats snapshot); chasm-web's persona module turns it
/// into a SillyTavern-style user-persona description via a vision-capable LLM
/// when one is reachable, else a stats-only text generation.
///
/// `#[serde(default)]` so older settings files (no `persona` key) load fine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PersonaSettings {
    /// Master switch: when off, incoming captures are still stored (image +
    /// stats visible on the Persona page) but no LLM generation runs and no
    /// description is injected into prompts.
    pub enabled: bool,
    /// Optional SEPARATE vision-capable OpenAI-compatible endpoint base URL
    /// (the `/v1/chat/completions` suffix is appended, mirroring the main LLM
    /// client). Blank = try the main LLM endpoint with the image first. Set
    /// this when the main model has no multimodal projector.
    pub vision_endpoint: String,
    /// Optional model id sent to the vision endpoint (blank = the endpoint's
    /// first advertised `/v1/models` entry / server default).
    pub vision_model: String,
    /// Optional bearer token for the vision endpoint (`Authorization: Bearer
    /// <key>`). Stored as-is; never logged.
    pub vision_api_key: String,
    /// Hard cap on the stored persona description length, in characters
    /// (truncated at a word boundary). 0 = use the default.
    pub max_chars: u32,
}

impl Default for PersonaSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            vision_endpoint: String::new(),
            vision_model: String::new(),
            vision_api_key: String::new(),
            max_chars: PERSONA_MAX_CHARS_DEFAULT,
        }
    }
}

impl PersonaSettings {
    /// The effective description cap (the default when the stored value is 0).
    pub fn effective_max_chars(&self) -> usize {
        if self.max_chars == 0 {
            PERSONA_MAX_CHARS_DEFAULT as usize
        } else {
            self.max_chars as usize
        }
    }
}

/// UI appearance settings, all CSS-expressible. Rendered into the dynamic
/// `/theme.css` document (a small set of `:root{}` overrides + a few helper
/// rules) that the layout links AFTER `app.css`, so each field demonstrably
/// changes the rendered UI on the next page load — no per-page threading.
///
/// `#[serde(default)]` so older settings files (no `interface` key) load fine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InterfaceSettings {
    /// Theme preset id (`"midnight" | "slate" | "ocean"`). Sets the base palette;
    /// the accent override below still applies on top.
    pub theme: String,
    /// Accent colour, a CSS hex like `#55a7ff`. Overrides `--accent`.
    pub accent: String,
    /// UI density (`"comfortable" | "compact"`). Drives the `--pad`/`--gap`
    /// spacing vars + compact paddings.
    pub density: String,
    /// Base font scale in percent (90..=120). Emitted as `html{font-size}`.
    pub font_scale: u32,
    /// When on, disables CSS transitions/animations app-wide.
    pub reduce_motion: bool,
    /// When off, message timestamps are hidden via CSS.
    pub show_timestamps: bool,
    /// When off, the right-hand prompt-inspector column is collapsed via CSS.
    pub show_prompt_panel: bool,
}

impl Default for InterfaceSettings {
    fn default() -> Self {
        Self {
            theme: INTERFACE_DEFAULT_THEME.to_string(),
            accent: INTERFACE_DEFAULT_ACCENT.to_string(),
            density: INTERFACE_DEFAULT_DENSITY.to_string(),
            font_scale: INTERFACE_FONT_SCALE_DEFAULT,
            reduce_motion: false,
            show_timestamps: true,
            show_prompt_panel: true,
        }
    }
}

/// Settings for the Tracing page. `trace_dir` overrides where per-request trace
/// JSONL files are read from; blank means "auto-discover" (the web layer reads
/// the helper config JSON's `nativeBridgeRoots[0]` + `/traces`, falling back to
/// the known FNV overwrite path). Mirrors how `LauncherSettings` persists.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TracingSettings {
    /// Override for the traces directory (blank = auto-discover from the helper
    /// config / fallback path).
    pub trace_dir: String,
}

/// Persisted overrides for the game launcher. Every field is optional: a blank
/// value means "auto-detect" (the `game_launcher` module resolves the MO2 exe,
/// the instance under `%LOCALAPPDATA%\ModOrganizer`, the profile/executable, and
/// the game dir from MO2's `ModOrganizer.ini` / the Steam default). Mirrors how
/// `LlmSettings` / `RetrievalSettings` persist as JSON with serde defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LauncherSettings {
    /// Override for `ModOrganizer.exe`'s path (blank = default path / PATH).
    pub mo2_exe: String,
    /// Override for the MO2 instance name (blank = the single instance found).
    pub instance: String,
    /// Override for the MO2 profile (blank = `Default`).
    pub profile: String,
    /// Override for the MO2 launch-executable title (blank = `NVSE`).
    pub executable: String,
    /// Override for the base game dir (blank = MO2 ini / Steam default).
    pub game_dir: String,
    /// FNV bridge helper — node.exe path (blank = built-in default / PATH).
    /// Legacy: the bridge now runs in-process inside chasm. When set, Play also
    /// starts this Node helper, which spawns the local runtimes per its own config.
    pub helper_node: String,
    /// Path to the helper script `nvbridge-helper.mjs` (blank = built-in default).
    /// If the resolved path does not exist, Play skips the AI-stack start.
    pub helper_script: String,
    /// Path to the helper config json (blank = built-in default).
    pub helper_config: String,
    /// Working directory for the helper (blank = the helper script's folder).
    pub helper_cwd: String,
    /// Optional Nexus Mods personal API key, used by the auto-setup to download
    /// Nexus-hosted mods (JIP LN, NVTF) that have no public GitHub release. Blank
    /// = those mods are left for the user to install manually (a "Get" link is
    /// shown instead). Stored as-is; never logged. NOTE: programmatic Nexus
    /// downloads via this key require a Nexus **Premium** account — free keys can
    /// authenticate the API but not generate direct download links.
    pub nexus_api_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmSettings {
    pub provider: String,
    pub model: String,
    /// Per-request generation sampling sent to the local llama.cpp
    /// (OpenAI-compatible) server on every NPC / admin turn. Defaults match the
    /// previous hard-coded behaviour so nothing changes until the user tweaks.
    pub sampling: LlmSamplingSettings,
    /// Live chat orchestrator: when off, the first eligible NPC always speaks
    /// (no director LLM call). When on, multi-NPC scenes get one director call.
    pub orchestrator_enabled: bool,
    /// Max speakers the director may pick for one turn (clamped 1..=10 on read).
    pub orchestrator_max_speakers: u32,
    /// Director sampling temperature (clamped 0.0..=2.0 on read).
    pub orchestrator_temperature: f32,
    /// Editable director system prompt (blank → the built-in default).
    pub orchestrator_system_prompt: String,
}

/// Per-request LLM sampling knobs forwarded to the local llama.cpp
/// OpenAI-compatible `/v1/chat/completions` endpoint. Only fields that endpoint
/// actually honours in the request body are exposed here (verified against
/// llama.cpp's OpenAI server: `temperature`, `top_p`, `top_k`, `min_p`,
/// `repeat_penalty`, `max_tokens`/`n_predict`, `seed`, and `n_ctx` as a runtime
/// context hint). Each is wired through `request_body` in `chasm-web`'s
/// `llm.rs`, so a change takes effect on the very next turn (settings are read
/// fresh per request) with no restart.
///
/// `#[serde(default)]` so older settings files (no `sampling` key) still load,
/// and every field has a serde default so a partial object fills the rest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmSamplingSettings {
    /// Sampling temperature. Default 0.7 (the prior hard-coded value).
    pub temperature: f32,
    /// Nucleus sampling cutoff. Default 1.0 (llama.cpp default; disabled).
    pub top_p: f32,
    /// Top-k cutoff. `0` = disabled (llama.cpp default 40, but we send the value
    /// only when > 0 so the default behaviour is unchanged until set).
    pub top_k: u32,
    /// Min-p cutoff. Default 0.0 (disabled); sent only when > 0.
    pub min_p: f32,
    /// Repetition penalty. Default 1.0 (off / llama.cpp default).
    pub repeat_penalty: f32,
    /// Max tokens to generate (`max_tokens` / `n_predict`). `0` = no limit
    /// (the server decides). Default 0 (unchanged from prior behaviour).
    pub max_tokens: u32,
    /// Context window hint (`n_ctx`). `0` = use the loaded model's default. Sent
    /// only when > 0. Note: llama.cpp's server fixes context at load; this is a
    /// best-effort per-request hint honoured by servers that support it.
    pub n_ctx: u32,
    /// RNG seed. `-1` = random each call (llama.cpp default). Sent only when >= 0.
    pub seed: i64,
}

// --- LLM sampling defaults + ranges (see [`LlmSamplingSettings`]) ------------
pub const LLM_TEMPERATURE_MIN: f32 = 0.0;
pub const LLM_TEMPERATURE_MAX: f32 = 2.0;
pub const LLM_TEMPERATURE_STEP: f32 = 0.05;
pub const LLM_TEMPERATURE_DEFAULT: f32 = 0.7;

pub const LLM_TOP_P_MIN: f32 = 0.0;
pub const LLM_TOP_P_MAX: f32 = 1.0;
pub const LLM_TOP_P_STEP: f32 = 0.01;
pub const LLM_TOP_P_DEFAULT: f32 = 1.0;

pub const LLM_TOP_K_MIN: u32 = 0;
pub const LLM_TOP_K_MAX: u32 = 200;
pub const LLM_TOP_K_DEFAULT: u32 = 0;

pub const LLM_MIN_P_MIN: f32 = 0.0;
pub const LLM_MIN_P_MAX: f32 = 1.0;
pub const LLM_MIN_P_STEP: f32 = 0.01;
pub const LLM_MIN_P_DEFAULT: f32 = 0.0;

pub const LLM_REPEAT_PENALTY_MIN: f32 = 0.0;
pub const LLM_REPEAT_PENALTY_MAX: f32 = 2.0;
pub const LLM_REPEAT_PENALTY_STEP: f32 = 0.01;
pub const LLM_REPEAT_PENALTY_DEFAULT: f32 = 1.0;

pub const LLM_MAX_TOKENS_MIN: u32 = 0;
pub const LLM_MAX_TOKENS_MAX: u32 = 8_192;
pub const LLM_MAX_TOKENS_DEFAULT: u32 = 0;

pub const LLM_N_CTX_MIN: u32 = 0;
pub const LLM_N_CTX_MAX: u32 = 131_072;
pub const LLM_N_CTX_DEFAULT: u32 = 0;

pub const LLM_SEED_DEFAULT: i64 = -1;

impl Default for LlmSamplingSettings {
    fn default() -> Self {
        Self {
            temperature: LLM_TEMPERATURE_DEFAULT,
            top_p: LLM_TOP_P_DEFAULT,
            top_k: LLM_TOP_K_DEFAULT,
            min_p: LLM_MIN_P_DEFAULT,
            repeat_penalty: LLM_REPEAT_PENALTY_DEFAULT,
            max_tokens: LLM_MAX_TOKENS_DEFAULT,
            n_ctx: LLM_N_CTX_DEFAULT,
            seed: LLM_SEED_DEFAULT,
        }
    }
}

impl LlmSamplingSettings {
    /// Returns a copy with every field clamped/normalized to its documented
    /// range, the way the panel view + the request builder should see it.
    pub fn normalized(&self) -> Self {
        Self {
            temperature: clamp_finite(self.temperature, LLM_TEMPERATURE_MIN, LLM_TEMPERATURE_MAX),
            top_p: clamp_finite(self.top_p, LLM_TOP_P_MIN, LLM_TOP_P_MAX),
            top_k: self.top_k.min(LLM_TOP_K_MAX),
            min_p: clamp_finite(self.min_p, LLM_MIN_P_MIN, LLM_MIN_P_MAX),
            repeat_penalty: clamp_finite(
                self.repeat_penalty,
                LLM_REPEAT_PENALTY_MIN,
                LLM_REPEAT_PENALTY_MAX,
            ),
            max_tokens: self.max_tokens.min(LLM_MAX_TOKENS_MAX),
            n_ctx: self.n_ctx.min(LLM_N_CTX_MAX),
            seed: self.seed.max(-1),
        }
    }
}

/// Defaults for the orchestrator knobs (also the clamp targets).
pub const ORCHESTRATOR_DEFAULT_ENABLED: bool = true;
pub const ORCHESTRATOR_DEFAULT_MAX_SPEAKERS: u32 = 3;
pub const ORCHESTRATOR_DEFAULT_TEMPERATURE: f32 = 0.2;
pub const ORCHESTRATOR_MAX_SPEAKERS_MIN: u32 = 1;
pub const ORCHESTRATOR_MAX_SPEAKERS_MAX: u32 = 10;
pub const ORCHESTRATOR_TEMPERATURE_MIN: f32 = 0.0;
pub const ORCHESTRATOR_TEMPERATURE_MAX: f32 = 2.0;

/// The built-in director system prompt (used as the persisted default and the
/// reset value when the form prompt is left blank).
pub const ORCHESTRATOR_DEFAULT_SYSTEM_PROMPT: &str = "You are the director of a live conversation. You are given the characters currently present and the recent conversation. Decide which characters should speak next and in what order.

Rules:
- Only choose from the listed characters. Never invent characters and never choose the player.
- Usually exactly one character speaks next. Choose more than one only if several would naturally respond, and list them in the order they should speak.
- If no character would naturally speak, return an empty list.
- Reply only with the required JSON object.";

impl Default for LlmSettings {
    fn default() -> Self {
        Self {
            provider: String::new(),
            model: String::new(),
            sampling: LlmSamplingSettings::default(),
            orchestrator_enabled: ORCHESTRATOR_DEFAULT_ENABLED,
            orchestrator_max_speakers: ORCHESTRATOR_DEFAULT_MAX_SPEAKERS,
            orchestrator_temperature: ORCHESTRATOR_DEFAULT_TEMPERATURE,
            orchestrator_system_prompt: ORCHESTRATOR_DEFAULT_SYSTEM_PROMPT.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SttSettings {
    /// Selected STT provider value (matches [`STT_PROVIDERS`]). Only `whisper`
    /// (koboldcpp's local Whisper) is available.
    pub provider: String,
    /// Transcription model sent to koboldcpp — the GGML `.bin` filename of the
    /// active Whisper model (a [`WhisperModel::file`] value). Blank/legacy values
    /// fall back to [`STT_WHISPER_DEFAULT_MODEL`].
    pub model: String,
    /// Optional default language hint (e.g. `en`). Empty = auto.
    pub language: String,
    /// Optional default transcription prompt — a biasing hint forwarded as the
    /// OpenAI `prompt` multipart field when a request doesn't supply its own.
    /// Wired in `speech_recognize`: it reaches koboldcpp's Whisper request form,
    /// so it genuinely affects decoding. Empty = none.
    pub prompt: String,
    /// Default request timeout in ms for a transcription call, used when the
    /// request body omits `timeoutMs`. Wired as the actual reqwest timeout on the
    /// transcription POST. Clamped to [`STT_TIMEOUT_MS_MIN`]..=[`STT_TIMEOUT_MS_MAX`].
    pub timeout_ms: u64,
}

impl Default for SttSettings {
    fn default() -> Self {
        Self {
            provider: STT_DEFAULT_PROVIDER.to_string(),
            // No default Whisper model: the user must download + pick one (see
            // `stt_effective_model`). Empty = "none selected".
            model: String::new(),
            language: String::new(),
            prompt: String::new(),
            timeout_ms: STT_TIMEOUT_MS_DEFAULT,
        }
    }
}

/// STT request-timeout default + clamp range (mirrors the web layer's old
/// hard-coded constants, now the settings default + bounds).
pub const STT_TIMEOUT_MS_MIN: u64 = 1_000;
pub const STT_TIMEOUT_MS_MAX: u64 = 300_000;
pub const STT_TIMEOUT_MS_DEFAULT: u64 = 45_000;

/// Clamps an STT timeout to the documented range.
pub fn normalize_stt_timeout_ms(value: u64) -> u64 {
    value.clamp(STT_TIMEOUT_MS_MIN, STT_TIMEOUT_MS_MAX)
}

/// Semantic-retrieval (embed + rerank) settings. Mirrors the `RetrieverConfig`
/// the `chasm-embed` crate consumes, plus per-source toggles/limits for
/// the phase-2 consumers (chat memory, lore, quests). Persisted as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RetrievalSettings {
    /// Master switch. When off, no retriever is loaded and consumers skip
    /// semantic retrieval entirely.
    pub enabled: bool,
    /// Per-source toggles.
    pub chat_memory_enabled: bool,
    pub lore_semantic_enabled: bool,
    pub action_semantic_enabled: bool,
    pub quest_semantic_enabled: bool,
    /// `"small" | "base" | "quality"`.
    pub embedder_tier: String,
    pub reranker_enabled: bool,
    /// `"small" | "large"`.
    pub reranker_tier: String,
    /// `"cpu" | "gpu"`.
    pub execution: String,
    /// Final results returned to the prompt (after rerank).
    pub top_k: u32,
    /// Recall candidates considered before reranking (top-N).
    pub candidates: u32,
    /// Minimum (rerank) score for a hit to be kept (lore / quest / chat memory).
    pub min_score: f32,
    /// Separate floor for ACTION hits. Action commands score systematically lower
    /// than lore passages on the cross-encoder (a terse command vs. a descriptive
    /// passage), so they need a lower bar or they all get cut. Kept distinct from
    /// `min_score` so lore can stay tight while actions still surface.
    pub action_min_score: f32,
    /// Per-source caps on how many hits each source may contribute.
    pub chat_memory_limit: u32,
    pub lore_limit: u32,
    pub quest_limit: u32,
}

impl Default for RetrievalSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            chat_memory_enabled: true,
            lore_semantic_enabled: true,
            action_semantic_enabled: true,
            quest_semantic_enabled: true,
            embedder_tier: "small".to_string(),
            // Off by default: for small, hand-authored corpora the reranker is
            // overkill (and the small tier's reranker mis-ranks more than it
            // helps). Pure bi-encoder cosine + curated keywords is the lean
            // default; the reranker stays available for big-corpus / GPU users.
            reranker_enabled: false,
            reranker_tier: "small".to_string(),
            execution: "cpu".to_string(),
            top_k: 6,
            candidates: 40,
            min_score: 0.2,
            action_min_score: 0.16,
            chat_memory_limit: 4,
            lore_limit: 4,
            quest_limit: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsSettings {
    /// `"local"` or `"api"`.
    pub mode: String,
    /// Selected local engine value (e.g. `"pockettts"`).
    pub local_engine: String,
    /// Selected API provider value (matches [`TTS_API_PROVIDERS`]).
    pub api_provider: String,
    pub streaming_enabled: bool,
    pub streaming_chunk_ms: u32,
    /// MAX size (ms of audio) of a streamed TTS chunk. The backend ramps slices (a
    /// small first slice for fast first-audio, doubling up to this cap) so the plugin
    /// gets few, big chunk files — bounding its per-frame file I/O. Read fresh per
    /// request. Clamped to [`STREAM_SLICE_MS_MIN`]..=[`STREAM_SLICE_MS_MAX`] via
    /// [`normalize_stream_slice_ms`].
    pub stream_slice_ms: u32,
    /// Max characters per on-screen caption segment (display only; never affects
    /// audio). The FNV plugin splits the NPC's full line into <= this many chars at
    /// word boundaries and reveals the segments in sync with playback. 0 = whole
    /// line in one caption. Clamped via [`normalize_caption_max_chars`].
    pub caption_max_chars: u32,
    /// Linear playback gain for ordinary, directional (in-world) NPC voices.
    /// `1.0` = unchanged. Applied to the synthesized PCM samples, read fresh per
    /// request so a slider move is heard on the next line with no restart.
    /// Clamped via [`normalize_voice_volume`].
    pub npc_volume: f32,
    /// Linear playback gain for the non-positional "admin" voice — spoken straight
    /// into the player's ear with no 3D positioning (e.g. Todd). Separate from
    /// [`Self::npc_volume`] so the two can be balanced independently. `1.0` =
    /// unchanged. Clamped via [`normalize_voice_volume`].
    pub admin_volume: f32,
    pub default_voice: String,
    pub audio_tags: AudioTagsSettings,
    /// Per-request synthesis tuning (silence pads, gain, PocketTTS sampling).
    /// Applied live by the warm worker on every `/synthesize`, so changes take
    /// effect with no worker/app restart. Also what the voice-clone Test button
    /// sends so a tweak can be heard immediately before saving.
    pub tuning: TtsTuningSettings,
}

impl Default for TtsSettings {
    fn default() -> Self {
        Self {
            mode: "local".to_string(),
            // No default engine: the user must pick + install one (see
            // `normalize_local_engine`). Empty = "none selected".
            local_engine: String::new(),
            api_provider: "ElevenLabs".to_string(),
            streaming_enabled: true,
            streaming_chunk_ms: STREAMING_CHUNK_MS_DEFAULT,
            stream_slice_ms: STREAM_SLICE_MS_DEFAULT,
            caption_max_chars: CAPTION_MAX_CHARS_DEFAULT,
            npc_volume: NPC_VOLUME_DEFAULT,
            admin_volume: ADMIN_VOLUME_DEFAULT,
            default_voice: String::new(),
            audio_tags: AudioTagsSettings::default(),
            tuning: TtsTuningSettings::default(),
        }
    }
}

/// Live, per-request synthesis tuning knobs.
///
/// Two groups, both applied by the warm PocketTTS worker per request (no
/// reload):
///   * Post-processing of the rendered audio — silence pads + output gain.
///   * Real PocketTTS generation knobs. The four sampling knobs (`temperature`,
///     `lsd_decode_steps`, `noise_clamp`, `eos_threshold`) are read off the live
///     `TTSModel` instance at every generation step, so the worker mutates them
///     per request; `max_tokens` / `frames_after_eos` are passed straight to
///     `generate_audio(...)`. Only knobs the installed library actually accepts
///     are exposed here (verified against `pocket_tts.TTSModel`).
///
/// `#[serde(default)]` so older settings files (no `tuning` key) load fine, and
/// every field has a serde default so a partial object still fills the rest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsTuningSettings {
    /// Leading silence pad prepended to every line, in ms. Protects the speech
    /// onset from playback-startup clipping (replaces the worker's old hard-coded
    /// `LEAD_IN_SEC = 0.15`). Clamped to [`TUNING_PAD_MS_MIN`]..=[`TUNING_PAD_MS_MAX`].
    pub lead_in_ms: u32,
    /// Trailing silence pad appended to every line, in ms. NPC lines can clip the
    /// end on some playback paths too. Clamped to the same pad range.
    pub trailing_ms: u32,
    /// Silence inserted BETWEEN sentences of a line, in ms. Streaming engines
    /// (PocketTTS) synthesize a line sentence-by-sentence with very short tails, so
    /// without a gap the next sentence starts almost instantly and the delivery
    /// sounds rushed / run-together. Pure inserted silence — no model artifacts.
    /// Clamped to the same pad range.
    pub sentence_gap_ms: u32,
    /// Output gain in dB applied to the rendered samples. `0.0` = unchanged.
    /// Clamped to [`TUNING_GAIN_DB_MIN`]..=[`TUNING_GAIN_DB_MAX`].
    pub gain_db: f32,
    /// PocketTTS sampling temperature (`TTSModel.temp`). Higher = more varied,
    /// potentially less stable. Default 0.7. Clamped to
    /// [`TUNING_TEMPERATURE_MIN`]..=[`TUNING_TEMPERATURE_MAX`].
    pub temperature: f32,
    /// PocketTTS LSD (Lagrangian self-distillation) decode steps
    /// (`TTSModel.lsd_decode_steps`). More steps can raise quality at more
    /// compute. Must be >= 1 (the library asserts this). Default 1. Clamped to
    /// [`TUNING_LSD_STEPS_MIN`]..=[`TUNING_LSD_STEPS_MAX`].
    pub lsd_decode_steps: u32,
    /// PocketTTS EOS-detection threshold (`TTSModel.eos_threshold`). Higher makes
    /// the model more likely to keep generating (longer tails). Default -4.0.
    /// Clamped to [`TUNING_EOS_THRESHOLD_MIN`]..=[`TUNING_EOS_THRESHOLD_MAX`].
    pub eos_threshold: f32,
    /// PocketTTS noise clamp (`TTSModel.noise_clamp`). Bounds the sampling noise;
    /// `<= 0` means "no clamp" (library default `None`). Default 0.0 (off).
    /// Clamped to [`TUNING_NOISE_CLAMP_MIN`]..=[`TUNING_NOISE_CLAMP_MAX`].
    pub noise_clamp: f32,
    /// PocketTTS max tokens per chunk (`generate_audio(max_tokens=...)`): the
    /// size of the sentence chunks long text is split into. Default 50. Clamped
    /// to [`TUNING_MAX_TOKENS_MIN`]..=[`TUNING_MAX_TOKENS_MAX`].
    pub max_tokens: u32,
    /// PocketTTS frames generated after EOS (`generate_audio(frames_after_eos=...)`).
    /// `0` = let the library auto-pick (its default `None`, ~1-3). Otherwise a
    /// fixed tail length. Clamped to 0..=[`TUNING_FRAMES_AFTER_EOS_MAX`].
    pub frames_after_eos: u32,
}

impl Default for TtsTuningSettings {
    fn default() -> Self {
        Self {
            lead_in_ms: TUNING_LEAD_IN_MS_DEFAULT,
            trailing_ms: TUNING_TRAILING_MS_DEFAULT,
            sentence_gap_ms: TUNING_SENTENCE_GAP_MS_DEFAULT,
            gain_db: TUNING_GAIN_DB_DEFAULT,
            temperature: TUNING_TEMPERATURE_DEFAULT,
            lsd_decode_steps: TUNING_LSD_STEPS_DEFAULT,
            eos_threshold: TUNING_EOS_THRESHOLD_DEFAULT,
            noise_clamp: TUNING_NOISE_CLAMP_DEFAULT,
            max_tokens: TUNING_MAX_TOKENS_DEFAULT,
            frames_after_eos: TUNING_FRAMES_AFTER_EOS_DEFAULT,
        }
    }
}

impl TtsTuningSettings {
    /// Returns a copy with every field clamped/normalized to its documented
    /// range, the way the panel view + the worker body should see it. Mirrors how
    /// the other settings normalize on read rather than trusting stored values.
    pub fn normalized(&self) -> Self {
        Self {
            lead_in_ms: self.lead_in_ms.clamp(TUNING_PAD_MS_MIN, TUNING_PAD_MS_MAX),
            trailing_ms: self.trailing_ms.clamp(TUNING_PAD_MS_MIN, TUNING_PAD_MS_MAX),
            sentence_gap_ms: self
                .sentence_gap_ms
                .clamp(TUNING_PAD_MS_MIN, TUNING_PAD_MS_MAX),
            gain_db: clamp_finite(self.gain_db, TUNING_GAIN_DB_MIN, TUNING_GAIN_DB_MAX),
            temperature: clamp_finite(
                self.temperature,
                TUNING_TEMPERATURE_MIN,
                TUNING_TEMPERATURE_MAX,
            ),
            lsd_decode_steps: self
                .lsd_decode_steps
                .clamp(TUNING_LSD_STEPS_MIN, TUNING_LSD_STEPS_MAX),
            eos_threshold: clamp_finite(
                self.eos_threshold,
                TUNING_EOS_THRESHOLD_MIN,
                TUNING_EOS_THRESHOLD_MAX,
            ),
            noise_clamp: clamp_finite(
                self.noise_clamp,
                TUNING_NOISE_CLAMP_MIN,
                TUNING_NOISE_CLAMP_MAX,
            ),
            max_tokens: self
                .max_tokens
                .clamp(TUNING_MAX_TOKENS_MIN, TUNING_MAX_TOKENS_MAX),
            frames_after_eos: self.frames_after_eos.min(TUNING_FRAMES_AFTER_EOS_MAX),
        }
    }
}

/// Clamp an `f32` to `[min, max]`, treating a non-finite value (NaN/inf) as the
/// midpoint so a bad stored/posted value can't poison synthesis.
fn clamp_finite(value: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        (min + max) / 2.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioTagsSettings {
    pub enabled: bool,
    /// `"auto"`, `"custom"`, or a provider value.
    pub profile: String,
    pub max_tags_per_reply: u8,
    pub strip_game_subtitles: bool,
    pub custom_prompt: String,
}

impl Default for AudioTagsSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            profile: "auto".to_string(),
            max_tags_per_reply: DEFAULT_MAX_TAGS_PER_REPLY,
            strip_game_subtitles: true,
            custom_prompt: String::new(),
        }
    }
}

impl AppSettings {
    /// Loads settings from `path`, falling back to defaults if missing/invalid.
    pub fn load(path: &Path) -> Self {
        fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    /// Writes settings to `path` (pretty JSON), creating parent dirs as needed.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string());
        fs::write(path, json)
    }

    /// Resolves the active game-profile id used to scope per-profile content:
    /// `self.profile` when it is non-empty *and* a profile with that id exists
    /// under `profiles_dir`; otherwise the first profile from
    /// [`crate::GameProfile::list`]; otherwise `""` (no profile → legacy/global
    /// content paths).
    pub fn active_profile_id(&self, profiles_dir: &Path) -> String {
        let configured = self.profile.trim();
        if !configured.is_empty() && crate::GameProfile::read(profiles_dir, configured).is_some() {
            return configured.to_string();
        }
        crate::GameProfile::list(profiles_dir)
            .into_iter()
            .next()
            .map(|profile| profile.id)
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// TTS reference data (mirrored from SillyTavern)
// ---------------------------------------------------------------------------

pub const STREAMING_CHUNK_MS_MIN: u32 = 0;
pub const STREAMING_CHUNK_MS_MAX: u32 = 10_000;
pub const STREAMING_CHUNK_MS_STEP: u32 = 50;
pub const STREAMING_CHUNK_MS_DEFAULT: u32 = 500;

// `tts.stream_slice_ms`: the MAX size (ms of audio) of a streamed TTS chunk. The
// backend ramps slices — a small first slice (~200ms) for fast first-audio, then
// DOUBLING up to this cap — so the FNV plugin gets FEW, BIG chunk files instead of
// many tiny ones. Chunk-file count is what bounds the plugin's per-frame file I/O
// (it reads one WAV per chunk through MO2's usvfs, ~tens of ms each), so big steady
// slices are the key to smooth in-game playback. The single streaming buffer plays
// any chunk size gaplessly, and the ramp keeps it from underrunning early. Distinct
// from `streaming_chunk_ms` (the LLM opener text size), not the TTS audio slice.
pub const STREAM_SLICE_MS_MIN: u32 = 60;
pub const STREAM_SLICE_MS_MAX: u32 = 4_000;
pub const STREAM_SLICE_MS_DEFAULT: u32 = 1_500;

// `tts.caption_max_chars`: max characters per on-screen caption segment. The FNV
// plugin splits the NPC's full line into <= this many chars (at word boundaries)
// and reveals the segments progressively, timed to playback — a pure DISPLAY layer
// that never gates or reshapes the audio (a slight first-caption delay is fine).
// 0 = show the whole line in one caption.
pub const CAPTION_MAX_CHARS_MIN: u32 = 0;
pub const CAPTION_MAX_CHARS_MAX: u32 = 400;
pub const CAPTION_MAX_CHARS_STEP: u32 = 10;
pub const CAPTION_MAX_CHARS_DEFAULT: u32 = 120;

// `tts.npc_volume` / `tts.admin_volume`: linear playback gain (1.0 = unchanged)
// applied to the synthesized PCM. Two independent knobs so the directional in-world
// NPC voices and the non-positional "admin" voice (Todd, spoken straight into the
// player's ear) can be balanced separately, live and on the fly. Applied in the
// sample domain — unlike DirectSound's SetVolume, which can only attenuate, this
// lets values > 1.0 genuinely boost (hard-clamped to int16 at playback). The UI
// renders the multiplier as a percent (0–200 %, 100 % = unity).
pub const VOICE_VOLUME_MIN: f32 = 0.0;
pub const VOICE_VOLUME_MAX: f32 = 2.0;
pub const NPC_VOLUME_DEFAULT: f32 = 1.0;
pub const ADMIN_VOLUME_DEFAULT: f32 = 1.0;

// --- TTS tuning defaults + ranges (see [`TtsTuningSettings`]) ----------------
// Silence pads share one range; the default lead-in matches the worker's old
// hard-coded 0.15 s (150 ms).
pub const TUNING_PAD_MS_MIN: u32 = 0;
pub const TUNING_PAD_MS_MAX: u32 = 2_000;
pub const TUNING_PAD_MS_STEP: u32 = 10;
pub const TUNING_LEAD_IN_MS_DEFAULT: u32 = 150;
pub const TUNING_TRAILING_MS_DEFAULT: u32 = 60;
/// Default inter-sentence gap for streaming engines (PocketTTS). ~180 ms reads as
/// a natural sentence pause without sounding sluggish.
pub const TUNING_SENTENCE_GAP_MS_DEFAULT: u32 = 180;

// Output gain in dB. ±0 is unity; allow a useful boost/cut without inviting
// clipping at the top end.
pub const TUNING_GAIN_DB_MIN: f32 = -24.0;
pub const TUNING_GAIN_DB_MAX: f32 = 12.0;
pub const TUNING_GAIN_DB_STEP: f32 = 0.5;
pub const TUNING_GAIN_DB_DEFAULT: f32 = 0.0;

// PocketTTS `TTSModel.temp` (sampling temperature). Library default 0.7.
pub const TUNING_TEMPERATURE_MIN: f32 = 0.0;
pub const TUNING_TEMPERATURE_MAX: f32 = 2.0;
pub const TUNING_TEMPERATURE_STEP: f32 = 0.05;
pub const TUNING_TEMPERATURE_DEFAULT: f32 = 0.7;

// PocketTTS `TTSModel.lsd_decode_steps`. Library asserts `> 0`, so min is 1.
pub const TUNING_LSD_STEPS_MIN: u32 = 1;
pub const TUNING_LSD_STEPS_MAX: u32 = 16;
pub const TUNING_LSD_STEPS_DEFAULT: u32 = 1;

// PocketTTS `TTSModel.eos_threshold`. Library default -4.0; higher → longer.
pub const TUNING_EOS_THRESHOLD_MIN: f32 = -12.0;
pub const TUNING_EOS_THRESHOLD_MAX: f32 = 0.0;
pub const TUNING_EOS_THRESHOLD_STEP: f32 = 0.25;
pub const TUNING_EOS_THRESHOLD_DEFAULT: f32 = -4.0;

// PocketTTS `TTSModel.noise_clamp`. Library default None (off); we model "off"
// as `<= 0` and let positive values clamp the truncated-normal noise.
pub const TUNING_NOISE_CLAMP_MIN: f32 = 0.0;
pub const TUNING_NOISE_CLAMP_MAX: f32 = 4.0;
pub const TUNING_NOISE_CLAMP_STEP: f32 = 0.1;
pub const TUNING_NOISE_CLAMP_DEFAULT: f32 = 0.0;

// PocketTTS `generate_audio(max_tokens=...)` chunk size. Library default 50.
pub const TUNING_MAX_TOKENS_MIN: u32 = 8;
pub const TUNING_MAX_TOKENS_MAX: u32 = 200;
pub const TUNING_MAX_TOKENS_DEFAULT: u32 = 50;

// PocketTTS `generate_audio(frames_after_eos=...)`. 0 = auto (library `None`).
pub const TUNING_FRAMES_AFTER_EOS_MAX: u32 = 50;
pub const TUNING_FRAMES_AFTER_EOS_DEFAULT: u32 = 0;

pub const MAX_TAGS_MIN: u8 = 0;
pub const MAX_TAGS_MAX: u8 = 8;
const DEFAULT_MAX_TAGS_PER_REPLY: u8 = 2;

// ---------------------------------------------------------------------------
// STT reference data
// ---------------------------------------------------------------------------

/// The only STT provider value wired up: the local koboldcpp Whisper server
/// (OpenAI-compatible `/v1/audio/transcriptions`).
pub const STT_DEFAULT_PROVIDER: &str = "whisper";

/// Default Whisper transcription model — the GGML `.bin` filename koboldcpp
/// loads via `--whispermodel`. This is the OpenAI `model` field on the request
/// AND the on-disk filename probed under the Whisper models dir; the picker keeps
/// the two in sync. Matches the current `whisper-small-q5_1.bin` build.
pub const STT_WHISPER_DEFAULT_MODEL: &str = WHISPER_MODELS[2].file; // small (q5_1)

/// Available STT providers (value, label). Single entry: the local koboldcpp
/// Whisper server (the Parakeet server is gone).
pub const STT_PROVIDERS: &[(&str, &str)] = &[("whisper", "Whisper (koboldcpp, local)")];

/// Normalizes a saved/posted STT provider to a known value, defaulting to
/// Whisper for empty/unknown values (e.g. legacy `parakeet`/`sillytavern`).
pub fn normalize_stt_provider(provider: &str) -> String {
    let candidate = provider.trim();
    if STT_PROVIDERS.iter().any(|(value, _)| *value == candidate) {
        candidate.to_string()
    } else {
        STT_DEFAULT_PROVIDER.to_string()
    }
}

/// The effective transcription model: the saved Whisper `.bin` filename, or `""`
/// when nothing valid is selected. There is NO default model for a public release —
/// the user must download + pick one — so empty/stale values resolve to "none
/// selected" (empty), never a silent fallback. A saved Parakeet model name
/// (`nvidia/parakeet-…`) from an older settings file is treated as stale and
/// dropped, so the picker never shows a model koboldcpp can't load.
pub fn stt_effective_model(stt: &SttSettings) -> String {
    let model = stt.model.trim();
    if model.is_empty() || model.starts_with("nvidia/") || model.contains("parakeet") {
        String::new()
    } else {
        model.to_string()
    }
}

// ---------------------------------------------------------------------------
// Whisper model registry (downloadable GGML .bin builds for koboldcpp)
// ---------------------------------------------------------------------------

/// One downloadable Whisper model — a GGML `.bin` build koboldcpp loads via
/// `--whispermodel`. Download state is detected at runtime from the Whisper
/// models directory (the `file` present / a `.downloading` marker). Mirrors
/// [`LlmModel`] but the on-disk filename IS the OpenAI `model` value (koboldcpp
/// keys whisper off the loaded file, not a HF repo id).
#[derive(Debug, Clone, Copy)]
pub struct WhisperModel {
    /// Stable id used in routes/markers (e.g. `large-v3-turbo`).
    pub id: &'static str,
    /// Display name (e.g. `Large v3 Turbo`).
    pub name: &'static str,
    /// The GGML `.bin` filename — both the download target and the `model` field
    /// sent on each transcription request.
    pub file: &'static str,
    /// Approximate on-disk / VRAM footprint in GB (drives the recommended badge).
    pub size_gb: f64,
    /// Approximate VRAM/RAM the model needs at inference (whisper.cpp f16), in GB.
    /// Whisper runs the same on GPU or CPU, so one number covers both. Feeds the
    /// onboarding hardware-fit recommendation.
    pub vram_gb: f64,
}

/// The HuggingFace repo every Whisper GGML build is pulled from.
pub const WHISPER_REPO: &str = "ggerganov/whisper.cpp";

/// The standard Whisper models offered in the STT picker, smallest → largest.
/// Filenames are the exact GGML builds in `ggerganov/whisper.cpp` (`main`). The
/// quantized small build (`whisper-small-q5_1.bin`) is the current default; the
/// rest are the canonical full-precision builds. `large-v3-turbo` is the speed
/// pick (distilled large, near-large accuracy at a fraction of the cost).
pub const WHISPER_MODELS: &[WhisperModel] = &[
    WhisperModel {
        id: "tiny",
        name: "Tiny",
        file: "ggml-tiny.bin",
        size_gb: 0.08,
        vram_gb: 0.3,
    },
    WhisperModel {
        id: "base",
        name: "Base",
        file: "ggml-base.bin",
        size_gb: 0.15,
        vram_gb: 0.4,
    },
    WhisperModel {
        id: "small",
        name: "Small (q5_1, current)",
        file: "whisper-small-q5_1.bin",
        size_gb: 0.2,
        vram_gb: 0.9,
    },
    WhisperModel {
        id: "medium",
        name: "Medium",
        file: "ggml-medium.bin",
        size_gb: 1.5,
        vram_gb: 2.1,
    },
    WhisperModel {
        id: "large-v3",
        name: "Large v3",
        file: "ggml-large-v3.bin",
        size_gb: 3.1,
        vram_gb: 3.9,
    },
    WhisperModel {
        id: "large-v3-turbo",
        name: "Large v3 Turbo",
        file: "ggml-large-v3-turbo.bin",
        size_gb: 1.6,
        vram_gb: 1.8,
    },
];

/// Resolves a Whisper model by its registry id.
pub fn whisper_model_by_id(id: &str) -> Option<&'static WhisperModel> {
    WHISPER_MODELS.iter().find(|model| model.id == id)
}

/// Resolves the Whisper model whose `.bin` filename matches `file` (the saved
/// `model` value), so the picker can highlight the active model's radio.
pub fn whisper_model_by_file(file: &str) -> Option<&'static WhisperModel> {
    let candidate = file.trim();
    WHISPER_MODELS.iter().find(|model| model.file == candidate)
}

/// Human label for a Whisper model download status string (mirrors
/// [`llm_model_status_label`]).
pub fn whisper_model_status_label(status: &str) -> String {
    match status {
        "downloaded" => "Downloaded",
        "downloading" => "Downloading…",
        "failed" => "Download failed",
        _ => "Available",
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// Interface (appearance) reference data + dynamic theme stylesheet
// ---------------------------------------------------------------------------

pub const INTERFACE_DEFAULT_THEME: &str = "midnight";
pub const INTERFACE_DEFAULT_ACCENT: &str = "#55a7ff";
pub const INTERFACE_DEFAULT_DENSITY: &str = "comfortable";
pub const INTERFACE_FONT_SCALE_MIN: u32 = 90;
pub const INTERFACE_FONT_SCALE_MAX: u32 = 120;
pub const INTERFACE_FONT_SCALE_STEP: u32 = 5;
pub const INTERFACE_FONT_SCALE_DEFAULT: u32 = 100;

/// One selectable dark theme preset. Each supplies the handful of base palette
/// vars the shells read; the accent is applied separately so it composes with
/// any preset. Values are emitted verbatim into `/theme.css`.
#[derive(Debug, Clone, Copy)]
pub struct ThemePreset {
    pub id: &'static str,
    pub label: &'static str,
    pub bg: &'static str,
    pub panel: &'static str,
    pub panel_2: &'static str,
    pub sidebar: &'static str,
    pub sidebar_2: &'static str,
    pub line: &'static str,
    pub line_soft: &'static str,
}

/// The dark theme presets offered in the Interface settings. `midnight` mirrors
/// `app.css`'s built-in palette (so "default" is a real, named choice); the
/// others are cooler/warmer dark variants.
pub const INTERFACE_THEMES: &[ThemePreset] = &[
    ThemePreset {
        id: "midnight",
        label: "Midnight (default)",
        bg: "#101114",
        panel: "#16181d",
        panel_2: "#1d2027",
        sidebar: "#111318",
        sidebar_2: "#171a21",
        line: "#292d36",
        line_soft: "#222630",
    },
    ThemePreset {
        id: "slate",
        label: "Slate",
        bg: "#0e0f12",
        panel: "#191b20",
        panel_2: "#212530",
        sidebar: "#121419",
        sidebar_2: "#1b1e26",
        line: "#2f333d",
        line_soft: "#262a33",
    },
    ThemePreset {
        id: "ocean",
        label: "Ocean",
        bg: "#0c1016",
        panel: "#121823",
        panel_2: "#182030",
        sidebar: "#0d131c",
        sidebar_2: "#141c28",
        line: "#243243",
        line_soft: "#1d2937",
    },
];

/// Accent colour preset swatches shown next to the colour picker (value, label).
pub const INTERFACE_ACCENTS: &[(&str, &str)] = &[
    ("#55a7ff", "Blue"),
    ("#5fb784", "Green"),
    ("#d7b15d", "Amber"),
    ("#a68bd8", "Violet"),
    ("#e36d6d", "Coral"),
    ("#4fd1c5", "Teal"),
];

pub const INTERFACE_DENSITIES: &[(&str, &str)] =
    &[("comfortable", "Comfortable"), ("compact", "Compact")];

/// Resolves a theme id to its preset, defaulting to the first (midnight).
pub fn interface_theme(id: &str) -> &'static ThemePreset {
    INTERFACE_THEMES
        .iter()
        .find(|preset| preset.id == id.trim())
        .unwrap_or(&INTERFACE_THEMES[0])
}

/// Validates an accent colour to a safe `#rgb`/`#rrggbb` hex, else the default.
/// Guards the `/theme.css` output against injection (only hex chars allowed).
pub fn normalize_accent(value: &str) -> String {
    let candidate = value.trim();
    let hex = candidate.strip_prefix('#').unwrap_or("");
    let ok = (hex.len() == 3 || hex.len() == 6) && hex.chars().all(|c| c.is_ascii_hexdigit());
    if ok {
        format!("#{}", hex.to_ascii_lowercase())
    } else {
        INTERFACE_DEFAULT_ACCENT.to_string()
    }
}

/// Normalizes a density string to a known value (default: comfortable).
pub fn normalize_density(value: &str) -> String {
    normalize_option(value, INTERFACE_DENSITIES)
}

/// Clamps + steps a font scale to the slider's 90..=120 range.
pub fn normalize_font_scale(value: u32) -> u32 {
    let clamped = value.clamp(INTERFACE_FONT_SCALE_MIN, INTERFACE_FONT_SCALE_MAX);
    ((clamped + INTERFACE_FONT_SCALE_STEP / 2) / INTERFACE_FONT_SCALE_STEP)
        * INTERFACE_FONT_SCALE_STEP
}

/// Builds the dynamic theme stylesheet served at `GET /theme.css`. Every value
/// is read fresh from the saved [`InterfaceSettings`] and normalized first, so
/// the document only ever contains validated tokens (the accent is hex-checked).
/// Linked AFTER `app.css`, so these `:root{}` overrides win.
///
/// What each setting drives (proving the wiring):
/// * theme preset → base palette vars (`--bg`/`--panel`/`--sidebar`/`--line`…).
/// * accent → `--accent`.
/// * density → `--pad`/`--gap` + compact paddings on the shells/cards.
/// * font scale → `html{font-size}`.
/// * reduce motion → a global `*{transition:none;animation:none}` rule.
/// * show timestamps → hides `.message-meta time` / `.msg-time` when off.
/// * show prompt panel → collapses the 4th `.app-shell` column + hides `.prompt`.
pub fn build_theme_css(interface: &InterfaceSettings) -> String {
    let preset = interface_theme(&interface.theme);
    let accent = normalize_accent(&interface.accent);
    let density = normalize_density(&interface.density);
    let font_scale = normalize_font_scale(interface.font_scale);
    let compact = density == "compact";

    // Density vars: introduced here (consumed by app.css via var() with
    // fallbacks) so a single token controls the shells' breathing room.
    let (pad, gap, card_pad) = if compact {
        ("8px", "8px", "10px 12px")
    } else {
        ("16px", "14px", "15px 16px")
    };

    let mut css = String::new();
    css.push_str("/* Generated by Chasm from Interface settings. */\n");
    css.push_str(":root{\n");
    css.push_str(&format!("  --bg:{};\n", preset.bg));
    css.push_str(&format!("  --panel:{};\n", preset.panel));
    css.push_str(&format!("  --panel-2:{};\n", preset.panel_2));
    css.push_str(&format!("  --sidebar:{};\n", preset.sidebar));
    css.push_str(&format!("  --sidebar-2:{};\n", preset.sidebar_2));
    css.push_str(&format!("  --line:{};\n", preset.line));
    css.push_str(&format!("  --line-soft:{};\n", preset.line_soft));
    css.push_str(&format!("  --accent:{accent};\n"));
    css.push_str(&format!("  --pad:{pad};\n"));
    css.push_str(&format!("  --gap:{gap};\n"));
    css.push_str(&format!("  --card-pad:{card_pad};\n"));
    css.push_str("}\n");

    // Font scale.
    css.push_str(&format!("html{{font-size:{font_scale}%;}}\n"));

    // Compact density: tighten the main shells + cards. These rules only emit
    // when compact, so comfortable is exactly the app.css baseline.
    if compact {
        css.push_str(".settings-main{padding:16px 20px 40px;}\n");
        css.push_str(".settings-group{margin-bottom:12px;}\n");
        css.push_str("details.settings-group > .settings-legend{padding:9px 14px;}\n");
        css.push_str(".library-main{padding:18px 22px 40px;}\n");
        css.push_str(".library-card{padding:10px 12px;}\n");
        css.push_str(".message{padding-top:8px;padding-bottom:8px;}\n");
    }

    // Reduce motion: kill transitions + animations everywhere.
    if interface.reduce_motion {
        css.push_str("*,*::before,*::after{transition:none!important;animation:none!important;scroll-behavior:auto!important;}\n");
    }

    // Hide message timestamps when off (covers the live-chat + message-list
    // timestamp elements).
    if !interface.show_timestamps {
        css.push_str(
            ".message-meta time,.msg-time,.message-time,time.msg-meta{display:none!important;}\n",
        );
    }

    // Collapse the right-hand prompt-inspector column when off: drop the 4th
    // grid track and hide the panel itself.
    if !interface.show_prompt_panel {
        css.push_str(
            ".app-shell{grid-template-columns:76px minmax(220px,286px) minmax(0,1fr)!important;}\n",
        );
        css.push_str(".prompt{display:none!important;}\n");
    }

    css
}

// ---------------------------------------------------------------------------
// Retrieval reference data
// ---------------------------------------------------------------------------

/// Embedder tier options (value, label) for the Retrieval settings picker.
pub const RETRIEVAL_EMBEDDER_TIERS: &[(&str, &str)] = &[
    ("small", "Small (BGE-small, INT8 on CPU)"),
    ("base", "Base (BGE-base)"),
    ("quality", "Quality (BGE-large)"),
];

/// Reranker tier options (value, label).
pub const RETRIEVAL_RERANKER_TIERS: &[(&str, &str)] = &[
    ("small", "Small (jina-reranker-v1-turbo-en)"),
    ("large", "Large (bge-reranker-v2-m3)"),
];

/// Execution-provider options (value, label).
pub const RETRIEVAL_EXECUTIONS: &[(&str, &str)] = &[
    ("cpu", "CPU only"),
    ("gpu", "GPU (CUDA, falls back to CPU)"),
];

pub const RETRIEVAL_TOP_K_MIN: u32 = 1;
pub const RETRIEVAL_TOP_K_MAX: u32 = 50;
pub const RETRIEVAL_CANDIDATES_MIN: u32 = 1;
pub const RETRIEVAL_CANDIDATES_MAX: u32 = 500;
pub const RETRIEVAL_SOURCE_LIMIT_MIN: u32 = 0;
pub const RETRIEVAL_SOURCE_LIMIT_MAX: u32 = 50;

/// Normalizes an embedder tier string to a known value (default: first option).
pub fn normalize_embedder_tier(value: &str) -> String {
    normalize_option(value, RETRIEVAL_EMBEDDER_TIERS)
}

/// Normalizes a reranker tier string to a known value (default: first option).
pub fn normalize_reranker_tier(value: &str) -> String {
    normalize_option(value, RETRIEVAL_RERANKER_TIERS)
}

/// Normalizes an execution string to a known value (default: first option).
pub fn normalize_execution(value: &str) -> String {
    normalize_option(value, RETRIEVAL_EXECUTIONS)
}

/// Normalizes `value` to a known option value, falling back to the first
/// option in the list when empty/unknown (no "auto" fallback anymore).
fn normalize_option(value: &str, options: &[(&str, &str)]) -> String {
    let candidate = value.trim();
    if options.iter().any(|(v, _)| *v == candidate) {
        candidate.to_string()
    } else {
        options
            .first()
            .map(|(v, _)| v.to_string())
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// LLM model registry (downloadable local GGUFs)
// ---------------------------------------------------------------------------

/// One downloadable local LLM model. Download state is detected at runtime from
/// the LLM models directory (a matching `*.gguf` file / `.downloading` marker).
#[derive(Debug, Clone, Copy)]
pub struct LlmModel {
    /// Stable id used in routes/markers (e.g. `gemma-4-e2b`).
    pub id: &'static str,
    /// Display name (e.g. `Gemma 4 E2B`).
    pub name: &'static str,
    /// Hugging Face repo (e.g. `unsloth/gemma-4-E2B-it-GGUF`).
    pub repo: &'static str,
    /// Quant tag downloaded for this model (the file matches `*<quant>.gguf`).
    pub quant: &'static str,
    /// Approximate VRAM needed to run this model fully on GPU at ~8k context
    /// (UD-Q4_K_XL GGUF weights + KV cache + runtime overhead), in GB. Drives the
    /// onboarding fit search + the per-row "Recommended" badge.
    pub vram_gb: f64,
    /// Approximate system RAM needed to run this model on CPU (no GPU / CPU
    /// fallback), in GB — a bit above `vram_gb` to cover the OS keeping weights
    /// resident. Used by the onboarding recommender for CPU-only hosts.
    pub ram_gb: f64,
}

/// The quant variant Chasm downloads for every Gemma 4 model.
pub const LLM_QUANT_TAG: &str = "UD-Q4_K_XL";

/// Downloadable Gemma 4 models (Unsloth GGUFs, UD-Q4_K_XL quant). Mirrors
/// [`TTS_LOCAL_ENGINES`] as the reference list the UI renders.
pub const LLM_MODELS: &[LlmModel] = &[
    LlmModel {
        id: "gemma-4-e2b",
        name: "Gemma 4 E2B",
        repo: "unsloth/gemma-4-E2B-it-GGUF",
        quant: LLM_QUANT_TAG,
        // ~3.1 GB GGUF + KV/overhead.
        vram_gb: 3.0,
        ram_gb: 5.0,
    },
    LlmModel {
        id: "gemma-4-e4b",
        name: "Gemma 4 E4B",
        repo: "unsloth/gemma-4-E4B-it-GGUF",
        quant: LLM_QUANT_TAG,
        // ~5.0 GB GGUF + KV/overhead.
        vram_gb: 5.0,
        ram_gb: 7.0,
    },
    LlmModel {
        id: "gemma-4-12b",
        name: "Gemma 4 12B",
        repo: "unsloth/gemma-4-12b-it-GGUF",
        quant: LLM_QUANT_TAG,
        // ~6.7 GB GGUF + KV/overhead.
        vram_gb: 7.0,
        ram_gb: 9.0,
    },
    LlmModel {
        id: "gemma-4-26b-a4b",
        name: "Gemma 4 26B-A4B",
        repo: "unsloth/gemma-4-26B-A4B-it-GGUF",
        quant: LLM_QUANT_TAG,
        // ~16.9 GB GGUF + KV/overhead (MoE: big weights, light compute).
        vram_gb: 15.0,
        ram_gb: 19.0,
    },
    LlmModel {
        id: "gemma-4-31b",
        name: "Gemma 4 31B",
        repo: "unsloth/gemma-4-31B-it-GGUF",
        quant: LLM_QUANT_TAG,
        // ~18.3 GB GGUF + KV/overhead.
        vram_gb: 18.0,
        ram_gb: 22.0,
    },
];

// ---------------------------------------------------------------------------
// Whisper download-detection helper (shared by the STT page + onboarding)
// ---------------------------------------------------------------------------

/// A lowercase stem used to detect any download of a Whisper model on disk: a
/// `.bin` whose lowercased name contains this stem counts as downloaded. Derived
/// from the filename without the `ggml-` prefix / `.bin` suffix.
pub fn whisper_model_match_stem(model: &WhisperModel) -> String {
    model
        .file
        .strip_prefix("ggml-")
        .unwrap_or(model.file)
        .strip_suffix(".bin")
        .unwrap_or(model.file)
        .to_lowercase()
}

/// The basename of a repo (the part after `/`), used to build the conventional
/// GGUF filename: `<repo-basename>-<quant>.gguf`.
pub fn llm_repo_basename(repo: &str) -> &str {
    repo.rsplit('/').next().unwrap_or(repo)
}

/// The conventional GGUF filename for a model's quant, e.g.
/// `gemma-4-E2B-it-GGUF-UD-Q4_K_XL.gguf`. Unsloth strips the trailing `-GGUF`
/// from the repo basename in the actual filenames, so we drop it too.
pub fn llm_model_filename(model: &LlmModel) -> String {
    let base = llm_repo_basename(model.repo)
        .strip_suffix("-GGUF")
        .unwrap_or_else(|| llm_repo_basename(model.repo));
    format!("{base}-{}.gguf", model.quant)
}

/// A lowercase "stem" used to match any-quant downloads of a model on disk
/// (e.g. `gemma-4-26b-a4b-it`). A `*.gguf` whose lowercased name contains this
/// stem counts as that model being downloaded, regardless of quant.
pub fn llm_model_match_stem(model: &LlmModel) -> String {
    llm_repo_basename(model.repo)
        .strip_suffix("-GGUF")
        .unwrap_or_else(|| llm_repo_basename(model.repo))
        .to_lowercase()
}

/// Resolves the on-disk GGUF path to load for a model `id`, looking in
/// `models_dir`. Prefers any already-present `*.gguf` whose lowercased name
/// contains the model's match stem (so a different quant the user already has is
/// used, matching how download *status* is detected), and otherwise falls back
/// to the conventional `<basename>-<quant>.gguf` path this app would download.
/// Returns `None` for an unknown id. The returned path may not exist yet (the
/// fallback) — callers that need a real file should check `.exists()`.
pub fn llm_model_gguf_path(models_dir: &Path, id: &str) -> Option<std::path::PathBuf> {
    let model = LLM_MODELS.iter().find(|model| model.id == id)?;
    let stem = llm_model_match_stem(model);
    if let Ok(entries) = fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_lowercase();
            // A vision projector (mmproj-*.gguf) can legitimately share the
            // model's name stem — it is NOT the model.
            if name.ends_with(".gguf") && name.contains(&stem) && !name.contains("mmproj") {
                return Some(entry.path());
            }
        }
    }
    Some(models_dir.join(llm_model_filename(model)))
}

/// Human label for an LLM model download status string.
pub fn llm_model_status_label(status: &str) -> String {
    match status {
        "downloaded" => "Downloaded",
        "downloading" => "Downloading…",
        "failed" => "Download failed",
        _ => "Available",
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// Retrieval model registry (downloadable embedder / reranker weights)
// ---------------------------------------------------------------------------

/// A downloadable retrieval model (an embedder or a reranker). Mirrors the
/// shape of [`TTS_LOCAL_ENGINES`] but with extra metadata the UI needs:
/// hardware fit (footprint) and the on-disk cache dir we probe for status.
///
/// `cache_dir` is the `models--<org>--<repo>` directory fastembed/hf-hub creates
/// under the embed cache dir on first use; its presence means "downloaded".
#[derive(Debug, Clone, Copy)]
pub struct RetrievalModelDef {
    /// Stable id used in URLs + the download route (e.g. `bge-small`).
    pub id: &'static str,
    /// Human label shown in the list.
    pub label: &'static str,
    /// `"embedder"` or `"reranker"`.
    pub kind: &'static str,
    /// The settings tier value this model backs: embedders use
    /// `small`/`base`/`quality`, rerankers use `small`/`large`.
    pub tier: &'static str,
    /// The `models--<org>--<repo>` directory fastembed creates on download.
    pub cache_dir: &'static str,
    /// Approximate on-disk / VRAM footprint in GB (drives the recommended badge).
    pub footprint_gb: f64,
}

/// The embedder + reranker models that can be downloaded for retrieval. The
/// fastembed model each `tier` resolves to (per device) lives in the
/// `chasm-embed` crate; this registry mirrors the ids so the UI + the
/// download route stay in sync. Cache-dir names are the exact directories
/// fastembed creates (verified on disk for `bge-small` + `jina-turbo`).
pub const RETRIEVAL_MODELS: &[RetrievalModelDef] = &[
    RetrievalModelDef {
        id: "bge-small",
        label: "BGE-small (en v1.5, INT8)",
        kind: "embedder",
        tier: "small",
        cache_dir: "models--Qdrant--bge-small-en-v1.5-onnx-Q",
        footprint_gb: 0.3,
    },
    RetrievalModelDef {
        id: "bge-base",
        label: "BGE-base (en v1.5, INT8)",
        kind: "embedder",
        tier: "base",
        cache_dir: "models--Qdrant--bge-base-en-v1.5-onnx-Q",
        footprint_gb: 0.5,
    },
    RetrievalModelDef {
        id: "bge-large",
        label: "BGE-large (en v1.5)",
        kind: "embedder",
        tier: "quality",
        cache_dir: "models--Xenova--bge-large-en-v1.5",
        footprint_gb: 1.3,
    },
    RetrievalModelDef {
        id: "jina-turbo",
        label: "jina-reranker-v1-turbo-en",
        kind: "reranker",
        tier: "small",
        cache_dir: "models--jinaai--jina-reranker-v1-turbo-en",
        footprint_gb: 0.2,
    },
    RetrievalModelDef {
        id: "bge-reranker-v2-m3",
        label: "bge-reranker-v2-m3",
        kind: "reranker",
        tier: "large",
        cache_dir: "models--rozgo--bge-reranker-v2-m3",
        footprint_gb: 2.2,
    },
];

/// Human label for a retrieval-model download status string.
pub fn retrieval_model_status_label(status: &str) -> String {
    match status {
        "downloaded" => "Downloaded",
        "downloading" => "Downloading…",
        "failed" => "Download failed",
        _ => "Available",
    }
    .to_string()
}

/// Local TTS engines (id, label). Both stream on :5002 via the same OpenAI
/// `/v1/audio/speech` contract (faster-qwen3-tts → `qwen3_tts_server.py`,
/// PocketTTS → `pockettts_server.py`), so the picker can swap between them.
/// faster-qwen3-tts runs from its own venv (helper config `localRuntimes.tts`);
/// PocketTTS install state is detected from markers under the engines directory.
pub const TTS_LOCAL_ENGINES: &[(&str, &str)] = &[
    ("faster-qwen3-tts", "faster-qwen3-tts (streaming)"),
    ("pockettts", "PocketTTS (streaming)"),
];

/// Normalizes a saved/posted local TTS engine to a known engine value. Returns the
/// matched engine id, or `""` for empty/unknown/removed values (e.g. the retired
/// `omnivoice` / plain `qwen3`). NOTE: empty means "none selected" — there is NO
/// default engine. For a public release nothing is auto-selected or auto-started;
/// the user must actively pick (and install) an engine. Callers must treat `""` as
/// "don't launch / show nothing selected", never as a silent fallback to a specific
/// engine.
pub fn normalize_local_engine(engine: &str) -> String {
    let candidate = engine.trim();
    if TTS_LOCAL_ENGINES.iter().any(|(value, _)| *value == candidate) {
        candidate.to_string()
    } else {
        String::new()
    }
}

/// Human label for an engine install status string.
pub fn engine_status_label(status: &str) -> String {
    match status {
        "installed" => "Installed",
        "installing" => "Installing…",
        "failed" => "Install failed",
        _ => "Not installed",
    }
    .to_string()
}

/// All API TTS providers available in SillyTavern (the TTS provider dropdown).
pub const TTS_API_PROVIDERS: &[&str] = &[
    "AllTalk",
    "Azure",
    "Chatterbox",
    "Chutes",
    "Coqui",
    "CosyVoice (Unofficial)",
    "Edge",
    "ElevenLabs",
    "Electron Hub",
    "Google Gemini TTS",
    "Google Translate",
    "GPT-SoVITS-Adapter",
    "GPT-SoVITS-V2 (Unofficial)",
    "GSVI",
    "Inworld",
    "Kokoro",
    "MiniMax",
    "Novel",
    "OpenAI",
    "OpenAI Compatible",
    "Pollinations",
    "SBVits2",
    "Silero",
    "SpeechT5",
    "System",
    "TTS WebUI",
    "VITS",
    "Volcengine",
    "XTTSv2",
];

/// Audio-tag profile options (`Auto`/`Custom` + every provider), mirroring
/// `TTS_AUDIO_TAG_PROFILE_OPTIONS`.
pub fn tts_audio_tag_profiles() -> Vec<(String, String)> {
    let mut options = vec![
        ("auto".to_string(), "Auto (current provider)".to_string()),
        ("custom".to_string(), "Custom".to_string()),
    ];
    options.extend(
        TTS_API_PROVIDERS
            .iter()
            .map(|name| (name.to_string(), name.to_string())),
    );
    options
}

const INWORLD_TTS2_STEERING_TAGS: &[&str] = &[
    "[say with force]",
    "[articulate clearly]",
    "[say with deliberate pauses]",
    "[say with a falling pitch]",
    "[say with a rising pitch]",
    "[very loud]",
    "[very quiet]",
    "[say in a low tone]",
    "[say in a high pitch]",
    "[say playfully]",
    "[say with no pitch variation]",
    "[very fast]",
    "[very slow]",
    "[sing joyfully]",
    "[whisper in a hushed style]",
    "[give a nasal quality]",
];

const INWORLD_TTS2_NONVERBAL_TAGS: &[&str] = &[
    "[laugh]",
    "[breathe]",
    "[clear throat]",
    "[sigh]",
    "[cough]",
    "[yawn]",
];

const ELEVEN_V3_VOICE_TAGS: &[&str] = &[
    "[laughs]",
    "[laughs harder]",
    "[starts laughing]",
    "[wheezing]",
    "[whispers]",
    "[sighs]",
    "[exhales]",
    "[sarcastic]",
    "[curious]",
    "[excited]",
    "[crying]",
    "[snorts]",
    "[mischievously]",
    "[excitedly]",
    "[curiously]",
    "[impressed]",
    "[dramatically]",
    "[giggling]",
    "[with genuine belly laugh]",
    "[delighted]",
    "[amazed]",
    "[warmly]",
    "[frustrated sigh]",
    "[happy gasp]",
    "[happy]",
    "[sad]",
    "[angry]",
    "[whisper]",
    "[annoyed]",
    "[appalled]",
    "[thoughtful]",
    "[surprised]",
    "[laughing]",
    "[chuckles]",
    "[clears throat]",
    "[short pause]",
    "[long pause]",
    "[exhales sharply]",
    "[inhales deeply]",
    "[singing]",
    "[muttering]",
];

const ELEVEN_V3_SOUND_EFFECT_TAGS: &[&str] = &[
    "[gunshot]",
    "[applause]",
    "[clapping]",
    "[explosion]",
    "[swallows]",
    "[gulps]",
];

const ELEVEN_V3_SPECIAL_TAGS: &[&str] = &["[strong X accent]", "[sings]", "[woo]", "[fart]"];

/// The audio-tag prompt + model note that "comes with" a provider, mirroring
/// `getTtsAudioTagProfile`. Returns `None` for providers without tag support.
/// For ElevenLabs/Inworld the tag-supporting model variant's prompt is shown.
pub fn tts_provider_audio_tags(provider: &str) -> Option<(String, String)> {
    match provider {
        "Inworld" => Some((
            [
                "Inworld TTS-2 supports square-bracket natural-language steering in English.".to_string(),
                format!(
                    "Documented steering tag examples: {}.",
                    INWORLD_TTS2_STEERING_TAGS.join(", ")
                ),
                format!(
                    "Documented non-verbal tags, exact spelling: {}.",
                    INWORLD_TTS2_NONVERBAL_TAGS.join(", ")
                ),
                "Use one steering tag at the start of the spoken line when delivery needs direction. Non-verbal tags may appear inline where the sound occurs.".to_string(),
            ]
            .join("\n"),
            "Inworld TTS-2; the TTS-1.5 model uses a smaller compact tag set.".to_string(),
        )),
        "ElevenLabs" => Some((
            [
                "ElevenLabs v3 supports square-bracket audio tags; earlier ElevenLabs models in this provider profile do not.".to_string(),
                format!("Documented voice and delivery tags: {}.", ELEVEN_V3_VOICE_TAGS.join(", ")),
                format!("Documented sound-effect tags: {}.", ELEVEN_V3_SOUND_EFFECT_TAGS.join(", ")),
                format!("Documented special/experimental tags: {}.", ELEVEN_V3_SPECIAL_TAGS.join(", ")),
                "Use only the tags above. Place a tag immediately before or after the phrase it affects.".to_string(),
                "Prefer voice/delivery tags for NPC dialogue; use sound-effect or special tags only when the scene explicitly needs that audible effect.".to_string(),
            ]
            .join("\n"),
            "ElevenLabs v3 models only.".to_string(),
        )),
        _ => None,
    }
}

/// Clamp/round max-tags to `[0, 8]`, mirroring `normalizeMaxTagsPerReply`.
pub fn normalize_max_tags(value: u8) -> u8 {
    value.min(MAX_TAGS_MAX)
}

/// Clamp/round a streaming chunk value to the slider's range and step.
pub fn normalize_streaming_chunk_ms(value: u32) -> u32 {
    let clamped = value.clamp(STREAMING_CHUNK_MS_MIN, STREAMING_CHUNK_MS_MAX);
    ((clamped + STREAMING_CHUNK_MS_STEP / 2) / STREAMING_CHUNK_MS_STEP) * STREAMING_CHUNK_MS_STEP
}

/// Clamp a TTS mini-chunk slice size to [`STREAM_SLICE_MS_MIN`]..=[`STREAM_SLICE_MS_MAX`].
pub fn normalize_stream_slice_ms(value: u32) -> u32 {
    value.clamp(STREAM_SLICE_MS_MIN, STREAM_SLICE_MS_MAX)
}

/// Clamp a TTS voice-volume multiplier to [`VOICE_VOLUME_MIN`]..=[`VOICE_VOLUME_MAX`]
/// (and drop NaN/inf to unity). Shared by the settings form, the panel view, and
/// the synth path so a stored, posted, or displayed value is always in range.
pub fn normalize_voice_volume(value: f32) -> f32 {
    if !value.is_finite() {
        return NPC_VOLUME_DEFAULT;
    }
    value.clamp(VOICE_VOLUME_MIN, VOICE_VOLUME_MAX)
}

/// Clamp/round a caption segment size to the slider range/step. 0 stays 0 (= show
/// the whole line in one caption).
pub fn normalize_caption_max_chars(value: u32) -> u32 {
    let clamped = value.min(CAPTION_MAX_CHARS_MAX);
    if clamped == 0 {
        0
    } else {
        ((clamped + CAPTION_MAX_CHARS_STEP / 2) / CAPTION_MAX_CHARS_STEP) * CAPTION_MAX_CHARS_STEP
    }
}

/// The provider whose audio tags drive the prompt, mirroring
/// `getHeadlessTtsAudioTagsPrompt`: the explicit profile, else the current
/// (API) provider.
fn audio_tags_current_provider(tts: &TtsSettings) -> &str {
    let profile = tts.audio_tags.profile.as_str();
    if profile == "auto" || profile == "custom" {
        tts.api_provider.as_str()
    } else {
        profile
    }
}

/// The resolved audio-tag prompt that would be injected, mirroring
/// `buildTtsAudioTagsPrompt`.
pub fn build_audio_tags_preview(tts: &TtsSettings) -> String {
    let at = &tts.audio_tags;
    if !at.enabled {
        return String::new();
    }
    if at.profile == "custom" {
        return at.custom_prompt.trim().to_string();
    }
    let provider = audio_tags_current_provider(tts);
    let Some((prompt, _note)) = tts_provider_audio_tags(provider) else {
        return String::new();
    };
    let max = normalize_max_tags(at.max_tags_per_reply);
    let plural = if max == 1 { "" } else { "s" };
    [
        "Audio Tags:".to_string(),
        "These instructions apply only to the next spoken assistant reply and should be visible in the SillyTavern chat text.".to_string(),
        format!("Use no more than {max} short audio tag{plural} in a reply unless the scene absolutely requires more."),
        "The allowed provider tags/control forms are listed below. Do not invent bracket tags outside these provider rules.".to_string(),
        "Never explain the tags, never put tags on their own line, and never let tags replace the actual dialogue.".to_string(),
        prompt,
    ]
    .join("\n")
}

// ---------------------------------------------------------------------------
// View models for the settings page
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct SettingsNavItem {
    pub key: String,
    pub label: String,
    pub active: bool,
}

/// One required mod as rendered in the settings "Mods" list (status pill + a
/// "Get" link, or a "bundled" label). Built by the web layer from
/// `game_launcher::mod_status`.
#[derive(Debug, Clone, Serialize)]
pub struct RequiredModView {
    pub id: String,
    pub display: String,
    pub installed: bool,
    pub required: bool,
    /// `installed` / `missing` (drives the pill style + label).
    pub status: String,
    pub status_label: String,
    /// `GitHub` / `Nexus` / `Bundled`.
    pub source_label: String,
    /// "Get" link, or `None` for bundled mods (shown as a plain label).
    pub source_url: Option<String>,
    /// The absolute path probed (shown as a hint / tooltip).
    pub detect_path: String,
    /// The on-disk install folder for this mod — `<instance>/mods/<name>/` for a
    /// mod folder, or the game dir for a game-file mod (xNVSE). Shown as a "drop it
    /// here" path with an Open-folder button so a manually-downloaded mod can be
    /// dropped straight into the place detection checks.
    pub install_folder: String,
    /// `true` when this mod needs a manual download (Nexus, no auto-install) — the
    /// row shows a "download, extract, drop it here" hint.
    pub manual: bool,
}

/// The "Game" (launcher) settings panel: detected MO2 + the FNV install + the
/// editable override fields. Built by the web layer (the detected booleans depend
/// on the filesystem). chasm is a passive backend now — it neither launches the
/// game nor installs mods — so this view is a read-only detection summary plus the
/// override fields the bridge/trace resolution still reads.
#[derive(Debug, Clone, Serialize)]
pub struct GameLauncherView {
    /// Resolved (effective) values, shown alongside the editable overrides.
    pub mo2_exe: String,
    pub instance: String,
    pub profile: String,
    pub executable: String,
    pub game_dir: String,
    /// The raw saved overrides (blank = auto-detected), for the form inputs.
    pub mo2_exe_override: String,
    pub instance_override: String,
    pub profile_override: String,
    pub executable_override: String,
    pub game_dir_override: String,
    /// Detection results.
    pub mo2_detected: bool,
    pub nvse_detected: bool,
    /// `FalloutNV.exe` present in `game_dir` → the install path is valid.
    pub falloutnv_detected: bool,
    /// The exact launch command preview (`ModOrganizer.exe "moshortcut://…"`).
    pub launch_command: String,
    pub moshortcut_arg: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LocalEngineView {
    pub value: String,
    pub label: String,
    pub selected: bool,
    /// `installed`, `installing`, `failed`, or `not_installed`.
    pub status: String,
    pub status_label: String,
    pub installed: bool,
    pub installing: bool,
    pub can_download: bool,
    /// Whether this engine is the one currently serving the TTS port (the active
    /// engine marker matches AND :5002 is reachable). Drives the "Running" badge.
    pub running: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct LlmModelView {
    pub id: String,
    pub name: String,
    pub repo: String,
    /// Approximate VRAM (GB) needed to run fully on GPU.
    pub vram_gb: f64,
    /// `downloaded`, `downloading`, `failed`, or `available`.
    pub status: String,
    pub status_label: String,
    pub downloaded: bool,
    pub downloading: bool,
    pub can_download: bool,
    /// `true` for the single best-fit model (the "Recommended" badge).
    pub recommended: bool,
    /// Short per-row fit hint from [`GpuFit`]: e.g. "fits", "tight", "CPU only".
    pub fit_label: String,
    /// `true` when this is the active model (the saved `llm.model`), so the
    /// picker radio renders checked. Set in [`llm_models_panel_view`].
    pub selected: bool,
    /// `true` when the radio may be chosen: the model is downloaded, or it is the
    /// current selection (so a save never silently drops the active model even if
    /// its file went missing). Non-selectable rows render the radio disabled.
    pub selectable: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct LlmModelsPanelView {
    /// "Detected: RTX 5090, 32 GB VRAM" style summary of the host.
    pub host_summary: String,
    pub models: Vec<LlmModelView>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiProviderView {
    pub value: String,
    pub label: String,
    pub selected: bool,
    pub has_audio_tags: bool,
    pub audio_tags_prompt: String,
    pub audio_tags_note: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfileOptionView {
    pub value: String,
    pub label: String,
    pub selected: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TtsPanelView {
    pub mode: String,
    pub is_local: bool,
    pub is_api: bool,
    pub local_engines: Vec<LocalEngineView>,
    pub api_providers: Vec<ApiProviderView>,
    pub streaming_enabled: bool,
    pub streaming_chunk_ms: u32,
    pub streaming_chunk_min: u32,
    pub streaming_chunk_max: u32,
    pub streaming_chunk_step: u32,
    pub caption_max_chars: u32,
    pub caption_max_min: u32,
    pub caption_max_max: u32,
    pub caption_max_step: u32,
    /// Live voice-volume sliders (percent: 100 = unity). NPC = directional voices,
    /// admin = the non-positional voice (Todd). Range/step shared by both.
    pub npc_volume_pct: u32,
    pub admin_volume_pct: u32,
    pub voice_volume_min_pct: u32,
    pub voice_volume_max_pct: u32,
    pub voice_volume_step_pct: u32,
    pub default_voice: String,
    pub audio_tags_enabled: bool,
    pub audio_tags_profile: String,
    pub audio_tags_profiles: Vec<ProfileOptionView>,
    pub audio_tags_max_tags: u8,
    pub audio_tags_max_min: u8,
    pub audio_tags_max_max: u8,
    pub audio_tags_strip_subtitles: bool,
    pub audio_tags_custom_prompt: String,
    pub audio_tags_preview: String,
    /// Live synthesis tuning (current saved values + each control's range/step).
    pub tuning: TtsTuningView,
}

/// The TTS-tuning card view: the current (normalized) value of every knob plus
/// the min/max/step each input should render. Built in [`tts_panel_view`].
#[derive(Debug, Clone, Serialize)]
pub struct TtsTuningView {
    pub lead_in_ms: u32,
    pub trailing_ms: u32,
    pub sentence_gap_ms: u32,
    pub pad_ms_min: u32,
    pub pad_ms_max: u32,
    pub pad_ms_step: u32,
    pub gain_db: f32,
    pub gain_db_min: f32,
    pub gain_db_max: f32,
    pub gain_db_step: f32,
    pub temperature: f32,
    pub temperature_min: f32,
    pub temperature_max: f32,
    pub temperature_step: f32,
    pub lsd_decode_steps: u32,
    pub lsd_steps_min: u32,
    pub lsd_steps_max: u32,
    pub eos_threshold: f32,
    pub eos_threshold_min: f32,
    pub eos_threshold_max: f32,
    pub eos_threshold_step: f32,
    pub noise_clamp: f32,
    pub noise_clamp_min: f32,
    pub noise_clamp_max: f32,
    pub noise_clamp_step: f32,
    pub max_tokens: u32,
    pub max_tokens_min: u32,
    pub max_tokens_max: u32,
    pub frames_after_eos: u32,
    pub frames_after_eos_max: u32,
}

/// Builds the tuning card view from the persisted tuning settings, clamping each
/// value to its documented range first (so the inputs always show a valid value).
pub fn tts_tuning_view(tuning: &TtsTuningSettings) -> TtsTuningView {
    let t = tuning.normalized();
    TtsTuningView {
        lead_in_ms: t.lead_in_ms,
        trailing_ms: t.trailing_ms,
        sentence_gap_ms: t.sentence_gap_ms,
        pad_ms_min: TUNING_PAD_MS_MIN,
        pad_ms_max: TUNING_PAD_MS_MAX,
        pad_ms_step: TUNING_PAD_MS_STEP,
        gain_db: t.gain_db,
        gain_db_min: TUNING_GAIN_DB_MIN,
        gain_db_max: TUNING_GAIN_DB_MAX,
        gain_db_step: TUNING_GAIN_DB_STEP,
        temperature: t.temperature,
        temperature_min: TUNING_TEMPERATURE_MIN,
        temperature_max: TUNING_TEMPERATURE_MAX,
        temperature_step: TUNING_TEMPERATURE_STEP,
        lsd_decode_steps: t.lsd_decode_steps,
        lsd_steps_min: TUNING_LSD_STEPS_MIN,
        lsd_steps_max: TUNING_LSD_STEPS_MAX,
        eos_threshold: t.eos_threshold,
        eos_threshold_min: TUNING_EOS_THRESHOLD_MIN,
        eos_threshold_max: TUNING_EOS_THRESHOLD_MAX,
        eos_threshold_step: TUNING_EOS_THRESHOLD_STEP,
        noise_clamp: t.noise_clamp,
        noise_clamp_min: TUNING_NOISE_CLAMP_MIN,
        noise_clamp_max: TUNING_NOISE_CLAMP_MAX,
        noise_clamp_step: TUNING_NOISE_CLAMP_STEP,
        max_tokens: t.max_tokens,
        max_tokens_min: TUNING_MAX_TOKENS_MIN,
        max_tokens_max: TUNING_MAX_TOKENS_MAX,
        frames_after_eos: t.frames_after_eos,
        frames_after_eos_max: TUNING_FRAMES_AFTER_EOS_MAX,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SttProviderView {
    pub value: String,
    pub label: String,
    pub selected: bool,
}

/// One downloadable Whisper model as rendered in the STT picker (radio +
/// download button + status pill + recommended badge + hardware hint). Mirrors
/// [`LlmModelView`] / [`RetrievalModelView`]; `selected` drives the radio's
/// `checked` state (the active model is the saved `model` filename).
#[derive(Debug, Clone, Serialize)]
pub struct WhisperModelView {
    pub id: String,
    pub name: String,
    /// The GGML `.bin` filename — the radio's value AND the `model` field saved.
    pub file: String,
    /// Approx footprint, pre-formatted (e.g. `~0.2 GB`).
    pub size_label: String,
    /// `downloaded`, `downloading`, `failed`, or `available`.
    pub status: String,
    pub status_label: String,
    pub downloaded: bool,
    pub downloading: bool,
    pub can_download: bool,
    /// `true` for the single best-fit model (the "Recommended" badge).
    pub recommended: bool,
    /// Short hardware hint, e.g. `Fits GPU comfortably` / `CPU only`.
    pub fit_hint: String,
    /// `true` when this model's `.bin` is the currently-selected one (drives the
    /// radio's `checked` state). Set in [`stt_panel_view`].
    pub selected: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SttPanelView {
    pub provider: String,
    pub providers: Vec<SttProviderView>,
    pub model: String,
    pub language: String,
    pub prompt: String,
    pub timeout_ms: u64,
    pub timeout_ms_min: u64,
    pub timeout_ms_max: u64,
    /// Downloadable Whisper models (status + recommended badge + active radio).
    /// Built by the web layer (depends on on-disk download status + hardware).
    pub models: Vec<WhisperModelView>,
    /// Detected-host summary shown above the model list (reuses the retrieval
    /// host view shape).
    pub host: RetrievalHostView,
}

/// The LLM generation-sampling card view: the current (normalized) value of
/// every knob plus the min/max/step each input should render.
#[derive(Debug, Clone, Serialize)]
pub struct LlmSamplingView {
    pub temperature: f32,
    pub temperature_min: f32,
    pub temperature_max: f32,
    pub temperature_step: f32,
    pub top_p: f32,
    pub top_p_min: f32,
    pub top_p_max: f32,
    pub top_p_step: f32,
    pub top_k: u32,
    pub top_k_min: u32,
    pub top_k_max: u32,
    pub min_p: f32,
    pub min_p_min: f32,
    pub min_p_max: f32,
    pub min_p_step: f32,
    pub repeat_penalty: f32,
    pub repeat_penalty_min: f32,
    pub repeat_penalty_max: f32,
    pub repeat_penalty_step: f32,
    pub max_tokens: u32,
    pub max_tokens_min: u32,
    pub max_tokens_max: u32,
    pub n_ctx: u32,
    pub n_ctx_min: u32,
    pub n_ctx_max: u32,
    pub seed: i64,
}

/// Builds the LLM sampling card view from the persisted settings, clamping each
/// value to its documented range first (so the inputs always show a valid value).
pub fn llm_sampling_view(sampling: &LlmSamplingSettings) -> LlmSamplingView {
    let s = sampling.normalized();
    LlmSamplingView {
        temperature: s.temperature,
        temperature_min: LLM_TEMPERATURE_MIN,
        temperature_max: LLM_TEMPERATURE_MAX,
        temperature_step: LLM_TEMPERATURE_STEP,
        top_p: s.top_p,
        top_p_min: LLM_TOP_P_MIN,
        top_p_max: LLM_TOP_P_MAX,
        top_p_step: LLM_TOP_P_STEP,
        top_k: s.top_k,
        top_k_min: LLM_TOP_K_MIN,
        top_k_max: LLM_TOP_K_MAX,
        min_p: s.min_p,
        min_p_min: LLM_MIN_P_MIN,
        min_p_max: LLM_MIN_P_MAX,
        min_p_step: LLM_MIN_P_STEP,
        repeat_penalty: s.repeat_penalty,
        repeat_penalty_min: LLM_REPEAT_PENALTY_MIN,
        repeat_penalty_max: LLM_REPEAT_PENALTY_MAX,
        repeat_penalty_step: LLM_REPEAT_PENALTY_STEP,
        max_tokens: s.max_tokens,
        max_tokens_min: LLM_MAX_TOKENS_MIN,
        max_tokens_max: LLM_MAX_TOKENS_MAX,
        n_ctx: s.n_ctx,
        n_ctx_min: LLM_N_CTX_MIN,
        n_ctx_max: LLM_N_CTX_MAX,
        seed: s.seed,
    }
}

/// One theme-preset radio option in the Interface panel.
#[derive(Debug, Clone, Serialize)]
pub struct ThemeOptionView {
    pub id: String,
    pub label: String,
    pub selected: bool,
    /// Background swatch colour for the preview chip.
    pub bg: String,
    /// Panel swatch colour for the preview chip.
    pub panel: String,
}

/// One accent-swatch option in the Interface panel.
#[derive(Debug, Clone, Serialize)]
pub struct AccentOptionView {
    pub value: String,
    pub label: String,
    pub selected: bool,
}

/// The Interface (appearance) panel view: the current saved values + the option
/// lists/ranges each control renders. Everything here is wired to `/theme.css`.
#[derive(Debug, Clone, Serialize)]
pub struct InterfacePanelView {
    pub themes: Vec<ThemeOptionView>,
    pub accent: String,
    pub accents: Vec<AccentOptionView>,
    pub densities: Vec<SelectOptionView>,
    pub density: String,
    pub font_scale: u32,
    pub font_scale_min: u32,
    pub font_scale_max: u32,
    pub font_scale_step: u32,
    pub reduce_motion: bool,
    pub show_timestamps: bool,
    pub show_prompt_panel: bool,
}

/// Builds the Interface panel view from the persisted settings + reference data,
/// normalizing every value the same way `/theme.css` will.
pub fn interface_panel_view(interface: &InterfaceSettings) -> InterfacePanelView {
    let theme_id = interface_theme(&interface.theme).id;
    let accent = normalize_accent(&interface.accent);
    let density = normalize_density(&interface.density);
    let themes = INTERFACE_THEMES
        .iter()
        .map(|preset| ThemeOptionView {
            id: preset.id.to_string(),
            label: preset.label.to_string(),
            selected: preset.id == theme_id,
            bg: preset.bg.to_string(),
            panel: preset.panel.to_string(),
        })
        .collect();
    let accents = INTERFACE_ACCENTS
        .iter()
        .map(|(value, label)| AccentOptionView {
            selected: value.eq_ignore_ascii_case(&accent),
            value: value.to_string(),
            label: label.to_string(),
        })
        .collect();
    InterfacePanelView {
        themes,
        accent,
        accents,
        densities: select_options(INTERFACE_DENSITIES, &density),
        density,
        font_scale: normalize_font_scale(interface.font_scale),
        font_scale_min: INTERFACE_FONT_SCALE_MIN,
        font_scale_max: INTERFACE_FONT_SCALE_MAX,
        font_scale_step: INTERFACE_FONT_SCALE_STEP,
        reduce_motion: interface.reduce_motion,
        show_timestamps: interface.show_timestamps,
        show_prompt_panel: interface.show_prompt_panel,
    }
}

/// One game-profile card in the Profiles settings panel. Built by the web layer
/// (the content counts + cloned-voices flag come from the repository / disk),
/// mirroring how [`GameLauncherView`] is assembled outside core.
#[derive(Debug, Clone, Serialize)]
pub struct ProfileCardView {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Two-letter badge initials (e.g. "FN").
    pub initials: String,
    /// The currently-active profile gets the ACTIVE badge + disabled Activate.
    pub active: bool,
    pub character_count: usize,
    pub lorebook_count: usize,
    pub quest_count: usize,
    pub action_count: usize,
    /// Number of characters whose selected-engine voice is cloned (sample.wav).
    pub cloned_voice_count: usize,
}

/// The Profiles settings panel: the active profile id + every profile as a card.
#[derive(Debug, Clone, Serialize)]
pub struct ProfilesPanelView {
    pub active_id: String,
    pub profiles: Vec<ProfileCardView>,
    /// The profiles directory path (shown in the "how to add a profile" note).
    pub profiles_dir: String,
}

/// Builds the STT panel view from the persisted settings + reference data.
/// `models` + `host` are built by the caller (the web layer) because they depend
/// on detected hardware and the on-disk download status of each Whisper `.bin`.
/// The active model (the saved `model` filename) is marked `selected` so its
/// radio renders checked.
pub fn stt_panel_view(
    stt: &SttSettings,
    mut models: Vec<WhisperModelView>,
    host: RetrievalHostView,
) -> SttPanelView {
    let provider = normalize_stt_provider(&stt.provider);
    let providers = STT_PROVIDERS
        .iter()
        .map(|(value, label)| SttProviderView {
            selected: *value == provider,
            value: value.to_string(),
            label: label.to_string(),
        })
        .collect();
    let active_model = stt_effective_model(stt);
    for model in &mut models {
        model.selected = model.file == active_model;
    }
    SttPanelView {
        provider,
        providers,
        model: active_model,
        language: stt.language.clone(),
        prompt: stt.prompt.clone(),
        timeout_ms: normalize_stt_timeout_ms(stt.timeout_ms),
        timeout_ms_min: STT_TIMEOUT_MS_MIN,
        timeout_ms_max: STT_TIMEOUT_MS_MAX,
        models,
        host,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SelectOptionView {
    pub value: String,
    pub label: String,
    pub selected: bool,
}

/// One downloadable retrieval model as rendered in the list (status pill +
/// download button + recommended badge + hardware hint).
#[derive(Debug, Clone, Serialize)]
pub struct RetrievalModelView {
    pub id: String,
    pub label: String,
    pub kind: String,
    pub tier: String,
    /// Approx footprint, pre-formatted (e.g. `~0.3 GB`).
    pub size_label: String,
    /// `downloaded`, `downloading`, `failed`, or `available`.
    pub status: String,
    pub status_label: String,
    pub downloaded: bool,
    pub downloading: bool,
    pub can_download: bool,
    /// Recommended pick for this host (only ever set on one embedder).
    pub recommended: bool,
    /// Short hardware hint, e.g. `Fits GPU comfortably` / `CPU only`.
    pub fit_hint: String,
    /// `true` when this model's tier is the currently-selected one for its kind
    /// (drives the picker radio's `checked` state). Set in `retrieval_panel_view`.
    pub selected: bool,
}

/// Detected-host summary shown above the model list.
#[derive(Debug, Clone, Serialize, Default)]
pub struct RetrievalHostView {
    /// Pre-formatted one-liner, e.g. `RTX 5090, 32 GB VRAM / 24 cores`.
    pub summary: String,
    pub has_gpu: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RetrievalPanelView {
    pub enabled: bool,
    pub chat_memory_enabled: bool,
    pub lore_semantic_enabled: bool,
    pub action_semantic_enabled: bool,
    pub quest_semantic_enabled: bool,
    pub embedder_tiers: Vec<SelectOptionView>,
    pub reranker_enabled: bool,
    pub reranker_tiers: Vec<SelectOptionView>,
    pub executions: Vec<SelectOptionView>,
    pub top_k: u32,
    pub top_k_min: u32,
    pub top_k_max: u32,
    pub candidates: u32,
    pub candidates_min: u32,
    pub candidates_max: u32,
    pub min_score: f32,
    pub action_min_score: f32,
    pub chat_memory_limit: u32,
    pub lore_limit: u32,
    pub quest_limit: u32,
    pub source_limit_min: u32,
    pub source_limit_max: u32,
    /// Downloadable embedder/reranker models (status + recommended badge).
    pub models: Vec<RetrievalModelView>,
    /// Detected-host summary shown above the model list.
    pub host: RetrievalHostView,
}

fn select_options(options: &[(&str, &str)], selected: &str) -> Vec<SelectOptionView> {
    options
        .iter()
        .map(|(value, label)| SelectOptionView {
            selected: *value == selected,
            value: value.to_string(),
            label: label.to_string(),
        })
        .collect()
}

/// Builds the Retrieval panel view from the persisted settings + reference data.
/// `models` + `host` are built by the caller (the web layer) because they depend
/// on detected hardware and the on-disk download status of each model.
pub fn retrieval_panel_view(
    r: &RetrievalSettings,
    mut models: Vec<RetrievalModelView>,
    host: RetrievalHostView,
) -> RetrievalPanelView {
    // Mark the model row whose tier matches the selected embedder/reranker tier
    // so the picker radios render checked. Embedders use `embedder_tier`,
    // rerankers use `reranker_tier`.
    let selected_embedder = normalize_embedder_tier(&r.embedder_tier);
    let selected_reranker = normalize_reranker_tier(&r.reranker_tier);
    for model in &mut models {
        model.selected = match model.kind.as_str() {
            "embedder" => model.tier == selected_embedder,
            "reranker" => model.tier == selected_reranker,
            _ => false,
        };
    }
    RetrievalPanelView {
        enabled: r.enabled,
        chat_memory_enabled: r.chat_memory_enabled,
        lore_semantic_enabled: r.lore_semantic_enabled,
        action_semantic_enabled: r.action_semantic_enabled,
        quest_semantic_enabled: r.quest_semantic_enabled,
        embedder_tiers: select_options(
            RETRIEVAL_EMBEDDER_TIERS,
            &normalize_embedder_tier(&r.embedder_tier),
        ),
        reranker_enabled: r.reranker_enabled,
        reranker_tiers: select_options(
            RETRIEVAL_RERANKER_TIERS,
            &normalize_reranker_tier(&r.reranker_tier),
        ),
        executions: select_options(RETRIEVAL_EXECUTIONS, &normalize_execution(&r.execution)),
        top_k: r.top_k.clamp(RETRIEVAL_TOP_K_MIN, RETRIEVAL_TOP_K_MAX),
        top_k_min: RETRIEVAL_TOP_K_MIN,
        top_k_max: RETRIEVAL_TOP_K_MAX,
        candidates: r
            .candidates
            .clamp(RETRIEVAL_CANDIDATES_MIN, RETRIEVAL_CANDIDATES_MAX),
        candidates_min: RETRIEVAL_CANDIDATES_MIN,
        candidates_max: RETRIEVAL_CANDIDATES_MAX,
        min_score: r.min_score,
        action_min_score: r.action_min_score,
        chat_memory_limit: r
            .chat_memory_limit
            .clamp(RETRIEVAL_SOURCE_LIMIT_MIN, RETRIEVAL_SOURCE_LIMIT_MAX),
        lore_limit: r
            .lore_limit
            .clamp(RETRIEVAL_SOURCE_LIMIT_MIN, RETRIEVAL_SOURCE_LIMIT_MAX),
        quest_limit: r
            .quest_limit
            .clamp(RETRIEVAL_SOURCE_LIMIT_MIN, RETRIEVAL_SOURCE_LIMIT_MAX),
        source_limit_min: RETRIEVAL_SOURCE_LIMIT_MIN,
        source_limit_max: RETRIEVAL_SOURCE_LIMIT_MAX,
        models,
        host,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct VoiceCloneCharacterView {
    pub name: String,
    /// `cloned`, `cloning`, `failed`, or `pending`.
    pub status: String,
    pub status_label: String,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct VoiceCloneView {
    pub has_profile: bool,
    pub profile_id: String,
    pub profile_name: String,
    pub engine_id: String,
    pub engine_label: String,
    pub characters: Vec<VoiceCloneCharacterView>,
    pub any_cloning: bool,
    pub cloned_count: usize,
}

/// Human label for a voice-clone status string.
pub fn clone_status_label(status: &str) -> String {
    match status {
        "cloned" => "Cloned",
        "cloning" => "Cloning…",
        "failed" => "Failed",
        _ => "Not cloned",
    }
    .to_string()
}

/// The "drop files here to add models manually" folder paths, one per model
/// category. Each is an absolute path string computed in the web layer (the
/// dirs are filesystem-dependent), rendered on the matching category's panel.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ModelPathsView {
    /// LLM GGUF folder (`llm_models_dir`) — drop `*.gguf` files here.
    pub llm: String,
    /// Whisper `.bin` folder (`whisper_models_dir`) — drop `*.bin` files here.
    pub stt: String,
    /// Active profile's voice-samples folder (`active_voices_dir`) — drop voice
    /// audio here to add clone voices.
    pub tts_voices: String,
    /// Where the TTS engine venvs are installed (`engines_dir`) — shown for
    /// reference (installed via the picker, not really drag-droppable).
    pub tts_engines: String,
}

/// The runtime-requirement status shown on a model settings page: which runtime
/// runs the page's models (koboldcpp for LLM/STT, the TTS engine for TTS) and
/// whether it's present. `status` is the pill class suffix (`installed` /
/// `downloading` / `missing`), `status_label` the short pill text, and `detail`
/// the one-line explanation under it. Resolved in the web layer (filesystem +
/// helper-config dependent), mirroring [`ModelPathsView`].
#[derive(Debug, Clone, Default, Serialize)]
pub struct RuntimeStatusView {
    /// Runtime name shown to the user, e.g. `"koboldcpp"` or a TTS engine label.
    pub name: String,
    /// Pill class suffix: `installed` | `downloading` | `missing`.
    pub status: String,
    /// Short pill text, e.g. `"Installed"` / `"Downloading…"` / `"Not installed"`.
    pub status_label: String,
    /// One-line explanation, e.g. "Downloaded with your first model".
    pub detail: String,
    pub installed: bool,
    pub downloading: bool,
    pub missing: bool,
}

/// Builds the koboldcpp [`RuntimeStatusView`] from a status string (`installed` /
/// `downloading` / `missing`). Shared by the LLM + STT pages, since koboldcpp runs
/// both the LLM and Whisper STT.
pub fn koboldcpp_runtime_status(status: &str) -> RuntimeStatusView {
    let (status_label, detail) = match status {
        "installed" => ("Installed", "Ready — the runtime that runs your models is present."),
        "downloading" => (
            "Downloading…",
            "Fetching koboldcpp from GitHub in the background — click Refresh to update.",
        ),
        _ => (
            "Not installed",
            "Will be downloaded automatically with your first model.",
        ),
    };
    RuntimeStatusView {
        name: "koboldcpp".to_string(),
        status: status.to_string(),
        status_label: status_label.to_string(),
        detail: detail.to_string(),
        installed: status == "installed",
        downloading: status == "downloading",
        missing: status != "installed" && status != "downloading",
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SettingsPageView {
    pub category: String,
    pub nav: Vec<SettingsNavItem>,
    /// Grouped nav (small section labels above clusters of categories).
    pub nav_groups: Vec<SettingsNavGroup>,
    pub saved: bool,
    pub settings_path: String,
    /// On-disk folder where each model category's files live, surfaced on the
    /// matching settings page so power users can drop files in by hand (they then
    /// show up in the picker). Resolved as absolute strings in the web layer
    /// (filesystem-dependent), mirroring `settings_path`.
    pub model_paths: ModelPathsView,
    /// Runtime-requirement status for the LLM + STT pages: koboldcpp (which runs
    /// both the LLM and Whisper STT). Resolved in the web layer.
    pub kobold_runtime: RuntimeStatusView,
    pub tts: TtsPanelView,
    pub llm: LlmSettings,
    pub stt: SttSettings,
    pub stt_panel: SttPanelView,
    pub retrieval: RetrievalPanelView,
    pub voice_clone: VoiceCloneView,
    pub llm_models: LlmModelsPanelView,
    /// LLM generation sampling controls (wired into the llama.cpp request).
    pub llm_sampling: LlmSamplingView,
    /// The "Game" (launcher) panel: MO2 status + required mods.
    pub game: GameLauncherView,
    /// The "Interface" (appearance) panel, wired via `/theme.css`.
    pub interface: InterfacePanelView,
    /// The "Profiles" panel: the drop-in game profiles as cards.
    pub profiles: ProfilesPanelView,
}

/// Builds the TTS panel view from the persisted settings + reference data.
/// `engine_status` maps a local-engine id to its install status string.
pub fn tts_panel_view(
    tts: &TtsSettings,
    engine_status: &HashMap<String, String>,
    running_engine: Option<&str>,
) -> TtsPanelView {
    // Compare the saved engine after normalization. A legacy/removed value
    // (e.g. `omnivoice`/`qwen3`) or an empty one normalizes to "" (none selected),
    // so no engine row is highlighted until the user picks one.
    let selected_engine = normalize_local_engine(&tts.local_engine);
    let local_engines = TTS_LOCAL_ENGINES
        .iter()
        .map(|(value, label)| {
            let status = engine_status
                .get(*value)
                .map(String::as_str)
                .unwrap_or("not_installed");
            LocalEngineView {
                value: value.to_string(),
                label: label.to_string(),
                selected: selected_engine == *value,
                status: status.to_string(),
                status_label: engine_status_label(status),
                installed: status == "installed",
                installing: status == "installing",
                // Both engines now install into a chasm-managed `engines/<id>` venv,
                // so both offer a Download/Install button when not yet installed.
                can_download: status == "not_installed" || status == "failed",
                running: running_engine == Some(*value),
            }
        })
        .collect();

    let api_providers = TTS_API_PROVIDERS
        .iter()
        .map(|name| {
            let tags = tts_provider_audio_tags(name);
            let (prompt, note) = tags.unwrap_or_default();
            ApiProviderView {
                value: name.to_string(),
                label: name.to_string(),
                selected: tts.api_provider == *name,
                has_audio_tags: !prompt.is_empty(),
                audio_tags_prompt: prompt,
                audio_tags_note: note,
            }
        })
        .collect();

    let audio_tags_profiles = tts_audio_tag_profiles()
        .into_iter()
        .map(|(value, label)| ProfileOptionView {
            selected: tts.audio_tags.profile == value,
            value,
            label,
        })
        .collect();

    TtsPanelView {
        mode: tts.mode.clone(),
        is_local: tts.mode != "api",
        is_api: tts.mode == "api",
        local_engines,
        api_providers,
        streaming_enabled: tts.streaming_enabled,
        streaming_chunk_ms: normalize_streaming_chunk_ms(tts.streaming_chunk_ms),
        streaming_chunk_min: STREAMING_CHUNK_MS_MIN,
        streaming_chunk_max: STREAMING_CHUNK_MS_MAX,
        streaming_chunk_step: STREAMING_CHUNK_MS_STEP,
        caption_max_chars: normalize_caption_max_chars(tts.caption_max_chars),
        caption_max_min: CAPTION_MAX_CHARS_MIN,
        caption_max_max: CAPTION_MAX_CHARS_MAX,
        caption_max_step: CAPTION_MAX_CHARS_STEP,
        npc_volume_pct: (normalize_voice_volume(tts.npc_volume) * 100.0).round() as u32,
        admin_volume_pct: (normalize_voice_volume(tts.admin_volume) * 100.0).round() as u32,
        voice_volume_min_pct: (VOICE_VOLUME_MIN * 100.0) as u32,
        voice_volume_max_pct: (VOICE_VOLUME_MAX * 100.0) as u32,
        voice_volume_step_pct: 5,
        default_voice: tts.default_voice.clone(),
        audio_tags_enabled: tts.audio_tags.enabled,
        audio_tags_profile: tts.audio_tags.profile.clone(),
        audio_tags_profiles,
        audio_tags_max_tags: normalize_max_tags(tts.audio_tags.max_tags_per_reply),
        audio_tags_max_min: MAX_TAGS_MIN,
        audio_tags_max_max: MAX_TAGS_MAX,
        audio_tags_strip_subtitles: tts.audio_tags.strip_game_subtitles,
        audio_tags_custom_prompt: tts.audio_tags.custom_prompt.clone(),
        audio_tags_preview: build_audio_tags_preview(tts),
        tuning: tts_tuning_view(&tts.tuning),
    }
}

/// Short fit hint for a model on this host's GPU, used per-row in the UI.
fn llm_fit_label(fit: GpuFit) -> String {
    match fit {
        GpuFit::Comfortable => "fits",
        GpuFit::Tight => "tight",
        GpuFit::Exceeds => "exceeds VRAM",
        GpuFit::NoGpu => "CPU only",
    }
    .to_string()
}

/// "Detected: RTX 5090, 32 GB VRAM" style summary of the host hardware.
pub fn llm_host_summary(system: &SystemInfo) -> String {
    match (&system.gpu_name, system.vram_total_gb) {
        (Some(name), Some(vram)) => format!("Detected: {name}, {vram:.0} GB VRAM"),
        (Some(name), None) => format!("Detected: {name}"),
        _ => {
            let ram = system
                .ram_gb
                .map(|v| format!(", {v:.0} GB RAM"))
                .unwrap_or_default();
            format!("Detected: no GPU ({} cores{ram})", system.cpu_cores)
        }
    }
}

/// Resolves which LLM model id is the active one for the picker: the saved
/// `selected` id when it matches a known model, else the first *downloaded*
/// model, else `""` (nothing checked). Keeps the radio honest when settings are
/// empty or name a model that isn't in the registry.
pub fn selected_llm_model_id(saved: &str, model_status: &HashMap<String, String>) -> String {
    let saved = saved.trim();
    if LLM_MODELS.iter().any(|model| model.id == saved) {
        return saved.to_string();
    }
    LLM_MODELS
        .iter()
        .find(|model| model_status.get(model.id).map(String::as_str) == Some("downloaded"))
        .map(|model| model.id.to_string())
        .unwrap_or_default()
}

/// Builds the LLM models panel: each model's download status (from
/// `model_status`), its per-host fit hint, the single "Recommended" pick, and
/// which row is the active selection (`selected_id`, the saved `llm.model`) so
/// the picker radio renders checked.
pub fn llm_models_panel_view(
    model_status: &HashMap<String, String>,
    system: &SystemInfo,
    selected_id: &str,
) -> LlmModelsPanelView {
    let vram_reqs: Vec<f64> = LLM_MODELS.iter().map(|model| model.vram_gb).collect();
    let recommended = recommended_index(&vram_reqs, system);
    let models = LLM_MODELS
        .iter()
        .enumerate()
        .map(|(index, model)| {
            let status = model_status
                .get(model.id)
                .map(String::as_str)
                .unwrap_or("available");
            LlmModelView {
                id: model.id.to_string(),
                name: model.name.to_string(),
                repo: model.repo.to_string(),
                vram_gb: model.vram_gb,
                status: status.to_string(),
                status_label: llm_model_status_label(status),
                downloaded: status == "downloaded",
                downloading: status == "downloading",
                can_download: status == "available" || status == "failed",
                recommended: recommended == Some(index),
                fit_label: llm_fit_label(system.gpu_fit(model.vram_gb)),
                selected: model.id == selected_id,
                selectable: status == "downloaded" || model.id == selected_id,
            }
        })
        .collect();
    LlmModelsPanelView {
        host_summary: llm_host_summary(system),
        models,
    }
}

/// Looks up an LLM model by id.
pub fn llm_model_by_id(id: &str) -> Option<&'static LlmModel> {
    LLM_MODELS.iter().find(|m| m.id == id)
}

/// The settings categories in nav order, as `(key, label)`. The single source of
/// truth for the left-nav on every settings page (including Tracing). Append new
/// categories here.
pub const SETTINGS_NAV: &[(&str, &str)] = &[
    ("interface", "Interface"),
    ("profiles", "Profiles"),
    ("llm", "LLM"),
    ("tts", "TTS"),
    ("stt", "STT"),
    ("retrieval", "Retrieval"),
    ("game", "Bridge"),
    ("tracing", "Tracing"),
];

/// Settings nav grouping: each section gets a small label above its categories.
/// Drives the grouped left-nav. Keys must all appear in [`SETTINGS_NAV`].
pub const SETTINGS_NAV_GROUPS: &[(&str, &[&str])] = &[
    ("Appearance", &["interface"]),
    ("Content", &["profiles"]),
    ("AI", &["llm", "tts", "stt", "retrieval"]),
    ("System", &["game", "tracing"]),
];

/// Builds the left-nav items for a settings page, marking `category` active.
pub fn settings_nav_items(category: &str) -> Vec<SettingsNavItem> {
    SETTINGS_NAV
        .iter()
        .map(|(key, label)| SettingsNavItem {
            key: key.to_string(),
            label: label.to_string(),
            active: *key == category,
        })
        .collect()
}

/// One labeled group of settings-nav links (e.g. "AI" → LLM/TTS/STT/Retrieval).
#[derive(Debug, Clone, Serialize)]
pub struct SettingsNavGroup {
    pub label: String,
    pub items: Vec<SettingsNavItem>,
}

/// Builds the grouped left-nav for a settings page (small section label above
/// each cluster of categories), marking `category` active. Looks up each key's
/// human label from [`SETTINGS_NAV`] so the two stay in sync.
pub fn settings_nav_groups(category: &str) -> Vec<SettingsNavGroup> {
    SETTINGS_NAV_GROUPS
        .iter()
        .map(|(label, keys)| SettingsNavGroup {
            label: label.to_string(),
            items: keys
                .iter()
                .filter_map(|key| {
                    SETTINGS_NAV.iter().find(|(nav_key, _)| nav_key == key).map(
                        |(nav_key, nav_label)| SettingsNavItem {
                            key: nav_key.to_string(),
                            label: nav_label.to_string(),
                            active: *nav_key == category,
                        },
                    )
                })
                .collect(),
        })
        .collect()
}

/// Builds the full settings page view for `category` (`llm`/`tts`/`stt`).
#[allow(clippy::too_many_arguments)]
pub fn settings_page_view(
    settings: &AppSettings,
    category: &str,
    saved: bool,
    settings_path: String,
    model_paths: ModelPathsView,
    kobold_runtime: RuntimeStatusView,
    engine_status: &HashMap<String, String>,
    voice_clone: VoiceCloneView,
    llm_models: LlmModelsPanelView,
    retrieval_models: Vec<RetrievalModelView>,
    retrieval_host: RetrievalHostView,
    whisper_models: Vec<WhisperModelView>,
    whisper_host: RetrievalHostView,
    game: GameLauncherView,
    profiles: ProfilesPanelView,
    running_engine: Option<String>,
) -> SettingsPageView {
    let nav = settings_nav_items(category);
    let nav_groups = settings_nav_groups(category);

    SettingsPageView {
        category: category.to_string(),
        nav,
        nav_groups,
        saved,
        settings_path,
        model_paths,
        kobold_runtime,
        tts: tts_panel_view(&settings.tts, engine_status, running_engine.as_deref()),
        llm: settings.llm.clone(),
        stt: settings.stt.clone(),
        stt_panel: stt_panel_view(&settings.stt, whisper_models, whisper_host),
        retrieval: retrieval_panel_view(&settings.retrieval, retrieval_models, retrieval_host),
        voice_clone,
        llm_models,
        llm_sampling: llm_sampling_view(&settings.llm.sampling),
        game,
        interface: interface_panel_view(&settings.interface),
        profiles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_sillytavern() {
        let tts = TtsSettings::default();
        assert_eq!(tts.streaming_chunk_ms, 500);
        assert!(tts.streaming_enabled);
        assert_eq!(tts.audio_tags.max_tags_per_reply, 2);
        assert!(!tts.audio_tags.enabled);
    }

    #[test]
    fn whisper_stem_matches_bin_name() {
        let turbo = whisper_model_by_id("large-v3-turbo").unwrap();
        let stem = whisper_model_match_stem(turbo);
        assert_eq!(stem, "large-v3-turbo");
        // A real on-disk filename should contain the stem (case-insensitive).
        assert!("ggml-large-v3-turbo.bin".to_lowercase().contains(&stem));
    }

    #[test]
    fn tuning_defaults_match_pockettts() {
        // Defaults mirror the worker's old lead-in + PocketTTS library defaults.
        let t = TtsTuningSettings::default();
        assert_eq!(t.lead_in_ms, 150);
        assert_eq!(t.trailing_ms, 60);
        assert_eq!(t.gain_db, 0.0);
        assert_eq!(t.temperature, 0.7); // pocket_tts DEFAULT_TEMPERATURE
        assert_eq!(t.lsd_decode_steps, 1); // DEFAULT_LSD_DECODE_STEPS
        assert_eq!(t.eos_threshold, -4.0); // DEFAULT_EOS_THRESHOLD
        assert_eq!(t.noise_clamp, 0.0); // off == library None
        assert_eq!(t.max_tokens, 50); // MAX_TOKEN_PER_CHUNK
        assert_eq!(t.frames_after_eos, 0); // 0 == auto (library None)
                                           // TtsSettings embeds the tuning defaults.
        assert_eq!(TtsSettings::default().tuning.lead_in_ms, 150);
    }

    #[test]
    fn tuning_normalizes_out_of_range_values() {
        let raw = TtsTuningSettings {
            lead_in_ms: 99_999,
            trailing_ms: 99_999,
            sentence_gap_ms: 99_999,
            gain_db: 999.0,
            temperature: -5.0,
            lsd_decode_steps: 0, // library requires >= 1
            eos_threshold: 50.0,
            noise_clamp: -3.0,
            max_tokens: 0,
            frames_after_eos: 9_999,
        };
        let t = raw.normalized();
        assert_eq!(t.lead_in_ms, TUNING_PAD_MS_MAX);
        assert_eq!(t.trailing_ms, TUNING_PAD_MS_MAX);
        assert_eq!(t.sentence_gap_ms, TUNING_PAD_MS_MAX);
        assert_eq!(t.gain_db, TUNING_GAIN_DB_MAX);
        assert_eq!(t.temperature, TUNING_TEMPERATURE_MIN);
        assert_eq!(t.lsd_decode_steps, TUNING_LSD_STEPS_MIN); // clamped up to 1
        assert_eq!(t.eos_threshold, TUNING_EOS_THRESHOLD_MAX);
        assert_eq!(t.noise_clamp, TUNING_NOISE_CLAMP_MIN);
        assert_eq!(t.max_tokens, TUNING_MAX_TOKENS_MIN);
        assert_eq!(t.frames_after_eos, TUNING_FRAMES_AFTER_EOS_MAX);
    }

    #[test]
    fn tuning_nonfinite_floats_fall_to_midpoint() {
        let raw = TtsTuningSettings {
            gain_db: f32::NAN,
            temperature: f32::INFINITY,
            ..TtsTuningSettings::default()
        };
        let t = raw.normalized();
        assert_eq!(t.gain_db, (TUNING_GAIN_DB_MIN + TUNING_GAIN_DB_MAX) / 2.0);
        assert_eq!(
            t.temperature,
            (TUNING_TEMPERATURE_MIN + TUNING_TEMPERATURE_MAX) / 2.0
        );
    }

    #[test]
    fn tuning_defaults_when_key_absent() {
        // Older settings files have no `tuning` key → serde default fills it,
        // and a partial tuning object fills only the rest from defaults.
        let json = r#"{"tts":{"mode":"local"}}"#;
        let settings: AppSettings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.tts.tuning.lead_in_ms, 150);
        assert_eq!(settings.tts.tuning.temperature, 0.7);

        let partial = r#"{"tts":{"tuning":{"lead_in_ms":300}}}"#;
        let s2: AppSettings = serde_json::from_str(partial).unwrap();
        assert_eq!(s2.tts.tuning.lead_in_ms, 300); // honored
        assert_eq!(s2.tts.tuning.trailing_ms, 60); // default
        assert_eq!(s2.tts.tuning.eos_threshold, -4.0); // default
    }

    #[test]
    fn tuning_view_clamps_and_carries_ranges() {
        let raw = TtsTuningSettings {
            lead_in_ms: 50_000,
            ..TtsTuningSettings::default()
        };
        let view = tts_tuning_view(&raw);
        assert_eq!(view.lead_in_ms, TUNING_PAD_MS_MAX); // value clamped
        assert_eq!(view.pad_ms_max, TUNING_PAD_MS_MAX); // range carried
        assert_eq!(view.temperature_step, TUNING_TEMPERATURE_STEP);
        assert_eq!(view.lsd_steps_min, 1);
    }

    #[test]
    fn only_inworld_and_elevenlabs_have_tags() {
        assert!(tts_provider_audio_tags("Inworld").is_some());
        assert!(tts_provider_audio_tags("ElevenLabs").is_some());
        assert!(tts_provider_audio_tags("OpenAI").is_none());
        assert!(tts_provider_audio_tags("Kokoro").is_none());
    }

    #[test]
    fn streaming_chunk_is_clamped_and_stepped() {
        assert_eq!(normalize_streaming_chunk_ms(99_999), 10_000);
        assert_eq!(normalize_streaming_chunk_ms(524), 500);
        assert_eq!(normalize_streaming_chunk_ms(525), 550);
    }

    #[test]
    fn voice_volume_clamps_and_defaults() {
        assert_eq!(normalize_voice_volume(1.0), 1.0);
        assert_eq!(normalize_voice_volume(1.5), 1.5);
        assert_eq!(normalize_voice_volume(5.0), VOICE_VOLUME_MAX);
        assert_eq!(normalize_voice_volume(-1.0), VOICE_VOLUME_MIN);
        assert_eq!(normalize_voice_volume(f32::NAN), NPC_VOLUME_DEFAULT);
        // Defaults are unity, and the panel renders them as 100 %.
        let tts = TtsSettings::default();
        assert_eq!(tts.npc_volume, 1.0);
        assert_eq!(tts.admin_volume, 1.0);
        let view = tts_panel_view(&tts, &HashMap::new(), None);
        assert_eq!(view.npc_volume_pct, 100);
        assert_eq!(view.admin_volume_pct, 100);
        assert_eq!(view.voice_volume_max_pct, 200);
    }

    #[test]
    fn stt_defaults_to_whisper_provider_but_no_model() {
        let stt = SttSettings::default();
        assert_eq!(stt.provider, "whisper");
        // No default model for a public release — the user must pick one.
        assert_eq!(stt.model, "");
        let view = stt_panel_view(&stt, Vec::new(), RetrievalHostView::default());
        assert_eq!(view.providers.len(), 1);
        assert!(view.providers[0].selected);
        // No model selected by default.
        assert_eq!(view.model, "");
        // Legacy provider values (parakeet/sillytavern) normalize to whisper.
        assert_eq!(normalize_stt_provider("parakeet"), "whisper");
        assert_eq!(normalize_stt_provider("sillytavern"), "whisper");
        assert_eq!(normalize_stt_provider("whisper"), "whisper");
    }

    #[test]
    fn stt_effective_model_drops_stale_and_empty() {
        // A stale Parakeet model name from an old settings file → none (empty).
        let mut stt = SttSettings::default();
        stt.model = "nvidia/parakeet-tdt-0.6b-v3".to_string();
        assert_eq!(stt_effective_model(&stt), "");
        // A real whisper .bin is kept as-is.
        stt.model = "ggml-large-v3-turbo.bin".to_string();
        assert_eq!(stt_effective_model(&stt), "ggml-large-v3-turbo.bin");
        // Blank → none (no default).
        stt.model = "  ".to_string();
        assert_eq!(stt_effective_model(&stt), "");
    }

    #[test]
    fn whisper_registry_is_well_formed() {
        // Ids + filenames unique; every file is a .bin; the picker can find the
        // active model by filename; the default resolves.
        let mut ids = std::collections::HashSet::new();
        let mut files = std::collections::HashSet::new();
        for m in WHISPER_MODELS {
            assert!(ids.insert(m.id), "duplicate whisper id {}", m.id);
            assert!(files.insert(m.file), "duplicate whisper file {}", m.file);
            assert!(m.file.ends_with(".bin"), "{} is not a .bin", m.file);
            assert!(m.size_gb > 0.0);
        }
        assert!(whisper_model_by_id("large-v3-turbo").is_some());
        assert!(whisper_model_by_file("whisper-small-q5_1.bin").is_some());
        assert!(whisper_model_by_file("nope.bin").is_none());
        // The default constant points at a real registry entry.
        assert!(whisper_model_by_file(STT_WHISPER_DEFAULT_MODEL).is_some());
    }

    #[test]
    fn stt_panel_marks_active_model_selected() {
        let mut stt = SttSettings::default();
        stt.model = "ggml-large-v3-turbo.bin".to_string();
        let models = WHISPER_MODELS
            .iter()
            .map(|m| WhisperModelView {
                id: m.id.to_string(),
                name: m.name.to_string(),
                file: m.file.to_string(),
                size_label: format!("~{:.1} GB", m.size_gb),
                status: "available".to_string(),
                status_label: whisper_model_status_label("available"),
                downloaded: false,
                downloading: false,
                can_download: true,
                recommended: false,
                fit_hint: String::new(),
                selected: false,
            })
            .collect();
        let view = stt_panel_view(&stt, models, RetrievalHostView::default());
        let selected: Vec<&str> = view
            .models
            .iter()
            .filter(|m| m.selected)
            .map(|m| m.id.as_str())
            .collect();
        assert_eq!(selected, vec!["large-v3-turbo"]);
        assert_eq!(view.model, "ggml-large-v3-turbo.bin");
    }

    #[test]
    fn llm_filename_and_match_stem() {
        let e2b = LLM_MODELS.iter().find(|m| m.id == "gemma-4-e2b").unwrap();
        assert_eq!(llm_model_filename(e2b), "gemma-4-E2B-it-UD-Q4_K_XL.gguf");
        assert_eq!(llm_model_match_stem(e2b), "gemma-4-e2b-it");
        // The user's existing 26B file (a *different* quant) still matches its stem.
        let a4b = LLM_MODELS
            .iter()
            .find(|m| m.id == "gemma-4-26b-a4b")
            .unwrap();
        let stem = llm_model_match_stem(a4b);
        assert!("gemma-4-26b-a4b-it-ud-q4_k_s.gguf".contains(&stem));
    }

    #[test]
    fn llm_panel_marks_one_recommended() {
        // A 16 GB GPU: 12B fits comfortably, 26B/31B exceed → 12B recommended.
        let system = SystemInfo {
            cpu_cores: 8,
            ram_gb: Some(32.0),
            gpu_name: Some("Test GPU".into()),
            vram_total_gb: Some(16.0),
        };
        let panel = llm_models_panel_view(&HashMap::new(), &system, "");
        let recommended: Vec<&str> = panel
            .models
            .iter()
            .filter(|m| m.recommended)
            .map(|m| m.id.as_str())
            .collect();
        assert_eq!(recommended, vec!["gemma-4-12b"]);
        assert!(panel.host_summary.contains("16 GB VRAM"));
        // Every model defaults to "available" with no on-disk files.
        assert!(panel.models.iter().all(|m| m.status == "available"));
        // Nothing selected/selectable when no model is saved and none downloaded.
        assert!(panel.models.iter().all(|m| !m.selected && !m.selectable));
    }

    #[test]
    fn llm_panel_marks_selected_and_selectable() {
        let system = SystemInfo {
            cpu_cores: 8,
            ram_gb: Some(32.0),
            gpu_name: Some("Test GPU".into()),
            vram_total_gb: Some(16.0),
        };
        // E2B downloaded; E4B is the saved selection (even though not downloaded).
        let mut status = HashMap::new();
        status.insert("gemma-4-e2b".to_string(), "downloaded".to_string());
        let panel = llm_models_panel_view(&status, &system, "gemma-4-e4b");
        let e2b = panel.models.iter().find(|m| m.id == "gemma-4-e2b").unwrap();
        let e4b = panel.models.iter().find(|m| m.id == "gemma-4-e4b").unwrap();
        let e12b = panel.models.iter().find(|m| m.id == "gemma-4-12b").unwrap();
        // Downloaded → selectable, not the active one.
        assert!(e2b.selectable && !e2b.selected);
        // Saved selection → selected + selectable even though not downloaded.
        assert!(e4b.selected && e4b.selectable);
        // Neither downloaded nor selected → not selectable.
        assert!(!e12b.selectable && !e12b.selected);
    }

    #[test]
    fn selected_llm_model_id_resolves() {
        // Saved id that exists wins.
        let status = HashMap::new();
        assert_eq!(
            selected_llm_model_id("gemma-4-12b", &status),
            "gemma-4-12b"
        );
        // Unknown saved id → first downloaded model.
        let mut status = HashMap::new();
        status.insert("gemma-4-e4b".to_string(), "downloaded".to_string());
        assert_eq!(selected_llm_model_id("bogus", &status), "gemma-4-e4b");
        // Nothing saved + nothing downloaded → empty.
        assert_eq!(selected_llm_model_id("", &HashMap::new()), "");
    }

    #[test]
    fn llm_gguf_path_prefers_existing_file() {
        let dir = std::env::temp_dir().join(format!("sb-llm-gguf-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        // A different-quant file the user already has matches by stem.
        let existing = dir.join("gemma-4-26B-A4B-it-UD-Q4_K_S.gguf");
        fs::write(&existing, b"x").unwrap();
        let path = llm_model_gguf_path(&dir, "gemma-4-26b-a4b").unwrap();
        assert_eq!(path, existing);
        // A model with no file on disk falls back to the conventional filename.
        let fallback = llm_model_gguf_path(&dir, "gemma-4-e2b").unwrap();
        assert_eq!(fallback, dir.join("gemma-4-E2B-it-UD-Q4_K_XL.gguf"));
        // Unknown id → None.
        assert!(llm_model_gguf_path(&dir, "nope").is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn retrieval_registry_is_well_formed() {
        // Ids unique; kinds + tiers valid; cache dirs use the fastembed pattern.
        let mut ids = std::collections::HashSet::new();
        for m in RETRIEVAL_MODELS {
            assert!(ids.insert(m.id), "duplicate model id {}", m.id);
            assert!(
                m.kind == "embedder" || m.kind == "reranker",
                "bad kind {}",
                m.kind
            );
            assert!(
                m.cache_dir.starts_with("models--"),
                "cache dir {} not a fastembed dir",
                m.cache_dir
            );
            assert!(m.footprint_gb > 0.0);
        }
        // The two pre-downloaded models keep their verified on-disk names.
        let small = RETRIEVAL_MODELS
            .iter()
            .find(|m| m.id == "bge-small")
            .unwrap();
        assert_eq!(small.cache_dir, "models--Qdrant--bge-small-en-v1.5-onnx-Q");
        let jina = RETRIEVAL_MODELS
            .iter()
            .find(|m| m.id == "jina-turbo")
            .unwrap();
        assert_eq!(jina.cache_dir, "models--jinaai--jina-reranker-v1-turbo-en");
    }

    #[test]
    fn retrieval_status_labels() {
        assert_eq!(retrieval_model_status_label("downloaded"), "Downloaded");
        assert_eq!(retrieval_model_status_label("available"), "Available");
        assert_eq!(retrieval_model_status_label("weird"), "Available");
    }

    #[test]
    fn launcher_settings_round_trip() {
        // Defaults are all-blank (everything auto-detects).
        let mut settings = AppSettings::default();
        assert!(settings.launcher.mo2_exe.is_empty());
        assert!(settings.launcher.instance.is_empty());

        settings.launcher.mo2_exe = r"C:\Modding\MO2\ModOrganizer.exe".to_string();
        settings.launcher.instance = "New Vegas".to_string();
        settings.launcher.profile = "Default".to_string();
        settings.launcher.executable = "NVSE".to_string();
        settings.launcher.game_dir = r"C:\Games\FNV".to_string();

        let json = serde_json::to_string(&settings).unwrap();
        let back: AppSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.launcher.mo2_exe, r"C:\Modding\MO2\ModOrganizer.exe");
        assert_eq!(back.launcher.instance, "New Vegas");
        assert_eq!(back.launcher.profile, "Default");
        assert_eq!(back.launcher.executable, "NVSE");
        assert_eq!(back.launcher.game_dir, r"C:\Games\FNV");
    }

    #[test]
    fn launcher_settings_default_when_field_absent() {
        // Older settings files have no `launcher` key → serde default fills it.
        let json = r#"{"profile":"fallout-new-vegas"}"#;
        let settings: AppSettings = serde_json::from_str(json).unwrap();
        assert!(settings.launcher.mo2_exe.is_empty());
        assert!(settings.launcher.game_dir.is_empty());
    }

    #[test]
    fn preview_empty_when_disabled_or_unsupported() {
        let mut tts = TtsSettings::default();
        assert!(build_audio_tags_preview(&tts).is_empty());
        tts.audio_tags.enabled = true;
        tts.api_provider = "ElevenLabs".to_string();
        assert!(build_audio_tags_preview(&tts).contains("ElevenLabs v3"));
        tts.api_provider = "OpenAI".to_string();
        assert!(build_audio_tags_preview(&tts).is_empty());
    }

    #[test]
    fn interface_defaults_and_serde_backfill() {
        let i = InterfaceSettings::default();
        assert_eq!(i.theme, "midnight");
        assert_eq!(i.accent, "#55a7ff");
        assert_eq!(i.density, "comfortable");
        assert_eq!(i.font_scale, 100);
        assert!(i.show_timestamps && i.show_prompt_panel && !i.reduce_motion);
        // Older settings files (no `interface` key) backfill via serde default.
        let s: AppSettings = serde_json::from_str(r#"{"profile":"x"}"#).unwrap();
        assert_eq!(s.interface.accent, "#55a7ff");
    }

    #[test]
    fn accent_is_validated_to_hex() {
        assert_eq!(normalize_accent("#ABCDEF"), "#abcdef");
        assert_eq!(normalize_accent("#fff"), "#fff");
        // Injection / garbage falls back to the default (never reaches CSS raw).
        assert_eq!(normalize_accent("red;}body{display:none"), "#55a7ff");
        assert_eq!(normalize_accent("55a7ff"), "#55a7ff"); // missing '#'
        assert_eq!(normalize_accent(""), "#55a7ff");
    }

    #[test]
    fn font_scale_clamps_and_steps() {
        assert_eq!(normalize_font_scale(50), 90);
        assert_eq!(normalize_font_scale(999), 120);
        assert_eq!(normalize_font_scale(102), 100);
        assert_eq!(normalize_font_scale(103), 105);
    }

    #[test]
    fn theme_css_reflects_each_setting() {
        let mut i = InterfaceSettings::default();
        i.accent = "#5fb784".to_string();
        i.density = "compact".to_string();
        i.font_scale = 110;
        i.reduce_motion = true;
        i.show_timestamps = false;
        i.show_prompt_panel = false;
        i.theme = "ocean".to_string();
        let css = build_theme_css(&i);
        // Accent override + the ocean preset bg.
        assert!(css.contains("--accent:#5fb784;"));
        assert!(css.contains("--bg:#0c1016;"));
        // Font scale on the root.
        assert!(css.contains("font-size:110%;"));
        // Compact density emits the tighter padding rules.
        assert!(css.contains(".settings-main{padding:16px 20px 40px;}"));
        // Reduce motion + hide timestamps + collapse prompt panel.
        assert!(css.contains("transition:none!important"));
        assert!(css.contains(".message-meta time"));
        assert!(css.contains(".prompt{display:none!important;}"));

        // Defaults: no reduce-motion/timestamp/prompt rules emitted.
        let def = build_theme_css(&InterfaceSettings::default());
        assert!(def.contains("--accent:#55a7ff;"));
        assert!(!def.contains("transition:none!important"));
        assert!(!def.contains(".prompt{display:none!important;}"));
    }

    #[test]
    fn llm_sampling_defaults_match_prior_behaviour() {
        let s = LlmSamplingSettings::default();
        assert_eq!(s.temperature, 0.7); // the prior hard-coded temperature
        assert_eq!(s.top_p, 1.0);
        assert_eq!(s.top_k, 0); // off
        assert_eq!(s.min_p, 0.0); // off
        assert_eq!(s.repeat_penalty, 1.0); // off
        assert_eq!(s.max_tokens, 0); // no cap
        assert_eq!(s.n_ctx, 0); // model default
        assert_eq!(s.seed, -1); // random
                                // Backfills on an older settings file.
        let app: AppSettings = serde_json::from_str(r#"{"llm":{"model":"m"}}"#).unwrap();
        assert_eq!(app.llm.sampling.temperature, 0.7);
        assert_eq!(app.llm.model, "m");
    }

    #[test]
    fn llm_sampling_normalizes_out_of_range() {
        let raw = LlmSamplingSettings {
            temperature: 9.0,
            top_p: 5.0,
            top_k: 9_999,
            min_p: -1.0,
            repeat_penalty: 9.0,
            max_tokens: 999_999,
            n_ctx: 9_999_999,
            seed: -50,
        };
        let n = raw.normalized();
        assert_eq!(n.temperature, LLM_TEMPERATURE_MAX);
        assert_eq!(n.top_p, LLM_TOP_P_MAX);
        assert_eq!(n.top_k, LLM_TOP_K_MAX);
        assert_eq!(n.min_p, LLM_MIN_P_MIN);
        assert_eq!(n.repeat_penalty, LLM_REPEAT_PENALTY_MAX);
        assert_eq!(n.max_tokens, LLM_MAX_TOKENS_MAX);
        assert_eq!(n.n_ctx, LLM_N_CTX_MAX);
        assert_eq!(n.seed, -1); // a sub--1 seed normalizes to "random"
    }

    #[test]
    fn nav_groups_cover_every_category_once() {
        use std::collections::HashSet;
        let mut seen: HashSet<&str> = HashSet::new();
        for (_, keys) in SETTINGS_NAV_GROUPS {
            for key in *keys {
                assert!(seen.insert(key), "duplicate category {key} in nav groups");
                assert!(
                    SETTINGS_NAV.iter().any(|(k, _)| k == key),
                    "nav group key {key} missing from SETTINGS_NAV"
                );
            }
        }
        // Every nav category is placed in exactly one group.
        for (key, _) in SETTINGS_NAV {
            assert!(seen.contains(key), "category {key} not in any nav group");
        }
        // The active flag lands on the right item.
        let groups = settings_nav_groups("profiles");
        let active: Vec<&str> = groups
            .iter()
            .flat_map(|g| g.items.iter())
            .filter(|i| i.active)
            .map(|i| i.key.as_str())
            .collect();
        assert_eq!(active, vec!["profiles"]);
    }
}
