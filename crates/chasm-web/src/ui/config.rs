//! UI config domain — the per-engine CONFIGURATION fields (LLM sampling, TTS
//! tuning/volumes, STT params, Retrieval tuning) for the four AI settings
//! screens.
//!
//! The model PICKER (which model/engine is active) lives in [`super::models`];
//! THIS module surfaces the per-engine knobs the legacy Askama settings pages
//! exposed (temperature, top-p, voice volumes, min-score, …) so the redesigned
//! React screens can edit + persist them again.
//!
//! It is a thin ADAPTER over the EXISTING legacy apply/normalize path: the POST
//! handler reconstructs the same `HashMap<String, String>` form body the Askama
//! save built and calls the SAME `apply_*_form` functions
//! (`crate::apply_llm_form` / `apply_tts_form` / `apply_stt_form` /
//! `apply_retrieval_form`), so a saved value round-trips and takes effect
//! exactly as it did before — no forked behaviour. The GET handler reads the
//! same `AppSettings` (normalized) the request path reads.
//!
//! IMPORTANT: config only. It deliberately does NOT carry the model/engine
//! selection (that stays with the picker's `/models/:domain/select`) and never
//! drives the AI-stack lifecycle. Every field here is read fresh per request by
//! the runtime, so a save takes effect on the next turn with no restart. Stays
//! under `/api/ui/v1`.

use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use chasm_core::{AppSettings, LlmSamplingSettings, TtsTuningSettings};

use crate::{AppState, WebResult};

// ---------------------------------------------------------------------------
// LLM config (generation sampling)
// ---------------------------------------------------------------------------

/// The LLM generation-sampling knobs the Askama LLM page exposed (mirrors
/// [`LlmSamplingSettings`], read normalized). Round-trips through
/// `crate::apply_llm_sampling_form` via the `sampling_*` form keys.
#[derive(Serialize, Deserialize)]
pub(crate) struct LlmConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub max_tokens: u32,
    pub n_ctx: u32,
    pub seed: i64,
}

impl LlmConfig {
    fn from_settings(s: &LlmSamplingSettings) -> Self {
        let n = s.normalized();
        Self {
            temperature: n.temperature,
            top_p: n.top_p,
            top_k: n.top_k,
            min_p: n.min_p,
            repeat_penalty: n.repeat_penalty,
            max_tokens: n.max_tokens,
            n_ctx: n.n_ctx,
            seed: n.seed,
        }
    }

    /// Builds the `sampling_*` form body the legacy `apply_llm_form` reads.
    fn to_form(&self) -> HashMap<String, String> {
        let mut form = HashMap::new();
        form.insert("sampling_temperature".into(), self.temperature.to_string());
        form.insert("sampling_top_p".into(), self.top_p.to_string());
        form.insert("sampling_top_k".into(), self.top_k.to_string());
        form.insert("sampling_min_p".into(), self.min_p.to_string());
        form.insert(
            "sampling_repeat_penalty".into(),
            self.repeat_penalty.to_string(),
        );
        form.insert("sampling_max_tokens".into(), self.max_tokens.to_string());
        form.insert("sampling_n_ctx".into(), self.n_ctx.to_string());
        form.insert("sampling_seed".into(), self.seed.to_string());
        form
    }
}

// ---------------------------------------------------------------------------
// TTS config (volumes + per-request synthesis tuning)
// ---------------------------------------------------------------------------

/// The TTS config the Askama TTS page exposed: voice volumes + the live
/// synthesis tuning ([`TtsTuningSettings`]). Volumes are surfaced as the same
/// percent (100 = unity) the legacy slider posted, so `apply_tts_form` stores
/// the identical multiplier. Round-trips via `apply_tts_form`.
#[derive(Serialize, Deserialize)]
pub(crate) struct TtsConfig {
    /// NPC voice volume as a percent (100 = unity).
    pub npc_volume_pct: f32,
    /// Admin voice volume as a percent (100 = unity).
    pub admin_volume_pct: f32,
    pub lead_in_ms: u32,
    pub trailing_ms: u32,
    pub sentence_gap_ms: u32,
    pub gain_db: f32,
    pub temperature: f32,
    pub lsd_decode_steps: u32,
    pub eos_threshold: f32,
    pub noise_clamp: f32,
    pub max_tokens: u32,
    pub frames_after_eos: u32,
}

impl TtsConfig {
    fn from_settings(s: &chasm_core::TtsSettings) -> Self {
        let t: TtsTuningSettings = s.tuning.normalized();
        Self {
            // Stored as a multiplier; the slider/legacy form speaks percent.
            npc_volume_pct: chasm_core::normalize_voice_volume(s.npc_volume) * 100.0,
            admin_volume_pct: chasm_core::normalize_voice_volume(s.admin_volume) * 100.0,
            lead_in_ms: t.lead_in_ms,
            trailing_ms: t.trailing_ms,
            sentence_gap_ms: t.sentence_gap_ms,
            gain_db: t.gain_db,
            temperature: t.temperature,
            lsd_decode_steps: t.lsd_decode_steps,
            eos_threshold: t.eos_threshold,
            noise_clamp: t.noise_clamp,
            max_tokens: t.max_tokens,
            frames_after_eos: t.frames_after_eos,
        }
    }

    /// Builds the volume + `tuning_*` form body `apply_tts_form` reads. Only the
    /// config keys are emitted; the engine/mode/streaming/audio-tag keys are left
    /// absent so `apply_tts_form` leaves those (and the engine selection) untouched.
    fn to_form(&self) -> HashMap<String, String> {
        let mut form = HashMap::new();
        form.insert("npc_volume".into(), self.npc_volume_pct.to_string());
        form.insert("admin_volume".into(), self.admin_volume_pct.to_string());
        form.insert("tuning_lead_in_ms".into(), self.lead_in_ms.to_string());
        form.insert("tuning_trailing_ms".into(), self.trailing_ms.to_string());
        form.insert(
            "tuning_sentence_gap_ms".into(),
            self.sentence_gap_ms.to_string(),
        );
        form.insert("tuning_gain_db".into(), self.gain_db.to_string());
        form.insert("tuning_temperature".into(), self.temperature.to_string());
        form.insert(
            "tuning_lsd_decode_steps".into(),
            self.lsd_decode_steps.to_string(),
        );
        form.insert(
            "tuning_eos_threshold".into(),
            self.eos_threshold.to_string(),
        );
        form.insert("tuning_noise_clamp".into(), self.noise_clamp.to_string());
        form.insert("tuning_max_tokens".into(), self.max_tokens.to_string());
        form.insert(
            "tuning_frames_after_eos".into(),
            self.frames_after_eos.to_string(),
        );
        form
    }
}

// ---------------------------------------------------------------------------
// STT config (language / prompt / timeout)
// ---------------------------------------------------------------------------

/// The STT config the Askama STT page exposed (besides the model picker):
/// language hint, biasing prompt, and per-request timeout. Round-trips via
/// `apply_stt_form` (the `model`/`provider` keys are left absent so the picker's
/// selection is untouched).
#[derive(Serialize, Deserialize)]
pub(crate) struct SttConfig {
    pub language: String,
    pub prompt: String,
    pub timeout_ms: u64,
}

impl SttConfig {
    fn from_settings(s: &chasm_core::SttSettings) -> Self {
        Self {
            language: s.language.clone(),
            prompt: s.prompt.clone(),
            timeout_ms: chasm_core::normalize_stt_timeout_ms(s.timeout_ms),
        }
    }

    fn to_form(&self) -> HashMap<String, String> {
        let mut form = HashMap::new();
        form.insert("language".into(), self.language.clone());
        form.insert("prompt".into(), self.prompt.clone());
        form.insert("timeout_ms".into(), self.timeout_ms.to_string());
        form
    }
}

// ---------------------------------------------------------------------------
// Retrieval config (tiers / toggles / limits / scores)
// ---------------------------------------------------------------------------

/// The Retrieval config the Askama Retrieval page exposed (besides the embedder
/// picker): master + per-source toggles, reranker tier/toggle, execution
/// provider, and the recall/score knobs. Round-trips via `apply_retrieval_form`.
///
/// `embedder_tier` is intentionally NOT included — that is the picker's
/// selection (`/models/retrieval/select`), so leaving its form key absent means
/// `apply_retrieval_form` keeps whatever the picker last set.
#[derive(Serialize, Deserialize)]
pub(crate) struct RetrievalConfig {
    pub enabled: bool,
    pub chat_memory_enabled: bool,
    pub lore_semantic_enabled: bool,
    pub action_semantic_enabled: bool,
    pub quest_semantic_enabled: bool,
    pub reranker_enabled: bool,
    pub reranker_tier: String,
    pub execution: String,
    pub top_k: u32,
    pub candidates: u32,
    pub min_score: f32,
    pub action_min_score: f32,
    pub chat_memory_limit: u32,
    pub lore_limit: u32,
    pub quest_limit: u32,
}

impl RetrievalConfig {
    fn from_settings(s: &chasm_core::RetrievalSettings) -> Self {
        Self {
            enabled: s.enabled,
            chat_memory_enabled: s.chat_memory_enabled,
            lore_semantic_enabled: s.lore_semantic_enabled,
            action_semantic_enabled: s.action_semantic_enabled,
            quest_semantic_enabled: s.quest_semantic_enabled,
            reranker_enabled: s.reranker_enabled,
            reranker_tier: chasm_core::normalize_reranker_tier(&s.reranker_tier),
            execution: chasm_core::normalize_execution(&s.execution),
            top_k: s.top_k,
            candidates: s.candidates,
            min_score: s.min_score,
            action_min_score: s.action_min_score,
            chat_memory_limit: s.chat_memory_limit,
            lore_limit: s.lore_limit,
            quest_limit: s.quest_limit,
        }
    }

    /// Builds the form body `apply_retrieval_form` reads. The toggles use
    /// checkbox semantics (present = on), so a `false` toggle omits its key.
    fn to_form(&self) -> HashMap<String, String> {
        let mut form = HashMap::new();
        let flag = |form: &mut HashMap<String, String>, key: &str, on: bool| {
            if on {
                form.insert(key.to_string(), "on".to_string());
            }
        };
        flag(&mut form, "enabled", self.enabled);
        flag(&mut form, "chat_memory_enabled", self.chat_memory_enabled);
        flag(&mut form, "lore_semantic_enabled", self.lore_semantic_enabled);
        flag(
            &mut form,
            "action_semantic_enabled",
            self.action_semantic_enabled,
        );
        flag(
            &mut form,
            "quest_semantic_enabled",
            self.quest_semantic_enabled,
        );
        flag(&mut form, "reranker_enabled", self.reranker_enabled);
        form.insert("reranker_tier".into(), self.reranker_tier.clone());
        form.insert("execution".into(), self.execution.clone());
        form.insert("top_k".into(), self.top_k.to_string());
        form.insert("candidates".into(), self.candidates.to_string());
        form.insert("min_score".into(), self.min_score.to_string());
        form.insert(
            "action_min_score".into(),
            self.action_min_score.to_string(),
        );
        form.insert(
            "chat_memory_limit".into(),
            self.chat_memory_limit.to_string(),
        );
        form.insert("lore_limit".into(), self.lore_limit.to_string());
        form.insert("quest_limit".into(), self.quest_limit.to_string());
        form
    }
}

// ---------------------------------------------------------------------------
// Combined payload (one shape per domain; only one field is populated)
// ---------------------------------------------------------------------------

/// The config payload for one AI domain. Exactly one field is populated for a
/// given `:domain`; the others are `None`. Mirrors how [`super::models`] returns
/// one shape per domain.
#[derive(Serialize)]
pub(crate) struct UiConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm: Option<LlmConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tts: Option<TtsConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stt: Option<SttConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval: Option<RetrievalConfig>,
}

impl UiConfig {
    fn empty() -> Self {
        Self {
            llm: None,
            tts: None,
            stt: None,
            retrieval: None,
        }
    }

    fn build(settings: &AppSettings, domain: &str) -> Self {
        let mut out = Self::empty();
        match domain {
            "llm" => out.llm = Some(LlmConfig::from_settings(&settings.llm.sampling)),
            "tts" => out.tts = Some(TtsConfig::from_settings(&settings.tts)),
            "stt" => out.stt = Some(SttConfig::from_settings(&settings.stt)),
            "retrieval" => {
                out.retrieval = Some(RetrievalConfig::from_settings(&settings.retrieval))
            }
            _ => {}
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Routing handlers
// ---------------------------------------------------------------------------

/// `GET /api/ui/v1/config/:domain` — the per-engine config for one AI domain,
/// read (normalized) from the same `AppSettings` the runtime reads.
pub(crate) async fn get_config(
    State(state): State<Arc<AppState>>,
    Path(domain): Path<String>,
) -> Json<UiConfig> {
    let settings = AppSettings::load(&state.config.settings_path);
    Json(UiConfig::build(&settings, &domain))
}

/// `POST /api/ui/v1/config/:domain` — persist the edited config and return the
/// fresh view. Reconstructs the legacy form body and calls the SAME
/// `apply_*_form` function the Askama save used, so the saved values normalize +
/// take effect exactly as before. Only the posted domain's section is touched;
/// the model/engine selection (the picker's job) is never disturbed.
pub(crate) async fn save_config(
    State(state): State<Arc<AppState>>,
    Path(domain): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> WebResult<Json<UiConfig>> {
    let mut settings = AppSettings::load(&state.config.settings_path);
    match domain.as_str() {
        "llm" => {
            if let Ok(cfg) = serde_json::from_value::<LlmConfig>(body) {
                crate::apply_llm_form(&mut settings.llm, &cfg.to_form());
            }
        }
        "tts" => {
            if let Ok(cfg) = serde_json::from_value::<TtsConfig>(body) {
                crate::apply_tts_form(&mut settings.tts, &cfg.to_form());
            }
        }
        "stt" => {
            if let Ok(cfg) = serde_json::from_value::<SttConfig>(body) {
                crate::apply_stt_form(&mut settings.stt, &cfg.to_form());
            }
        }
        "retrieval" => {
            if let Ok(cfg) = serde_json::from_value::<RetrievalConfig>(body) {
                crate::apply_retrieval_form(&mut settings.retrieval, &cfg.to_form());
            }
        }
        _ => return Ok(Json(UiConfig::empty())),
    }
    settings.save(&state.config.settings_path)?;
    Ok(Json(UiConfig::build(&settings, &domain)))
}
