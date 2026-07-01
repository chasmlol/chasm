//! UI models domain — LLM / TTS / STT / Retrieval settings endpoints.
//!
//! The four AI settings screens render through the SHARED React `<ModelPicker>`
//! and hit `GET /api/ui/v1/models/:domain` plus `POST .../select` and
//! `.../download`. Every domain returns the same [`UiModelSettings`] shape (a
//! `models: Vec<UiModel>` catalog + the selected id + the on-disk folder), so the
//! four screens are identical layouts fed different data.
//!
//! This module is a thin ADAPTER: it reuses the existing model cores rather than
//! reimplementing registries / downloads / swaps:
//!   * llm      → `LLM_MODELS` + `llm_models_panel_view` / `selected_llm_model_id`
//!                (status via `crate::llm_model_statuses`); download via
//!                `crate::start_llm_download` (+ `ensure_koboldcpp`); select via
//!                `launcher::apply_selected_llm_model`.
//!   * stt      → `crate::build_whisper_models` (WHISPER_MODELS + vram_gb fit);
//!                download via `crate::start_whisper_download`; select via
//!                `launcher::apply_selected_whisper_model`.
//!   * retrieval→ `crate::build_retrieval_models` (RETRIEVAL_MODELS); download via
//!                `crate::start_retrieval_download`; select persists the tier.
//!   * tts      → `TTS_LOCAL_ENGINES` + `crate::engine_statuses` (+ the running
//!                badge); download via `crate::start_engine_install`; select via
//!                `launcher::apply_selected_tts_engine`.
//!
//! Stays under `/api/ui/v1`; it configures models only and must not drive the
//! AI-stack lifecycle (the swap helpers only kill/respawn the one runtime whose
//! model changed — the same thing the Askama settings save already does).

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use chasm_core::{
    engine_status_label, llm_models_panel_view, normalize_embedder_tier, normalize_local_engine,
    retrieval_panel_view, selected_llm_model_id, stt_panel_view, whisper_model_by_id, AppSettings,
    LlmModelView, RetrievalModelView, TTS_LOCAL_ENGINES, WhisperModelView,
};

use crate::AppState;

/// A model-card status pill (tone + label), mapped 1:1 to the React `StatusTone`.
#[derive(Serialize)]
pub(crate) struct UiModelStatus {
    /// One of `ok` / `warn` / `error` / `busy` / `idle` (the React `StatusTone`).
    pub tone: &'static str,
    pub label: String,
}

/// One model card for the React `<ModelPicker>` (backend-shaped subset).
#[derive(Serialize)]
pub(crate) struct UiModel {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub installed: bool,
    pub recommended: bool,
    /// Free-form meta chips (size / VRAM / params).
    pub meta: Vec<UiModelMeta>,
    /// Explicit status pill (download/active/running). When present the picker
    /// renders it verbatim; when omitted it derives one from `installed`/selected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<UiModelStatus>,
}

#[derive(Serialize)]
pub(crate) struct UiModelMeta {
    pub label: String,
    pub value: String,
}

/// A model-settings payload: the catalog + selected id + the drop-files folder.
#[derive(Serialize)]
pub(crate) struct UiModelSettings {
    pub models: Vec<UiModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub folder: Option<String>,
}

impl UiModelSettings {
    fn empty() -> Self {
        Self {
            models: Vec::new(),
            selected_id: None,
            folder: None,
        }
    }
}

/// `{ "id": "<model-id>" }` body for select/download.
#[derive(Deserialize)]
pub(crate) struct ModelIdBody {
    #[serde(default)]
    id: String,
}

// ---------------------------------------------------------------------------
// Status-pill mapping (download status string → tone + label)
// ---------------------------------------------------------------------------

/// Maps a download-status string (`downloaded`/`downloading`/`failed`/…) plus a
/// `selected` flag to the `<ModelPicker>` status pill. `selected` only matters
/// when the model is on disk: an active downloaded model reads "Active".
fn download_status_pill(status: &str, label: String, selected: bool) -> UiModelStatus {
    match status {
        "downloaded" if selected => UiModelStatus { tone: "ok", label: "Active".to_string() },
        "downloaded" => UiModelStatus { tone: "ok", label: "Ready".to_string() },
        "downloading" => UiModelStatus { tone: "busy", label },
        "failed" => UiModelStatus { tone: "error", label },
        _ => UiModelStatus { tone: "idle", label },
    }
}

// ---------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------

/// Builds the LLM catalog from `LLM_MODELS` + on-disk GGUF status + host fit,
/// reusing the exact panel builder the Askama settings page uses.
fn llm_settings(state: &AppState) -> UiModelSettings {
    let settings = AppSettings::load(&state.config.settings_path);
    let model_status = crate::llm_model_statuses(&state.config.llm_models_dir);
    let selected = selected_llm_model_id(&settings.llm.model, &model_status);
    let panel = llm_models_panel_view(&model_status, &state.system_info, &selected);

    let models = panel
        .models
        .into_iter()
        .map(|m: LlmModelView| {
            let status = download_status_pill(&m.status, m.status_label, m.selected);
            UiModel {
                description: Some(m.repo),
                installed: m.downloaded,
                recommended: m.recommended,
                meta: vec![
                    UiModelMeta { label: "VRAM".to_string(), value: format!("~{:.0} GB", m.vram_gb) },
                    UiModelMeta { label: "Fit".to_string(), value: m.fit_label },
                ],
                status: Some(status),
                id: m.id,
                name: m.name,
            }
        })
        .collect();

    UiModelSettings {
        models,
        selected_id: (!selected.is_empty()).then_some(selected),
        folder: Some(state.config.llm_models_dir.display().to_string()),
    }
}

// ---------------------------------------------------------------------------
// STT (Whisper)
// ---------------------------------------------------------------------------

/// Builds the STT catalog from the Whisper registry (`WHISPER_MODELS`, vram_gb
/// fit) via the same `build_whisper_models` + `stt_panel_view` the Askama page
/// uses, so the recommended badge / fit hint / active selection all match.
fn stt_settings(state: &AppState) -> UiModelSettings {
    let settings = AppSettings::load(&state.config.settings_path);
    let (whisper_models, host) = crate::build_whisper_models(&settings, &state.system_info);
    let panel = stt_panel_view(&settings.stt, whisper_models, host);
    let selected_file = panel.model.clone();

    let models = panel
        .models
        .into_iter()
        .map(|m: WhisperModelView| {
            let status = download_status_pill(&m.status, m.status_label, m.selected);
            UiModel {
                description: Some(m.fit_hint),
                installed: m.downloaded,
                recommended: m.recommended,
                meta: vec![UiModelMeta { label: "Size".to_string(), value: m.size_label }],
                status: Some(status),
                id: m.id,
                name: m.name,
            }
        })
        .collect();

    // The picker's selected id is the registry id of the active `.bin` file.
    let selected_id = whisper_model_by_id(&selected_file)
        .map(|m| m.id.to_string())
        .or_else(|| {
            chasm_core::WHISPER_MODELS
                .iter()
                .find(|m| m.file == selected_file)
                .map(|m| m.id.to_string())
        });

    UiModelSettings {
        models,
        selected_id,
        folder: Some(
            crate::launcher::whisper_models_dir(&settings)
                .display()
                .to_string(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Retrieval (embedder / reranker)
// ---------------------------------------------------------------------------

/// Builds the Retrieval catalog from `RETRIEVAL_MODELS` via the same
/// `build_retrieval_models` + `retrieval_panel_view` the Askama page uses. The
/// "selected" card is the active EMBEDDER (the model the picker swaps); rerankers
/// are shown with their own kind/tier meta but the picker selects an embedder.
fn retrieval_settings(state: &AppState) -> UiModelSettings {
    let settings = AppSettings::load(&state.config.settings_path);
    let (retrieval_models, host) = crate::build_retrieval_models(&state.system_info);
    let panel = retrieval_panel_view(&settings.retrieval, retrieval_models, host);

    let mut selected_id: Option<String> = None;
    let models = panel
        .models
        .into_iter()
        .map(|m: RetrievalModelView| {
            // The embedder whose tier is active is the picker's selection.
            let is_selected_embedder = m.kind == "embedder" && m.selected;
            if is_selected_embedder {
                selected_id = Some(m.id.clone());
            }
            // Mark the active embedder AND the active reranker (per its tier) as
            // selected in their pills, so each split section shows its own pick.
            let status = download_status_pill(&m.status, m.status_label, m.selected);
            let kind_label = if m.kind == "reranker" { "Reranker" } else { "Embedder" };
            UiModel {
                description: Some(m.fit_hint),
                installed: m.downloaded,
                recommended: m.recommended,
                meta: vec![
                    UiModelMeta { label: "Kind".to_string(), value: kind_label.to_string() },
                    UiModelMeta { label: "Tier".to_string(), value: m.tier },
                    UiModelMeta { label: "Size".to_string(), value: m.size_label },
                ],
                status: Some(status),
                id: m.id,
                name: m.label,
            }
        })
        .collect();

    UiModelSettings {
        models,
        selected_id,
        folder: Some(chasm_embed::embed_cache_dir().display().to_string()),
    }
}

// ---------------------------------------------------------------------------
// TTS (engine picker)
// ---------------------------------------------------------------------------

/// Builds the TTS catalog from `TTS_LOCAL_ENGINES` + per-engine install status,
/// reusing `engine_statuses` (and `faster_qwen3_tts_installed` / `tts_running_engine`
/// for the running badge) — the same sources the Askama TTS page uses. The
/// selected card is the saved local engine. Takes the `Arc` directly because the
/// running-badge check (`tts_running_engine`) needs it.
fn tts_settings(state: &Arc<AppState>) -> UiModelSettings {
    let settings = AppSettings::load(&state.config.settings_path);
    let selected = normalize_local_engine(&settings.tts.local_engine);
    let faster_installed = crate::launcher::faster_qwen3_tts_installed(&settings, &state.config);
    let statuses = crate::engine_statuses(&state.config.engines_dir, faster_installed);
    let running = crate::launcher::tts_running_engine(state);

    let models = TTS_LOCAL_ENGINES
        .iter()
        .map(|(id, label)| {
            let status = statuses.get(*id).map(String::as_str).unwrap_or("not_installed");
            let installed = status == "installed";
            let is_running = running.as_deref() == Some(*id);
            let is_selected = *id == selected;
            let pill = if is_running {
                UiModelStatus { tone: "ok", label: "Running".to_string() }
            } else {
                match status {
                    "installed" if is_selected => UiModelStatus { tone: "ok", label: "Selected".to_string() },
                    "installed" => UiModelStatus { tone: "ok", label: "Installed".to_string() },
                    "installing" => UiModelStatus { tone: "busy", label: engine_status_label(status) },
                    "failed" => UiModelStatus { tone: "error", label: engine_status_label(status) },
                    _ => UiModelStatus { tone: "idle", label: engine_status_label(status) },
                }
            };
            UiModel {
                description: Some("Streaming OpenAI /v1/audio/speech engine".to_string()),
                installed,
                recommended: false,
                meta: Vec::new(),
                status: Some(pill),
                id: (*id).to_string(),
                name: (*label).to_string(),
            }
        })
        .collect();

    UiModelSettings {
        models,
        // Empty = no engine selected (no default), mirror LLM/STT and emit None so
        // the picker shows nothing checked instead of an empty-string selection.
        selected_id: (!selected.is_empty()).then_some(selected),
        folder: Some(state.config.engines_dir.display().to_string()),
    }
}

// ---------------------------------------------------------------------------
// Routing handlers
// ---------------------------------------------------------------------------

/// `GET /api/ui/v1/models/:domain` — the model catalog for one AI domain.
pub(crate) async fn get_models(
    State(state): State<Arc<AppState>>,
    Path(domain): Path<String>,
) -> Json<UiModelSettings> {
    Json(build_models(&state, &domain))
}

/// Builds the catalog for `domain`; an unknown domain returns the empty catalog.
fn build_models(state: &Arc<AppState>, domain: &str) -> UiModelSettings {
    match domain {
        "llm" => llm_settings(state),
        "stt" => stt_settings(state),
        "retrieval" => retrieval_settings(state),
        "tts" => tts_settings(state),
        _ => UiModelSettings::empty(),
    }
}

/// `POST /api/ui/v1/models/:domain/select` — set the active model and return the
/// fresh catalog. Persists into `AppSettings` exactly as the Askama save does,
/// then applies the swap (kill/respawn the one runtime whose model changed) the
/// same way the settings save does — off the async path, best-effort.
pub(crate) async fn select_model(
    State(state): State<Arc<AppState>>,
    Path(domain): Path<String>,
    Json(body): Json<ModelIdBody>,
) -> Json<UiModelSettings> {
    let id = body.id.trim().to_string();
    if !id.is_empty() {
        apply_select(&state, &domain, &id);
    }
    Json(build_models(&state, &domain))
}

/// Persists the selection for `domain` and applies the live swap. Mirrors the
/// per-category save logic in `save_settings`.
fn apply_select(state: &Arc<AppState>, domain: &str, id: &str) {
    let mut settings = AppSettings::load(&state.config.settings_path);
    match domain {
        "llm" => {
            // Only accept a known LLM id, like apply_llm_form.
            if !chasm_core::LLM_MODELS.iter().any(|m| m.id == id) {
                return;
            }
            let prev = settings.llm.model.trim().to_string();
            settings.llm.model = id.to_string();
            if settings.save(&state.config.settings_path).is_err() {
                return;
            }
            if settings.llm.model.trim() != prev {
                let state = Arc::clone(state);
                tokio::task::spawn_blocking(move || {
                    crate::launcher::apply_selected_llm_model(&state);
                });
            }
        }
        "stt" => {
            // The picker carries a registry id; persist the model's `.bin` file
            // (the value `stt.model` stores, like apply_stt_form).
            let Some(model) = whisper_model_by_id(id) else {
                return;
            };
            let prev = chasm_core::stt_effective_model(&settings.stt);
            settings.stt.model = model.file.to_string();
            if settings.save(&state.config.settings_path).is_err() {
                return;
            }
            let new = chasm_core::stt_effective_model(&settings.stt);
            if new != prev {
                let state = Arc::clone(state);
                let file = new.clone();
                tokio::task::spawn_blocking(move || {
                    crate::launcher::apply_selected_whisper_model(&state, &file);
                });
            }
        }
        "retrieval" => {
            // The picker selects an EMBEDDER; persist its tier (like apply_retrieval_form).
            let Some(model) = chasm_core::RETRIEVAL_MODELS
                .iter()
                .find(|m| m.id == id && m.kind == "embedder")
            else {
                return;
            };
            settings.retrieval.embedder_tier = normalize_embedder_tier(model.tier);
            let _ = settings.save(&state.config.settings_path);
            // The retriever loads lazily on the next turn that needs it; no live
            // kill/respawn (unlike koboldcpp/TTS, the embedder is in-process).
        }
        "tts" => {
            let prev = normalize_local_engine(&settings.tts.local_engine);
            settings.tts.local_engine = normalize_local_engine(id);
            if settings.save(&state.config.settings_path).is_err() {
                return;
            }
            if settings.tts.local_engine != prev {
                let state = Arc::clone(state);
                tokio::task::spawn_blocking(move || {
                    crate::launcher::apply_selected_tts_engine(&state);
                });
            }
        }
        _ => {}
    }
}

/// `POST /api/ui/v1/models/:domain/download` — start a model download and return
/// the fresh catalog (the card flips to a "Downloading" pill via its on-disk
/// `.downloading` marker on the next poll). Reuses the exact same per-category
/// download starters the Askama settings page fires.
pub(crate) async fn download_model(
    State(state): State<Arc<AppState>>,
    Path(domain): Path<String>,
    Json(body): Json<ModelIdBody>,
) -> Json<UiModelSettings> {
    let id = body.id.trim().to_string();
    if !id.is_empty() {
        match domain.as_str() {
            "llm" => {
                let _ = crate::start_llm_download(&state, &id);
                // One download also pulls the koboldcpp runtime if absent.
                crate::ensure_koboldcpp(&state);
            }
            "stt" => {
                let _ = crate::start_whisper_download(&state, &id);
                crate::ensure_koboldcpp(&state);
            }
            "retrieval" => {
                let _ = crate::start_retrieval_download(&state, &id);
            }
            "tts" => {
                let _ = crate::start_engine_install(&state, &id);
            }
            _ => {}
        }
    }
    Json(build_models(&state, &domain))
}
