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
//!                `crate::start_llm_download` (+ `ensure_llamacpp`); select via
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
    retrieval_panel_view, selected_llm_model_id, AppSettings, LlmModelView, RetrievalModelView,
    TTS_LOCAL_ENGINES,
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
        // Curated recommended list: drop the small E2B / E4B Gemma variants — the
        // LLM page now recommends the larger, higher-quality models (12B and up).
        .filter(|m: &LlmModelView| m.id != "gemma-4-e2b" && m.id != "gemma-4-e4b")
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

/// Builds the STT "local model" catalog: a single card for the managed Parakeet
/// engine — the only managed local STT (hosted-API STT is configured through the
/// providers surface, not this picker). The card's status reflects the engine
/// install/run state so the LLM/STT pages can show whether local STT is ready.
fn stt_settings(state: &Arc<AppState>) -> UiModelSettings {
    let settings = AppSettings::load(&state.config.settings_path);
    let provider = chasm_core::normalize_stt_provider(&settings.stt.provider);
    let local_selected = provider == chasm_core::PROVIDER_LOCAL;

    let parakeet_status = crate::parakeet_engine_status(state);
    let parakeet_installed = parakeet_status == "installed";
    let parakeet_pill = if local_selected && parakeet_installed {
        if crate::launcher::parakeet_running(state) {
            UiModelStatus { tone: "ok", label: "Running".to_string() }
        } else {
            UiModelStatus { tone: "ok", label: "Active".to_string() }
        }
    } else {
        match parakeet_status.as_str() {
            "installed" => UiModelStatus { tone: "ok", label: "Ready".to_string() },
            "installing" => UiModelStatus { tone: "busy", label: "Installing…".to_string() },
            "failed" => UiModelStatus { tone: "error", label: "Install failed".to_string() },
            _ => UiModelStatus { tone: "idle", label: "Available".to_string() },
        }
    };
    let models = vec![UiModel {
        id: chasm_core::STT_PARAKEET_PICKER_ID.to_string(),
        name: "Parakeet TDT 0.6B v3".to_string(),
        description: Some(
            "NVIDIA Parakeet on its own local server (GPU) — voice input never \
             waits for the LLM. The only managed local STT engine."
                .to_string(),
        ),
        installed: parakeet_installed,
        recommended: true,
        meta: vec![
            UiModelMeta { label: "Size".to_string(), value: "~2.4 GB".to_string() },
            UiModelMeta { label: "Port".to_string(), value: "5003".to_string() },
        ],
        status: Some(parakeet_pill),
    }];

    UiModelSettings {
        models,
        selected_id: local_selected.then(|| chasm_core::STT_PARAKEET_PICKER_ID.to_string()),
        folder: Some(state.config.engines_dir.display().to_string()),
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
// Runtime (LLM runtime picker: the managed llama.cpp runtime)
// ---------------------------------------------------------------------------

/// Builds the Runtimes catalog: the single managed LLM runtime, llama.cpp
/// `llama-server`, with install state from the resolved exe / download markers.
/// This is the one-click auto-install card the Runtimes page keeps (only MODEL
/// FILES moved to guided manual placement). Rendered through the same
/// `<ModelPicker>` as the other domains.
fn runtime_settings(state: &Arc<AppState>) -> UiModelSettings {
    let status = crate::launcher::llamacpp_status(&state.config);
    let installed = status == crate::launcher::RuntimeStatus::Installed;
    let running = installed && crate::launcher::llm_runtime_running(state);
    let pill = if running {
        UiModelStatus { tone: "ok", label: "Running".to_string() }
    } else {
        match status {
            crate::launcher::RuntimeStatus::Installed => {
                UiModelStatus { tone: "ok", label: "Installed".to_string() }
            }
            crate::launcher::RuntimeStatus::Downloading => {
                UiModelStatus { tone: "busy", label: "Downloading…".to_string() }
            }
            crate::launcher::RuntimeStatus::Missing => {
                UiModelStatus { tone: "idle", label: "Not installed".to_string() }
            }
        }
    };

    let models = vec![UiModel {
        id: chasm_core::LLM_RUNTIME_LLAMACPP.to_string(),
        name: "llama.cpp (llama-server)".to_string(),
        description: Some(
            "The managed local LLM runtime (OpenAI-compatible on :5001), with \
             multiple prompt-cache slots so group-scene speaker swaps skip the \
             full prompt reprocess. Auto-downloads with one click."
                .to_string(),
        ),
        installed,
        recommended: true,
        meta: vec![
            UiModelMeta { label: "Slots".to_string(), value: "2 × 8k ctx".to_string() },
            UiModelMeta { label: "STT".to_string(), value: "Parakeet".to_string() },
        ],
        status: Some(pill),
    }];

    UiModelSettings {
        models,
        selected_id: Some(chasm_core::LLM_RUNTIME_LLAMACPP.to_string()),
        folder: Some(
            crate::launcher::llamacpp_managed_default(&state.config)
                .parent()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        ),
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
        "runtime" => runtime_settings(state),
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
            // The only local STT card is Parakeet; selecting it sets the provider
            // to the managed-local option and spawns the server.
            if id != chasm_core::STT_PARAKEET_PICKER_ID {
                return;
            }
            let prev = chasm_core::normalize_stt_provider(&settings.stt.provider);
            settings.stt.provider = chasm_core::PROVIDER_LOCAL.to_string();
            if settings.save(&state.config.settings_path).is_err() {
                return;
            }
            if prev != chasm_core::PROVIDER_LOCAL {
                let state = Arc::clone(state);
                tokio::task::spawn_blocking(move || {
                    crate::launcher::apply_selected_stt_provider(&state);
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
            // kill/respawn (unlike the LLM runtime/TTS, the embedder is in-process).
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
        "runtime" => {
            // llama.cpp is the only managed LLM runtime — there is nothing to
            // switch. Selection is a no-op (the card is informational; installing
            // it happens via the download endpoint).
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
                // LLM MODELS are now placed manually (guided browser-download +
                // drag-drop); this legacy path just ensures the managed runtime is
                // present. The React LLM page no longer exposes a model download.
                let _ = crate::start_llm_download(&state, &id);
                crate::ensure_llamacpp(&state);
            }
            "stt" => {
                // The Parakeet card installs the engine venv + prefetches the
                // .nemo (same install shape as the TTS engines). This is the only
                // managed local STT.
                if id == chasm_core::STT_PARAKEET_PICKER_ID {
                    let _ = crate::start_engine_install(&state, chasm_core::PARAKEET_ENGINE_ID);
                }
            }
            "retrieval" => {
                let _ = crate::start_retrieval_download(&state, &id);
            }
            "tts" => {
                let _ = crate::start_engine_install(&state, &id);
            }
            "runtime" => {
                if id == chasm_core::LLM_RUNTIME_LLAMACPP {
                    crate::ensure_llamacpp(&state);
                }
            }
            _ => {}
        }
    }
    Json(build_models(&state, &domain))
}
