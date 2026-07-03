use std::{
    collections::HashMap,
    fs,
    net::SocketAddr,
    process::{Command, Stdio},
    sync::{Arc, OnceLock},
};

use askama::Template;
use axum::{
    body::Body,
    extract::{Form, Path, Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Redirect, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::Serialize;
use chasm_core::{
    clone_status_label, falloutnv_detected, llm_model_filename,
    llm_model_match_stem, llm_models_panel_view, mo2_detected, normalize_embedder_tier,
    normalize_execution, normalize_max_tags, normalize_reranker_tier,
    normalize_stt_provider, nvse_detected,
    recommended_index, retrieval_model_status_label, settings_page_view,
    stt_effective_model, whisper_model_by_id, whisper_model_status_label,
    AppConfig, AppSettings, GameLauncherView, GameProfile, GpuFit, InterfaceSettings, LauncherConfig,
    LauncherSettings, LiveChatView, LlmSamplingSettings, LlmSettings, MessageView,
    ParticipantView, ProfileCardView, ProfilesPanelView, PromptAssemblyView,
    RetrievalHostView, RetrievalModelView, RetrievalSettings, SettingsPageView,
    SttSettings, SystemInfo, TtsSettings, TtsTuningSettings, VoiceCloneCharacterView, VoiceCloneView,
    WhisperModelView, LLM_MODELS, ORCHESTRATOR_DEFAULT_SYSTEM_PROMPT, ORCHESTRATOR_MAX_SPEAKERS_MAX,
    ORCHESTRATOR_MAX_SPEAKERS_MIN, ORCHESTRATOR_TEMPERATURE_MAX, ORCHESTRATOR_TEMPERATURE_MIN,
    RETRIEVAL_CANDIDATES_MAX, RETRIEVAL_CANDIDATES_MIN, RETRIEVAL_MODELS, RETRIEVAL_SOURCE_LIMIT_MAX,
    RETRIEVAL_SOURCE_LIMIT_MIN, RETRIEVAL_TOP_K_MAX, RETRIEVAL_TOP_K_MIN, TTS_LOCAL_ENGINES,
    WHISPER_MODELS, WHISPER_REPO,
};
use chasm_embed::{embed_cache_dir, EmbeddingCache, Retriever, RetrieverConfig};
use chasm_st_compat::{CompatError, LiveChatRepository};
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::info;

mod books;
mod capture;
mod connection;
mod fnv_bridge;
mod game_bridge;
mod gamemaster;
mod generate;
mod launcher;
mod llm;
mod orchestrator;
mod persona;
mod save_sync;
mod stack_lifecycle;
mod trace_routes;
mod ui;
mod warmup;

pub struct AppState {
    pub config: AppConfig,
    pub repository: LiveChatRepository,
    /// Lazily-loaded shared retriever. Loading downloads/loads ONNX models and is
    /// expensive, so we never load at startup: the first turn that needs
    /// retrieval triggers the load, and the result (`Some` on success, `None` on
    /// disabled/offline/failed) is memoized here. The keyword path keeps working
    /// regardless, so a `None` is a graceful degradation, not an error.
    retriever: OnceLock<Option<Retriever>>,
    /// Persistent embedding cache, opened once under the active profile
    /// (`profiles/<id>/embed-cache`, legacy fallback). `None` only if the cache
    /// dir can't be created.
    embed_cache: OnceLock<Option<EmbeddingCache>>,
    /// Host hardware detected once at boot; drives the "recommended" model badges.
    pub system_info: SystemInfo,
    /// Connection-driven AI stack lifecycle phase. Driven by the
    /// [`stack_lifecycle`] task (game connect → start stack, disconnect → stop),
    /// read by `connection_status`. Shared via `Arc<AppState>`.
    pub(crate) lifecycle: stack_lifecycle::StackLifecycle,
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        // The repository resolves content under the ACTIVE profile per request
        // (it re-reads settings + profile each call), with a legacy data-root
        // fallback. Voices + embed-cache legacy bases are passed for the two
        // content kinds whose legacy location is not under `data_root`.
        let repository = LiveChatRepository::with_profiles(
            config.data_root.clone(),
            config.profiles_dir.clone(),
            config.settings_path.clone(),
            config.voices_dir.clone(),
            config.legacy_embed_cache_dir(),
        );
        let system_info = SystemInfo::detect();
        info!(
            "host: {} cores, {} RAM, GPU {} ({})",
            system_info.cpu_cores,
            system_info
                .ram_gb
                .map(|v| format!("{v:.0} GB"))
                .unwrap_or_else(|| "?".into()),
            system_info.gpu_name.as_deref().unwrap_or("none"),
            system_info
                .vram_total_gb
                .map(|v| format!("{v:.0} GB VRAM"))
                .unwrap_or_else(|| "no VRAM".into()),
        );
        Self {
            config,
            repository,
            retriever: OnceLock::new(),
            embed_cache: OnceLock::new(),
            system_info,
            lifecycle: stack_lifecycle::StackLifecycle::default(),
        }
    }

    /// Returns the shared retriever, loading it ON FIRST USE from the persisted
    /// `RetrievalSettings`. Returns `None` (and logs) when retrieval is disabled
    /// or the model can't be loaded (e.g. offline / no weights), so callers fall
    /// back to the keyword path. Never blocks app startup — only the first turn
    /// that asks for retrieval pays the load cost, and the outcome is memoized.
    pub fn retriever(&self) -> Option<&Retriever> {
        self.retriever
            .get_or_init(|| {
                let settings = AppSettings::load(&self.config.settings_path);
                let r = &settings.retrieval;
                if !r.enabled {
                    info!("retrieval disabled in settings; using keyword path only");
                    return None;
                }
                let cfg = RetrieverConfig {
                    embedder_tier: r.embedder_tier.clone(),
                    reranker_enabled: r.reranker_enabled,
                    reranker_tier: r.reranker_tier.clone(),
                    execution: r.execution.clone(),
                };
                // Gate on presence: `Retriever::load` DOWNLOADS the ONNX weights
                // (~144MB) on first use. We never auto-download — only load when the
                // selected model is already on disk. The user fetches it explicitly
                // via Settings → Retrieval; until then we use the keyword path.
                if !chasm_embed::models_present(&cfg) {
                    info!(
                        "retrieval model not downloaded; download it in Settings → Retrieval — using keyword path until then"
                    );
                    return None;
                }
                match Retriever::load(&cfg) {
                    Ok(retriever) => {
                        info!(
                            "semantic retriever loaded (model {}, {:?}, reranker: {})",
                            retriever.model_id(),
                            retriever.device(),
                            if retriever.has_reranker() {
                                "on"
                            } else {
                                "off (cosine fallback)"
                            }
                        );
                        Some(retriever)
                    }
                    Err(error) => {
                        info!("retriever load failed ({error}); falling back to keyword path");
                        None
                    }
                }
            })
            .as_ref()
    }

    /// Returns the shared embedding cache, opened on first use. The cache dir is
    /// the ACTIVE profile's `embed-cache` (`profiles/<id>/embed-cache`), falling
    /// back to the legacy `CHASM_EMBED_DIR` / `<data_root>/embed-cache` when
    /// the profile has no such dir. Resolved once (on first use) and memoized.
    pub fn embed_cache(&self) -> Option<&EmbeddingCache> {
        self.embed_cache
            .get_or_init(|| {
                let dir = self.config.active_profile_paths().embed_cache_dir();
                match EmbeddingCache::open(&dir) {
                    Ok(cache) => Some(cache),
                    Err(error) => {
                        info!("embedding cache open failed ({error}); retrieval disabled");
                        None
                    }
                }
            })
            .as_ref()
    }

    /// Non-triggering view of whether the retriever is ALREADY loaded (embedder,
    /// plus reranker if enabled). Unlike [`Self::retriever`], this NEVER kicks off
    /// a (potentially multi-second) load — it just reports the memoized slot — so
    /// the status endpoint can poll it cheaply every couple of seconds.
    pub(crate) fn retriever_loaded(&self) -> Option<&Retriever> {
        self.retriever.get().and_then(|slot| slot.as_ref())
    }
}

#[derive(Template)]
#[template(path = "live_chat.html")]
struct LiveChatTemplate {
    live_chat: LiveChatView,
    selected_participant: Option<ParticipantView>,
    messages: Vec<MessageView>,
    prompt: PromptAssemblyView,
    sidebar: SidebarView,
}

/// One profile option in the sidebar profile-selector menu.
struct ProfileOptionView {
    id: String,
    name: String,
    description: String,
    /// Two-letter initials used by the menu's badge.
    initials: String,
    active: bool,
}

/// Everything the left sidebar needs: the active profile (name) plus the full
/// profile list (for the switcher) and the in-scene roster count. The Library
/// now lives on the rail, so no book counts are surfaced here.
struct SidebarView {
    /// Active profile name (e.g. "Fallout: New Vegas"), or a fallback label.
    profile_name: String,
    /// Active profile initials for the badge (e.g. "FN").
    profile_initials: String,
    /// All profiles for the switch menu, active one flagged.
    profiles: Vec<ProfileOptionView>,
    /// Roster size (participants in the scene), shown on the Characters header.
    character_count: usize,
}

impl SidebarView {
    /// Builds the sidebar view from the active profile + repository counts.
    /// `participant_count` is the live-chat roster size (used as the Characters
    /// count so the nav reflects who's actually in the scene).
    fn build(state: &AppState, participant_count: usize) -> Self {
        let profiles_dir = &state.config.profiles_dir;
        let settings = AppSettings::load(&state.config.settings_path);
        let active_id = settings.active_profile_id(profiles_dir);

        let all = GameProfile::list(profiles_dir);
        let active = all.iter().find(|profile| profile.id == active_id);
        let profile_name = active
            .map(|profile| profile.name.clone())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "No profile".to_string());
        let active_initials = profile_initials(&profile_name);

        let profiles = all
            .iter()
            .map(|profile| {
                let name = if profile.name.is_empty() {
                    profile.id.clone()
                } else {
                    profile.name.clone()
                };
                ProfileOptionView {
                    initials: profile_initials(&name),
                    id: profile.id.clone(),
                    active: profile.id == active_id,
                    description: profile.description.clone(),
                    name,
                }
            })
            .collect();

        Self {
            profile_name,
            profile_initials: active_initials,
            profiles,
            character_count: participant_count,
        }
    }
}

/// First letters of up to two words, uppercased — the profile badge label.
/// "Fallout: New Vegas" -> "FN"; single word -> first two letters; empty -> "?".
fn profile_initials(name: &str) -> String {
    let words: Vec<&str> = name
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    let letters: String = match words.as_slice() {
        [] => return "?".to_string(),
        [single] => single.chars().take(2).collect(),
        [first, second, ..] => first
            .chars()
            .take(1)
            .chain(second.chars().take(1))
            .collect(),
    };
    letters.to_uppercase()
}

#[derive(Template)]
#[template(path = "partials/prompt_panel.html")]
struct PromptPanelTemplate {
    live_chat: LiveChatView,
    prompt: PromptAssemblyView,
    /// The selected participant's visible messages, so the panel can render each
    /// one's injected lore/quest/action entries + chosen actions on demand.
    messages: Vec<MessageView>,
}

#[derive(Template)]
#[template(path = "partials/character_list.html")]
struct CharacterListTemplate {
    live_chat: LiveChatView,
    sidebar: SidebarView,
}

#[derive(Template)]
#[template(path = "partials/message_list.html")]
struct MessageListTemplate {
    messages: Vec<MessageView>,
    selected_participant: Option<ParticipantView>,
}

#[derive(Template)]
#[template(path = "settings.html")]
struct SettingsTemplate {
    page: SettingsPageView,
}

#[derive(Template)]
#[template(path = "tracing.html")]
struct TracingTemplate {
    page: trace_routes::TracingPageView,
}

#[derive(Template)]
#[template(path = "partials/voice_clone.html")]
struct VoiceClonePartialTemplate {
    voice_clone: VoiceCloneView,
}

#[derive(Debug)]
pub struct WebError(anyhow::Error);

impl<E> From<E> for WebError
where
    E: Into<anyhow::Error>,
{
    fn from(error: E) -> Self {
        Self(error.into())
    }
}

impl IntoResponse for WebError {
    fn into_response(self) -> Response {
        let status = if self.0.downcast_ref::<CompatError>().is_some_and(|error| {
            matches!(
                error,
                CompatError::LiveChatNotFound(_)
                    | CompatError::ActionBookNotFound(_)
                    | CompatError::InvalidSessionId
            )
        }) {
            StatusCode::NOT_FOUND
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        let body = format!(
            "<main class=\"error-page\"><h1>chasm</h1><p>{}</p></main>",
            html_escape(&self.0.to_string())
        );
        (
            status,
            [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
            body,
        )
            .into_response()
    }
}

pub type WebResult<T> = Result<T, WebError>;

pub fn router(config: AppConfig) -> Router {
    let static_dir = config.workspace_root.join("static");
    let state = Arc::new(AppState::new(config));

    // The retriever (embedder + reranker) is NOT warmed at startup: it comes up with
    // the rest of the stack — on "Start models" (`stack_start`) or when the game
    // connects (`stack_lifecycle`) — so its status lights don't read "online" before
    // anything else has started. It still lazy-loads on first use as a fallback.

    // Section 7 fold: run the FNV bridge loop in-process (no Node helper, no
    // localhost hop) when `CHASM_FNV_BRIDGE` is set. Shares this runtime +
    // AppState; off by default so the standalone bin / HTTP path is unaffected.
    if tokio::runtime::Handle::try_current().is_ok() && fnv_bridge::in_process_enabled() {
        let bridge_state = Arc::clone(&state);
        tokio::spawn(async move {
            fnv_bridge::spawn_in_process(bridge_state).await;
        });

        // Connection-driven AI stack lifecycle: while the bridge is running, watch
        // the in-game plugin's heartbeat and bring the AI stack up when the game
        // connects / tear it down when it leaves. Gated on the same bridge-mode
        // flag (it only makes sense when chasm is the active backend) and spawned
        // the same way.
        let lifecycle_state = Arc::clone(&state);
        tokio::spawn(async move {
            stack_lifecycle::spawn_lifecycle(lifecycle_state).await;
        });
    }

    Router::new()
        // The new React SPA is the primary UI now; `/` sends you into it. The
        // legacy server-rendered UI stays reachable at `/legacy` as a fallback.
        .route("/", get(|| async { Redirect::to("/app/") }))
        .route("/legacy", get(index))
        .route("/health", get(health))
        .route("/live/:live_chat_id/:participant_id", get(live_chat))
        .route(
            "/live/:live_chat_id/:participant_id/clear-history",
            post(clear_participant_history),
        )
        .route(
            "/partials/live/:live_chat_id/participants",
            get(participants_partial),
        )
        .route(
            "/partials/live/:live_chat_id/messages/:participant_id",
            get(messages_partial),
        )
        .route(
            "/partials/live/:live_chat_id/prompt/:participant_id",
            get(prompt_partial),
        )
        // Library: each profile ships one book per kind, so the rail links go
        // straight to it (no index). All three are full editors (read the single
        // book file as a Value, overlay edits in place, write back).
        .route(
            "/lorebook",
            get(books::lorebook_editor).post(books::lorebook_save),
        )
        .route(
            "/questbook",
            get(books::questbook_editor).post(books::questbook_save),
        )
        .route(
            "/actionbook",
            get(books::actionbook_editor).post(books::actionbook_save),
        )
        .route("/settings", get(settings_index))
        .route(
            "/settings/:category",
            get(settings_page).post(save_settings),
        )
        .route("/partials/settings/voice-clone", get(voice_clone_partial))
        // --- Tracing (per-request waterfall) --------------------------------
        .route("/traces", get(trace_routes::list_traces_endpoint))
        .route("/traces/:id", get(trace_routes::get_trace_endpoint))
        // --- Connection status (in-game plugin heartbeat) -------------------
        // chasm is a passive backend now; the in-game plugin writes a heartbeat
        // file while running, and the chat rail polls this to show Connected.
        .route("/connection/status", get(connection::connection_status))
        // --- Model stack: manual "Start models" + per-service status lights ---
        .route("/api/stack/status", get(stack_status))
        .route("/api/stack/start", post(stack_start))
        .route("/engines/:id/install", post(install_engine))
        .route("/llm-models/:id/download", post(download_llm_model))
        .route("/whisper-models/:id/download", post(download_whisper_model))
        // Opens a model-category folder in Windows Explorer. The dir is resolved
        // from fixed config (keyed on `:category`), never from user input.
        .route("/open-folder/:category", post(open_model_folder))
        .route(
            "/retrieval-models/:id/download",
            post(download_retrieval_model),
        )
        .route("/voices/clone", post(clone_voices))
        // JSON voice-clone for the React TTS page: GET status, POST to start.
        .route(
            "/api/voices/clone",
            get(voice_clone_status).post(voice_clone_start),
        )
        // Switch the active game profile (relaunches the TTS worker at the new
        // profile's voices dir). The sidebar UI calls this in a later phase.
        .route("/profile/select", post(select_profile))
        .route("/profile/select/:id", post(select_profile_path))
        // --- Headless API (FNV helper contract) -----------------------------
        // Everything the helper reaches via `apiBase`/`ttsApiBase`/`sttApiBase`
        // lives under this prefix, so a single base can point the whole helper
        // (speech, live-chat generation, admin generate, save-sync) at Rust.
        .nest(
            "/api/headless/v1",
            Router::new()
                .route("/speech/synthesize", post(speech_synthesize))
                .route("/speech/synthesize/stream", post(speech_synthesize_stream))
                .route("/speech/recognize", post(speech_recognize))
                .route("/live-chats", post(generate::create_live_chat))
                .route("/live-chats/:id", get(generate::get_live_chat))
                .route("/live-chats/:id/presence", post(generate::update_presence))
                .route("/live-chats/:id/generate", post(generate::generate))
                .route(
                    "/live-chats/:id/generate/stream",
                    post(generate::generate_stream),
                )
                // Admin / "Todd" single-character generation (non-live).
                .route("/generate", post(generate::generate_headless))
                .route("/generate/stream", post(generate::generate_headless_stream))
                // Game save/load checkpoint + restore.
                .route("/save-sync/events", post(save_sync::save_sync_event)),
        )
        // --- Game transport (HTTP successor to the NVBridge file folder) -----
        // A SEPARATE namespace from the file bridge and the /app UI: one streaming
        // endpoint that runs an NPC turn and returns reply text + audio chunks +
        // the game_master action as NDJSON. The file bridge stays the default; this
        // is purely additive. The C++ plugin becomes its HTTP client in a later
        // stage (different repo). See [`game_bridge`].
        .route("/api/game/v1/turn", post(game_bridge::turn))
        // Player-persona capture upload (stats snapshot + optional stealth
        // screenshot). Stored profile-aware; generation runs on a background
        // task and NEVER blocks a turn. See [`persona`]. The route-scoped body
        // limit raises axum's 2 MB default so a base64 screenshot up to the
        // handler's documented 8 MB decoded cap actually fits on the wire.
        .route(
            "/api/game/v1/persona",
            post(persona::receive_capture)
                .layer(axum::extract::DefaultBodyLimit::max(persona::MAX_BODY_BYTES)),
        )
        // Settings → Updates: reports the running version + the latest GitHub
        // release so the UI can offer a "Download update" link. Always succeeds.
        .route("/api/app/version", get(app_version))
        .route("/api/app/update/install", post(app_update_install))
        // Voices are served from the ACTIVE profile's voices dir (resolved per
        // request, legacy `{workspace}/voices` fallback), so a profile switch
        // changes which clips are served with no restart.
        .route("/voices/*path", get(serve_voice_file))
        // Dynamic appearance stylesheet computed from the saved InterfaceSettings,
        // linked after app.css so its :root overrides win.
        .route("/theme.css", get(theme_css))
        .nest_service("/static", ServeDir::new(static_dir))
        // ===== NEW REACT UI (chasm-ui) — added as one block ====================
        // The new React SPA (crates/chasm-web/ui) is built to static assets
        // and served HERE by this same axum on :7341, alongside the existing
        // Askama pages during the phased migration. It adds ONLY: the SPA assets
        // under `/app` (+ SPA fallback) and a UI-only JSON API under
        // `/api/ui/v1`. It does NOT touch `/api/headless/*` or `/api/game/*`
        // (the bridge/game contract). See `ui.rs`.
        .route("/app", get(ui::app_root_redirect))
        .nest("/api/ui/v1", ui::api_router())
        .nest_service("/app/", ui::spa_service(&state))
        // ===== END NEW REACT UI ===============================================
        .layer(TraceLayer::new_for_http())
        // Outermost: when CHASM_CAPTURE_DIR is set, record every request
        // verbatim for 1:1 replay (no-op otherwise). See `capture`.
        .layer(axum::middleware::from_fn(capture::capture_request))
        .with_state(state)
}

/// Tears down the AI stack (koboldcpp + TTS) that chasm's connection lifecycle
/// may have started, so a desktop-shell **Quit** doesn't orphan those runtimes.
/// Builds a throwaway [`AppState`] from `config` — [`launcher::stop_ai_stack`]
/// only reads the settings/endpoint paths off it and kills the runtimes by
/// process name + port, so it doesn't need the live, server-owned state. A no-op
/// in practice when nothing is running (the kills just find no targets).
pub fn shutdown_ai_stack(config: AppConfig) {
    let state = Arc::new(AppState::new(config));
    launcher::stop_ai_stack(&state);
}

/// Prepends app-local CUDA runtime dirs to PATH so ONNX Runtime's CUDA
/// execution provider (GPU retrieval: embedder + reranker) can load its
/// dependency DLLs (cublas, cuDNN 9, cufft). Without them on PATH, ort
/// SILENTLY falls back to CPU even when `retrieval.execution = "gpu"`.
/// Two well-known locations, no user PATH surgery required:
///   * `<llm models dir>/../cuda` — a drop-in folder the user (or a future
///     installer) can fill with the CUDA runtime DLLs.
///   * the managed llama.cpp runtime dir — its CUDA build ships cudart/cublas,
///     which covers part of the set for free.
/// Missing dirs are skipped; CPU fallback behavior is unchanged.
fn bootstrap_cuda_path(config: &AppConfig) {
    let mut extra: Vec<std::path::PathBuf> = Vec::new();
    if let Some(models_root) = config.llm_models_dir.parent() {
        extra.push(models_root.join("cuda"));
        extra.push(models_root.join("llamacpp"));
    }
    let existing: Vec<_> = extra.into_iter().filter(|dir| dir.is_dir()).collect();
    if existing.is_empty() {
        return;
    }
    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut parts: Vec<std::path::PathBuf> = existing.clone();
    parts.extend(std::env::split_paths(&current));
    if let Ok(joined) = std::env::join_paths(parts) {
        for dir in &existing {
            info!("CUDA runtime dir on PATH: {}", dir.display());
        }
        std::env::set_var("PATH", joined);
    }
}

pub async fn serve(config: AppConfig) -> anyhow::Result<()> {
    bootstrap_cuda_path(&config);
    let addr: SocketAddr = config.bind_addr.parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!(
        "chasm — Agentic NPC Engine listening on http://{}",
        listener.local_addr()?
    );
    // TTS, LLM and STT are all served by koboldcpp (spawned by the launcher on
    // Play); no separate Python TTS worker is started here.
    axum::serve(listener, router(config)).await?;
    Ok(())
}

/// The active profile's voices dir (legacy `{workspace}/voices` fallback),
/// resolved per call so a profile switch repoints a freshly-spawned TTS engine.
/// Used by the PocketTTS engine spawn ([`crate::launcher`]) and the live
/// voices-file route.
pub(crate) fn active_voices_dir(config: &AppConfig) -> std::path::PathBuf {
    config.active_profile_paths().voices_dir()
}

/// Best-effort kill of the process (if any) listening on `127.0.0.1:<port>`.
/// Windows-specific: resolves the owning PID via `netstat -ano` and `taskkill`s
/// it. A worker not found / already gone is a no-op.
pub(crate) fn kill_process_on_port(port: u16) {
    let Ok(output) = Command::new("netstat").args(["-ano", "-p", "tcp"]).output() else {
        return;
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let needle = format!(":{port}");
    let mut pids: Vec<String> = Vec::new();
    for line in text.lines() {
        // Match lines whose LOCAL address ends with :<port> and that are LISTENING.
        if !line.contains(&needle) {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        // netstat -ano: Proto  Local  Foreign  State  PID
        let local_matches = cols.get(1).is_some_and(|local| local.ends_with(&needle));
        if !local_matches {
            continue;
        }
        if let Some(pid) = cols.last() {
            if pid.chars().all(|c| c.is_ascii_digit()) && *pid != "0" {
                pids.push((*pid).to_string());
            }
        }
    }
    pids.sort();
    pids.dedup();
    for pid in pids {
        let _ = Command::new("taskkill")
            .args(["/F", "/PID", &pid])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        info!("stopped TTS worker pid {pid} on port {port} for profile switch");
    }
}

/// Extracts `(text, characterName)` from a speech request body.
fn speech_request(req: &serde_json::Value) -> (String, String) {
    let text = req
        .get("text")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    let character = req
        .get("characterName")
        .and_then(|value| value.as_str())
        .or_else(|| req.get("character").and_then(|value| value.as_str()))
        .unwrap_or("")
        .to_string();
    (text, character)
}

/// Resolves the synthesis tuning for one request: start from the saved settings,
/// then let any per-request override fields win. The game sends a bare
/// `{text, characterName}` (no `tuning`), so it always uses the saved values;
/// the voice-clone Test button sends a `tuning` object with the live form values
/// so a tweak is heard before saving. Unknown/partial overrides only replace the
/// fields they carry, and the result is clamped to each field's documented range.
fn resolve_tuning(saved: &TtsTuningSettings, req: &serde_json::Value) -> TtsTuningSettings {
    let mut tuning = saved.clone();
    if let Some(over) = req.get("tuning").and_then(|v| v.as_object()) {
        let u32_field = |key: &str, current: u32| -> u32 {
            over.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(current)
        };
        let f32_field = |key: &str, current: f32| -> f32 {
            over.get(key)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .filter(|v| v.is_finite())
                .unwrap_or(current)
        };
        tuning.lead_in_ms = u32_field("lead_in_ms", tuning.lead_in_ms);
        tuning.trailing_ms = u32_field("trailing_ms", tuning.trailing_ms);
        tuning.gain_db = f32_field("gain_db", tuning.gain_db);
        tuning.temperature = f32_field("temperature", tuning.temperature);
        tuning.lsd_decode_steps = u32_field("lsd_decode_steps", tuning.lsd_decode_steps);
        tuning.eos_threshold = f32_field("eos_threshold", tuning.eos_threshold);
        tuning.noise_clamp = f32_field("noise_clamp", tuning.noise_clamp);
        tuning.max_tokens = u32_field("max_tokens", tuning.max_tokens);
        tuning.frames_after_eos = u32_field("frames_after_eos", tuning.frames_after_eos);
    }
    tuning.normalized()
}

/// The `tuning` object sent to the worker's `/synthesize` body. The worker reads
/// these per request, so changes are live with no worker restart.
#[allow(dead_code)] // legacy: tuning payload for the native Python TTS worker
fn tuning_json(tuning: &TtsTuningSettings) -> serde_json::Value {
    serde_json::json!({
        "lead_in_ms": tuning.lead_in_ms,
        "trailing_ms": tuning.trailing_ms,
        "gain_db": tuning.gain_db,
        "temperature": tuning.temperature,
        "lsd_decode_steps": tuning.lsd_decode_steps,
        "eos_threshold": tuning.eos_threshold,
        "noise_clamp": tuning.noise_clamp,
        "max_tokens": tuning.max_tokens,
        "frames_after_eos": tuning.frames_after_eos,
    })
}

/// faster-qwen3-tts model output: 24 kHz mono int16. The FNV plugin's WAV loader
/// handles 24 kHz (it played koboldcpp's 24 kHz clips), so we keep the native
/// rate and let DirectSound resample on playback.
const TTS_SAMPLE_RATE: u32 = 24_000;

/// Synthesizes `text` in `character`'s cloned voice via the faster-qwen3-tts
/// service and returns a complete WAV. `base` is the faster-qwen3-tts base URL
/// (`{base}/v1/audio/speech`). We always request raw `pcm` and build the WAV
/// container ourselves — the service's `wav` mode emits a streaming header with
/// 0xFFFFFFFF (unknown) sizes that the FNV plugin's WAV loader can't parse.
async fn synthesize_via_worker(
    base: &str,
    text: &str,
    character: &str,
    _tuning: &TtsTuningSettings,
    gain: f32,
) -> anyhow::Result<Vec<u8>> {
    let mut pcm = synthesize_pcm(base, text, character).await?;
    apply_pcm_gain(&mut pcm, gain);
    Ok(pcm16_to_wav(&pcm, TTS_SAMPLE_RATE, 1, 16))
}

/// Requests raw int16-LE mono 24 kHz PCM for `text` in `character`'s cloned voice
/// from faster-qwen3-tts (`{base}/v1/audio/speech`, `response_format: "pcm"`).
/// The `voice` field is the plain NPC name — a key in the service's voices.json
/// (NO extension, unlike koboldcpp's filename-keyed `--ttsdir`). Retries while the
/// service warms up / captures CUDA graphs on the first request.
async fn synthesize_pcm(base: &str, text: &str, character: &str) -> anyhow::Result<Vec<u8>> {
    let url = format!("{}/v1/audio/speech", base.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": "qwen3-tts",
        "input": text,
        "voice": character,
        "response_format": "pcm",
    });
    let mut last = String::from("no response");
    for _ in 0..30 {
        match client.post(&url).json(&body).send().await {
            Ok(response) if response.status().is_success() => {
                return Ok(response.bytes().await?.to_vec());
            }
            Ok(response) => {
                let status = response.status();
                last = format!("faster-qwen3-tts status {status}");
                if status.is_client_error() {
                    break;
                }
            }
            Err(error) => last = error.to_string(),
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Err(anyhow::anyhow!(
        "TTS (faster-qwen3-tts) unavailable ({last})"
    ))
}

/// Wraps raw int16-LE PCM in a minimal RIFF/WAVE container with CORRECT finite
/// sizes, used for both the buffered line (whole clip) and each streamed slice so
/// the FNV plugin's WAV loader always gets a parseable header.
fn pcm16_to_wav(pcm: &[u8], sample_rate: u32, channels: u16, bits: u16) -> Vec<u8> {
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits / 8);
    let block_align = channels * (bits / 8);
    let data_len = pcm.len() as u32;
    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(pcm);
    wav
}

/// Scales 16-bit-LE mono PCM in place by `gain` (1.0 = unchanged), hard-clamping
/// to the int16 range. This is the TTS volume control: applied to the synthesized
/// samples (not DirectSound, which can only attenuate), so `gain > 1.0` genuinely
/// boosts. A no-op at unity so the common case copies nothing.
fn apply_pcm_gain(pcm: &mut [u8], gain: f32) {
    if (gain - 1.0).abs() < 1e-4 {
        return;
    }
    for frame in pcm.chunks_exact_mut(2) {
        let scaled = f32::from(i16::from_le_bytes([frame[0], frame[1]])) * gain;
        let clamped = scaled.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
        frame.copy_from_slice(&clamped.to_le_bytes());
    }
}

/// The live TTS volume for one synth request: `admin_volume` when the request is
/// the non-positional "admin" voice (the helper sets `nonPositional` for Todd's
/// 2D, straight-into-your-ear path), otherwise `npc_volume` for ordinary
/// directional NPCs. Read fresh per request, so a slider move is heard next line.
fn resolve_voice_volume(settings: &AppSettings, req: &serde_json::Value) -> f32 {
    let non_positional = req
        .get("nonPositional")
        .or_else(|| req.get("non_positional"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let raw = if non_positional {
        settings.tts.admin_volume
    } else {
        settings.tts.npc_volume
    };
    chasm_core::normalize_voice_volume(raw)
}

/// Buffered TTS — mirrors ST `/speech/synthesize`. Returns base64 wav in JSON so
/// the FNV mod can point its `/speech/*` calls at the Rust port unchanged. Loads
/// the TTS tuning live per request (saved settings, with any per-request override
/// from the body winning), so the in-game path always uses the current settings.
async fn speech_synthesize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> WebResult<Json<serde_json::Value>> {
    let (text, character) = speech_request(&req);
    let settings = AppSettings::load(&state.config.settings_path);
    let tuning = resolve_tuning(&settings.tts.tuning, &req);
    let volume = resolve_voice_volume(&settings, &req);
    let bytes = synthesize_via_worker(
        &state.config.tts_endpoint,
        &text,
        &character,
        &tuning,
        volume,
    )
    .await?;
    Ok(Json(serde_json::json!({
        "audio": { "data": STANDARD.encode(&bytes) },
        "mimeType": "audio/wav",
    })))
}

/// One `audio.chunk` NDJSON line: a base64 WAV slice the helper writes to disk and
/// the FNV plugin gaplessly plays. `text` (the line subtitle) is set on the first
/// chunk and empty after, so the plugin shows it once and reuses it.
fn audio_chunk_line(index: usize, wav: &[u8], text: &str, caption_max_chars: u32) -> String {
    let event = serde_json::json!({
        "type": "audio.chunk",
        "index": index,
        "audio": { "data": STANDARD.encode(wav) },
        "mimeType": "audio/wav",
        "text": text,
        // Display-only: how the FNV plugin should split this line's caption (0 =
        // whole line). Carried with the audio chunk but never affects synthesis.
        "captionMaxChars": caption_max_chars,
    });
    format!("{}\n", serde_json::to_string(&event).unwrap_or_default())
}

/// One `speech.error` NDJSON line; the helper treats it as terminal.
fn speech_error_line(message: &str) -> String {
    let event = serde_json::json!({
        "type": "speech.error",
        "error": { "message": message },
    });
    format!("{}\n", serde_json::to_string(&event).unwrap_or_default())
}

/// Streaming TTS — NDJSON of `audio.chunk` events. Opens ONE faster-qwen3-tts PCM
/// stream for the whole line and slices it into ~`stream_slice_ms` mini-WAVs as
/// frames arrive, so the helper writes (and the FNV plugin gaplessly plays) many
/// small chunks. First audio ≈ engine TTFA (~150 ms) + one slice, vs rendering the
/// whole line first. `stream_slice_ms` is read fresh per request (live-tunable).
async fn speech_synthesize_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> WebResult<Response> {
    use futures_util::StreamExt;
    let stream = speech_synthesize_stream_core(state, req);
    Ok((
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        Body::from_stream(stream.map(Ok::<String, std::convert::Infallible>)),
    )
        .into_response())
}

/// In-process core of [`speech_synthesize_stream`]: the raw NDJSON audio-chunk
/// line stream, with no localhost socket. The HTTP handler streams it as the
/// response body; the in-process bridge client parses each line and feeds
/// `audio.chunk` events to its callback.
pub(crate) fn speech_synthesize_stream_core(
    state: Arc<AppState>,
    req: serde_json::Value,
) -> impl futures_util::Stream<Item = String> + Send {
    use futures_util::StreamExt;
    let (text, character) = speech_request(&req);
    let settings = AppSettings::load(&state.config.settings_path);
    // Ramp the streamed slice size: a small FIRST slice for fast first-audio, then
    // DOUBLE up to stream_slice_ms (the cap). Few big chunk files (vs many tiny ones)
    // keep the in-game plugin's per-frame file I/O cheap — it reads one WAV per chunk
    // via MO2's usvfs (~tens of ms each), and that count was the in-game lag. The
    // single streaming buffer plays any chunk size gaplessly; the ramp keeps it from
    // underrunning before the engine's faster-than-real-time lead builds up.
    let max_slice_ms =
        chasm_core::normalize_stream_slice_ms(settings.tts.stream_slice_ms) as usize;
    let first_slice_ms = 200usize.min(max_slice_ms.max(1));
    // Caption chunking is a pure display hint forwarded to the plugin; it never
    // changes the audio. Read fresh per request (live-tunable).
    let caption_max_chars =
        chasm_core::normalize_caption_max_chars(settings.tts.caption_max_chars);
    let base = state.config.tts_endpoint.clone();
    // Live voice volume for this line (admin vs directional), read fresh per request.
    let volume = resolve_voice_volume(&settings, &req);
    // Live PocketTTS tuning, forwarded per request so the Settings → TTS → Tuning
    // sliders take effect with no restart. PocketTTS reads these from the request
    // body (silence pads it inserts itself; the sampling knobs it sets on the model
    // before each generation); the Qwen server ignores the extra fields. Pads: a
    // lead-in protects the speech onset from playback-startup clipping, an
    // inter-sentence gap stops PocketTTS's sentence-chunked output running together.
    let pad = settings.tts.tuning.normalized();
    let lead_in_ms = pad.lead_in_ms;
    let sentence_gap_ms = pad.sentence_gap_ms;
    let trailing_ms = pad.trailing_ms;
    let pt_temperature = pad.temperature;
    let pt_lsd_decode_steps = pad.lsd_decode_steps;
    let pt_eos_threshold = pad.eos_threshold;
    let pt_noise_clamp = pad.noise_clamp;
    let pt_max_tokens = pad.max_tokens;
    let pt_frames_after_eos = pad.frames_after_eos;

    let stream = async_stream::stream! {
        let url = format!("{}/v1/audio/speech", base.trim_end_matches('/'));
        let client = reqwest::Client::new();
        let body = serde_json::json!({
            "model": "qwen3-tts",
            "input": text,
            "voice": character,
            "response_format": "pcm",
            "lead_in_ms": lead_in_ms,
            "sentence_gap_ms": sentence_gap_ms,
            "trailing_ms": trailing_ms,
            "temperature": pt_temperature,
            "lsd_decode_steps": pt_lsd_decode_steps,
            "eos_threshold": pt_eos_threshold,
            "noise_clamp": pt_noise_clamp,
            "max_tokens": pt_max_tokens,
            "frames_after_eos": pt_frames_after_eos,
        });
        let response = match client.post(&url).json(&body).send().await {
            Ok(response) if response.status().is_success() => response,
            Ok(response) => {
                yield speech_error_line(&format!("faster-qwen3-tts status {}", response.status()));
                return;
            }
            Err(error) => {
                yield speech_error_line(&error.to_string());
                return;
            }
        };

        // Slice the chunked PCM body into ramping mini-WAVs as bytes arrive (small
        // first slice, doubling to the cap). The engine streams faster than real time,
        // so slices flow out continuously and the plugin stitches them seamlessly.
        let mut byte_stream = response.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut index = 0usize;
        let mut cur_slice_ms = first_slice_ms;
        while let Some(item) = byte_stream.next().await {
            let bytes = match item {
                Ok(bytes) => bytes,
                Err(error) => {
                    yield speech_error_line(&format!("TTS stream error: {error}"));
                    return;
                }
            };
            buf.extend_from_slice(&bytes);
            loop {
                let target = ((cur_slice_ms * (TTS_SAMPLE_RATE as usize) * 2 / 1000).max(2)) & !1usize;
                if buf.len() < target {
                    break;
                }
                let mut slice: Vec<u8> = buf.drain(..target).collect();
                apply_pcm_gain(&mut slice, volume);
                let wav = pcm16_to_wav(&slice, TTS_SAMPLE_RATE, 1, 16);
                yield audio_chunk_line(index, &wav, if index == 0 { &text } else { "" }, caption_max_chars);
                index += 1;
                cur_slice_ms = (cur_slice_ms * 2).min(max_slice_ms);
            }
        }
        // Flush the trailing partial slice (also guarantees >= 1 chunk for short lines).
        if !buf.is_empty() {
            apply_pcm_gain(&mut buf, volume);
            let wav = pcm16_to_wav(&buf, TTS_SAMPLE_RATE, 1, 16);
            yield audio_chunk_line(index, &wav, if index == 0 { &text } else { "" }, caption_max_chars);
        }
    };

    stream
}

// Per-request STT timeout clamp bounds (the saved default lives in core as
// `STT_TIMEOUT_MS_DEFAULT`; the request value is clamped to these).
const STT_MIN_TIMEOUT_MS: u64 = chasm_core::STT_TIMEOUT_MS_MIN;
const STT_MAX_TIMEOUT_MS: u64 = chasm_core::STT_TIMEOUT_MS_MAX;

/// Pulls the base64 audio payload (+ format/encoding/mimeType) out of a
/// `/speech/recognize` body, accepting both the nested `audio: {...}` object the
/// FNV helper sends and the flat `audio`/`audioBase64`/`data` string forms ST
/// also accepts.
fn stt_audio_payload(req: &serde_json::Value) -> (String, String, String, String) {
    let mut data = String::new();
    let mut encoding = "base64".to_string();
    let mut format = "wav".to_string();
    let mut mime_type = "audio/wav".to_string();

    let audio = req.get("audio");
    if let Some(obj) = audio.and_then(|value| value.as_object()) {
        if let Some(value) = obj.get("data").and_then(|v| v.as_str()) {
            data = value.to_string();
        }
        if let Some(value) = obj.get("encoding").and_then(|v| v.as_str()) {
            encoding = value.to_string();
        }
        if let Some(value) = obj.get("format").and_then(|v| v.as_str()) {
            format = value.to_string();
        }
        if let Some(value) = obj.get("mimeType").and_then(|v| v.as_str()) {
            mime_type = value.to_string();
        }
    } else if let Some(value) = audio
        .and_then(|v| v.as_str())
        .or_else(|| req.get("audioBase64").and_then(|v| v.as_str()))
        .or_else(|| req.get("data").and_then(|v| v.as_str()))
    {
        data = value.to_string();
    }

    if let Some(value) = req.get("format").and_then(|v| v.as_str()) {
        if audio.and_then(|a| a.as_object()).is_none() {
            format = value.to_string();
        }
    }
    if let Some(value) = req.get("mimeType").and_then(|v| v.as_str()) {
        if audio.and_then(|a| a.as_object()).is_none() {
            mime_type = value.to_string();
        }
    }

    (data, encoding, format, mime_type)
}

/// Strips an optional `data:` URI prefix and removes whitespace from a base64
/// audio string, mirroring ST's `normalizeAudioBase64`.
fn normalize_audio_base64(input: &str) -> String {
    let trimmed = input.trim();
    let body = match trimmed.find("data:").filter(|index| *index == 0) {
        Some(_) => trimmed.split_once(',').map(|(_, rest)| rest).unwrap_or(""),
        None => trimmed,
    };
    body.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Extracts transcript text from an OpenAI-compatible / Parakeet response,
/// mirroring ST's `extractTranscriptText` field fallbacks.
fn extract_transcript_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.trim().to_string(),
        serde_json::Value::Array(items) => items
            .iter()
            .map(extract_transcript_text)
            .collect::<String>()
            .trim()
            .to_string(),
        serde_json::Value::Object(map) => {
            for field in ["text", "transcription", "transcript", "result"] {
                if let Some(text) = map.get(field).and_then(|v| v.as_str()) {
                    return text.trim().to_string();
                }
            }
            for field in ["segments", "chunks"] {
                if let Some(parts) = map.get(field).and_then(|v| v.as_array()) {
                    return parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(|v| v.as_str()))
                        .collect::<String>()
                        .trim()
                        .to_string();
                }
            }
            String::new()
        }
        _ => String::new(),
    }
}

/// Minimum audio length (ms) handed to whisper. koboldcpp's whisper returns
/// EMPTY text for clips shorter than ~1s because — unlike reference whisper — it
/// does not pad short audio up to a usable encoder window, so a quick "hi" comes
/// back blank (and the FNV helper then surfaces a bridge error). Parakeet (the
/// previous STT) had no such floor, which is why short utterances used to work.
const STT_MIN_AUDIO_MS: u32 = 2000;

/// Pads a PCM WAV with trailing silence so it is at least `min_ms` long, returning
/// a canonical PCM WAV. Audio that is already long enough — or that isn't a PCM
/// WAV we can parse — is returned byte-for-byte unchanged. whisper ignores the
/// trailing silence, so the spoken words still transcribe; this just guarantees
/// the encoder gets enough samples (see [`STT_MIN_AUDIO_MS`]).
fn pad_wav_to_min_duration(bytes: &[u8], min_ms: u32) -> Vec<u8> {
    // Canonical WAV starts `RIFF....WAVE`; pass anything else through unchanged.
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return bytes.to_vec();
    }
    let read_u16 = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let read_u32 =
        |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

    // Walk the chunk list for `fmt ` (format) and `data` (samples).
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, rate, bits)
    let mut data: Option<(usize, usize)> = None; // (offset, len)
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = read_u32(pos + 4) as usize;
        let body = pos + 8;
        if id == b"fmt " && body + 16 <= bytes.len() {
            fmt = Some((
                read_u16(body),      // audio format (1 = PCM)
                read_u16(body + 2),  // channels
                read_u32(body + 4),  // sample rate
                read_u16(body + 14), // bits per sample
            ));
        } else if id == b"data" {
            data = Some((body, size.min(bytes.len().saturating_sub(body))));
            break; // `data` is conventionally last; the samples follow here
        }
        pos = body + size + (size & 1); // chunks are word-aligned
    }

    let (Some((format, channels, rate, bits)), Some((off, len))) = (fmt, data) else {
        return bytes.to_vec();
    };
    // Only safe to pad uncompressed PCM; leave anything else untouched.
    if format != 1 || channels == 0 || rate == 0 || bits < 8 {
        return bytes.to_vec();
    }
    let block_align = channels as usize * (bits as usize / 8);
    if block_align == 0 {
        return bytes.to_vec();
    }
    let byte_rate = rate as usize * block_align;
    let min_bytes = byte_rate * min_ms as usize / 1000;
    if len >= min_bytes {
        return bytes.to_vec(); // already long enough
    }
    // Trailing silence, rounded up to a whole frame.
    let mut pad = min_bytes - len;
    pad += (block_align - pad % block_align) % block_align;
    let new_len = len + pad;

    let mut out = Vec::with_capacity(44 + new_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + new_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(byte_rate as u32).to_le_bytes());
    out.extend_from_slice(&(block_align as u16).to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(new_len as u32).to_le_bytes());
    out.extend_from_slice(&bytes[off..off + len]); // original samples
    out.resize(44 + new_len, 0); // + trailing silence
    out
}

/// The transcription endpoint the active STT provider serves: the dedicated
/// Parakeet server when the provider is `parakeet` AND the engine is installed,
/// else the koboldcpp Whisper endpoint. Falling back on "selected but not
/// installed" means voice input keeps working (on Whisper) instead of dying
/// against a port nothing listens on.
pub(crate) fn effective_stt_endpoint(
    config: &chasm_core::AppConfig,
    settings: &AppSettings,
) -> String {
    if launcher::stt_uses_parakeet(settings, config) {
        config.parakeet_stt_endpoint.clone()
    } else {
        config.stt_endpoint.clone()
    }
}

/// Speech recognition — mirrors ST `/speech/recognize`. Forwards the base64 WAV
/// payload the FNV helper sends to the active local OpenAI-compatible STT server
/// (koboldcpp Whisper, or the dedicated Parakeet server when selected;
/// `/v1/audio/transcriptions`, multipart `file` upload) and returns the
/// transcription in the same JSON shape ST returns so the helper parses
/// `recognition.text` unchanged.
async fn speech_recognize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> WebResult<Json<serde_json::Value>> {
    let settings = AppSettings::load(&state.config.settings_path);

    let (raw_data, encoding, format, mime_type) = stt_audio_payload(&req);
    let audio_b64 = normalize_audio_base64(&raw_data);
    if audio_b64.is_empty() {
        return Err(WebError(anyhow::anyhow!("audio is required.")));
    }
    let audio_bytes = STANDARD
        .decode(audio_b64.as_bytes())
        .map_err(|_| anyhow::anyhow!("audio must be valid base64."))?;
    let byte_length = audio_bytes.len();
    // koboldcpp's whisper returns empty text for clips shorter than ~1s (it does
    // not pad short audio up to a usable encoder window like reference whisper),
    // so a quick "hi" comes back blank. Pad short PCM WAV audio with trailing
    // silence first; whisper ignores the silence, so the words still transcribe.
    let audio_bytes = pad_wav_to_min_duration(&audio_bytes, STT_MIN_AUDIO_MS);

    // Provider/model/language: request overrides the saved STT settings.
    let requested_provider = req
        .get("provider")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_default();
    let resolved_provider = normalize_stt_provider(&settings.stt.provider);
    let model = req
        .get("model")
        .and_then(|v| v.as_str())
        .or_else(|| req.get("modelId").and_then(|v| v.as_str()))
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| stt_effective_model(&settings.stt));
    let language = req
        .get("language")
        .and_then(|v| v.as_str())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| settings.stt.language.trim().to_string());
    // Prompt: request value wins, else the saved STT default biasing prompt
    // (forwarded as the OpenAI `prompt` multipart field below).
    let prompt = req
        .get("prompt")
        .and_then(|v| v.as_str())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| settings.stt.prompt.trim().to_string());
    // Timeout: request value (clamped) wins, else the saved STT default timeout,
    // which becomes the actual reqwest deadline on the Parakeet POST.
    let timeout_ms = req
        .get("timeoutMs")
        .or_else(|| req.get("timeout_ms"))
        .and_then(|v| v.as_u64())
        .map(|value| value.clamp(STT_MIN_TIMEOUT_MS, STT_MAX_TIMEOUT_MS))
        .unwrap_or_else(|| chasm_core::normalize_stt_timeout_ms(settings.stt.timeout_ms));

    let filename = format!("audio.{format}");
    let file_part = reqwest::multipart::Part::bytes(audio_bytes)
        .file_name(filename)
        .mime_str(&mime_type)
        .unwrap_or_else(|_| {
            reqwest::multipart::Part::bytes(Vec::new())
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .expect("static wav mime")
        });
    let mut form = reqwest::multipart::Form::new()
        .part("file", file_part)
        .text("model", model.clone());
    if !language.is_empty() {
        form = form.text("language", language.clone());
    }
    if !prompt.is_empty() {
        form = form.text("prompt", prompt.clone());
    }

    let endpoint = effective_stt_endpoint(&state.config, &settings);
    tracing::debug!("speech recognize: provider={resolved_provider} endpoint={endpoint}");
    let client = reqwest::Client::new();
    let start = std::time::Instant::now();
    let response = client
        .post(&endpoint)
        .timeout(std::time::Duration::from_millis(timeout_ms))
        .multipart(form)
        .send()
        .await
        .map_err(|error| {
            anyhow::anyhow!(
                "{resolved_provider} speech-to-text request failed ({endpoint}): {error}"
            )
        })?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let snippet: String = body.chars().take(1000).collect();
        return Err(WebError(anyhow::anyhow!(
            "{resolved_provider} speech-to-text request failed (status {status}): {snippet}"
        )));
    }
    let provider_result: serde_json::Value = response.json().await.map_err(|error| {
        anyhow::anyhow!("{resolved_provider} returned an unreadable response: {error}")
    })?;
    let duration_ms = start.elapsed().as_millis() as u64;

    let text = extract_transcript_text(&provider_result);
    if text.is_empty() {
        return Err(WebError(anyhow::anyhow!(
            "Speech recognition returned no text."
        )));
    }

    let requested = if requested_provider.trim().is_empty() {
        "sillytavern".to_string()
    } else {
        requested_provider
    };

    Ok(Json(serde_json::json!({
        "provider": resolved_provider,
        "resolvedProvider": resolved_provider,
        "configuredProvider": true,
        "requestedProvider": requested,
        "text": text,
        "audio": {
            "format": format,
            "encoding": encoding,
            "mimeType": mime_type,
            "byteLength": byte_length,
        },
        "metadata": {
            "model": if model.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(model) },
            "language": if language.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(language) },
            "task": "transcribe",
            "timeoutMs": timeout_ms,
            "durationMs": duration_ms,
        },
    })))
}

async fn index(State(state): State<Arc<AppState>>) -> WebResult<Html<String>> {
    let Some(live_chat) = state.repository.list_live_chats()?.into_iter().next() else {
        return Ok(Html(landing_html(&state.config)));
    };
    let selected = choose_participant_id(&state.repository, &live_chat)?;
    render_live_chat_page(state, live_chat.id.as_str(), selected.as_str()).await
}

/// Friendly landing shown when no Live Chat data is found (e.g. a fresh clone
/// before `CHASM_DATA_ROOT` is pointed at a SillyTavern `default-user`).
fn landing_html(config: &AppConfig) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
<title>chasm</title><link rel=\"stylesheet\" href=\"/static/app.css\"></head>\
<body><main class=\"error-page\"><h1>chasm</h1>\
<p>No Live Chat data was found at:</p>\
<p><code>{data_root}</code></p>\
<p>Set <code>CHASM_DATA_ROOT</code> to a SillyTavern <code>default-user</code> folder \
(or drop data there), then refresh. <code>/health</code> is available meanwhile.</p>\
</main></body></html>",
        data_root = html_escape(&config.data_root.display().to_string())
    )
}

async fn live_chat(
    State(state): State<Arc<AppState>>,
    Path((live_chat_id, participant_id)): Path<(String, String)>,
) -> WebResult<Html<String>> {
    render_live_chat_page(state, &live_chat_id, &participant_id).await
}

/// Clears the chat history of one character: deletes every message in the live
/// chat where that participant is the speaker or is in `audibleTo`, then redirects
/// back to their view. Triggered by the sidebar right-click "Clear history" item.
async fn clear_participant_history(
    State(state): State<Arc<AppState>>,
    Path((live_chat_id, participant_id)): Path<(String, String)>,
) -> WebResult<Redirect> {
    let live_chat = state.repository.get_live_chat(&live_chat_id)?;
    let removed = state
        .repository
        .clear_participant_history(&live_chat, &participant_id)?;
    // Also scrub the participant from save-sync checkpoint snapshots, so a later
    // game load / quickload restore can't bring the cleared conversation back
    // (save-sync ties chat history to game saves and restores it on load).
    let scrubbed = save_sync::scrub_participant_from_checkpoints(
        &state.config.active_profile_paths().content_root(),
        &live_chat_id,
        &participant_id,
    );
    info!(
        "cleared {removed} message(s) for participant {participant_id} in live chat \
         {live_chat_id} (+{scrubbed} from save-sync checkpoints)"
    );
    Ok(Redirect::to(&participant_url(
        &live_chat_id,
        &participant_id,
    )))
}

async fn render_live_chat_page(
    state: Arc<AppState>,
    live_chat_id: &str,
    participant_id: &str,
) -> WebResult<Html<String>> {
    let live_chat = state.repository.get_live_chat(live_chat_id)?;
    let view = state
        .repository
        .live_chat_view(&live_chat, Some(participant_id))?;
    let selected = view
        .participants
        .iter()
        .find(|participant| participant.id == participant_id)
        .cloned()
        .or_else(|| view.participants.first().cloned());
    let selected_id = selected
        .as_ref()
        .map(|participant| participant.id.as_str())
        .unwrap_or(participant_id);
    let messages = state
        .repository
        .messages_for_participant(&live_chat, selected_id)?;
    let prompt = match selected.as_ref() {
        Some(participant) => {
            chasm_prompt::assemble_prompt(&state.repository, participant, &messages)
        }
        None => empty_prompt_assembly(selected_id),
    };
    let sidebar = SidebarView::build(&state, view.participants.len());
    Ok(Html(
        LiveChatTemplate {
            live_chat: view,
            selected_participant: selected,
            messages,
            prompt,
            sidebar,
        }
        .render()?,
    ))
}

async fn participants_partial(
    State(state): State<Arc<AppState>>,
    Path(live_chat_id): Path<String>,
) -> WebResult<Html<String>> {
    let live_chat = state.repository.get_live_chat(&live_chat_id)?;
    let selected = choose_participant_id(&state.repository, &live_chat)?;
    let view = state
        .repository
        .live_chat_view(&live_chat, Some(selected.as_str()))?;
    let sidebar = SidebarView::build(&state, view.participants.len());
    Ok(Html(
        CharacterListTemplate {
            live_chat: view,
            sidebar,
        }
        .render()?,
    ))
}

async fn messages_partial(
    State(state): State<Arc<AppState>>,
    Path((live_chat_id, participant_id)): Path<(String, String)>,
) -> WebResult<Html<String>> {
    let live_chat = state.repository.get_live_chat(&live_chat_id)?;
    let view = state
        .repository
        .live_chat_view(&live_chat, Some(participant_id.as_str()))?;
    let selected = view
        .participants
        .iter()
        .find(|participant| participant.id == participant_id)
        .cloned();
    let messages = state
        .repository
        .messages_for_participant(&live_chat, &participant_id)?;
    Ok(Html(
        MessageListTemplate {
            messages,
            selected_participant: selected,
        }
        .render()?,
    ))
}

async fn prompt_partial(
    State(state): State<Arc<AppState>>,
    Path((live_chat_id, participant_id)): Path<(String, String)>,
) -> WebResult<Html<String>> {
    let live_chat = state.repository.get_live_chat(&live_chat_id)?;
    let view = state
        .repository
        .live_chat_view(&live_chat, Some(participant_id.as_str()))?;
    let selected = view
        .participants
        .iter()
        .find(|participant| participant.id == participant_id)
        .cloned();
    let messages = state
        .repository
        .messages_for_participant(&live_chat, &participant_id)?;
    let prompt = match selected.as_ref() {
        Some(participant) => {
            chasm_prompt::assemble_prompt(&state.repository, participant, &messages)
        }
        None => empty_prompt_assembly(&participant_id),
    };
    Ok(Html(
        PromptPanelTemplate {
            live_chat: view,
            prompt,
            messages,
        }
        .render()?,
    ))
}

fn empty_prompt_assembly(participant_id: &str) -> PromptAssemblyView {
    PromptAssemblyView {
        participant_id: participant_id.to_string(),
        participant_name: String::new(),
        character_id: None,
        character_found: false,
        system_char_count: 0,
        history_count: 0,
        total_char_count: 0,
        components: Vec::new(),
        notes: vec!["No participant selected.".to_string()],
    }
}

const SETTINGS_CATEGORIES: [&str; 8] = [
    "interface",
    "profiles",
    "llm",
    "tts",
    "stt",
    "retrieval",
    "game",
    "tracing",
];

fn normalize_category(category: &str) -> String {
    if SETTINGS_CATEGORIES.contains(&category) {
        category.to_string()
    } else {
        "interface".to_string()
    }
}

async fn settings_index() -> Redirect {
    Redirect::to("/settings/interface")
}

/// `GET /theme.css` — the dynamic appearance stylesheet. Reads the saved
/// `InterfaceSettings` FRESH on every request and emits a small `:root{}`
/// override document (plus a few helper rules) from them, so changing an
/// appearance setting takes effect on the next page load with no restart. The
/// layout links this AFTER `app.css`, so these overrides win. `no-store` keeps a
/// stale theme from being cached after a change.
async fn theme_css(State(state): State<Arc<AppState>>) -> Response {
    let settings = AppSettings::load(&state.config.settings_path);
    let css = chasm_core::build_theme_css(&settings.interface);
    (
        [
            (header::CONTENT_TYPE, "text/css; charset=utf-8"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        css,
    )
        .into_response()
}

async fn settings_page(
    State(state): State<Arc<AppState>>,
    Path(category): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> WebResult<Html<String>> {
    let category = normalize_category(&category);
    let settings = AppSettings::load(&state.config.settings_path);
    let saved = params.get("saved").is_some_and(|value| value == "1");

    // The Tracing page has its own (richer) view model + template.
    if category == "tracing" {
        let nav = chasm_core::settings_nav_items(&category);
        let selected = params.get("trace").map(String::as_str);
        let page = trace_routes::build_tracing_view(
            &settings,
            nav,
            saved,
            state.config.settings_path.display().to_string(),
            selected,
        );
        return Ok(Html(TracingTemplate { page }.render()?));
    }

    let faster_installed = crate::launcher::faster_qwen3_tts_installed(&settings, &state.config);
    let engine_status = engine_statuses(&state.config.engines_dir, faster_installed);
    let running_engine = crate::launcher::tts_running_engine(&state);
    let voice_clone = build_voice_clone(&state.config, &settings, &settings.tts.local_engine);
    let model_status = llm_model_statuses(&state.config.llm_models_dir);
    let selected_llm = chasm_core::selected_llm_model_id(&settings.llm.model, &model_status);
    let llm_models = llm_models_panel_view(&model_status, &state.system_info, &selected_llm);
    let (retrieval_models, retrieval_host) = build_retrieval_models(&state.system_info);
    let (whisper_models, whisper_host) = build_whisper_models(&settings, &state.system_info);
    let game = build_game_launcher_view(&settings.launcher);
    let profiles = build_profiles_panel(&state, &settings);
    // Absolute "drop files here" folders, surfaced on each model page so power
    // users can add models by hand. Filesystem-dependent, so resolved here.
    let model_paths = chasm_core::ModelPathsView {
        llm: state.config.llm_models_dir.display().to_string(),
        stt: crate::launcher::whisper_models_dir(&settings)
            .display()
            .to_string(),
        tts_voices: active_voices_dir(&state.config).display().to_string(),
        tts_engines: state.config.engines_dir.display().to_string(),
    };
    // Runtime status for the LLM + STT pages: koboldcpp runs both, so one status
    // drives both. (TTS surfaces its per-engine status in the engine list itself.)
    let kobold_runtime = chasm_core::koboldcpp_runtime_status(
        crate::launcher::koboldcpp_status(&settings, &state.config).as_str(),
    );
    let page = settings_page_view(
        &settings,
        &category,
        saved,
        state.config.settings_path.display().to_string(),
        model_paths,
        kobold_runtime,
        &engine_status,
        voice_clone,
        llm_models,
        retrieval_models,
        retrieval_host,
        whisper_models,
        whisper_host,
        game,
        profiles,
        running_engine,
    );
    Ok(Html(SettingsTemplate { page }.render()?))
}

/// Renders just the voice-cloning panel for the engine named in `?engine=`,
/// so the settings page can swap it in when the local-engine dropdown changes
/// (clone status is per-engine). Unknown engines fall back to the saved one.
async fn voice_clone_partial(
    State(state): State<Arc<AppState>>,
    Query(params): Query<HashMap<String, String>>,
) -> WebResult<Html<String>> {
    let settings = AppSettings::load(&state.config.settings_path);
    let engine = params
        .get("engine")
        .cloned()
        .unwrap_or_else(|| settings.tts.local_engine.clone());
    let voice_clone = build_voice_clone(&state.config, &settings, &engine);
    Ok(Html(VoiceClonePartialTemplate { voice_clone }.render()?))
}

/// Builds the voice-cloning view for a specific engine from the active game
/// profile + on-disk clone status (read from the voices directory). Cloning is
/// per-engine, so the panel reflects whichever engine is passed in.
fn build_voice_clone(config: &AppConfig, settings: &AppSettings, engine: &str) -> VoiceCloneView {
    let engine_id = TTS_LOCAL_ENGINES
        .iter()
        .find(|(id, _)| *id == engine)
        .map(|(id, _)| (*id).to_string())
        .unwrap_or_else(|| settings.tts.local_engine.clone());
    let engine_label = TTS_LOCAL_ENGINES
        .iter()
        .find(|(id, _)| *id == engine_id)
        .map(|(_, label)| label.to_string())
        .unwrap_or_else(|| engine_id.clone());

    let Some(profile) = GameProfile::read(&config.profiles_dir, &settings.profile) else {
        return VoiceCloneView {
            has_profile: false,
            engine_id,
            engine_label,
            ..Default::default()
        };
    };

    // Clone status is read from the ACTIVE profile's voices dir (legacy
    // `{workspace}/voices` fallback) so the panel reflects the right profile.
    let voices_dir = config.active_profile_paths().voices_dir();
    let mut characters = Vec::new();
    let mut any_cloning = false;
    let mut cloned_count = 0;
    for character in &profile.characters {
        // Cloning is per-engine: the shared reference clip lives at
        // voices/<name>/reference.wav, but the selected engine's clone lives at
        // voices/<name>/<engine>/sample.wav. Status reflects the chosen engine.
        let dir = voices_dir.join(&character.name).join(&engine_id);
        let status = if dir.join(".cloning").exists() {
            any_cloning = true;
            "cloning"
        } else if dir.join("sample.wav").exists() {
            cloned_count += 1;
            "cloned"
        } else if dir.join(".failed").exists() {
            "failed"
        } else {
            "pending"
        };
        characters.push(VoiceCloneCharacterView {
            name: character.name.clone(),
            status: status.to_string(),
            status_label: clone_status_label(status),
        });
    }

    VoiceCloneView {
        has_profile: true,
        profile_id: profile.id,
        profile_name: profile.name,
        engine_id,
        engine_label,
        characters,
        any_cloning,
        cloned_count,
    }
}

/// Kicks off cloning the active profile's character voices: marks each
/// character `cloning` and spawns the detached clone orchestrator.
/// Kicks off a voice-clone run for the active profile with the CURRENTLY-SELECTED
/// TTS engine: marks each character `.cloning` under `voices/<name>/<engine>/` and
/// spawns clone-voices.ps1 (which runs the profile's own `extract_voices.py` to pull
/// per-NPC references from the game, then clones each with the engine). Shared by
/// the legacy redirect handler and the JSON API. Cloning is always per-engine, so a
/// PocketTTS run never touches faster-qwen3-tts clips and vice-versa.
pub(crate) fn start_voice_clone(state: &AppState) {
    let settings = AppSettings::load(&state.config.settings_path);
    let profile_id = settings.active_profile_id(&state.config.profiles_dir);
    let Some(profile) = GameProfile::read(&state.config.profiles_dir, &profile_id) else {
        return;
    };
    // Clones are written to (and read back from) the ACTIVE profile's voices dir.
    let voices_dir = state.config.profile_paths(&profile_id).voices_dir();
    let engine = &settings.tts.local_engine;
    for character in &profile.characters {
        let dir = voices_dir.join(&character.name).join(engine);
        let _ = fs::create_dir_all(&dir);
        let _ = fs::remove_file(dir.join(".failed"));
        let _ = fs::write(dir.join(".cloning"), "");
    }

    let script = state
        .config
        .workspace_root
        .join("scripts")
        .join("clone-voices.ps1");
    let profile_dir = state.config.profiles_dir.join(&profile_id);
    let _ = Command::new("powershell")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            &script.display().to_string(),
            "-ProfileDir",
            &profile_dir.display().to_string(),
            "-EnginesDir",
            &state.config.engines_dir.display().to_string(),
            "-VoicesDir",
            &voices_dir.display().to_string(),
            "-Engine",
            engine,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

async fn clone_voices(State(state): State<Arc<AppState>>) -> WebResult<Redirect> {
    start_voice_clone(&state);
    Ok(Redirect::to("/settings/tts?cloning=1"))
}

/// `GET /api/voices/clone` — per-character clone status for the React TTS page,
/// scoped to the currently-selected engine (each engine clones separately).
async fn voice_clone_status(State(state): State<Arc<AppState>>) -> Json<VoiceCloneView> {
    let settings = AppSettings::load(&state.config.settings_path);
    Json(build_voice_clone(
        &state.config,
        &settings,
        &settings.tts.local_engine,
    ))
}

/// `POST /api/voices/clone` — start cloning the active profile's voices with the
/// selected engine, then return the fresh status (characters flip to "cloning…").
async fn voice_clone_start(State(state): State<Arc<AppState>>) -> Json<VoiceCloneView> {
    start_voice_clone(&state);
    let settings = AppSettings::load(&state.config.settings_path);
    Json(build_voice_clone(
        &state.config,
        &settings,
        &settings.tts.local_engine,
    ))
}

/// Serves a file from the ACTIVE profile's voices dir (resolved per request,
/// legacy `{workspace}/voices` fallback). Replaces the old static `ServeDir`
/// mount so a profile switch repoints `/voices/...` with no restart. The path is
/// sanitized component-by-component and verified to resolve inside the voices
/// dir (path-traversal guard), mirroring `ServeDir`'s safety.
async fn serve_voice_file(
    State(state): State<Arc<AppState>>,
    Path(path): Path<String>,
) -> Response {
    let base = active_voices_dir(&state.config);
    // Reject any traversal/absolute/drive components; only plain names allowed.
    let mut full = base.clone();
    for segment in path.split(['/', '\\']) {
        if segment.is_empty() || segment == "." {
            continue;
        }
        if segment == ".."
            || segment.contains(':')
            || matches!(segment, "<" | ">" | "|" | "?" | "*")
        {
            return (StatusCode::NOT_FOUND, "not found").into_response();
        }
        full.push(segment);
    }
    if !full.starts_with(&base) || !full.is_file() {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    match fs::read(&full) {
        Ok(bytes) => ([(header::CONTENT_TYPE, voice_content_type(&full))], bytes).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

/// Content-type for a served voice file by extension (audio clips + the small
/// JSON/text sidecars the clone pipeline may write). Defaults to octet-stream.
fn voice_content_type(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        Some("ogg") => "audio/ogg",
        Some("flac") => "audio/flac",
        Some("json") => "application/json; charset=utf-8",
        Some("txt") | Some("log") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// JSON `POST /profile/select` body: `{ "id": "<profile-id>" }` (also accepts
/// `profile`). The path variant `POST /profile/select/:id` carries the id in the
/// URL instead.
#[derive(serde::Deserialize)]
struct SelectProfileBody {
    #[serde(default)]
    id: String,
    #[serde(default)]
    profile: String,
}

/// `POST /profile/select` — switch the active game profile from a JSON body.
async fn select_profile(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SelectProfileBody>,
) -> WebResult<Json<serde_json::Value>> {
    let id = if body.id.trim().is_empty() {
        body.profile
    } else {
        body.id
    };
    select_profile_inner(&state, id.trim())
}

/// `POST /profile/select/:id` — switch the active game profile via path param.
async fn select_profile_path(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> WebResult<Json<serde_json::Value>> {
    select_profile_inner(&state, id.trim())
}

/// Validates `id` against the available profiles, persists it as the active
/// profile, and relaunches the warm TTS worker at the new profile's voices dir.
/// Returns `{ok, profile: {id, name}}` on success. An unknown id is a 400-style
/// error so the caller can surface it.
fn select_profile_inner(state: &Arc<AppState>, id: &str) -> WebResult<Json<serde_json::Value>> {
    if id.is_empty() {
        return Err(WebError(anyhow::anyhow!("profile id is required.")));
    }
    let Some(profile) = GameProfile::read(&state.config.profiles_dir, id) else {
        return Err(WebError(anyhow::anyhow!("unknown profile id '{id}'.")));
    };

    let mut settings = AppSettings::load(&state.config.settings_path);
    settings.profile = profile.id.clone();
    settings.save(&state.config.settings_path)?;

    // koboldcpp serves TTS with voices fixed at launch (--ttsdir), so a profile
    // switch does not respawn a per-profile TTS worker.
    info!("active profile switched to '{}'", profile.id);
    Ok(Json(serde_json::json!({
        "ok": true,
        "profile": { "id": profile.id, "name": profile.name },
    })))
}

/// Reads the install status of each local engine from on-disk markers under the
/// engines directory: `.installed`, `.installing`, `.failed`, else not installed.
/// HF model repo backing each local TTS engine — used to verify the engine's
/// *weights* are present, not just its venv/code. The install marker alone is
/// not enough: model weights are pulled separately and can be missing or
/// mid-download, which is why an engine could show "installed" before it was.
const TTS_ENGINE_MODELS: &[(&str, &str)] = &[
    ("pockettts", "kyutai/pocket-tts"),
    ("faster-qwen3-tts", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
];

/// HuggingFace hub cache dir (honors `HF_HUB_CACHE` / `HF_HOME`, else
/// `~/.cache/huggingface/hub`).
fn hf_hub_cache_dir() -> Option<std::path::PathBuf> {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE") {
        return Some(std::path::PathBuf::from(dir));
    }
    if let Some(home) = std::env::var_os("HF_HOME") {
        return Some(std::path::PathBuf::from(home).join("hub"));
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    Some(
        std::path::PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub"),
    )
}

/// Whether an engine's model weights are fully present: the HF cache dir exists
/// and has no `*.incomplete` blob still downloading. `None` = engine has no known
/// model repo (then trust the install marker).
fn engine_model_present(engine_id: &str) -> Option<bool> {
    let repo = TTS_ENGINE_MODELS.iter().find(|(id, _)| *id == engine_id)?.1;
    let Some(cache) = hf_hub_cache_dir() else {
        return Some(true); // can't locate the cache → don't contradict the marker
    };
    let model_dir = cache.join(format!("models--{}", repo.replace('/', "--")));
    if !model_dir.is_dir() {
        return Some(false);
    }
    let downloading = std::fs::read_dir(model_dir.join("blobs"))
        .map(|entries| {
            entries
                .flatten()
                .any(|entry| entry.file_name().to_string_lossy().ends_with(".incomplete"))
        })
        .unwrap_or(false);
    Some(!downloading)
}

/// Install status for each local TTS engine. Both engines now install into a
/// chasm-managed `engines/<id>` venv, so both key off the on-disk markers
/// (`.installing`/`.failed`/`.installed`) the install script writes — that is what
/// surfaces an in-flight install or a failure to the picker.
///
/// `faster_qwen3_installed` is the launcher's belt-and-suspenders check (the managed
/// venv + server script resolve, OR a developer helper config points at an existing
/// install). For faster-qwen3-tts it forces "installed" even without an `.installed`
/// marker, so a hand-configured dev install (which never ran our script) still
/// reports correctly.
pub(crate) fn engine_statuses(
    engines_dir: &std::path::Path,
    faster_qwen3_installed: bool,
) -> HashMap<String, String> {
    TTS_LOCAL_ENGINES
        .iter()
        .map(|(id, _label)| {
            let dir = engines_dir.join(id);
            // Flip a stalled .installing marker to .failed (progress = install.log).
            flip_marker_if_stale(
                &dir.join(".installing"),
                &dir.join(".failed"),
                &[dir.join("install.log")],
            );
            let status = if dir.join(".installing").exists() {
                "installing"
            } else if dir.join(".failed").exists() {
                "failed"
            } else if dir.join(".installed").exists() {
                // The marker is written after the venv/code; the engine is only
                // truly installed when its model weights are present too.
                match engine_model_present(id) {
                    Some(true) | None => "installed",
                    Some(false) => "not_installed",
                }
            } else if *id == "faster-qwen3-tts" && faster_qwen3_installed {
                // No marker, but the managed venv + script resolve (or a dev helper
                // config points at an existing install).
                "installed"
            } else {
                "not_installed"
            };
            ((*id).to_string(), status.to_string())
        })
        .collect()
}

/// Install status of the Parakeet STT engine's `engines/parakeet` dir, using the
/// same markers [`engine_statuses`] reads for the TTS engines (`.installing` /
/// `.failed` / `.installed`, stalled-marker backstop included). The `.installed`
/// marker is only trusted when the venv actually resolves (belt and suspenders).
pub(crate) fn parakeet_engine_status(state: &AppState) -> String {
    let dir = state
        .config
        .engines_dir
        .join(chasm_core::PARAKEET_ENGINE_ID);
    flip_marker_if_stale(
        &dir.join(".installing"),
        &dir.join(".failed"),
        &[dir.join("install.log")],
    );
    if dir.join(".installing").exists() {
        "installing"
    } else if dir.join(".failed").exists() {
        "failed"
    } else if launcher::parakeet_installed(&state.config) {
        "installed"
    } else {
        "not_installed"
    }
    .to_string()
}

/// Kicks off a local-engine install: writes an `.installing` marker and spawns
/// the detached install script (which writes `.installed`/`.failed` on finish).
/// Returns `false` for an unknown engine; a no-op (already installed/installing)
/// returns `true`. Shared by the settings endpoint + onboarding "Use recommended".
pub(crate) fn start_engine_install(state: &AppState, id: &str) -> std::io::Result<bool> {
    // TTS engines + the Parakeet STT engine share the same venv install shape
    // (engines/<id>/.venv via scripts/install-engine.ps1).
    let known = TTS_LOCAL_ENGINES
        .iter()
        .any(|(engine_id, _)| engine_id == &id)
        || id == chasm_core::PARAKEET_ENGINE_ID;
    if !known {
        return Ok(false);
    }
    let engine_dir = state.config.engines_dir.join(id);
    if engine_dir.join(".installed").exists() || engine_dir.join(".installing").exists() {
        return Ok(true);
    }
    fs::create_dir_all(&engine_dir)?;
    let _ = fs::remove_file(engine_dir.join(".failed"));

    // Surface a missing installer immediately (Requirement E): if the bundled
    // install script (or the bundled uv) is absent, write `.failed` + a short log
    // and bail, so the card flips to an error instead of hanging on "Installing…".
    let script = state
        .config
        .workspace_root
        .join("scripts")
        .join("install-engine.ps1");
    if !script.exists() {
        let _ = fs::write(
            engine_dir.join(".failed"),
            format!("installer missing: {}", script.display()),
        );
        let _ = fs::write(
            engine_dir.join("install.log"),
            format!("install-engine.ps1 not found at {}\n", script.display()),
        );
        tracing::warn!(
            "engine install '{id}': installer missing at {}; wrote .failed",
            script.display()
        );
        return Ok(true);
    }

    fs::write(engine_dir.join(".installing"), "")?;

    let spawned = Command::new("powershell")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            &script.display().to_string(),
            "-Engine",
            id,
            "-EnginesDir",
            &state.config.engines_dir.display().to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    if let Err(error) = spawned {
        let _ = fs::remove_file(engine_dir.join(".installing"));
        let _ = fs::write(engine_dir.join(".failed"), format!("spawn failed: {error}"));
    }
    Ok(true)
}

async fn install_engine(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> WebResult<Redirect> {
    if !start_engine_install(&state, &id)? {
        return Ok(Redirect::to("/settings/tts"));
    }
    Ok(Redirect::to("/settings/tts?installing=1"))
}

/// How long a `.downloading`/`.installing` marker may sit with NO progress before a
/// status read flips it to `.failed`. Generous (30 min) so a slow multi-GB download
/// over a poor connection is never killed — this is purely a backstop so a *dead*
/// spawn (the script never ran / crashed before writing `.failed`) can't hang a card
/// forever. "Progress" = the newest mtime among the marker + its sibling progress
/// files (`.log`/`install.log`/`*.part`), which advance while a real download runs.
const STALE_MARKER_SECS: u64 = 30 * 60;

/// The age in seconds of the newest of `paths` (most-recently-modified), or `None`
/// when none exist / are unreadable.
fn newest_age_secs(paths: &[std::path::PathBuf]) -> Option<u64> {
    paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
        .filter_map(|mtime| std::time::SystemTime::now().duration_since(mtime).ok())
        .map(|d| d.as_secs())
        .min()
}

/// Backstop for a stalled marker: if `marker` exists but neither it nor any of its
/// `progress` siblings has advanced within [`STALE_MARKER_SECS`], the spawn is
/// presumed dead — rename the marker to `failed_marker` so the card surfaces an
/// error instead of an eternal "Downloading…"/"Installing…". Returns `true` when it
/// flipped a stale marker (so the caller treats the state as `failed` this read).
/// Best-effort: any filesystem error leaves the marker as-is (reads as in-flight).
pub(crate) fn flip_marker_if_stale(
    marker: &std::path::Path,
    failed_marker: &std::path::Path,
    progress: &[std::path::PathBuf],
) -> bool {
    if !marker.exists() {
        return false;
    }
    let mut watch: Vec<std::path::PathBuf> = vec![marker.to_path_buf()];
    watch.extend_from_slice(progress);
    match newest_age_secs(&watch) {
        Some(age) if age >= STALE_MARKER_SECS => {
            let _ = std::fs::write(
                failed_marker,
                format!("stalled: no progress for {age}s (presumed dead)"),
            );
            let _ = std::fs::remove_file(marker);
            true
        }
        _ => false,
    }
}

/// Reads the download status of each LLM model from the models directory: a
/// matching `*.gguf` (any quant of that model) → `downloaded`; a `.downloading`
/// marker → `downloading`; a `.failed` marker → `failed`; else `available`.
/// Mirrors [`engine_statuses`] but keys off on-disk GGUF files, since a model
/// the user already has (e.g. a different quant) should show as downloaded. A
/// stalled `.downloading` marker (no progress for [`STALE_MARKER_SECS`]) is flipped
/// to `failed` so a dead spawn never hangs the card.
pub(crate) fn llm_model_statuses(models_dir: &std::path::Path) -> HashMap<String, String> {
    // Lowercased names of every *.gguf present, for any-quant matching.
    let gguf_names: Vec<String> = fs::read_dir(models_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_lowercase();
            name.ends_with(".gguf").then_some(name)
        })
        .collect();

    LLM_MODELS
        .iter()
        .map(|model| {
            let stem = llm_model_match_stem(model);
            let downloading = models_dir.join(format!("{}.downloading", model.id));
            let failed = models_dir.join(format!("{}.failed", model.id));
            // Flip a stalled marker to failed (progress = its .log + the .part file).
            let progress = [
                models_dir.join(format!("{}.log", model.id)),
                models_dir.join(format!("{}.part", llm_model_filename(model))),
            ];
            flip_marker_if_stale(&downloading, &failed, &progress);
            let status = if gguf_names.iter().any(|name| name.contains(&stem)) {
                "downloaded"
            } else if downloading.exists() {
                "downloading"
            } else if failed.exists() {
                "failed"
            } else {
                "available"
            };
            (model.id.to_string(), status.to_string())
        })
        .collect()
}

/// Reads the download status of each Whisper model from the Whisper models dir:
/// the model's `.bin` file present → `downloaded`; a `<id>.downloading` marker →
/// `downloading`; a `<id>.failed` marker → `failed`; else `available`. Mirrors
/// [`llm_model_statuses`] but keys off the exact `.bin` filename (koboldcpp loads
/// Whisper by file, not a HF repo id).
fn whisper_model_statuses(models_dir: &std::path::Path) -> HashMap<String, String> {
    WHISPER_MODELS
        .iter()
        .map(|model| {
            let downloading = models_dir.join(format!("{}.downloading", model.id));
            let failed = models_dir.join(format!("{}.failed", model.id));
            let progress = [
                models_dir.join(format!("{}.log", model.id)),
                models_dir.join(format!("{}.part", model.file)),
            ];
            flip_marker_if_stale(&downloading, &failed, &progress);
            let status = if models_dir.join(model.file).is_file() {
                "downloaded"
            } else if downloading.exists() {
                "downloading"
            } else if failed.exists() {
                "failed"
            } else {
                "available"
            };
            (model.id.to_string(), status.to_string())
        })
        .collect()
}

/// Builds the Whisper model list + host summary for the STT picker. The
/// "recommended" badge is the largest model that fits this host's GPU comfortably
/// (via [`recommended_index`] over the models' footprints); the per-model hint
/// comes from [`SystemInfo::gpu_fit`]. Mirrors [`build_retrieval_models`].
pub(crate) fn build_whisper_models(
    settings: &AppSettings,
    system: &SystemInfo,
) -> (Vec<WhisperModelView>, RetrievalHostView) {
    let models_dir = crate::launcher::whisper_models_dir(settings);
    let statuses = whisper_model_statuses(&models_dir);

    let footprints: Vec<f64> = WHISPER_MODELS.iter().map(|m| m.size_gb).collect();
    let recommended = recommended_index(&footprints, system);

    let models = WHISPER_MODELS
        .iter()
        .enumerate()
        .map(|(index, m)| {
            let status = statuses
                .get(m.id)
                .map(String::as_str)
                .unwrap_or("available");
            WhisperModelView {
                id: m.id.to_string(),
                name: m.name.to_string(),
                file: m.file.to_string(),
                size_label: format!("~{:.1} GB", m.size_gb),
                status: status.to_string(),
                status_label: whisper_model_status_label(status),
                downloaded: status == "downloaded",
                downloading: status == "downloading",
                can_download: status == "available" || status == "failed",
                recommended: recommended == Some(index),
                fit_hint: fit_hint(system, m.size_gb),
                // Set by stt_panel_view once the saved model filename is known.
                selected: false,
            }
        })
        .collect();

    let host = RetrievalHostView {
        summary: host_summary(system),
        has_gpu: system.vram_total_gb.is_some(),
    };
    (models, host)
}

/// Reads the download status of each retrieval model. A model is `downloaded`
/// when its `models--<org>--<repo>` weight dir exists under the embed cache dir,
/// `downloading` when a `.downloading` marker (under a per-id markers dir) is
/// present, `failed` on a `.failed` marker, else `available`. Mirrors
/// [`engine_statuses`].
pub(crate) fn retrieval_model_statuses(cache_dir: &std::path::Path) -> HashMap<String, String> {
    RETRIEVAL_MODELS
        .iter()
        .map(|model| {
            let markers = retrieval_marker_dir(cache_dir, model.id);
            let status = if chasm_embed::model_downloaded(model.id) {
                "downloaded"
            } else if markers.join(".downloading").exists() {
                "downloading"
            } else if markers.join(".failed").exists() {
                "failed"
            } else {
                "available"
            };
            (model.id.to_string(), status.to_string())
        })
        .collect()
}

/// Ensures the koboldcpp runtime (the exe that serves the LLM AND Whisper STT) is
/// present, kicking off its detached GitHub-release download when it's `Missing`.
/// A no-op when koboldcpp is already `Installed` (existing users keep their build,
/// never re-downloaded) or already `Downloading`. Called alongside every LLM /
/// Whisper model download so one "Download" click also pulls the runtime if needed.
/// Best-effort: download/marker errors are logged, never surfaced to the caller.
pub(crate) fn ensure_koboldcpp(state: &AppState) {
    let settings = AppSettings::load(&state.config.settings_path);
    let status = crate::launcher::koboldcpp_status(&settings, &state.config);
    if status != crate::launcher::KoboldcppStatus::Missing {
        return;
    }
    let exe = crate::launcher::koboldcpp_exe_path(&settings, &state.config);
    let Some(dir) = exe.parent() else {
        return;
    };
    if let Err(error) = fs::create_dir_all(dir) {
        tracing::warn!("could not create koboldcpp dir {}: {error}", dir.display());
        return;
    }
    let _ = fs::remove_file(dir.join("koboldcpp.failed"));
    let script = state
        .config
        .workspace_root
        .join("scripts")
        .join("download-koboldcpp.ps1");
    // Surface a missing downloader immediately (Requirement E): write `.failed`
    // instead of a `.downloading` marker that would hang the runtime card forever.
    if !script.exists() {
        let _ = fs::write(
            dir.join("koboldcpp.failed"),
            format!("downloader missing: {}", script.display()),
        );
        tracing::warn!("koboldcpp download: downloader missing at {}; wrote .failed", script.display());
        return;
    }
    if let Err(error) = fs::write(dir.join("koboldcpp.downloading"), "") {
        tracing::warn!("could not write koboldcpp.downloading marker: {error}");
        return;
    }
    let spawned = Command::new("powershell")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            &script.display().to_string(),
            "-ExePath",
            &exe.display().to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    match spawned {
        Ok(_) => tracing::info!("koboldcpp missing; started runtime download -> {}", exe.display()),
        Err(error) => {
            let _ = fs::remove_file(dir.join("koboldcpp.downloading"));
            let _ = fs::write(
                dir.join("koboldcpp.failed"),
                format!("spawn failed: {error}"),
            );
            tracing::warn!("could not start koboldcpp download: {error}");
        }
    }
}

/// Ensures the llama.cpp runtime exists: when `llamacpp_status` is Missing,
/// spawns `scripts/download-llamacpp.ps1` detached (markers `llamacpp.downloading`
/// / `.done` / `.failed` beside the exe). Mirrors [`ensure_koboldcpp`].
pub(crate) fn ensure_llamacpp(state: &AppState) {
    let status = crate::launcher::llamacpp_status(&state.config);
    if status != crate::launcher::KoboldcppStatus::Missing {
        return;
    }
    let exe = crate::launcher::llamacpp_exe_path(&state.config);
    let Some(dir) = exe.parent() else {
        return;
    };
    if let Err(error) = fs::create_dir_all(dir) {
        tracing::warn!("could not create llamacpp dir {}: {error}", dir.display());
        return;
    }
    let _ = fs::remove_file(dir.join("llamacpp.failed"));
    let script = state
        .config
        .workspace_root
        .join("scripts")
        .join("download-llamacpp.ps1");
    if !script.exists() {
        let _ = fs::write(
            dir.join("llamacpp.failed"),
            format!("downloader missing: {}", script.display()),
        );
        tracing::warn!(
            "llamacpp download: downloader missing at {}; wrote .failed",
            script.display()
        );
        return;
    }
    if let Err(error) = fs::write(dir.join("llamacpp.downloading"), "") {
        tracing::warn!("could not write llamacpp.downloading marker: {error}");
        return;
    }
    let spawned = Command::new("powershell")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            &script.display().to_string(),
            "-ExePath",
            &exe.display().to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    match spawned {
        Ok(_) => tracing::info!(
            "llama.cpp missing; started runtime download -> {}",
            exe.display()
        ),
        Err(error) => {
            let _ = fs::remove_file(dir.join("llamacpp.downloading"));
            let _ = fs::write(dir.join("llamacpp.failed"), format!("spawn failed: {error}"));
            tracing::warn!("could not start llama.cpp download: {error}");
        }
    }
}

/// Kicks off downloading an LLM GGUF: writes a `<id>.downloading` marker and
/// spawns the detached download script (which writes `<id>.done`/`<id>.failed`
/// and removes the marker on finish). Returns `false` for an unknown id; a no-op
/// (already present / in flight) returns `true`. Shared by the settings endpoint +
/// onboarding "Use recommended".
pub(crate) fn start_llm_download(state: &AppState, id: &str) -> std::io::Result<bool> {
    let Some(model) = LLM_MODELS.iter().find(|model| model.id == id) else {
        return Ok(false);
    };
    let models_dir = &state.config.llm_models_dir;
    fs::create_dir_all(models_dir)?;

    // Already present (any quant) or already in flight → no-op.
    let status = llm_model_statuses(models_dir);
    if matches!(
        status.get(id).map(String::as_str),
        Some("downloaded") | Some("downloading")
    ) {
        return Ok(true);
    }

    let _ = fs::remove_file(models_dir.join(format!("{id}.failed")));

    let file = llm_model_filename(model);
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}?download=true",
        model.repo, file
    );
    let script = state
        .config
        .workspace_root
        .join("scripts")
        .join("download-llm.ps1");
    // Surface a missing downloader immediately (Requirement E): write `.failed`
    // instead of a `.downloading` marker that would hang the card forever.
    if !script.exists() {
        let _ = fs::write(
            models_dir.join(format!("{id}.failed")),
            format!("downloader missing: {}", script.display()),
        );
        tracing::warn!("LLM download '{id}': downloader missing at {}; wrote .failed", script.display());
        return Ok(true);
    }
    fs::write(models_dir.join(format!("{id}.downloading")), "")?;

    let spawned = Command::new("powershell")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            &script.display().to_string(),
            "-Id",
            id,
            "-Url",
            &url,
            "-File",
            &file,
            "-ModelsDir",
            &models_dir.display().to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    if let Err(error) = spawned {
        let _ = fs::remove_file(models_dir.join(format!("{id}.downloading")));
        let _ = fs::write(
            models_dir.join(format!("{id}.failed")),
            format!("spawn failed: {error}"),
        );
    }
    Ok(true)
}

async fn download_llm_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> WebResult<Redirect> {
    if !start_llm_download(&state, &id)? {
        return Ok(Redirect::to("/settings/llm"));
    }
    // One "Download" click also pulls the koboldcpp runtime if it isn't present,
    // so the model has something to run on (no-op when already installed).
    ensure_koboldcpp(&state);
    Ok(Redirect::to("/settings/llm?downloading=1"))
}

/// Kicks off downloading a Whisper GGML `.bin` into the Whisper models dir (the
/// dir koboldcpp loads `--whispermodel` from). Thin wrapper over
/// [`start_whisper_download`] so the STT page and the onboarding "Use recommended"
/// flow share one downloader. Mirrors [`download_llm_model`].
async fn download_whisper_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> WebResult<Redirect> {
    let _ = start_whisper_download(&state, &id);
    // koboldcpp serves Whisper STT too, so ensure the runtime alongside the model
    // (no-op when already installed).
    ensure_koboldcpp(&state);
    Ok(Redirect::to("/settings/stt?downloading=1"))
}

/// Opens a model-category folder in Windows Explorer, then redirects back to the
/// matching settings page. `category` is a fixed key (not user input) mapped to a
/// config dir, so there's no path injection: unknown keys just redirect with no
/// side effect. Creates the dir first (a freshly-set-up machine may not have it
/// yet) so Explorer opens the right place instead of erroring.
async fn open_model_folder(
    State(state): State<Arc<AppState>>,
    Path(category): Path<String>,
) -> WebResult<Redirect> {
    let settings = AppSettings::load(&state.config.settings_path);
    let (dir, back) = match category.as_str() {
        "llm" => (state.config.llm_models_dir.clone(), "/settings/llm"),
        "stt" => (crate::launcher::whisper_models_dir(&settings), "/settings/stt"),
        "tts-voices" => (active_voices_dir(&state.config), "/settings/tts"),
        "tts-engines" => (state.config.engines_dir.clone(), "/settings/tts"),
        _ => return Ok(Redirect::to("/settings")),
    };
    let _ = fs::create_dir_all(&dir);
    let _ = Command::new("explorer")
        .arg(dir.display().to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    Ok(Redirect::to(back))
}

/// Kicks off a Whisper `.bin` download into the Whisper models dir, returning
/// whether the model is already present / in flight (so onboarding can fire it the
/// same fire-and-forget way it does the LLM). Writes a `<id>.downloading` marker
/// and spawns the detached generic `download-llm.ps1` (Id/Url/File/ModelsDir — it
/// writes `<id>.done`/`<id>.failed` and clears the marker). The `.bin` lands beside
/// the model koboldcpp loads, pulled from `ggerganov/whisper.cpp`.
pub(crate) fn start_whisper_download(state: &AppState, id: &str) -> std::io::Result<bool> {
    let Some(model) = whisper_model_by_id(id) else {
        return Ok(false);
    };
    let settings = AppSettings::load(&state.config.settings_path);
    let models_dir = crate::launcher::whisper_models_dir(&settings);
    fs::create_dir_all(&models_dir)?;

    // Already present (the .bin exists) or already in flight → no-op.
    let status = whisper_model_statuses(&models_dir);
    if matches!(
        status.get(id).map(String::as_str),
        Some("downloaded") | Some("downloading")
    ) {
        return Ok(true);
    }

    let _ = fs::remove_file(models_dir.join(format!("{id}.failed")));

    let file = model.file.to_string();
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}?download=true",
        WHISPER_REPO, file
    );
    let script = state
        .config
        .workspace_root
        .join("scripts")
        .join("download-llm.ps1");
    // Surface a missing downloader immediately (Requirement E).
    if !script.exists() {
        let _ = fs::write(
            models_dir.join(format!("{id}.failed")),
            format!("downloader missing: {}", script.display()),
        );
        tracing::warn!("Whisper download '{id}': downloader missing at {}; wrote .failed", script.display());
        return Ok(true);
    }
    fs::write(models_dir.join(format!("{id}.downloading")), "")?;

    let spawned = Command::new("powershell")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            &script.display().to_string(),
            "-Id",
            id,
            "-Url",
            &url,
            "-File",
            &file,
            "-ModelsDir",
            &models_dir.display().to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    if let Err(error) = spawned {
        let _ = fs::remove_file(models_dir.join(format!("{id}.downloading")));
        let _ = fs::write(
            models_dir.join(format!("{id}.failed")),
            format!("spawn failed: {error}"),
        );
    }
    Ok(true)
}

/// Per-model markers live under `<cache>/.markers/<id>` so they never collide
/// with the `models--…` weight dirs fastembed creates.
fn retrieval_marker_dir(cache_dir: &std::path::Path, id: &str) -> std::path::PathBuf {
    cache_dir.join(".markers").join(id)
}

/// Builds the retrieval model list + host summary for the settings page. The
/// "recommended" badge is the largest embedder that fits this host's GPU
/// comfortably (via [`recommended_index`] over the embedders' footprints); the
/// per-model hint comes from [`SystemInfo::gpu_fit`].
pub(crate) fn build_retrieval_models(
    system: &SystemInfo,
) -> (Vec<RetrievalModelView>, RetrievalHostView) {
    let cache_dir = embed_cache_dir();
    let statuses = retrieval_model_statuses(&cache_dir);

    // Recommended pick is over the embedders only (the list the user actually
    // chooses an embed tier from). Map that back to the registry index.
    let embedder_indices: Vec<usize> = RETRIEVAL_MODELS
        .iter()
        .enumerate()
        .filter(|(_, m)| m.kind == "embedder")
        .map(|(i, _)| i)
        .collect();
    let embedder_footprints: Vec<f64> = embedder_indices
        .iter()
        .map(|&i| RETRIEVAL_MODELS[i].footprint_gb)
        .collect();
    let recommended_registry_index =
        recommended_index(&embedder_footprints, system).map(|local| embedder_indices[local]);

    let models = RETRIEVAL_MODELS
        .iter()
        .enumerate()
        .map(|(index, m)| {
            let status = statuses
                .get(m.id)
                .map(String::as_str)
                .unwrap_or("available");
            RetrievalModelView {
                id: m.id.to_string(),
                label: m.label.to_string(),
                kind: m.kind.to_string(),
                tier: m.tier.to_string(),
                size_label: format!("~{:.1} GB", m.footprint_gb),
                status: status.to_string(),
                status_label: retrieval_model_status_label(status),
                downloaded: status == "downloaded",
                downloading: status == "downloading",
                can_download: status == "available" || status == "failed",
                recommended: Some(index) == recommended_registry_index,
                fit_hint: fit_hint(system, m.footprint_gb),
                // Set by retrieval_panel_view once the saved tiers are known.
                selected: false,
            }
        })
        .collect();

    let host = RetrievalHostView {
        summary: host_summary(system),
        has_gpu: system.vram_total_gb.is_some(),
    };
    (models, host)
}

/// Builds the "Game" (launcher) settings panel: resolves the effective launcher
/// config (MO2 exe, instance, profile, executable, game dir) from the saved
/// overrides + the environment and detects MO2/NVSE/the FNV install. Detection is
/// filesystem-dependent, so it lives in the web layer. chasm no longer launches
/// the game or installs mods — this page is now a read-only "what's detected" view
/// plus the override fields the bridge resolution still reads.
fn build_game_launcher_view(launcher: &LauncherSettings) -> GameLauncherView {
    let cfg = LauncherConfig::resolve(launcher);

    GameLauncherView {
        mo2_exe: cfg.mo2_exe.display().to_string(),
        instance: cfg.instance.clone(),
        profile: cfg.profile.clone(),
        executable: cfg.executable.clone(),
        game_dir: cfg.game_dir.display().to_string(),
        mo2_exe_override: launcher.mo2_exe.clone(),
        instance_override: launcher.instance.clone(),
        profile_override: launcher.profile.clone(),
        executable_override: launcher.executable.clone(),
        game_dir_override: launcher.game_dir.clone(),
        mo2_detected: mo2_detected(&cfg),
        nvse_detected: nvse_detected(&cfg),
        falloutnv_detected: falloutnv_detected(&cfg),
        launch_command: launcher::launch_command_string(&cfg),
        moshortcut_arg: cfg.moshortcut_arg(),
    }
}

/// Builds the "Profiles" settings panel: every drop-in [`GameProfile`] as a card
/// with its content counts (characters / lorebooks / quests / actions) and how
/// many of its characters have a cloned voice for the active TTS engine. The
/// active profile is flagged so its card shows ACTIVE + a disabled Activate.
///
/// Counts come straight from each profile's folders on disk (so non-active
/// profiles are counted too, not just the active one the repository scopes to):
/// characters from `profile.json`, lorebooks from `worlds/*.json`, quest/action
/// entries summed across `headless/{quest,action}-books/*.json`, cloned voices
/// from `voices/<name>/<engine>/sample.wav`.
fn build_profiles_panel(state: &AppState, settings: &AppSettings) -> ProfilesPanelView {
    let profiles_dir = &state.config.profiles_dir;
    let active_id = settings.active_profile_id(profiles_dir);
    let engine = &settings.tts.local_engine;

    let profiles = GameProfile::list(profiles_dir)
        .into_iter()
        .map(|profile| {
            let dir = profiles_dir.join(&profile.id);
            let name = if profile.name.is_empty() {
                profile.id.clone()
            } else {
                profile.name.clone()
            };
            let voices_dir = state.config.profile_paths(&profile.id).voices_dir();
            let cloned_voice_count = profile
                .characters
                .iter()
                .filter(|character| {
                    voices_dir
                        .join(&character.name)
                        .join(engine)
                        .join("sample.wav")
                        .is_file()
                })
                .count();
            ProfileCardView {
                initials: profile_initials(&name),
                active: profile.id == active_id,
                character_count: profile.characters.len(),
                lorebook_count: count_json_files(&dir.join("worlds")),
                quest_count: count_book_entries(&dir.join("headless").join("quest-books")),
                action_count: count_book_entries(&dir.join("headless").join("action-books")),
                cloned_voice_count,
                description: profile.description.clone(),
                name,
                id: profile.id,
            }
        })
        .collect();

    ProfilesPanelView {
        active_id,
        profiles,
        profiles_dir: profiles_dir.display().to_string(),
    }
}

/// Counts `*.json` files directly under `dir` (non-recursive). Used for the
/// per-profile lorebook count (`worlds/<id>.json`).
fn count_json_files(dir: &std::path::Path) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        })
        .count()
}

/// Sums the number of `entries` across every `*.json` book under `dir`
/// (quest-books / action-books). The books store `entries` as a JSON object
/// (map keyed by id) in this format, but tolerate an array too. Best-effort:
/// unreadable / entries-less files count 0.
fn count_book_entries(dir: &std::path::Path) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        })
        .map(|entry| {
            fs::read_to_string(entry.path())
                .ok()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
                .and_then(|value| value.get("entries").map(json_collection_len))
                .unwrap_or(0)
        })
        .sum()
}

/// The element count of a JSON value that holds a collection: an array's length
/// or an object's key count, else 0.
fn json_collection_len(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Array(items) => items.len(),
        serde_json::Value::Object(map) => map.len(),
        _ => 0,
    }
}

/// Short hardware hint for one model's footprint on this host.
fn fit_hint(system: &SystemInfo, footprint_gb: f64) -> String {
    match system.gpu_fit(footprint_gb) {
        GpuFit::Comfortable => "Fits GPU comfortably".to_string(),
        GpuFit::Tight => "Fits GPU (tight)".to_string(),
        GpuFit::Exceeds => "Exceeds VRAM — CPU/RAM".to_string(),
        GpuFit::NoGpu => "CPU only".to_string(),
    }
}

/// One-line detected-host summary, e.g. `RTX 5090, 32 GB VRAM / 24 cores`.
fn host_summary(system: &SystemInfo) -> String {
    let gpu = match (&system.gpu_name, system.vram_total_gb) {
        (Some(name), Some(vram)) => format!("{name}, {vram:.0} GB VRAM"),
        (Some(name), None) => name.clone(),
        _ => "no GPU".to_string(),
    };
    let ram = system
        .ram_gb
        .map(|v| format!(", {v:.0} GB RAM"))
        .unwrap_or_default();
    format!("{gpu}{ram} / {} cores", system.cpu_cores)
}

/// Kicks off a retrieval-model download: writes a `.downloading` marker and
/// spawns the detached download script (which forces the weights to download via
/// the app's own embed crate, then writes `.done`/`.failed`). Returns `false` for
/// an unknown id; a no-op (already present) returns `true`. Shared by the Askama
/// settings endpoint + the React UI models endpoint so both kick off the exact
/// same download. Mirrors [`start_llm_download`].
pub(crate) fn start_retrieval_download(state: &AppState, id: &str) -> std::io::Result<bool> {
    let Some(model) = RETRIEVAL_MODELS.iter().find(|m| m.id == id) else {
        return Ok(false);
    };

    let cache_dir = embed_cache_dir();
    // Already present (ALL variants the runtime may resolve) or in flight:
    // nothing to do.
    if chasm_embed::model_downloaded(model.id) {
        return Ok(true);
    }
    let markers = retrieval_marker_dir(&cache_dir, id);
    if markers.join(".downloading").exists() {
        return Ok(true);
    }
    fs::create_dir_all(&markers)?;
    let _ = fs::remove_file(markers.join(".failed"));
    fs::write(markers.join(".downloading"), "")?;

    // Download IN-PROCESS via the embed crate (fastembed → hf-hub, pure Rust, no
    // Python) on a background thread. The old path shelled out to a PowerShell
    // script that ran the `chasm` CLI — but the installed app ships only
    // chasm-desktop.exe (no CLI, no cargo, no target/release), so it always failed
    // ("drive is null"). `download_model` resolves the SAME cache dir the retriever
    // loads from, so a finished download is detected as "downloaded".
    let _ = state; // no longer needs the workspace root / a script
    let id = id.to_string();
    std::thread::spawn(move || {
        let outcome = chasm_embed::download_model(&id);
        let _ = fs::remove_file(markers.join(".downloading"));
        if let Err(error) = outcome {
            let _ = fs::write(markers.join(".failed"), format!("{error}"));
        }
    });
    Ok(true)
}

/// Kicks off a retrieval-model download, then redirects back to the Askama
/// settings page. Thin wrapper over [`start_retrieval_download`].
async fn download_retrieval_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> WebResult<Redirect> {
    let _ = start_retrieval_download(&state, &id)?;
    Ok(Redirect::to("/settings/retrieval?downloading=1"))
}

async fn save_settings(
    State(state): State<Arc<AppState>>,
    Path(category): Path<String>,
    Form(form): Form<HashMap<String, String>>,
) -> WebResult<Redirect> {
    let category = normalize_category(&category);
    let mut settings = AppSettings::load(&state.config.settings_path);
    // Remember the live TTS engine so we can hot-swap :5002 if the picker changed.
    let prev_tts_engine = chasm_core::normalize_local_engine(&settings.tts.local_engine);
    // Remember the live LLM model so we can relaunch koboldcpp if the picker changed.
    let model_status = llm_model_statuses(&state.config.llm_models_dir);
    // RAW saved id before the form is applied — compare raw-vs-raw to detect a real
    // selection change. Comparing the fallback-resolved id (selected_llm_model_id)
    // hid swaps from an empty/defaulted setting, because it falls back to the first
    // *downloaded* model and collapses both before/after sides to the same id.
    let prev_raw_llm_model = settings.llm.model.trim().to_string();
    // Remember the active Whisper model so we can swap it (and force koboldcpp to
    // reload) if the STT picker changed.
    let prev_whisper_model = stt_effective_model(&settings.stt);
    match category.as_str() {
        "tts" => apply_tts_form(&mut settings.tts, &form),
        "llm" => apply_llm_form(&mut settings.llm, &form),
        "stt" => apply_stt_form(&mut settings.stt, &form),
        "retrieval" => apply_retrieval_form(&mut settings.retrieval, &form),
        "game" => apply_game_form(&mut settings.launcher, &form),
        "interface" => apply_interface_form(&mut settings.interface, &form),
        // The Profiles category has no persisted form fields: activation is done
        // via POST /profile/select. A save here is a harmless no-op.
        "profiles" => {}
        "tracing" => {
            if let Some(value) = form.get("trace_dir") {
                settings.tracing.trace_dir = value.trim().to_string();
            }
        }
        _ => {}
    }
    settings.save(&state.config.settings_path)?;

    // If the TTS engine selection changed, apply it to :5002 now (kill + respawn)
    // so the in-settings voice Test and the next in-game line use the newly-picked
    // engine without waiting for a Play. Off the async path — it sleeps briefly
    // between kill + spawn — and best-effort (a down stack just means next Play).
    if category == "tts" {
        let new_tts_engine = chasm_core::normalize_local_engine(&settings.tts.local_engine);
        if new_tts_engine != prev_tts_engine {
            let state = Arc::clone(&state);
            tokio::task::spawn_blocking(move || {
                crate::launcher::apply_selected_tts_engine(&state);
            });
        }
    }

    // If the LLM model selection changed, relaunch koboldcpp on the new --model
    // now (kill the old -> load the new) so the swap takes effect without a Play.
    // koboldcpp loads --model only at launch, so this is a full reload - the old
    // model is unloaded before the new one loads. Off the async path (it sleeps
    // between kill + spawn) and best-effort: a down stack just means the next Play
    // loads it. Only when the *selected* model id actually changed.
    if category == "llm" {
        let new_llm_model =
            chasm_core::selected_llm_model_id(&settings.llm.model, &model_status);
        let raw_changed = settings.llm.model.trim() != prev_raw_llm_model;
        if raw_changed && !new_llm_model.is_empty() {
            let state = Arc::clone(&state);
            tokio::task::spawn_blocking(move || {
                crate::launcher::apply_selected_llm_model(&state);
            });
        }
    }

    // If the Whisper model selection changed, rewrite koboldcpp's --whispermodel
    // (config + start_kobold.bat) and stop koboldcpp so the OLD model is unloaded;
    // the next Play relaunches it with the new model. koboldcpp can't hot-swap just
    // the whisper slot, so a restart (which also reloads the LLM) is the only way
    // to GUARANTEE the previous model leaves VRAM. Off the async path (it kills a
    // process + rewrites files); best-effort.
    if category == "stt" {
        let new_whisper_model = stt_effective_model(&settings.stt);
        if new_whisper_model != prev_whisper_model {
            let state = Arc::clone(&state);
            tokio::task::spawn_blocking(move || {
                crate::launcher::apply_selected_whisper_model(&state, &new_whisper_model);
            });
        }
    }

    Ok(Redirect::to(&format!("/settings/{category}?saved=1")))
}

pub(crate) fn apply_tts_form(tts: &mut TtsSettings, form: &HashMap<String, String>) {
    if let Some(mode) = form.get("mode") {
        if mode == "api" || mode == "local" {
            tts.mode = mode.clone();
        }
    }
    if let Some(value) = form.get("local_engine") {
        tts.local_engine = chasm_core::normalize_local_engine(value);
    }
    if let Some(value) = form.get("api_provider") {
        tts.api_provider = value.clone();
    }
    if let Some(value) = form
        .get("caption_max_chars")
        .and_then(|v| v.parse::<u32>().ok())
    {
        tts.caption_max_chars = chasm_core::normalize_caption_max_chars(value);
    }
    // Voice-volume sliders post a percent (100 = unity); store the multiplier.
    if let Some(value) = form.get("npc_volume").and_then(|v| v.parse::<f32>().ok()) {
        tts.npc_volume = chasm_core::normalize_voice_volume(value / 100.0);
    }
    if let Some(value) = form.get("admin_volume").and_then(|v| v.parse::<f32>().ok()) {
        tts.admin_volume = chasm_core::normalize_voice_volume(value / 100.0);
    }
    if let Some(value) = form.get("default_voice") {
        tts.default_voice = value.clone();
    }

    tts.audio_tags.enabled = form.contains_key("audio_tags_enabled");
    if let Some(value) = form.get("audio_tags_profile") {
        tts.audio_tags.profile = value.clone();
    }
    if let Some(value) = form
        .get("audio_tags_max_tags")
        .and_then(|v| v.parse::<u8>().ok())
    {
        tts.audio_tags.max_tags_per_reply = normalize_max_tags(value);
    }
    tts.audio_tags.strip_game_subtitles = form.contains_key("audio_tags_strip_subtitles");
    if let Some(value) = form.get("audio_tags_custom_prompt") {
        tts.audio_tags.custom_prompt = value.clone();
    }

    apply_tts_tuning_form(&mut tts.tuning, form);
}

/// Parses the TTS-tuning controls out of the posted form. Each field is optional
/// (only updated when present + parseable), then the whole group is normalized to
/// its documented ranges — mirroring how the other settings clamp on save.
fn apply_tts_tuning_form(tuning: &mut TtsTuningSettings, form: &HashMap<String, String>) {
    if let Some(v) = form.get("tuning_lead_in_ms").and_then(|v| v.parse().ok()) {
        tuning.lead_in_ms = v;
    }
    if let Some(v) = form.get("tuning_trailing_ms").and_then(|v| v.parse().ok()) {
        tuning.trailing_ms = v;
    }
    if let Some(v) = form
        .get("tuning_sentence_gap_ms")
        .and_then(|v| v.parse().ok())
    {
        tuning.sentence_gap_ms = v;
    }
    if let Some(v) = form
        .get("tuning_gain_db")
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
    {
        tuning.gain_db = v;
    }
    if let Some(v) = form
        .get("tuning_temperature")
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
    {
        tuning.temperature = v;
    }
    if let Some(v) = form
        .get("tuning_lsd_decode_steps")
        .and_then(|v| v.parse().ok())
    {
        tuning.lsd_decode_steps = v;
    }
    if let Some(v) = form
        .get("tuning_eos_threshold")
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
    {
        tuning.eos_threshold = v;
    }
    if let Some(v) = form
        .get("tuning_noise_clamp")
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
    {
        tuning.noise_clamp = v;
    }
    if let Some(v) = form.get("tuning_max_tokens").and_then(|v| v.parse().ok()) {
        tuning.max_tokens = v;
    }
    if let Some(v) = form
        .get("tuning_frames_after_eos")
        .and_then(|v| v.parse().ok())
    {
        tuning.frames_after_eos = v;
    }
    *tuning = tuning.normalized();
}

pub(crate) fn apply_llm_form(llm: &mut LlmSettings, form: &HashMap<String, String>) {
    if let Some(value) = form.get("provider") {
        llm.provider = value.clone();
    }
    // The model picker posts a model id (radio value). Only accept a known id so
    // a stale/bogus value can't be stored; an absent field (no selectable radio)
    // leaves the current selection untouched.
    if let Some(value) = form.get("model") {
        let candidate = value.trim();
        if chasm_core::LLM_MODELS
            .iter()
            .any(|model| model.id == candidate)
        {
            llm.model = candidate.to_string();
        }
    }

    apply_llm_sampling_form(&mut llm.sampling, form);

    // Live chat orchestrator. The whole section posts together, so the checkbox
    // is authoritative: present = enabled, absent = disabled.
    llm.orchestrator_enabled = form.contains_key("orchestrator_enabled");
    if let Some(value) = form
        .get("orchestrator_max_speakers")
        .and_then(|v| v.parse::<u32>().ok())
    {
        llm.orchestrator_max_speakers =
            value.clamp(ORCHESTRATOR_MAX_SPEAKERS_MIN, ORCHESTRATOR_MAX_SPEAKERS_MAX);
    }
    if let Some(value) = form
        .get("orchestrator_temperature")
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
    {
        llm.orchestrator_temperature =
            value.clamp(ORCHESTRATOR_TEMPERATURE_MIN, ORCHESTRATOR_TEMPERATURE_MAX);
    }
    if let Some(value) = form.get("orchestrator_system_prompt") {
        // Blank/whitespace-only → reset to the default (don't persist empty).
        let trimmed = value.trim();
        llm.orchestrator_system_prompt = if trimmed.is_empty() {
            ORCHESTRATOR_DEFAULT_SYSTEM_PROMPT.to_string()
        } else {
            trimmed.to_string()
        };
    }
}

/// Parses the LLM generation-sampling controls out of the posted form. Each
/// field is optional (only updated when present + parseable), then the whole
/// group is normalized to its documented ranges (mirroring the other settings,
/// which clamp on save). The normalized values are what reach the request.
fn apply_llm_sampling_form(sampling: &mut LlmSamplingSettings, form: &HashMap<String, String>) {
    if let Some(v) = form
        .get("sampling_temperature")
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
    {
        sampling.temperature = v;
    }
    if let Some(v) = form
        .get("sampling_top_p")
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
    {
        sampling.top_p = v;
    }
    if let Some(v) = form.get("sampling_top_k").and_then(|v| v.parse().ok()) {
        sampling.top_k = v;
    }
    if let Some(v) = form
        .get("sampling_min_p")
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
    {
        sampling.min_p = v;
    }
    if let Some(v) = form
        .get("sampling_repeat_penalty")
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
    {
        sampling.repeat_penalty = v;
    }
    if let Some(v) = form.get("sampling_max_tokens").and_then(|v| v.parse().ok()) {
        sampling.max_tokens = v;
    }
    if let Some(v) = form.get("sampling_n_ctx").and_then(|v| v.parse().ok()) {
        sampling.n_ctx = v;
    }
    if let Some(v) = form.get("sampling_seed").and_then(|v| v.parse().ok()) {
        sampling.seed = v;
    }
    *sampling = sampling.normalized();
}

/// Applies the Interface (appearance) settings form. Selects/toggles post
/// together, so each toggle is authoritative (present = on). Every value is
/// normalized the same way `/theme.css` will read it, so a bad/blank value can't
/// poison the stylesheet.
pub(crate) fn apply_interface_form(
    interface: &mut InterfaceSettings,
    form: &HashMap<String, String>,
) {
    if let Some(value) = form.get("theme") {
        interface.theme = chasm_core::interface_theme(value).id.to_string();
    }
    if let Some(value) = form.get("accent") {
        interface.accent = chasm_core::normalize_accent(value);
    }
    if let Some(value) = form.get("density") {
        interface.density = chasm_core::normalize_density(value);
    }
    if let Some(value) = form.get("font_scale").and_then(|v| v.parse::<u32>().ok()) {
        interface.font_scale = chasm_core::normalize_font_scale(value);
    }
    interface.reduce_motion = form.contains_key("reduce_motion");
    interface.show_timestamps = form.contains_key("show_timestamps");
    interface.show_prompt_panel = form.contains_key("show_prompt_panel");
}

pub(crate) fn apply_stt_form(stt: &mut SttSettings, form: &HashMap<String, String>) {
    if let Some(value) = form.get("provider") {
        stt.provider = normalize_stt_provider(value);
    }
    if let Some(value) = form.get("model") {
        stt.model = value.trim().to_string();
    }
    if let Some(value) = form.get("language") {
        stt.language = value.trim().to_string();
    }
    if let Some(value) = form.get("prompt") {
        stt.prompt = value.trim().to_string();
    }
    if let Some(value) = form.get("timeout_ms").and_then(|v| v.parse::<u64>().ok()) {
        stt.timeout_ms = chasm_core::normalize_stt_timeout_ms(value);
    }
}

pub(crate) fn apply_retrieval_form(retrieval: &mut RetrievalSettings, form: &HashMap<String, String>) {
    // Checkboxes: present in the form body only when checked.
    retrieval.enabled = form.contains_key("enabled");
    retrieval.chat_memory_enabled = form.contains_key("chat_memory_enabled");
    retrieval.lore_semantic_enabled = form.contains_key("lore_semantic_enabled");
    retrieval.action_semantic_enabled = form.contains_key("action_semantic_enabled");
    retrieval.quest_semantic_enabled = form.contains_key("quest_semantic_enabled");
    retrieval.reranker_enabled = form.contains_key("reranker_enabled");

    if let Some(value) = form.get("embedder_tier") {
        retrieval.embedder_tier = normalize_embedder_tier(value);
    }
    if let Some(value) = form.get("reranker_tier") {
        retrieval.reranker_tier = normalize_reranker_tier(value);
    }
    if let Some(value) = form.get("execution") {
        retrieval.execution = normalize_execution(value);
    }
    if let Some(value) = form.get("top_k").and_then(|v| v.parse::<u32>().ok()) {
        retrieval.top_k = value.clamp(RETRIEVAL_TOP_K_MIN, RETRIEVAL_TOP_K_MAX);
    }
    if let Some(value) = form.get("candidates").and_then(|v| v.parse::<u32>().ok()) {
        retrieval.candidates = value.clamp(RETRIEVAL_CANDIDATES_MIN, RETRIEVAL_CANDIDATES_MAX);
    }
    if let Some(value) = form.get("min_score").and_then(|v| v.parse::<f32>().ok()) {
        retrieval.min_score = value.clamp(0.0, 1.0);
    }
    if let Some(value) = form
        .get("action_min_score")
        .and_then(|v| v.parse::<f32>().ok())
    {
        retrieval.action_min_score = value.clamp(0.0, 1.0);
    }
    if let Some(value) = form
        .get("chat_memory_limit")
        .and_then(|v| v.parse::<u32>().ok())
    {
        retrieval.chat_memory_limit =
            value.clamp(RETRIEVAL_SOURCE_LIMIT_MIN, RETRIEVAL_SOURCE_LIMIT_MAX);
    }
    if let Some(value) = form.get("lore_limit").and_then(|v| v.parse::<u32>().ok()) {
        retrieval.lore_limit = value.clamp(RETRIEVAL_SOURCE_LIMIT_MIN, RETRIEVAL_SOURCE_LIMIT_MAX);
    }
    if let Some(value) = form.get("quest_limit").and_then(|v| v.parse::<u32>().ok()) {
        retrieval.quest_limit = value.clamp(RETRIEVAL_SOURCE_LIMIT_MIN, RETRIEVAL_SOURCE_LIMIT_MAX);
    }
}

/// Applies the "Game" launcher settings form. Each field is an override; a blank
/// value means "auto-detect" (stored as empty, re-resolved on read). Values are
/// trimmed so a whitespace-only entry resets to auto-detect.
fn apply_game_form(launcher: &mut LauncherSettings, form: &HashMap<String, String>) {
    if let Some(value) = form.get("mo2_exe") {
        launcher.mo2_exe = value.trim().to_string();
    }
    if let Some(value) = form.get("instance") {
        launcher.instance = value.trim().to_string();
    }
    if let Some(value) = form.get("profile") {
        launcher.profile = value.trim().to_string();
    }
    if let Some(value) = form.get("executable") {
        launcher.executable = value.trim().to_string();
    }
    if let Some(value) = form.get("game_dir") {
        launcher.game_dir = value.trim().to_string();
    }
}

/// Per-service status for the sidebar "model lights" (`GET /api/stack/status`).
/// Each field is a `StatusTone`-compatible string the UI maps to a dot colour:
/// `"ok"` = up / loaded, `"idle"` = down / not loaded.
#[derive(Debug, Serialize)]
struct StackStatusResponse {
    llm: &'static str,
    stt: &'static str,
    tts: &'static str,
    embedder: &'static str,
    reranker: &'static str,
}

/// `GET /api/stack/status` — cheap, non-blocking snapshot of each model/service
/// for the sidebar lights. LLM + STT both ride koboldcpp (one process, one port):
/// LLM is up when that port is reachable; STT additionally needs a Whisper model
/// downloaded. TTS is up when its engine server answers on :5002. Embedder /
/// reranker are in-process — reported from the already-loaded retriever, which
/// never triggers a (multi-second) load here.
async fn stack_status(State(state): State<Arc<AppState>>) -> Json<StackStatusResponse> {
    let settings = AppSettings::load(&state.config.settings_path);

    let kobold_up = launcher::koboldcpp_running(&state);
    // Runtime missing → its auto-download may be in flight; surface that as
    // "busy" so the LLM/STT lights read "coming up", not "broken". Tracks the
    // SELECTED runtime's markers (koboldcpp or llama.cpp).
    let runtime_llamacpp = chasm_core::normalize_llm_runtime(&settings.runtime.llm_runtime)
        == chasm_core::LLM_RUNTIME_LLAMACPP;
    let kobold_downloading = if runtime_llamacpp {
        launcher::llamacpp_status(&state.config) == launcher::KoboldcppStatus::Downloading
    } else {
        launcher::koboldcpp_status(&settings, &state.config)
            == launcher::KoboldcppStatus::Downloading
    };
    let tts_up = launcher::tts_running_engine(&state).is_some();

    // STT: the dedicated Parakeet server when selected + installed, else the
    // koboldcpp Whisper path (rides koboldcpp; needs a Whisper model present).
    let stt_parakeet = launcher::stt_uses_parakeet(&settings, &state.config);
    let parakeet_up = stt_parakeet && launcher::parakeet_running(&state);
    let whisper_dir = launcher::whisper_models_dir(&settings);
    let whisper_present = whisper_model_statuses(&whisper_dir)
        .values()
        .any(|status| status == "downloaded");

    let retriever = state.retriever_loaded();

    // "ok" = up · "busy" = coming up (runtime still downloading) · "idle" = down.
    let up_or_busy = |up: bool, busy: bool| {
        if up {
            "ok"
        } else if busy {
            "busy"
        } else {
            "idle"
        }
    };
    let flag = |up: bool| if up { "ok" } else { "idle" };

    let stt_up = if stt_parakeet {
        parakeet_up
    } else {
        kobold_up && whisper_present
    };
    Json(StackStatusResponse {
        llm: up_or_busy(kobold_up, kobold_downloading),
        // Whisper rides koboldcpp specifically: with the llama.cpp runtime the
        // whisper path can never come up, so don't show it as "coming up".
        stt: up_or_busy(stt_up, !stt_parakeet && !runtime_llamacpp && kobold_downloading),
        tts: flag(tts_up),
        embedder: flag(retriever.is_some()),
        reranker: flag(retriever.map(|r| r.has_reranker()).unwrap_or(false)),
    })
}

/// `POST /api/stack/start` — manually bring the whole model stack up without
/// waiting for the game to connect: spawn koboldcpp (LLM + Whisper STT) and the
/// selected TTS engine, and warm the in-process retriever (embedder + reranker).
/// Idempotent — already-running services are left alone. Returns immediately; the
/// sidebar lights flip to green as each service becomes reachable.
async fn stack_start(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    // start_ai_stack blocks (reachability probes + process spawn) → blocking pool.
    let stack_state = state.clone();
    tokio::task::spawn_blocking(move || {
        // Ensure the selected LLM RUNTIME exists — without it, an LLM model can't
        // be served and Start would silently do nothing for LLM/STT. This also
        // flips a stale `.downloading` marker (e.g. a download that died on an
        // older build) so a fresh fetch can start. llama.cpp is opt-in: its
        // downloader only runs when it IS the selected runtime.
        let runtime_status = |stack_state: &Arc<AppState>| {
            let settings = AppSettings::load(&stack_state.config.settings_path);
            if chasm_core::normalize_llm_runtime(&settings.runtime.llm_runtime)
                == chasm_core::LLM_RUNTIME_LLAMACPP
            {
                launcher::llamacpp_status(&stack_state.config)
            } else {
                launcher::koboldcpp_status(&settings, &stack_state.config)
            }
        };
        {
            let settings = AppSettings::load(&stack_state.config.settings_path);
            if chasm_core::normalize_llm_runtime(&settings.runtime.llm_runtime)
                == chasm_core::LLM_RUNTIME_LLAMACPP
            {
                ensure_llamacpp(&stack_state);
            } else {
                ensure_koboldcpp(&stack_state);
            }
        }
        // Spawn whatever is present now (the TTS engine, and the LLM runtime if
        // it's already installed).
        launcher::start_ai_stack(&stack_state);
        // If the runtime was still downloading, wait (bounded) for its exe to
        // land, then spawn it — so a first click brings LLM/STT up without a
        // second one.
        if runtime_status(&stack_state) == launcher::KoboldcppStatus::Downloading {
            for _ in 0..180 {
                std::thread::sleep(std::time::Duration::from_secs(5)); // ≤15 min
                match runtime_status(&stack_state) {
                    launcher::KoboldcppStatus::Installed => {
                        launcher::start_ai_stack(&stack_state);
                        break;
                    }
                    launcher::KoboldcppStatus::Missing => break, // download failed → give up
                    launcher::KoboldcppStatus::Downloading => continue,
                }
            }
        }
    });
    // Warm the whole stack (retriever, LLM prefix, Whisper, TTS first-inference)
    // off the request path. Permit-guarded, so overlapping Start clicks / a
    // concurrent game connect never run two warm-ups at once, and the lifecycle
    // can abort it if the game disconnects mid-warm-up.
    warmup::spawn_stack_warmup(&state);
    Json(serde_json::json!({ "started": true }))
}

/// Response for `GET /api/app/version` — the Settings → Updates check.
#[derive(Debug, Serialize)]
struct AppVersionResponse {
    /// The running version (`CARGO_PKG_VERSION`).
    current: String,
    /// The latest release tag (leading `v` stripped), or `null` on any error.
    latest: Option<String>,
    /// Whether `latest` is a higher semver than `current`.
    update_available: bool,
    /// The installer `.exe` asset download URL, or `null`.
    download_url: Option<String>,
    /// The release page URL, or `null`.
    release_url: Option<String>,
    /// `"nightly"` when the commit-based nightly comparison drove the result,
    /// `"release"` when the semver fallback did (local/dev builds).
    channel: String,
    /// Short commit this build was made from (CI stamps it), or `null` for
    /// local builds.
    current_commit: Option<String>,
    /// Short commit the rolling `nightly` tag currently points at, or `null`.
    latest_commit: Option<String>,
}

/// Commit the running binary was built from — stamped by the nightly CI
/// workflow (`CHASM_BUILD_COMMIT: ${{ github.sha }}`); `None` for local builds.
const BUILD_COMMIT: Option<&str> = option_env!("CHASM_BUILD_COMMIT");

/// Short (7-char) form of a commit sha.
fn short_sha(sha: &str) -> String {
    sha.chars().take(7).collect()
}

/// The commit sha the `nightly` tag points at, via the git ref API (the
/// workflow force-pushes the lightweight tag to the exact commit it built).
async fn fetch_nightly_commit() -> Option<String> {
    let resp = crate::llm::http_client()
        .get("https://api.github.com/repos/chasmlol/chasm/git/ref/tags/nightly")
        .header("User-Agent", "chasm")
        .header("Accept", "application/vnd.github+json")
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body: serde_json::Value = resp.json().await.ok()?;
    body.get("object")?
        .get("sha")?
        .as_str()
        .map(str::to_string)
}

/// A release fetched by exact tag (used for `nightly` on both repos).
async fn fetch_release_by_tag(repo: &str, tag: &str) -> Option<GithubRelease> {
    let url = format!("https://api.github.com/repos/{repo}/releases/tags/{tag}");
    let resp = crate::llm::http_client()
        .get(&url)
        .header("User-Agent", "chasm")
        .header("Accept", "application/vnd.github+json")
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json::<GithubRelease>().await.ok()
}

/// The subset of the GitHub "latest release" JSON we care about.
#[derive(Debug, serde::Deserialize)]
struct GithubRelease {
    tag_name: Option<String>,
    html_url: Option<String>,
    #[serde(default)]
    assets: Vec<GithubAsset>,
}

#[derive(Debug, serde::Deserialize)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
}

/// Reports the running version + the latest GitHub release. NEVER errors: on any
/// network/parse failure it returns the current version with `latest = null` and
/// `update_available = false`, so the UI degrades to "you're on the latest".
async fn app_version() -> Json<AppVersionResponse> {
    let current = env!("CARGO_PKG_VERSION").to_string();

    // NIGHTLY channel (CI builds): this build carries the commit it was built
    // from; a new nightly exists whenever the rolling tag points elsewhere.
    // Version numbers can't drive this — every nightly reports the same
    // CARGO_PKG_VERSION.
    if let Some(build_commit) = BUILD_COMMIT.filter(|c| !c.is_empty()) {
        let nightly_sha = fetch_nightly_commit().await;
        let nightly_release = fetch_release_by_tag("chasmlol/chasm", "nightly").await;
        let (download_url, release_url) = match nightly_release {
            Some(release) => (
                release
                    .assets
                    .into_iter()
                    .find(|a| a.name.to_ascii_lowercase().ends_with(".exe"))
                    .map(|a| a.browser_download_url),
                release.html_url,
            ),
            None => (None, None),
        };
        let current_short = short_sha(build_commit);
        let latest_short = nightly_sha.as_deref().map(short_sha);
        let update_available = matches!(&latest_short, Some(latest) if *latest != current_short);
        return Json(AppVersionResponse {
            current,
            latest: latest_short.as_ref().map(|s| format!("nightly ({s})")),
            update_available,
            download_url,
            release_url,
            channel: "nightly".to_string(),
            current_commit: Some(current_short),
            latest_commit: latest_short,
        });
    }

    // RELEASE fallback (local/dev builds without a stamped commit): highest
    // semver-tagged release vs the running version.
    let fetched = fetch_latest_release().await;
    let (latest, download_url, release_url) = match fetched {
        Some(release) => {
            let latest = release
                .tag_name
                .map(|t| t.trim().trim_start_matches('v').to_string())
                .filter(|t| !t.is_empty());
            let download_url = release
                .assets
                .into_iter()
                .find(|a| a.name.to_ascii_lowercase().ends_with(".exe"))
                .map(|a| a.browser_download_url);
            (latest, download_url, release.html_url)
        }
        None => (None, None, None),
    };

    let update_available = match (&latest, semver::Version::parse(&current)) {
        (Some(latest_str), Ok(current_ver)) => semver::Version::parse(latest_str)
            .map(|latest_ver| latest_ver > current_ver)
            .unwrap_or(false),
        _ => false,
    };

    Json(AppVersionResponse {
        current,
        latest,
        update_available,
        download_url,
        release_url,
        channel: "release".to_string(),
        current_commit: None,
        latest_commit: None,
    })
}

/// Fetches the latest release from the public chasm repo. Returns `None` on any
/// network/parse error (the endpoint never fails). GitHub requires a User-Agent.
async fn fetch_latest_release() -> Option<GithubRelease> {
    // NOT `/releases/latest`: that endpoint only ever returns a NON-pre-release,
    // and this project marks every release as a pre-release while pre-1.0 (plus
    // a rolling `nightly` tag). List releases instead and take the highest
    // semver-tagged one, skipping non-version tags like `nightly` — the in-app
    // updater tracks versioned milestones; the nightly is a manual download.
    let client = reqwest::Client::new();
    let resp = client
        .get("https://api.github.com/repos/chasmlol/chasm/releases?per_page=20")
        .header("User-Agent", "chasm")
        .header("Accept", "application/vnd.github+json")
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let releases = resp.json::<Vec<GithubRelease>>().await.ok()?;
    releases
        .into_iter()
        .filter_map(|release| {
            let tag = release.tag_name.as_deref()?.trim();
            let version = parse_semver(tag.trim_start_matches('v'))?;
            Some((version, release))
        })
        .max_by_key(|(version, _)| *version)
        .map(|(_, release)| release)
}

/// `"0.3.0"` -> `(0, 3, 0)`; anything non-semver (e.g. `nightly`) -> None.
fn parse_semver(s: &str) -> Option<(u64, u64, u64)> {
    let mut parts = s.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

/// Whether a process with the given image name is running (Windows tasklist).
#[cfg(windows)]
fn process_running(image: &str) -> bool {
    std::process::Command::new("tasklist")
        .args(["/FI", &format!("IMAGENAME eq {image}"), "/NH", "/FO", "CSV"])
        .output()
        .map(|out| {
            String::from_utf8_lossy(&out.stdout)
                .to_ascii_lowercase()
                .contains(&image.to_ascii_lowercase())
        })
        .unwrap_or(false)
}

/// `POST /api/app/update/install` — one-click self-update. Downloads the latest
/// release's installer `.exe` to the temp dir, then spawns a DETACHED helper that
/// (after a short grace so this HTTP response flushes) closes the running app,
/// runs the installer silently (`/S`), and relaunches chasm. Detached + a written
/// `.bat` so it outlives this process when the app is stopped, and so quoting of
/// paths-with-spaces is robust. Returns `{ started, error }`.
#[cfg(windows)]
async fn app_update_install() -> Json<serde_json::Value> {
    use std::os::windows::process::CommandExt;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    // The update also swaps the bridge mod DLL, which the game holds locked.
    // Refuse up front instead of silently failing (or killing a live game).
    if process_running("FalloutNV.exe") {
        return Json(serde_json::json!({
            "started": false,
            "error": "Fallout: New Vegas is running — close the game first (the update replaces the bridge mod)."
        }));
    }

    // Prefer the rolling nightly build; fall back to the newest versioned
    // release for installs that predate the nightly pipeline.
    let release = match fetch_release_by_tag("chasmlol/chasm", "nightly").await {
        Some(r) => Some(r),
        None => fetch_latest_release().await,
    };
    let Some(release) = release else {
        return Json(serde_json::json!({"started": false, "error": "could not reach GitHub"}));
    };
    let Some(url) = release
        .assets
        .into_iter()
        .find(|a| a.name.to_ascii_lowercase().ends_with(".exe"))
        .map(|a| a.browser_download_url)
    else {
        return Json(
            serde_json::json!({"started": false, "error": "no installer in the latest release"}),
        );
    };

    // Download the installer to the temp dir.
    let installer = std::env::temp_dir().join("chasm-update-setup.exe");
    let resp = match reqwest::Client::new()
        .get(&url)
        .header("User-Agent", "chasm")
        .timeout(std::time::Duration::from_secs(600))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            return Json(
                serde_json::json!({"started": false, "error": format!("download HTTP {}", r.status())}),
            )
        }
        Err(e) => {
            return Json(serde_json::json!({"started": false, "error": format!("download failed: {e}")}))
        }
    };
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return Json(serde_json::json!({"started": false, "error": format!("download failed: {e}")}))
        }
    };
    if let Err(e) = std::fs::write(&installer, &bytes) {
        return Json(
            serde_json::json!({"started": false, "error": format!("could not save installer: {e}")}),
        );
    }

    // Bridge mod: download the nightly NVBridge zip too, so one click updates
    // the app AND the game mod. Only when an MO2 NVBridge mod folder exists —
    // users without the game setup still get the app update.
    let mo2_mod_dir = std::env::var_os("LOCALAPPDATA").map(|base| {
        std::path::Path::new(&base)
            .join("ModOrganizer")
            .join("New Vegas")
            .join("mods")
            .join("NVBridge")
    });
    let mut bridge_step = String::new();
    let mut bridge_planned = false;
    if let Some(mo2_dir) = mo2_mod_dir.filter(|d| d.is_dir()) {
        let bridge_zip_url = fetch_release_by_tag("chasmlol/chasm-bridge-fnv", "nightly")
            .await
            .and_then(|r| {
                r.assets
                    .into_iter()
                    .find(|a| a.name.to_ascii_lowercase().ends_with(".zip"))
                    .map(|a| a.browser_download_url)
            });
        if let Some(zip_url) = bridge_zip_url {
            let zip_path = std::env::temp_dir().join("chasm-update-nvbridge.zip");
            let downloaded = match crate::llm::http_client()
                .get(&zip_url)
                .header("User-Agent", "chasm")
                .timeout(std::time::Duration::from_secs(600))
                .send()
                .await
            {
                Ok(r) if r.status().is_success() => match r.bytes().await {
                    Ok(b) => std::fs::write(&zip_path, &b).is_ok(),
                    Err(_) => false,
                },
                _ => false,
            };
            if downloaded {
                // Expand to temp then copy over the MO2 mod: Expand-Archive
                // (always present) + robocopy /E (overwrite, keep extra user
                // files like generated facegen). Exit code quirk: robocopy
                // returns 1 on success, so mask it with `& exit /b 0` style.
                let staging = std::env::temp_dir().join("chasm-update-nvbridge");
                bridge_step = format!(
                    "powershell -NoProfile -Command \"Remove-Item -Recurse -Force '{staging}' -ErrorAction SilentlyContinue; Expand-Archive -Force '{zip}' '{staging}'\" >nul 2>&1\r\n\
                     robocopy \"{staging}\\NVBridge\" \"{dst}\" /E /NFL /NDL /NJH /NJS >nul 2>&1\r\n",
                    staging = staging.display(),
                    zip = zip_path.display(),
                    dst = mo2_dir.display(),
                );
                bridge_planned = true;
            }
        }
    }

    // Write a detached updater .bat: wait for this response to flush, close the
    // running app so its exe isn't locked, update the bridge mod in MO2, then
    // silently install and relaunch the (now-updated) exe in place.
    let current_exe = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let bat = std::env::temp_dir().join("chasm-update.bat");
    let script = format!(
        "@echo off\r\ntimeout /t 2 /nobreak >nul\r\ntaskkill /IM chasm-desktop.exe /F >nul 2>&1\r\n{}\"{}\" /S\r\nstart \"\" \"{}\"\r\n",
        bridge_step,
        installer.display(),
        current_exe
    );
    if let Err(e) = std::fs::write(&bat, script) {
        return Json(
            serde_json::json!({"started": false, "error": format!("could not stage updater: {e}")}),
        );
    }

    match std::process::Command::new("cmd")
        .args(["/C", &bat.display().to_string()])
        .creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW)
        .spawn()
    {
        Ok(_) => Json(serde_json::json!({"started": true, "bridge_update": bridge_planned})),
        Err(e) => Json(
            serde_json::json!({"started": false, "error": format!("could not start updater: {e}")}),
        ),
    }
}

/// Non-Windows stub: self-update is Windows-only (the only packaged target).
#[cfg(not(windows))]
async fn app_update_install() -> Json<serde_json::Value> {
    Json(serde_json::json!({"started": false, "error": "self-update is only supported on Windows"}))
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    ok: bool,
    app: &'static str,
    data_root: String,
    live_chats: usize,
}

async fn health(State(state): State<Arc<AppState>>) -> WebResult<Json<HealthResponse>> {
    let live_chats = state.repository.list_live_chats()?.len();
    Ok(Json(HealthResponse {
        ok: true,
        app: "chasm-rs",
        data_root: state.config.data_root.display().to_string(),
        live_chats,
    }))
}

fn choose_participant_id(
    repository: &LiveChatRepository,
    live_chat: &chasm_st_compat::LiveChat,
) -> WebResult<String> {
    let view = repository.live_chat_view(live_chat, None)?;
    Ok(view
        .participants
        .iter()
        .find(|participant| participant.present && participant.kind == "npc")
        .or_else(|| {
            view.participants
                .iter()
                .find(|participant| participant.kind == "npc")
        })
        .or_else(|| view.participants.first())
        .map(|participant| participant.id.clone())
        .unwrap_or_else(|| "player".to_string()))
}

pub fn participant_url(live_chat_id: &str, participant_id: &str) -> String {
    format!(
        "/live/{}/{}",
        urlencoding::encode(live_chat_id),
        urlencoding::encode(participant_id)
    )
}

pub fn messages_partial_url(live_chat_id: &str, participant_id: &str) -> String {
    format!(
        "/partials/live/{}/messages/{}",
        urlencoding::encode(live_chat_id),
        urlencoding::encode(participant_id)
    )
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

mod filters {
    pub fn participant_url(live_chat_id: &str, participant_id: &str) -> askama::Result<String> {
        Ok(crate::participant_url(live_chat_id, participant_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_gain_scales_and_clamps_and_is_noop_at_unity() {
        // Unity leaves bytes untouched (and copies nothing).
        let mut pcm = 1000i16.to_le_bytes().to_vec();
        apply_pcm_gain(&mut pcm, 1.0);
        assert_eq!(i16::from_le_bytes([pcm[0], pcm[1]]), 1000);
        // 2x doubles a sample.
        apply_pcm_gain(&mut pcm, 2.0);
        assert_eq!(i16::from_le_bytes([pcm[0], pcm[1]]), 2000);
        // Boost past the int16 ceiling hard-clamps instead of wrapping.
        let mut hot = 20000i16.to_le_bytes().to_vec();
        apply_pcm_gain(&mut hot, 2.0);
        assert_eq!(i16::from_le_bytes([hot[0], hot[1]]), i16::MAX);
    }

    #[test]
    fn voice_volume_picks_admin_for_non_positional() {
        let mut settings = AppSettings::default();
        settings.tts.npc_volume = 1.2;
        settings.tts.admin_volume = 0.5;
        let npc = serde_json::json!({ "text": "hi", "characterName": "Easy Pete" });
        let admin =
            serde_json::json!({ "text": "hi", "characterName": "Todd", "nonPositional": true });
        assert_eq!(resolve_voice_volume(&settings, &npc), 1.2);
        assert_eq!(resolve_voice_volume(&settings, &admin), 0.5);
    }

    #[test]
    fn tuning_uses_saved_when_request_has_no_override() {
        // The game sends a bare {text, characterName}: saved settings are used.
        let saved = TtsTuningSettings {
            lead_in_ms: 300,
            temperature: 1.1,
            ..TtsTuningSettings::default()
        };
        let req = serde_json::json!({ "text": "hi", "characterName": "Easy Pete" });
        let resolved = resolve_tuning(&saved, &req);
        assert_eq!(resolved.lead_in_ms, 300);
        assert_eq!(resolved.temperature, 1.1);
        assert_eq!(resolved.trailing_ms, 60); // untouched default
    }

    /// Builds a minimal canonical PCM WAV for the padding tests.
    fn build_pcm_wav(channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let block_align = channels * (bits / 8);
        let byte_rate = rate * u32::from(block_align);
        let mut wav = Vec::with_capacity(44 + data.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((36 + data.len()) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&bits.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(data);
        wav
    }

    #[test]
    fn pad_wav_lengthens_short_pcm_and_preserves_samples() {
        // 0.5s of 16kHz mono 16-bit (the short-clip case whisper drops).
        let rate = 16_000u32;
        let samples = (rate / 2) as usize;
        let data: Vec<u8> = (0..samples)
            .flat_map(|i| (i as i16).wrapping_mul(7).to_le_bytes())
            .collect();
        let padded = pad_wav_to_min_duration(&build_pcm_wav(1, rate, 16, &data), 2000);

        // Valid canonical header.
        assert_eq!(&padded[0..4], b"RIFF");
        assert_eq!(&padded[8..12], b"WAVE");
        assert_eq!(&padded[36..40], b"data");
        // At least 2000ms of data (16000 bytes/s * 2s = 64000), header consistent.
        let data_len =
            u32::from_le_bytes([padded[40], padded[41], padded[42], padded[43]]) as usize;
        assert!(data_len >= 64_000, "data_len={data_len}");
        assert_eq!(padded.len(), 44 + data_len);
        let riff = u32::from_le_bytes([padded[4], padded[5], padded[6], padded[7]]) as usize;
        assert_eq!(riff, 36 + data_len);
        // Original samples preserved at the front; the rest is silence.
        assert_eq!(&padded[44..44 + data.len()], data.as_slice());
        assert!(padded[44 + data.len()..].iter().all(|&b| b == 0));
    }

    #[test]
    fn pad_wav_leaves_long_or_non_wav_unchanged() {
        // 3s clip is already past the 2000ms floor -> returned unchanged.
        let rate = 16_000u32;
        let long = build_pcm_wav(1, rate, 16, &vec![1u8; rate as usize * 2 * 3]);
        assert_eq!(pad_wav_to_min_duration(&long, 2000), long);
        // Non-WAV bytes pass through untouched.
        let junk = b"not a wav at all".to_vec();
        assert_eq!(pad_wav_to_min_duration(&junk, 2000), junk);
    }

    #[test]
    fn request_override_wins_over_saved() {
        // The Test button sends a `tuning` object: its fields beat the saved ones.
        let saved = TtsTuningSettings {
            lead_in_ms: 150,
            gain_db: 0.0,
            temperature: 0.7,
            ..TtsTuningSettings::default()
        };
        let req = serde_json::json!({
            "text": "hi",
            "characterName": "Easy Pete",
            "tuning": { "lead_in_ms": 500, "gain_db": 6.0, "temperature": 0.4 }
        });
        let resolved = resolve_tuning(&saved, &req);
        assert_eq!(resolved.lead_in_ms, 500); // override wins
        assert_eq!(resolved.gain_db, 6.0);
        assert_eq!(resolved.temperature, 0.4);
    }

    #[test]
    fn partial_override_keeps_saved_for_absent_fields() {
        let saved = TtsTuningSettings {
            lead_in_ms: 200,
            trailing_ms: 90,
            ..TtsTuningSettings::default()
        };
        // Only lead_in_ms is overridden; trailing_ms falls back to the saved 90.
        let req = serde_json::json!({ "tuning": { "lead_in_ms": 250 } });
        let resolved = resolve_tuning(&saved, &req);
        assert_eq!(resolved.lead_in_ms, 250);
        assert_eq!(resolved.trailing_ms, 90);
    }

    #[test]
    fn override_values_are_clamped() {
        let saved = TtsTuningSettings::default();
        let req = serde_json::json!({
            "tuning": { "lead_in_ms": 999999, "temperature": 99.0, "lsd_decode_steps": 0 }
        });
        let resolved = resolve_tuning(&saved, &req);
        // Out-of-range overrides are clamped to the documented ranges.
        assert_eq!(resolved.lead_in_ms, chasm_core::TUNING_PAD_MS_MAX);
        assert_eq!(
            resolved.temperature,
            chasm_core::TUNING_TEMPERATURE_MAX
        );
        assert_eq!(
            resolved.lsd_decode_steps,
            chasm_core::TUNING_LSD_STEPS_MIN
        );
    }

    #[test]
    fn tuning_json_carries_every_field() {
        let t = TtsTuningSettings::default();
        let json = tuning_json(&t);
        for key in [
            "lead_in_ms",
            "trailing_ms",
            "gain_db",
            "temperature",
            "lsd_decode_steps",
            "eos_threshold",
            "noise_clamp",
            "max_tokens",
            "frames_after_eos",
        ] {
            assert!(json.get(key).is_some(), "tuning_json missing {key}");
        }
    }

    #[test]
    fn apply_tts_tuning_form_parses_and_clamps() {
        let mut tuning = TtsTuningSettings::default();
        let mut form = HashMap::new();
        form.insert("tuning_lead_in_ms".to_string(), "320".to_string());
        form.insert("tuning_gain_db".to_string(), "3.5".to_string());
        form.insert("tuning_temperature".to_string(), "0.9".to_string());
        form.insert("tuning_lsd_decode_steps".to_string(), "0".to_string()); // clamps to 1
        form.insert("tuning_frames_after_eos".to_string(), "8".to_string());
        apply_tts_tuning_form(&mut tuning, &form);
        assert_eq!(tuning.lead_in_ms, 320);
        assert_eq!(tuning.gain_db, 3.5);
        assert_eq!(tuning.temperature, 0.9);
        assert_eq!(tuning.lsd_decode_steps, 1);
        assert_eq!(tuning.frames_after_eos, 8);
        // A field absent from the form keeps its prior value.
        assert_eq!(tuning.trailing_ms, 60);
    }

    #[test]
    fn apply_tts_form_parses_volume_percents() {
        let mut tts = TtsSettings::default();
        let mut form = HashMap::new();
        form.insert("npc_volume".to_string(), "150".to_string()); // 150% -> 1.5x
        form.insert("admin_volume".to_string(), "50".to_string()); // 50%  -> 0.5x
        apply_tts_form(&mut tts, &form);
        assert_eq!(tts.npc_volume, 1.5);
        assert_eq!(tts.admin_volume, 0.5);
        // Out-of-range percents clamp to the documented multiplier range.
        form.insert("npc_volume".to_string(), "500".to_string()); // -> 2.0x cap
        apply_tts_form(&mut tts, &form);
        assert_eq!(tts.npc_volume, chasm_core::VOICE_VOLUME_MAX);
    }

    /// A bare `MessageView` for render tests; `injected`/`turn_actions` are set by
    /// the caller to exercise the per-message panel.
    fn message_view_fixture(name: &str, content: &str) -> MessageView {
        MessageView {
            id: "m_0".to_string(),
            role: "npc".to_string(),
            speaker_participant_id: None,
            speaker_name: name.to_string(),
            speaker_initial: name
                .chars()
                .next()
                .map(|c| c.to_string())
                .unwrap_or_default(),
            content: content.to_string(),
            created_at: None,
            created_at_label: String::new(),
            segment_id: None,
            location: None,
            audible_to: Vec::new(),
            visible_reason: "speaker".to_string(),
            injected: None,
            turn_actions: Vec::new(),
        }
    }

    #[test]
    fn prompt_panel_renders_per_message_injections_and_no_data() {
        // Message 0: an NPC turn WITH injected lore/action + a chosen action.
        let mut with_blob = message_view_fixture("Sunny Smiles", "Right behind you.");
        with_blob.injected = Some(chasm_core::InjectedView {
            lore: vec![chasm_core::InjectedEntryView {
                source: "lore".to_string(),
                id: "Goodsprings".to_string(),
                title: "Goodsprings".to_string(),
                reason: "keyword".to_string(),
            }],
            quests: Vec::new(),
            actions: vec![chasm_core::InjectedEntryView {
                source: "action".to_string(),
                id: "movement.follow_target".to_string(),
                title: "Follow target".to_string(),
                reason: "vector".to_string(),
            }],
            activated_actions: Vec::new(),
        });
        with_blob.turn_actions = vec![chasm_core::ActionView {
            id: "movement.follow_target".to_string(),
            alias: "follow".to_string(),
            target: "player".to_string(),
            params: "{\"speed\":1}".to_string(),
            reason: "Player asked to be followed.".to_string(),
        }];
        // Message 1: a plain message with NO recorded blob -> "no data".
        let no_blob = message_view_fixture("Player", "Hey, follow me.");

        let panel = PromptPanelTemplate {
            live_chat: LiveChatView {
                id: "fnv-goodsprings".to_string(),
                title: "Goodsprings".to_string(),
                participants: Vec::new(),
                selected_participant_id: None,
            },
            prompt: empty_prompt_assembly("npc:sunny_smiles"),
            messages: vec![with_blob, no_blob],
        };
        let html = panel.render().expect("render prompt panel");
        // Print the rendered per-message panel for the verification report.
        println!("---PROMPT PANEL HTML START---\n{html}\n---PROMPT PANEL HTML END---");

        // Message 0 detail: keyed by list position, with lore + action chips.
        assert!(html.contains("data-msg-detail=\"0\""));
        assert!(html.contains("Goodsprings"));
        assert!(html.contains("inj-reason-keyword"));
        assert!(html.contains("movement.follow_target"));
        assert!(html.contains("inj-reason-vector"));
        // The chosen action with its alias + params + target.
        assert!(html.contains("Actions this turn"));
        assert!(html.contains(">follow<"));
        assert!(html.contains("target: player"));
        // Params are HTML-escaped by askama (renders as {"speed":1} in-browser).
        assert!(html.contains("{&quot;speed&quot;:1}"));
        // Message 1 detail: the graceful no-data note.
        assert!(html.contains("data-msg-detail=\"1\""));
        assert!(html.contains("No injection data recorded for this message."));
        // The default next-prompt view + the message container both exist.
        assert!(html.contains("id=\"prompt-next-view\""));
        assert!(html.contains("id=\"prompt-message-view\""));
        assert!(html.contains("id=\"prompt-back\""));
    }
}
