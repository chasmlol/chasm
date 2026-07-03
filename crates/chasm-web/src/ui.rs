//! New React SPA serving + the UI-only JSON API.
//!
//! This module is the ONLY thing the React frontend (crates/chasm-web/ui)
//! talks to, plus the read-only `/connection/status` it shares with the desktop
//! shell. It is deliberately isolated from the game/bridge contract:
//!
//!   * It serves the built Vite bundle (`ui/dist`) under `/app` with an SPA
//!     fallback, so the new UI lives alongside the existing Askama pages during
//!     the phased migration (the legacy `/`, `/settings/*`, … stay reachable).
//!   * It exposes a UI JSON API under `/api/ui/v1/*`. This namespace never
//!     overlaps `/api/headless/*` or `/api/game/*` (the bridge/game transport),
//!     so the parallel transport work and this UI work can't collide.
//!
//! The UI endpoints are organized into PER-DOMAIN submodules so each fill agent
//! edits only their module:
//!   * [`settings`] — Interface (appearance) round-trip + nav (DONE).
//!   * [`books`]    — Characters / Lore / Quest / Action          (stub).
//!   * [`models`]   — LLM / TTS / STT / Retrieval                 (stub).
//!   * [`chat`]     — read-only live-chat projection              (stub).
//! They are all registered in ONE block in [`api_router`] below.

use std::{path::PathBuf, sync::Arc};

use axum::{
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::set_header::SetResponseHeader;

use crate::AppState;

// Per-domain UI API submodules. Files live in `src/ui/`. Fill agents implement
// their domain's handlers in the matching module; routing is wired once below.
pub(crate) mod books;
pub(crate) mod bridge;
pub(crate) mod chat;
pub(crate) mod companions;
pub(crate) mod config;
pub(crate) mod gamestate;
pub(crate) mod globals;
pub(crate) mod hotkeys;
pub(crate) mod models;
pub(crate) mod persona;
pub(crate) mod profiles;
pub(crate) mod providers;
pub(crate) mod relationships;
pub(crate) mod settings;
pub(crate) mod system;

/// Where the built SPA lives: `crates/chasm-web/ui/dist`, resolved off the
/// workspace root (the same anchor `AppConfig` uses for `static/`). Built by
/// `npm run build` in that dir; absent until then (the route then 404s, which is
/// fine — the legacy UI is unaffected).
pub(crate) fn ui_dist_dir(state: &AppState) -> PathBuf {
    state
        .config
        .workspace_root
        .join("crates")
        .join("chasm-web")
        .join("ui")
        .join("dist")
}

/// The UI JSON API sub-router, nested at `/api/ui/v1` by `router()`.
///
/// ===== THE ONE /api/ui/v1 REGISTRATION BLOCK =============================
/// Every UI-only endpoint is registered HERE, grouped by domain, pointing at a
/// handler in the matching submodule. New UI endpoints belong in their domain
/// module + a line here — NOT under `/api/headless/*` or `/api/game/*`.
/// =========================================================================
pub(crate) fn api_router() -> Router<Arc<AppState>> {
    Router::new()
        // --- settings (Interface appearance; DONE) ---------------------------
        // The interface save lives on its OWN path (not a second method on the
        // `:category` route): axum 0.7 routes a static segment and a `:param`
        // segment independently, so registering both `GET /settings/:category`
        // and `POST /settings/interface` makes a GET to `/settings/interface`
        // resolve to the static node — which would 405 (POST-only). Keeping save
        // under `/settings/interface/save` avoids the static-vs-dynamic clash, so
        // GET `/settings/<any>` (interface included) all hit the read handler.
        .route("/settings/:category", get(settings::get_settings))
        .route("/settings/interface/save", post(settings::save_interface))
        // --- books (Characters / Lore / Quest / Action; STUB) ----------------
        .route("/books/:kind", get(books::list_book))
        .route("/books/:kind/:id", post(books::save_book))
        // --- models (LLM / TTS / STT / Retrieval) ----------------------------
        .route("/models/:domain", get(models::get_models))
        .route("/models/:domain/select", post(models::select_model))
        .route("/models/:domain/download", post(models::download_model))
        // Guided manual model placement (theme B): the user drops / picks a model
        // file and we stream it (raw bytes body) into the resolved folder. Disable
        // the 2MB default body limit — GGUFs are multi-GB.
        .route(
            "/models/:domain/place",
            post(system::place_model).layer(axum::extract::DefaultBodyLimit::disable()),
        )
        // --- providers (LLM / STT / TTS provider picker + per-provider config) -
        .route("/providers/:capability", get(providers::get_providers))
        .route(
            "/providers/:capability/select",
            post(providers::select_provider),
        )
        .route(
            "/providers/:capability/config",
            post(providers::save_provider_config),
        )
        // --- TTS API voice cloning (clone a character via the hosted provider) -
        .route("/tts/clone", post(providers::clone_api_voice))
        .route("/tts/api-voices", get(providers::list_api_voices))
        // --- system (OS integration the webview can't do itself) -------------
        .route("/system/open-url", post(system::open_url))
        .route("/system/open-folder", post(system::open_folder))
        // --- config (per-engine LLM/TTS/STT/Retrieval knobs; reuses the legacy
        // apply/normalize path so saved values round-trip identically) ---------
        .route(
            "/config/:domain",
            get(config::get_config).post(config::save_config),
        )
        // --- chat (read-only live-chat projection; STUB) ---------------------
        .route("/chat/view", get(chat::chat_view))
        // --- gamestate (latest recorded macro table + the substitution tester) -
        .route("/gamestate", get(gamestate::gamestate_view))
        .route("/gamestate/test", post(gamestate::gamestate_test))
        // --- persona (the generated player persona: description + screenshot
        // + stats snapshot; regenerate = the manual test hook) -----------------
        .route("/persona", get(persona::persona_view))
        .route("/persona/regenerate", post(persona::persona_regenerate))
        .route("/persona/custom", post(persona::persona_set_custom))
        // --- globals (global scenario template: the production macro surface,
        // replacing the per-character card scenario; + resolved preview) ------
        .route(
            "/globals/scenario",
            get(globals::get_scenario).put(globals::put_scenario),
        )
        .route(
            "/globals/scenario/preview",
            post(globals::preview_scenario),
        )
        // --- relationships (the Gamemaster's directional ledger: list + the
        // user's edit/clear correction surface; ids ride in the POST body
        // because character ids contain spaces) --------------------------------
        .route("/relationships", get(relationships::list_relationships))
        .route(
            "/relationships/save",
            post(relationships::save_relationship),
        )
        // --- companions (authored FNV followers; see mod-source
        // docs/companions-architecture.md: pool status + create + slot ops
        // relayed to the NVSE plugin as command files) -------------------------
        // Voice clips ride the create body as base64 (server cap: 64MB decoded),
        // so this route overrides axum's 2MB default body limit.
        .route(
            "/companions",
            get(companions::list_companions)
                .post(companions::create_companion)
                .layer(axum::extract::DefaultBodyLimit::max(100 * 1024 * 1024)),
        )
        .route("/companions/:slot/op", post(companions::companion_op))
        // --- profiles (list + activate the active game profile) --------------
        .route("/profiles", get(profiles::list_profiles))
        .route("/profiles/select", post(profiles::select_profile))
        // --- bridge (connection config + live status) ------------------------
        .route("/settings/bridge", get(bridge::get_bridge))
        .route("/settings/bridge/save", post(bridge::save_bridge))
        // --- hotkeys (in-game input bindings; save also pushes the bridge
        // control/hotkeys.cfg the NVSE plugin live-polls) ----------------------
        .route("/settings/hotkeys", get(hotkeys::get_hotkeys))
        .route("/settings/hotkeys/save", post(hotkeys::save_hotkeys))
        // --- tracing (READ-ONLY trace viewer; reuses the shared trace cores) -
        // The legacy `/traces` handlers already parse the per-request JSONL into
        // the same list + waterfall view; the SPA reads them under /api/ui/v1 too
        // so the namespace stays self-contained. They never mutate trace files.
        .route("/traces", get(crate::trace_routes::list_traces_endpoint))
        .route("/traces/:id", get(crate::trace_routes::get_trace_endpoint))
}

/// A `ServeDir` for the built SPA with an SPA fallback to `index.html`, so deep
/// links under `/app/...` resolve client-side. Mounted by `router()` at `/app`.
/// When `ui/dist` is absent (UI not built yet) this still constructs; requests
/// just 404 until `npm run build` runs.
///
/// Every response carries `Cache-Control: no-cache` so the desktop WebView
/// ALWAYS revalidates before serving from its disk cache. Without this the
/// WebView heuristically caches `index.html` (which has no hash in its name);
/// after an in-app update it would keep loading the stale HTML — and therefore
/// the stale hashed JS bundle it references — so UI changes appeared not to
/// take. `no-cache` still allows 304s (the hashed assets rarely change), it
/// just forbids using a cached copy without checking first.
pub(crate) fn spa_service(state: &AppState) -> SetResponseHeader<ServeDir<ServeFile>, HeaderValue> {
    let dist = ui_dist_dir(state);
    let index = dist.join("index.html");
    let serve = ServeDir::new(dist).fallback(ServeFile::new(index));
    SetResponseHeader::overriding(
        serve,
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache"),
    )
}

/// `GET /app` (exact) — redirect to `/app/` so the SPA's `base: '/app/'` asset
/// URLs resolve. Without the trailing slash the relative asset paths would 404.
pub(crate) async fn app_root_redirect() -> Response {
    (
        StatusCode::TEMPORARY_REDIRECT,
        [(axum::http::header::LOCATION, "/app/")],
    )
        .into_response()
}
