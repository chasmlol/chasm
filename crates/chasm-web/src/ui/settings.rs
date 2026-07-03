//! UI settings domain — the Interface (appearance) round-trip.
//!
//! This is the one settings screen ported end-to-end. Read returns a focused
//! view assembled from the public `chasm-core` builders; save reuses the
//! exact same `apply_interface_form` the server-rendered Askama page uses, so
//! both UIs round-trip identically and `/theme.css` updates on next load.
//!
//! Other settings categories (Profiles / Bridge / Tracing) add their endpoints
//! to their own modules; the AI categories live in [`super::models`]. This
//! module stays focused on the appearance settings + the shared nav.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Json,
};
use serde::Serialize;
use chasm_core::{
    interface_panel_view, settings_nav_groups, AppSettings, InterfacePanelView, SettingsNavGroup,
};

use crate::{apply_interface_form, AppState, WebResult};

/// The Music settings panel the React page renders: the saved values, the clamp
/// bounds for the length slider, the engine option(s), and the live install/run
/// status of the ACE-Step engine (so the page shows whether song generation is
/// ready, like the other capability pages).
#[derive(Serialize)]
pub(crate) struct MusicPanelView {
    pub enabled: bool,
    pub engine: String,
    pub style_tags: String,
    pub max_seconds: u32,
    pub max_seconds_min: u32,
    pub max_seconds_max: u32,
    /// Use the performing NPC's own voice clip as the ACE-Step style reference.
    pub match_npc_voice: bool,
    /// (id, label) options — only ACE-Step for now.
    pub engines: Vec<(String, String)>,
    /// Engine install status string (`installed` / `installing` / `failed` /
    /// `not_installed`) + whether the server is currently reachable.
    pub engine_status: String,
    pub engine_running: bool,
}

fn music_panel_view(state: &AppState) -> MusicPanelView {
    let settings = AppSettings::load(&state.config.settings_path);
    let status = crate::acestep_engine_status(state);
    // Reachability probe of the music server port (a closed localhost port refuses
    // instantly, so this never stalls the page). Mirrors `launcher::acestep_running`
    // but avoids threading an `Arc` through `build_ui_settings`'s `&AppState`.
    let running = "127.0.0.1:5004"
        .parse()
        .ok()
        .map(|addr| {
            std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(300))
                .is_ok()
        })
        .unwrap_or(false);
    MusicPanelView {
        enabled: settings.music.enabled,
        engine: chasm_core::normalize_music_engine(&settings.music.engine),
        style_tags: settings.music.style_tags,
        max_seconds: chasm_core::normalize_music_max_seconds(settings.music.max_seconds),
        max_seconds_min: chasm_core::MUSIC_MAX_SECONDS_MIN,
        max_seconds_max: chasm_core::MUSIC_MAX_SECONDS_MAX,
        match_npc_voice: settings.music.match_npc_voice,
        engines: chasm_core::MUSIC_ENGINES
            .iter()
            .map(|(id, label)| (id.to_string(), label.to_string()))
            .collect(),
        engine_status: status,
        engine_running: running,
    }
}

/// The focused settings view the React app consumes. A strict subset of the
/// server's `SettingsPageView` — only what the ported screen needs — built from
/// public core functions so it doesn't depend on the web layer's private
/// model-status plumbing. Extended as more screens are migrated.
#[derive(Serialize)]
pub(crate) struct UiSettingsView {
    pub category: String,
    pub nav_groups: Vec<SettingsNavGroup>,
    pub settings_path: String,
    pub interface: InterfacePanelView,
    pub music: MusicPanelView,
}

pub(crate) fn build_ui_settings(state: &AppState, category: &str) -> UiSettingsView {
    let settings = AppSettings::load(&state.config.settings_path);
    UiSettingsView {
        nav_groups: settings_nav_groups(category),
        settings_path: state.config.settings_path.display().to_string(),
        interface: interface_panel_view(&settings.interface),
        music: music_panel_view(state),
        category: category.to_string(),
    }
}

/// `GET /api/ui/v1/settings/:category` — the JSON the SPA renders a settings
/// screen from. Read-only; never mutates. (Only `interface` is fully consumed
/// by the React side today; other categories still return their nav + the
/// interface panel so the shell renders during migration.)
pub(crate) async fn get_settings(
    State(state): State<Arc<AppState>>,
    Path(category): Path<String>,
) -> Json<UiSettingsView> {
    Json(build_ui_settings(&state, &category))
}

/// `POST /api/ui/v1/settings/interface/save` — save the appearance settings
/// from the React form, then return the fresh view. Reuses `apply_interface_form`
/// (the same applier the Askama page posts through) so validation/normalization
/// and the persisted shape are identical across both UIs.
pub(crate) async fn save_interface(
    State(state): State<Arc<AppState>>,
    Json(form): Json<std::collections::HashMap<String, serde_json::Value>>,
) -> WebResult<Json<UiSettingsView>> {
    // The shared applier reads a string map (it was written for an HTML form
    // body). Coerce the JSON object to that shape: checkboxes are presence-keyed,
    // so only insert booleans that are true; everything else passes as a string.
    let mut string_form: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for (key, value) in &form {
        match value {
            serde_json::Value::Bool(true) => {
                string_form.insert(key.clone(), "on".to_string());
            }
            serde_json::Value::Bool(false) => { /* absent == unchecked */ }
            serde_json::Value::String(s) => {
                string_form.insert(key.clone(), s.clone());
            }
            serde_json::Value::Number(n) => {
                string_form.insert(key.clone(), n.to_string());
            }
            _ => {}
        }
    }

    let mut settings = AppSettings::load(&state.config.settings_path);
    apply_interface_form(&mut settings.interface, &string_form);
    settings.save(&state.config.settings_path)?;

    Ok(Json(build_ui_settings(&state, "interface")))
}

/// `POST /api/ui/v1/settings/music/save` — save the Music page's non-picker
/// fields (enable toggle, base style tags, max song length) and return the fresh
/// view. The engine SELECTION is handled by the `<ModelPicker>` (`/models/music`);
/// here we only persist the settings + start/stop the server on an enable-toggle.
pub(crate) async fn save_music(
    State(state): State<Arc<AppState>>,
    Json(body): Json<MusicSaveBody>,
) -> WebResult<Json<UiSettingsView>> {
    let mut settings = AppSettings::load(&state.config.settings_path);
    let was_enabled = settings.music.enabled;

    settings.music.enabled = body.enabled;
    if let Some(tags) = body.style_tags {
        settings.music.style_tags = tags.trim().to_string();
    }
    if let Some(secs) = body.max_seconds {
        settings.music.max_seconds = chasm_core::normalize_music_max_seconds(secs);
    }
    if let Some(match_voice) = body.match_npc_voice {
        settings.music.match_npc_voice = match_voice;
    }
    // There is exactly one music engine (ACE-Step). Requiring a SEPARATE "select"
    // click after enabling is a footgun — an enabled-but-no-engine config silently
    // aborts every song job. So when enabling with no engine chosen yet and the
    // engine is installed, auto-select it. (If it isn't installed, leave it empty so
    // the page still nudges to Runtimes.)
    if body.enabled
        && chasm_core::normalize_music_engine(&settings.music.engine).is_empty()
        && crate::launcher::acestep_installed(&state.config)
    {
        settings.music.engine = chasm_core::ACESTEP_ENGINE_ID.to_string();
    }
    settings.save(&state.config.settings_path)?;

    // Toggle -> start or stop the server so the change takes effect without waiting
    // for the next stack (re)start. Best-effort, off the async path.
    if body.enabled != was_enabled {
        let state2 = Arc::clone(&state);
        tokio::task::spawn_blocking(move || {
            if body.enabled {
                crate::launcher::start_music_engine(&state2);
            } else {
                crate::launcher::stop_music_engine(&state2);
            }
        });
    }

    Ok(Json(build_ui_settings(&state, "music")))
}

/// The Music save form body. `style_tags` / `max_seconds` are optional so a
/// partial save (just the toggle) leaves the rest untouched.
#[derive(serde::Deserialize)]
pub(crate) struct MusicSaveBody {
    pub enabled: bool,
    pub style_tags: Option<String>,
    pub max_seconds: Option<u32>,
    pub match_npc_voice: Option<bool>,
}
