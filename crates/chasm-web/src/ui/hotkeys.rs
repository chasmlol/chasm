//! UI hotkeys domain — the four in-game input bindings.
//!
//! The React Hotkeys screen edits `AppSettings.hotkeys` (canonical key names,
//! see `chasm_core::hotkeys::virtual_key_code` for the accepted set) through
//! the SAME `AppSettings::load`/`save` path the other settings screens use.
//!
//! On every save the bindings are ALSO written to the bridge rendezvous dir
//! (`<bridge_root>\control\hotkeys.cfg`, decimal VK codes) — the NVSE plugin
//! polls that file's mtime every second, so a save takes effect in a running
//! game without a restart. The file is additionally (re)written at bridge
//! startup (see `router()` in lib.rs) so it always reflects persisted settings.

use std::{collections::HashMap, sync::Arc};

use axum::{extract::State, Json};
use serde::Serialize;
use chasm_core::{
    hotkeys::{virtual_key_code, write_bridge_hotkeys_file},
    default_bridge_root, AppSettings, HotkeysSettings,
};

use crate::{AppState, WebError, WebResult};

/// The editable hotkey bindings + the built-in defaults (for per-row reset).
#[derive(Serialize)]
pub(crate) struct UiHotkeysView {
    pub settings_path: String,
    pub config: HotkeysSettings,
    pub defaults: HotkeysSettings,
}

fn build_hotkeys_view(state: &AppState, settings: &AppSettings) -> UiHotkeysView {
    UiHotkeysView {
        settings_path: state.config.settings_path.display().to_string(),
        config: settings.hotkeys.clone(),
        defaults: HotkeysSettings::default(),
    }
}

/// `GET /api/ui/v1/settings/hotkeys` — the current bindings + defaults.
pub(crate) async fn get_hotkeys(State(state): State<Arc<AppState>>) -> Json<UiHotkeysView> {
    let settings = AppSettings::load(&state.config.settings_path);
    Json(build_hotkeys_view(&state, &settings))
}

/// `POST /api/ui/v1/settings/hotkeys/save` — persist edited bindings, push
/// them to the bridge file, then return the fresh view. Each submitted name
/// must be a known canonical key name (the capture UI only produces those);
/// an unknown name is a 400 so a typo can never persist a dead binding.
pub(crate) async fn save_hotkeys(
    State(state): State<Arc<AppState>>,
    Json(form): Json<HashMap<String, String>>,
) -> WebResult<Json<UiHotkeysView>> {
    let mut settings = AppSettings::load(&state.config.settings_path);
    let fields: [(&str, &mut String); 4] = [
        ("push_to_talk", &mut settings.hotkeys.push_to_talk),
        ("enter_text", &mut settings.hotkeys.enter_text),
        ("todd_push_to_talk", &mut settings.hotkeys.todd_push_to_talk),
        ("todd_enter_text", &mut settings.hotkeys.todd_enter_text),
    ];
    for (key, slot) in fields {
        if let Some(value) = form.get(key) {
            let value = value.trim();
            if virtual_key_code(value).is_none() {
                return Err(WebError::from(anyhow::anyhow!(
                    "unknown key name {value:?} for {key}"
                )));
            }
            *slot = value.to_string();
        }
    }
    settings.save(&state.config.settings_path)?;

    // Deliver to the running game. Best-effort: the bridge dir may not exist
    // yet (game never launched) — that's fine, startup rewrites it too.
    if let Err(err) = write_bridge_hotkeys_file(&default_bridge_root(), &settings.hotkeys) {
        tracing::warn!("failed to write bridge hotkeys file: {err}");
    }

    Ok(Json(build_hotkeys_view(&state, &settings)))
}
