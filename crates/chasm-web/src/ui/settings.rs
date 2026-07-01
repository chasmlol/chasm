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
}

pub(crate) fn build_ui_settings(state: &AppState, category: &str) -> UiSettingsView {
    let settings = AppSettings::load(&state.config.settings_path);
    UiSettingsView {
        nav_groups: settings_nav_groups(category),
        settings_path: state.config.settings_path.display().to_string(),
        interface: interface_panel_view(&settings.interface),
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
