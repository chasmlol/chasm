//! UI profiles domain — list + activate the active game profile.
//!
//! The React Profiles screen renders the available drop-in [`GameProfile`]s as
//! cards (with their on-disk content counts) and activates one via
//! `POST /api/ui/v1/profiles/select`. Both endpoints reuse the public profile
//! cores (`GameProfile::list` / `read`, `AppSettings::active_profile_id`) and the
//! same persist path the legacy Askama selector used (`settings.profile = id;
//! settings.save(...)`), so a switch round-trips identically across both UIs.
//!
//! NOTE: profile *import* (drag-and-drop portability) is a planned follow-up and
//! is deliberately NOT implemented here — this module only lists + activates.
//! Stays under `/api/ui/v1`; reads/writes only the profile content + the active
//! profile id, never the game transport or AI-stack lifecycle.

use std::{path::Path, sync::Arc};

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use chasm_core::{AppSettings, GameProfile};
use tracing::info;

use crate::{AppState, WebError, WebResult};

/// One profile card in the Profiles list.
#[derive(Serialize)]
pub(crate) struct UiProfile {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Two-letter initials used by the card badge.
    pub initials: String,
    /// `true` for the currently-active profile.
    pub active: bool,
    pub character_count: usize,
    pub lorebook_count: usize,
    pub quest_count: usize,
    pub action_count: usize,
}

/// The Profiles screen payload: every profile + which one is active.
#[derive(Serialize)]
pub(crate) struct UiProfilesView {
    pub active_id: String,
    pub profiles_dir: String,
    pub profiles: Vec<UiProfile>,
}

fn build_profiles_view(state: &AppState) -> UiProfilesView {
    let profiles_dir = &state.config.profiles_dir;
    let settings = AppSettings::load(&state.config.settings_path);
    let active_id = settings.active_profile_id(profiles_dir);

    let profiles = GameProfile::list(profiles_dir)
        .into_iter()
        .map(|profile| {
            let dir = profiles_dir.join(&profile.id);
            let name = if profile.name.is_empty() {
                profile.id.clone()
            } else {
                profile.name.clone()
            };
            UiProfile {
                initials: profile_initials(&name),
                active: profile.id == active_id,
                character_count: profile.characters.len(),
                lorebook_count: count_json_files(&dir.join("worlds")),
                quest_count: count_book_entries(&dir.join("headless").join("quest-books")),
                action_count: count_book_entries(&dir.join("headless").join("action-books")),
                description: profile.description.clone(),
                name,
                id: profile.id,
            }
        })
        .collect();

    UiProfilesView {
        active_id,
        profiles_dir: profiles_dir.display().to_string(),
        profiles,
    }
}

/// `GET /api/ui/v1/profiles` — the list of profiles + the active id. Read-only.
pub(crate) async fn list_profiles(State(state): State<Arc<AppState>>) -> Json<UiProfilesView> {
    Json(build_profiles_view(&state))
}

/// `POST /api/ui/v1/profiles/select` body: `{ "id": "<profile-id>" }`.
#[derive(Deserialize)]
pub(crate) struct SelectBody {
    #[serde(default)]
    id: String,
}

/// `POST /api/ui/v1/profiles/select` — switch the active game profile, then
/// return the fresh list. Validates the id against the available profiles and
/// persists `settings.profile`, exactly like the legacy `/profile/select`.
pub(crate) async fn select_profile(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SelectBody>,
) -> WebResult<Json<UiProfilesView>> {
    let id = body.id.trim();
    if id.is_empty() {
        return Err(WebError(anyhow::anyhow!("profile id is required.")));
    }
    let Some(profile) = GameProfile::read(&state.config.profiles_dir, id) else {
        return Err(WebError(anyhow::anyhow!("unknown profile id '{id}'.")));
    };

    let mut settings = AppSettings::load(&state.config.settings_path);
    settings.profile = profile.id.clone();
    settings.save(&state.config.settings_path)?;
    info!("active profile switched to '{}'", profile.id);

    Ok(Json(build_profiles_view(&state)))
}

/// Two-letter uppercase initials for a profile-name badge (mirrors the legacy
/// `profile_initials`): the first letter of the first two whitespace-separated
/// words, else the first two chars.
fn profile_initials(name: &str) -> String {
    let words: Vec<&str> = name.split_whitespace().collect();
    let letters: String = if words.len() >= 2 {
        words
            .iter()
            .take(2)
            .filter_map(|w| w.chars().next())
            .collect()
    } else {
        name.chars().take(2).collect()
    };
    letters.to_uppercase()
}

/// Counts `*.json` files directly under `dir` (non-recursive) — the per-profile
/// lorebook count.
fn count_json_files(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
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

/// Sums the `entries` across every `*.json` book under `dir` (quest/action
/// books). Books store `entries` as a JSON object keyed by id (an array is
/// tolerated); unreadable / entries-less files count 0.
fn count_book_entries(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
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
        .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
        .filter_map(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .map(|value| match value.get("entries") {
            Some(serde_json::Value::Object(map)) => map.len(),
            Some(serde_json::Value::Array(arr)) => arr.len(),
            _ => 0,
        })
        .sum()
}
