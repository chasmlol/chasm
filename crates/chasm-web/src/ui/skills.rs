//! UI skills domain — the Skills page backend (the self-improving-NPC system).
//!
//! Endpoints under `/api/ui/v1`:
//!   * `GET  /skills`             — every event-triggered skill grouped per
//!     owner, plus the allowed-action catalogue, the system's settings toggles,
//!     and pass metadata for the header.
//!   * `POST /skills/settings`    — save the journaling / skill-creation /
//!     skill-execution toggles + the per-skill cooldown.
//!   * `POST /skills/:id/toggle`  — enable/disable one skill.
//!   * `POST /skills/:id/delete`  — remove one skill.
//!
//! Skills are CREATED only by the skill-creator pass; this surface toggles,
//! deletes, and configures. Firing is the LLM-free executor (`crate::skill_executor`).

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use chasm_core::AppSettings;

use crate::{AppState, WebResult};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiSkillAction {
    pub action_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiSkill {
    pub id: String,
    pub trigger_event: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger_filter: Option<String>,
    pub actions: Vec<UiSkillAction>,
    pub note: String,
    pub thought: String,
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiSkillOwner {
    pub owner_id: String,
    pub owner_name: String,
    pub skills: Vec<UiSkill>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiAllowedAction {
    pub action_id: String,
    pub title: String,
    pub description: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiSkillSettings {
    pub journaling_enabled: bool,
    pub skill_creation_enabled: bool,
    pub skill_execution_enabled: bool,
    pub skill_cooldown_secs: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiSkillsView {
    pub owners: Vec<UiSkillOwner>,
    pub allowed_actions: Vec<UiAllowedAction>,
    pub settings: UiSkillSettings,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_pass_at: Option<String>,
    pub pass_in_flight: bool,
}

fn view(state: &AppState) -> WebResult<UiSkillsView> {
    let store = state.repository.read_skills()?;
    let settings = AppSettings::load(&state.config.settings_path).self_improvement;
    let names: std::collections::BTreeMap<String, String> = state
        .repository
        .list_character_cards()
        .unwrap_or_default()
        .into_iter()
        .map(|card| (card.id, card.name))
        .collect();

    // Group skills by owner, preserving stored order within an owner.
    let mut owners: Vec<UiSkillOwner> = Vec::new();
    for skill in &store.skills {
        let display = names
            .get(&skill.owner)
            .filter(|n| !n.is_empty())
            .cloned()
            .or_else(|| Some(skill.owner_name.clone()).filter(|n| !n.is_empty()))
            .unwrap_or_else(|| skill.owner.clone());
        let ui_skill = UiSkill {
            id: skill.id.clone(),
            trigger_event: skill.trigger_event.clone(),
            trigger_filter: skill.trigger_filter.clone(),
            actions: skill
                .actions
                .iter()
                .map(|a| UiSkillAction {
                    action_id: a.action_id.clone(),
                    target: a.target.clone(),
                })
                .collect(),
            note: skill.note.clone(),
            thought: skill.thought.clone(),
            enabled: skill.enabled,
            created_at: skill.created_at.clone(),
            updated_at: skill.updated_at.clone(),
        };
        match owners.iter_mut().find(|o| o.owner_id == skill.owner) {
            Some(owner) => owner.skills.push(ui_skill),
            None => owners.push(UiSkillOwner {
                owner_id: skill.owner.clone(),
                owner_name: display,
                skills: vec![ui_skill],
            }),
        }
    }

    let allowed_actions = crate::skill_creator::allowed_actions(state)
        .into_iter()
        .map(|a| UiAllowedAction {
            action_id: a.action_id,
            title: a.title,
            description: a.description,
        })
        .collect();

    Ok(UiSkillsView {
        owners,
        allowed_actions,
        settings: UiSkillSettings {
            journaling_enabled: settings.journaling_enabled,
            skill_creation_enabled: settings.skill_creation_enabled,
            skill_execution_enabled: settings.skill_execution_enabled,
            skill_cooldown_secs: settings.skill_cooldown_secs,
        },
        last_pass_at: store.last_pass_at.clone(),
        pass_in_flight: crate::skill_creator::pass_in_flight(),
    })
}

/// `GET /api/ui/v1/skills` — the full grouped skill list + settings + catalogue.
pub(crate) async fn list_skills(
    State(state): State<Arc<AppState>>,
) -> WebResult<Json<UiSkillsView>> {
    Ok(Json(view(&state)?))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SkillSettingsBody {
    pub journaling_enabled: bool,
    pub skill_creation_enabled: bool,
    pub skill_execution_enabled: bool,
    pub skill_cooldown_secs: u32,
}

/// `POST /api/ui/v1/skills/settings` — save the self-improvement toggles.
pub(crate) async fn save_settings(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SkillSettingsBody>,
) -> WebResult<Json<UiSkillsView>> {
    let mut settings = AppSettings::load(&state.config.settings_path);
    settings.self_improvement.journaling_enabled = body.journaling_enabled;
    settings.self_improvement.skill_creation_enabled = body.skill_creation_enabled;
    settings.self_improvement.skill_execution_enabled = body.skill_execution_enabled;
    settings.self_improvement.skill_cooldown_secs = body.skill_cooldown_secs;
    settings.save(&state.config.settings_path)?;
    Ok(Json(view(&state)?))
}

/// `POST /api/ui/v1/skills/:id/toggle` — enable/disable one skill (the user's
/// override over the skill-creator).
pub(crate) async fn toggle_skill(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> WebResult<Json<UiSkillsView>> {
    state.repository.update_skills(|store| {
        if let Some(skill) = store.skills.iter_mut().find(|s| s.id == id) {
            skill.enabled = !skill.enabled;
        }
    })?;
    Ok(Json(view(&state)?))
}

/// `POST /api/ui/v1/skills/:id/delete` — remove one skill.
pub(crate) async fn delete_skill(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> WebResult<Json<UiSkillsView>> {
    state.repository.update_skills(|store| {
        store.skills.retain(|s| s.id != id);
    })?;
    Ok(Json(view(&state)?))
}
