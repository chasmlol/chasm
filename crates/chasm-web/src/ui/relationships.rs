//! UI relationships domain — the Relationships page backend.
//!
//! Read + edit surface over the Gamemaster-maintained relationships store
//! (`headless/relationships.json`, see `crate::gamemaster` and
//! `chasm_st_compat::RelationshipsStore`). Two endpoints under `/api/ui/v1`:
//!
//!   * `GET  /relationships`      — every directional entry across all
//!     characters, grouped per character, plus pass metadata for the header.
//!   * `POST /relationships/save` — edit or clear ONE pair's text (the user's
//!     correction channel over the GM). An empty text removes the pair, which
//!     is indistinguishable from it never having existed (nothing injected).
//!
//! Entries are only ever CREATED by the GM pass (or its target discovery);
//! this surface edits and clears existing pairs.

use std::sync::Arc;

use axum::{extract::State, Json};
use serde::Serialize;
use serde_json::Value;

use crate::{AppState, WebError, WebResult};

fn web_err(message: impl Into<String>) -> WebError {
    WebError::from(anyhow::anyhow!(message.into()))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiRelationshipEntry {
    pub target_id: String,
    pub target_name: String,
    /// `"player"` or `"npc"`.
    pub target_kind: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiRelationshipCharacter {
    pub character_id: String,
    /// Display name (card name when the card resolves, else the id).
    pub character_name: String,
    pub entries: Vec<UiRelationshipEntry>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiRelationshipsView {
    pub characters: Vec<UiRelationshipCharacter>,
    /// When the last Gamemaster pass completed, if any has run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_pass_at: Option<String>,
    /// True while a GM pass is currently running.
    pub pass_in_flight: bool,
}

fn view(state: &AppState) -> WebResult<UiRelationshipsView> {
    let store = state.repository.read_relationships()?;
    // Card display names for the group headers (id == PNG basename, which is
    // usually already the name; the card name wins when it differs).
    let names: std::collections::BTreeMap<String, String> = state
        .repository
        .list_character_cards()
        .unwrap_or_default()
        .into_iter()
        .map(|card| (card.id, card.name))
        .collect();
    let characters = store
        .characters
        .iter()
        .map(|(character_id, relationships)| UiRelationshipCharacter {
            character_id: character_id.clone(),
            character_name: names
                .get(character_id)
                .filter(|name| !name.is_empty())
                .cloned()
                .unwrap_or_else(|| character_id.clone()),
            entries: relationships
                .entries
                .iter()
                .map(|(target_id, entry)| UiRelationshipEntry {
                    target_id: target_id.clone(),
                    target_name: entry.target_name.clone(),
                    target_kind: entry.target_kind.clone(),
                    text: entry.text.clone(),
                    created_at: entry.created_at.clone(),
                    updated_at: entry.updated_at.clone(),
                })
                .collect(),
        })
        .collect();
    Ok(UiRelationshipsView {
        characters,
        last_pass_at: store.last_pass_at,
        pass_in_flight: crate::gamemaster::pass_in_flight(),
    })
}

/// `GET /api/ui/v1/relationships` — the full grouped ledger.
pub(crate) async fn list_relationships(
    State(state): State<Arc<AppState>>,
) -> WebResult<Json<UiRelationshipsView>> {
    Ok(Json(view(&state)?))
}

/// `POST /api/ui/v1/relationships/save` — edit or clear one pair.
///
/// Request: `{ "characterId": "…", "targetId": "…", "text": "…" }` (all
/// strings; ids in the body rather than the path because character ids contain
/// spaces). Empty/blank `text` clears the pair. The pair must already exist —
/// this is the correction surface over the GM, not an authoring tool.
pub(crate) async fn save_relationship(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> WebResult<Json<UiRelationshipsView>> {
    let field = |key: &str| {
        body.get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| web_err(format!("relationships save requires a string '{key}'")))
    };
    let character_id = field("characterId")?;
    let target_id = field("targetId")?;
    let text = field("text")?;

    let now = crate::persona::chrono_now_iso();
    state.repository.update_relationships(|store| {
        let Some(existing) = store
            .characters
            .get(&character_id)
            .and_then(|character| character.entries.get(&target_id))
            .cloned()
        else {
            return Err(web_err(format!(
                "no relationship entry {character_id} → {target_id}"
            )));
        };
        // `upsert` handles both paths: blank text removes the pair (and prunes
        // the character), non-blank rewrites it preserving created_at.
        store.upsert(
            &character_id,
            &target_id,
            &existing.target_name,
            &existing.target_kind,
            &text,
            &now,
        );
        Ok(())
    })??;
    Ok(Json(view(&state)?))
}
