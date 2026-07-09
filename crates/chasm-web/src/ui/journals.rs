//! UI journals domain — the read-only Journals page backend.
//!
//! One endpoint under `/api/ui/v1`: `GET /journals` returns every NPC's
//! append-only private journal (see [`chasm_st_compat::JournalStore`] and
//! `crate::journal`), grouped per character, plus pass metadata for the header.
//! Journals are only ever written by the journal pass; this surface is
//! read-only.

use std::sync::Arc;

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use crate::{AppState, WebResult};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiJournalEntry {
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub game_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub game_day: Option<i64>,
    pub text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiJournalCharacter {
    pub character_id: String,
    pub character_name: String,
    pub entries: Vec<UiJournalEntry>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiJournalsView {
    pub characters: Vec<UiJournalCharacter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_pass_at: Option<String>,
    pub pass_in_flight: bool,
}

fn view(state: &AppState) -> WebResult<UiJournalsView> {
    let store = state.repository.read_journals()?;
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
        .into_iter()
        .map(|(character_id, journal)| {
            let character_name = names
                .get(&character_id)
                .filter(|n| !n.is_empty())
                .cloned()
                .or_else(|| Some(journal.name.clone()).filter(|n| !n.is_empty()))
                .unwrap_or_else(|| character_id.clone());
            UiJournalCharacter {
                character_id,
                character_name,
                entries: journal
                    .entries
                    .into_iter()
                    .map(|e| UiJournalEntry {
                        created_at: e.created_at,
                        game_time: e.game_time,
                        game_day: e.game_day,
                        text: e.text,
                    })
                    .collect(),
            }
        })
        .collect();
    Ok(UiJournalsView {
        characters,
        last_pass_at: store.last_pass_at,
        pass_in_flight: crate::journal::pass_in_flight(),
    })
}

/// `GET /api/ui/v1/journals` — every NPC's journal, grouped per character.
pub(crate) async fn list_journals(
    State(state): State<Arc<AppState>>,
) -> WebResult<Json<UiJournalsView>> {
    Ok(Json(view(&state)?))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DeleteEntryBody {
    pub character_id: String,
    /// The entry's `createdAt` — unique within a character (one entry per pass),
    /// stable under the journal's append-only growth.
    pub created_at: String,
}

/// `POST /api/ui/v1/journals/delete-entry` — remove ONE journal entry. The pass
/// is otherwise append-only; this is the user curating a journal by hand. When
/// a character's last entry is removed, the (now empty) character is dropped so
/// it doesn't linger as a blank group. Returns the refreshed view.
pub(crate) async fn delete_entry(
    State(state): State<Arc<AppState>>,
    Json(body): Json<DeleteEntryBody>,
) -> WebResult<Json<UiJournalsView>> {
    let character_id = body.character_id.trim().to_string();
    let created_at = body.created_at.trim().to_string();
    if !character_id.is_empty() && !created_at.is_empty() {
        state.repository.update_journals(|store| {
            if let Some(journal) = store.characters.get_mut(&character_id) {
                journal.entries.retain(|entry| entry.created_at != created_at);
                if journal.entries.is_empty() {
                    store.characters.remove(&character_id);
                }
            }
        })?;
    }
    Ok(Json(view(&state)?))
}
