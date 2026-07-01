//! UI chat domain — read-only live-chat projection for the React Chat screen.
//!
//! The Chat screen reads `GET /api/ui/v1/chat/view`. This builds a read-only
//! projection of the active live chat from the repository (`state.repository`):
//! the conversation threads (one per in-scene NPC) and, for EACH message, the
//! per-turn context strip the redesigned UI shows inline:
//!
//!   * the Lore Book / Quest Book / world-info entries INJECTED into that turn,
//!   * the actions OFFERED to the model that turn,
//!   * which of those actions actually EXECUTED (highlighted green in the UI).
//!
//! Data source: the projection reuses chasm's existing core view-builder
//! ([`LiveChatRepository::messages_for_participant`] →
//! [`chasm_core::MessageView`]). Each `MessageView` already carries the
//! per-turn `injected` groups (lore / quests / actions) and the chosen
//! `turn_actions`, both recorded at generation time into the message's
//! `extra.chasm` blob. So "injected vs executed" is derived directly from
//! the persisted turn record — no game transport, no trace-file join required.
//! (The legacy `req_*.jsonl` traces remain the source for the Tracing waterfall;
//! they are NOT needed here because the turn's injected/offered/executed context
//! is persisted on the message itself.)
//!
//! "Offered → executed": an OFFERED action (an `injected.actions` entry) is
//! marked `executed = true` when its id matches one of the turn's `turn_actions`.
//! Any executed action with no matching offered entry is still surfaced (in the
//! per-message `executed` list) so a native/relayed action that wasn't in the
//! offered set is never hidden.
//!
//! IMPORTANT: this is a READ-ONLY projection under `/api/ui/v1`. The UI must
//! never reach the game transport (`/api/game/*`) or the headless contract
//! (`/api/headless/*`); chat data the UI needs is exposed here instead.

use std::sync::Arc;

use axum::{extract::State, Json};
use serde::Serialize;
use chasm_core::{ActionView, InjectedEntryView, MessageView};
use chasm_st_compat::LiveChat;

use crate::{AppState, WebResult};

/// One injected world-info entry (lore / quest / action) shown in a message's
/// context strip. Mirrors [`InjectedEntryView`] in a UI-stable shape.
#[derive(Serialize)]
pub(crate) struct UiInjectedEntry {
    /// `lore`, `quest`, or `action`.
    pub source: String,
    /// Stable display id (lore comment/index, quest id, action id).
    pub id: String,
    /// Human label (lore comment, quest title, action alias/title).
    pub title: String,
    /// Activation reason: `constant`, `keyword`, or `vector`.
    pub reason: String,
}

impl From<&InjectedEntryView> for UiInjectedEntry {
    fn from(entry: &InjectedEntryView) -> Self {
        Self {
            source: entry.source.clone(),
            id: entry.id.clone(),
            title: entry.title.clone(),
            reason: entry.reason.clone(),
        }
    }
}

/// One action OFFERED to the model for a turn, with whether it actually fired.
/// The UI renders `executed = true` ones in green.
#[derive(Serialize)]
pub(crate) struct UiOfferedAction {
    /// Canonical action id (e.g. `movement.follow_target`).
    pub id: String,
    /// Human label (alias/title), falling back to the id when blank.
    pub title: String,
    /// Why it was injected this turn: `constant`, `keyword`, or `vector`.
    pub reason: String,
    /// `true` when this offered action appears in the turn's executed actions.
    pub executed: bool,
}

/// One action the NPC actually EXECUTED this turn (the chosen `turn_actions`).
/// Always rendered green. Carries the target/params/reason for the strip.
#[derive(Serialize)]
pub(crate) struct UiExecutedAction {
    pub id: String,
    /// Alias if present, else the id (never blank).
    pub label: String,
    pub target: String,
    /// Compact JSON of the action parameters (`""` / `{}` hidden by the UI).
    pub params: String,
    pub reason: String,
    /// `true` when this executed action was also in the offered set (so the UI
    /// can avoid double-counting / show it was a sanctioned choice).
    pub offered: bool,
}

impl UiExecutedAction {
    fn from_action(action: &ActionView, offered: bool) -> Self {
        let label = if action.alias.trim().is_empty() {
            action.id.clone()
        } else {
            action.alias.clone()
        };
        Self {
            id: action.id.clone(),
            label,
            target: action.target.clone(),
            params: action.params.clone(),
            reason: action.reason.clone(),
            offered,
        }
    }
}

/// One message line in the UI chat projection, joined with its per-turn context.
#[derive(Serialize)]
pub(crate) struct UiChatMessage {
    pub id: String,
    pub speaker: String,
    /// First initial for the avatar.
    pub initial: String,
    /// `player`, `npc`, or `system`.
    pub role: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Human-friendly timestamp label (e.g. `Jun 20, 2026 · 9:28 PM`).
    #[serde(skip_serializing_if = "str::is_empty")]
    pub timestamp_label: String,
    // --- per-turn context strip (the key feature) ------------------------
    /// Injected Lore Book entries for this turn.
    pub injected_lore: Vec<UiInjectedEntry>,
    /// Injected Quest Book entries for this turn.
    pub injected_quests: Vec<UiInjectedEntry>,
    /// Actions OFFERED to the model this turn (each flagged executed-or-not).
    pub offered_actions: Vec<UiOfferedAction>,
    /// Actions the NPC actually EXECUTED this turn (always green in the UI).
    pub executed_actions: Vec<UiExecutedAction>,
    /// `true` when no injected/offered/executed context was recorded for this
    /// message (player turns + messages persisted before the feature existed),
    /// so the UI can show a quiet "no context recorded" note instead of nothing.
    pub no_context: bool,
}

/// One NPC conversation thread (everything addressed to / spoken by one NPC).
#[derive(Serialize)]
pub(crate) struct UiChatThread {
    /// Participant id (e.g. `npc:sunny_smiles`); the list row's stable value.
    pub participant_id: String,
    /// NPC display name.
    pub name: String,
    pub initial: String,
    /// Whether the NPC is currently in-scene (present), so the UI can hint it.
    pub present: bool,
    /// Number of visible messages in this thread (list-row count + sort key).
    pub message_count: usize,
    /// Short preview of the most recent message in this thread (list-row
    /// subtitle). Empty when the thread has no messages.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub last_message_preview: String,
    pub messages: Vec<UiChatMessage>,
}

#[derive(Serialize)]
pub(crate) struct UiChatView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_chat_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Every conversation that has chat history — one thread per character with
    /// at least one visible message — sorted by message count (busiest first).
    /// The UI renders these as a persistent conversation-list panel.
    pub threads: Vec<UiChatThread>,
    /// The thread the UI should select by default (the busiest one).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_participant_id: Option<String>,
}

/// Projects one [`MessageView`] into the UI message shape, deriving the
/// injected/offered/executed strip from the message's recorded turn context.
fn project_message(message: &MessageView) -> UiChatMessage {
    // Injected lore / quests come straight through. Offered actions are the
    // injected `actions`; each is marked executed when its id is among the
    // turn's chosen actions.
    let executed_ids: Vec<&str> = message
        .turn_actions
        .iter()
        .map(|action| action.id.trim())
        .filter(|id| !id.is_empty())
        .collect();
    let was_executed = |id: &str| {
        let id = id.trim();
        !id.is_empty() && executed_ids.iter().any(|exec| *exec == id)
    };

    let (injected_lore, injected_quests, offered_actions) = match &message.injected {
        Some(injected) => {
            let lore = injected.lore.iter().map(UiInjectedEntry::from).collect();
            let quests = injected.quests.iter().map(UiInjectedEntry::from).collect();
            let offered = injected
                .actions
                .iter()
                .map(|entry| UiOfferedAction {
                    title: if entry.title.trim().is_empty() {
                        entry.id.clone()
                    } else {
                        entry.title.clone()
                    },
                    executed: was_executed(&entry.id),
                    id: entry.id.clone(),
                    reason: entry.reason.clone(),
                })
                .collect();
            (lore, quests, offered)
        }
        None => (Vec::new(), Vec::new(), Vec::new()),
    };

    // Executed actions: which offered ids fired (to flag `offered` on each).
    let offered_ids: Vec<&str> = message
        .injected
        .as_ref()
        .map(|injected| {
            injected
                .actions
                .iter()
                .map(|entry| entry.id.trim())
                .filter(|id| !id.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let executed_actions = message
        .turn_actions
        .iter()
        .map(|action| {
            let offered = offered_ids.iter().any(|id| *id == action.id.trim());
            UiExecutedAction::from_action(action, offered)
        })
        .collect::<Vec<_>>();

    let no_context = injected_lore.is_empty()
        && injected_quests.is_empty()
        && offered_actions.is_empty()
        && executed_actions.is_empty();

    UiChatMessage {
        id: message.id.clone(),
        speaker: message.speaker_name.clone(),
        initial: message.speaker_initial.clone(),
        role: message.role.clone(),
        text: message.content.clone(),
        timestamp: message.created_at.clone(),
        timestamp_label: message.created_at_label.clone(),
        injected_lore,
        injected_quests,
        offered_actions,
        executed_actions,
        no_context,
    }
}

/// Builds the conversation-thread list for one live chat: one thread per NPC
/// character that has chat history (at least one visible message), each with its
/// visible messages projected. NPCs with no visible messages are omitted so the
/// list only surfaces real conversations.
///
/// This is the "all conversations" set the conversation-list panel renders: it
/// walks every merged participant (roster + presence + active ids, so away
/// characters with past history are included, not just in-scene NPCs) and keeps
/// the ones whose visible-message projection is non-empty. The result is sorted
/// by message count descending so the busiest chat is at the top / the default.
fn build_threads(state: &AppState, live_chat: &LiveChat) -> WebResult<Vec<UiChatThread>> {
    let view = state.repository.live_chat_view(live_chat, None)?;
    let mut threads = Vec::new();
    for participant in &view.participants {
        // Only NPC threads — the chat is "the conversation with the NPCs"; the
        // player isn't a thread of their own. NPC participants are keyed "npc:…";
        // their `kind` is "unknown" in the live data (npc-ness is the id prefix,
        // not the kind field), so filter on the id rather than kind.
        if !participant.id.starts_with("npc:") {
            continue;
        }
        let messages = state
            .repository
            .messages_for_participant(live_chat, &participant.id)?;
        if messages.is_empty() {
            continue;
        }
        let projected = messages.iter().map(project_message).collect::<Vec<_>>();
        let last_message_preview = projected
            .last()
            .map(|message| preview_text(&message.text))
            .unwrap_or_default();
        threads.push(UiChatThread {
            participant_id: participant.id.clone(),
            name: participant.name.clone(),
            initial: participant.initial.clone(),
            present: participant.present,
            message_count: projected.len(),
            last_message_preview,
            messages: projected,
        });
    }

    // Busiest conversation first (most messages at the top of the panel), with
    // present NPCs then name as stable tiebreakers when counts are equal.
    threads.sort_by(|a, b| {
        b.message_count
            .cmp(&a.message_count)
            .then(b.present.cmp(&a.present))
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    Ok(threads)
}

/// A compact single-line preview of a message body for the conversation-list
/// row: collapses whitespace/newlines and truncates to a readable length.
fn preview_text(text: &str) -> String {
    const MAX: usize = 80;
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= MAX {
        return collapsed;
    }
    let truncated: String = collapsed.chars().take(MAX).collect();
    format!("{}…", truncated.trim_end())
}

/// `GET /api/ui/v1/chat/view` — the current live-chat projection.
///
/// Reads the first (active) live chat from `state.repository`, builds one
/// conversation thread per NPC character that has chat history (busiest first),
/// and joins each message with its recorded injected / offered / executed turn
/// context. Returns an empty projection (no live chat / no NPC threads) rather
/// than erroring so the UI can render its empty state.
pub(crate) async fn chat_view(State(state): State<Arc<AppState>>) -> WebResult<Json<UiChatView>> {
    let Some(live_chat) = state.repository.list_live_chats()?.into_iter().next() else {
        return Ok(Json(UiChatView {
            live_chat_id: None,
            title: None,
            threads: Vec::new(),
            default_participant_id: None,
        }));
    };

    let title = if live_chat.title.trim().is_empty() {
        live_chat.id.clone()
    } else {
        live_chat.title.clone()
    };
    let threads = build_threads(&state, &live_chat)?;
    let default_participant_id = threads.first().map(|thread| thread.participant_id.clone());

    Ok(Json(UiChatView {
        live_chat_id: Some(live_chat.id.clone()),
        title: Some(title),
        threads,
        default_participant_id,
    }))
}
