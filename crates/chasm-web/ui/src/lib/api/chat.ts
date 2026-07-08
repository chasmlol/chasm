// Live-chat domain of the UI API.
//
// Read-only projection of the active live chat for the Chat screen. The backend
// (crates/chasm-web/src/ui/chat.rs) projects from the existing repository
// view-builder, joining each message with the per-turn context it recorded at
// generation time: injected Lore/Quest entries, the actions OFFERED to the
// model, and which of those actually EXECUTED.
//
// IMPORTANT: the UI must NOT call the game/bridge contract (`/api/headless/*`,
// `/api/game/*`). All chat data is exposed READ-ONLY under `/api/ui/v1/chat/*`.

import { getJson, postJson, UI_API } from "./http";

/** One injected world-info entry (lore / quest) shown in a message strip. */
export interface InjectedEntryDto {
  /** "lore" | "quest" | "action". */
  source: string;
  id: string;
  title: string;
  /** Activation reason: "constant" | "keyword" | "vector". */
  reason: string;
}

/** An action OFFERED to the model this turn, flagged with whether it fired. */
export interface OfferedActionDto {
  id: string;
  title: string;
  reason: string;
  /** True when this offered action appears in the turn's executed actions. */
  executed: boolean;
}

/** An action the NPC actually EXECUTED this turn (rendered green). */
export interface ExecutedActionDto {
  id: string;
  label: string;
  target: string;
  /** Compact JSON of parameters ("" / "{}" hidden by the UI). */
  params: string;
  reason: string;
  /** True when this executed action was also in the offered set. */
  offered: boolean;
}

/** One message line joined with its per-turn context strip. */
export interface ChatMessageDto {
  id: string;
  speaker: string;
  initial: string;
  role: "player" | "npc" | "system" | string;
  text: string;
  timestamp?: string;
  timestamp_label?: string;
  injected_lore: InjectedEntryDto[];
  injected_quests: InjectedEntryDto[];
  offered_actions: OfferedActionDto[];
  executed_actions: ExecutedActionDto[];
  /** True when no injected/offered/executed context was recorded. */
  no_context: boolean;
  /** True when this NPC turn was generated while the NPC was in combat. */
  in_combat: boolean;
  /** Display names of who the NPC was fighting this turn (empty unless in combat). */
  combat_with: string[];
  /** True for witnessed-event narration lines (rendered dim/italic, not as dialogue). */
  witnessed: boolean;
}

/** One NPC conversation thread (everything spoken by / to one NPC). */
export interface ChatThreadDto {
  participant_id: string;
  name: string;
  initial: string;
  present: boolean;
  message_count: number;
  /** Short preview of the thread's most recent message (list-row subtitle). */
  last_message_preview?: string;
  messages: ChatMessageDto[];
}

export interface ChatViewDto {
  live_chat_id: string | null;
  title?: string;
  threads: ChatThreadDto[];
  default_participant_id?: string;
}

export const chatApi = {
  /** The current live-chat projection (threads + per-message context). */
  view: () => getJson<ChatViewDto>(`${UI_API}/chat/view`),
  /** Fully clear a character's chat history — deletes their messages AND scrubs
   *  them from save-sync checkpoints, so a game load can't restore it. */
  clearHistory: (liveChatId: string, participantId: string) =>
    postJson<unknown>(
      `/live/${encodeURIComponent(liveChatId)}/${encodeURIComponent(participantId)}/clear-history`,
      {},
    ),
};
