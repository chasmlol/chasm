// Events domain — the read-only chronicle of everything that happened in-game.
//
// GET /api/ui/v1/events → the full event log, ascending by seq (oldest first)

import { getJson, UI_API } from "./http";

export interface EventActorDto {
  name: string;
  id: string;
}

export interface EventDto {
  id: string;
  seq: number;
  /**
   * "combat" | "death" | "location" | "item" | "conversation" | "quest" |
   * "level" | "day" | "companion" | "karma" | "world"
   */
  type: string;
  summary: string;
  /** ISO timestamp of when the event was recorded (wall clock). */
  realTime: string;
  /** Pre-formatted in-game clock, e.g. "13:42, 15 Nov 2281". */
  gameTime?: string;
  /** In-game day counter since the save began (1-based). */
  gameDay?: number;
  location?: string;
  actors?: EventActorDto[];
  /** Native NPC keys that actually WITNESSED the event (post sight/scope
   *  filtering). Empty array = it happened unobserved; absent = an event from
   *  before witness tracking. */
  witnessedBy?: string[];
  data?: Record<string, unknown>;
}

export interface EventsViewDto {
  events: EventDto[];
  total: number;
}

export const eventsApi = {
  list: () => getJson<EventsViewDto>(`${UI_API}/events`),
};
