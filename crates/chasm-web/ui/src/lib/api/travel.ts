// Travel domain of the UI API — the "Travel" page (NPC movement system).
//
// Mirrors crates/chasm-web/src/ui/travel.rs:
//   - GET  /api/ui/v1/travel            clock + movement settings + live journeys
//   - POST /api/ui/v1/travel/settings   save the movement settings
//   - POST /api/ui/v1/travel/:id/cancel cancel an in-progress journey

import { getJson, postJson, UI_API } from "./http";

/** The current in-game clock (null before a save loads / the plugin reports it). */
export interface TravelClockDto {
  day: number;
  hour: number;
  label: string;
}

/** The movement-engine settings (the reusable "walk NPCs to places" system). */
export interface MovementSettingsDto {
  enabled: boolean;
  /** Metres of world distance per in-game hour. */
  walkSpeed: number;
  offscreenSimulation: boolean;
  waypointStride: number;
}

/** One active/So-far journey row. */
export interface JourneyDto {
  id: string;
  npcName: string;
  characterName: string;
  destName: string;
  /** waiting | en route | arrived | cancelled | failed. */
  state: string;
  distanceMeters: number;
  /** "Day 4, 8:00 AM" — when they leave / left. */
  departLabel: string;
  /** "Day 4, 10:00 AM" — when they arrive. */
  arriveLabel: string;
  /** 0..100 — fraction of the route covered right now. */
  progress: number;
  createdAtMs: number;
}

export interface TravelViewDto {
  clock: TravelClockDto | null;
  settings: MovementSettingsDto;
  journeys: JourneyDto[];
}

// Local alias — the shared MutationResultDto is exported from ./scheduler; we
// don't re-export it here to avoid a duplicate-name barrel collision.
interface MutationResultDto {
  ok: boolean;
  error: string;
}

export const travelApi = {
  /** The in-game clock + movement settings + all journeys (newest first). */
  view: () => getJson<TravelViewDto>(`${UI_API}/travel`),
  /** Persist the movement settings; returns the fresh (normalized) settings. */
  saveSettings: (form: MovementSettingsDto) =>
    postJson<MovementSettingsDto>(`${UI_API}/travel/settings`, form),
  /** Cancel an in-progress journey (the NPC is left wherever they currently are). */
  cancel: (id: string) =>
    postJson<MutationResultDto>(
      `${UI_API}/travel/${encodeURIComponent(id)}/cancel`,
      {},
    ),
};
