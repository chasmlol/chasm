// Persona domain of the UI API — the Persona page.
//
// Two endpoints (crates/chasm-web/src/ui/persona.rs):
//   - GET  /api/ui/v1/persona            the stored player persona: generated
//     description + provenance, the character-data snapshot it used, and
//     timestamps (empty state before the first capture).
//   - POST /api/ui/v1/persona/regenerate re-runs generation from the last
//     received capture and returns the refreshed view (the manual test hook).
//
// The mod uploads captures on the game transport (POST /api/game/v1/persona);
// the UI NEVER calls `/api/headless/*` or `/api/game/*`.

import { getJson, postJson, UI_API } from "./http";

/** The stored player persona (all fields absent before the first capture). */
export interface PersonaViewDto {
  /** The generated third-person description. */
  description?: string | null;
  /** ISO timestamp of the last successful generation. */
  generated_at?: string | null;
  /** ISO timestamp of the capture the description came from. */
  captured_at?: string | null;
  /** "game_data" (older records may carry "vision" / "stats_only"). */
  source?: string | null;
  /** Human note on the generation outcome. */
  model_note?: string | null;
  /** Last generation error (a previous good description is kept alongside). */
  generation_error?: string | null;
  /** The exact prompt text sent to the LLM for the current description
   *  (absent on records generated before prompt persistence existed). */
  prompt?: string | null;
  /** The character-data snapshot used (stats + appearance display strings:
   *  player_name, level, special, skills, perks, equipped_weapon,
   *  equipped_apparel, location, sex, race, hair_*, eye_color, facial_hair). */
  stats: Record<string, string | number>;
  /** True while a generation task is currently running. */
  generating: boolean;
  /** True when a capture exists (Regenerate is meaningful). */
  has_capture: boolean;
  /** The user's custom addition — a free-text paragraph appended to the
   *  persona at injection, persisted separately so it survives regeneration.
   *  Empty string when never set. */
  custom_note: string;
}

export const personaApi = {
  /** The stored persona view (empty state before the first capture). */
  view: () => getJson<PersonaViewDto>(`${UI_API}/persona`),
  /** Re-run generation from the last received capture (the test hook). */
  regenerate: () => postJson<PersonaViewDto>(`${UI_API}/persona/regenerate`, {}),
  /** Persist the custom addition (an empty string clears it). Survives
   *  regeneration; applies on the next NPC turn with no restart. */
  setCustomNote: (note: string) =>
    postJson<PersonaViewDto>(`${UI_API}/persona/custom`, { note }),
};
