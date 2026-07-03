// Persona domain of the UI API — the Persona page.
//
// Three endpoints (crates/chasm-web/src/ui/persona.rs):
//   - GET  /api/ui/v1/persona            the stored player persona: generated
//     description + provenance, the stats snapshot it used, timestamps, and
//     whether a screenshot exists (empty state before the first capture).
//   - GET  /api/ui/v1/persona/image      the last stored screenshot bytes
//     (used directly as an <img src>; 404 before the first image capture).
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
  /** "vision" (described from the screenshot) or "stats_only". */
  source?: string | null;
  /** Which generation path produced it (vision endpoint / main LLM / …). */
  model_note?: string | null;
  /** Last generation error (a previous good description is kept alongside). */
  generation_error?: string | null;
  /** The exact prompt text sent to the LLM for the current description
   *  (absent on records generated before prompt persistence existed). */
  prompt?: string | null;
  /** The stats snapshot used (player_name, level, special, skills, perks,
   *  equipped_weapon, equipped_apparel, location — display strings). */
  stats: Record<string, string | number>;
  /** True when a screenshot is stored (render `${UI_API}/persona/image`). */
  has_image: boolean;
  /** True while a generation task is currently running. */
  generating: boolean;
  /** True when a capture exists (Regenerate is meaningful). */
  has_capture: boolean;
}

/** URL of the stored screenshot; `bust` forces a fresh fetch after updates. */
export function personaImageUrl(bust?: string | null): string {
  const base = `${UI_API}/persona/image`;
  return bust ? `${base}?t=${encodeURIComponent(bust)}` : base;
}

export const personaApi = {
  /** The stored persona view (empty state before the first capture). */
  view: () => getJson<PersonaViewDto>(`${UI_API}/persona`),
  /** Re-run generation from the last received capture (the test hook). */
  regenerate: () => postJson<PersonaViewDto>(`${UI_API}/persona/regenerate`, {}),
};
