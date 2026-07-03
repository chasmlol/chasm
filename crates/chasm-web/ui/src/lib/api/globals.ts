// Globals domain of the UI API — the Globals section (global scenario).
//
// The GLOBAL scenario template replaces the per-character card `scenario`
// field: one app-wide `{{macro}}` template, resolved per NPC turn with the
// gamestate macros the mod sends plus backend-computed macros
// (`{{participants}}`). Endpoints (crates/chasm-web/src/ui/globals.rs):
//   - GET  /api/ui/v1/globals/scenario          the effective template +
//     whether it is the built-in default (+ the default text for reset).
//   - PUT  /api/ui/v1/globals/scenario          save the template. Saving the
//     default text clears the override; saving "" disables the component.
//   - POST /api/ui/v1/globals/scenario/preview  resolve a (draft) template
//     through the latest recorded gamestate macros — no generation runs.
//
// IMPORTANT: the UI must NOT call the game/bridge contract (`/api/headless/*`,
// `/api/game/*`). All globals data is exposed under `/api/ui/v1/globals*`.

import { getJson, postJson, putJson, UI_API } from "./http";

/** The effective global scenario template. */
export interface GlobalsScenarioDto {
  /** The template in effect (saved value — may be "" = disabled — else the
   *  built-in default). */
  template: string;
  /** True when no override is saved (the built-in default is in effect). */
  is_default: boolean;
  /** The built-in default template (for "reset to default"). */
  default_template: string;
}

/** Request body for saving the template. */
export interface GlobalsScenarioSaveRequest {
  /** The template text ("" allowed = omit the scenario from prompts). */
  template: string;
}

/** Request body for the resolved preview. */
export interface GlobalsScenarioPreviewRequest {
  /** The draft template to preview; omitted → the saved/effective one. */
  template?: string;
}

/** The resolved preview of a template against the latest recorded macros. */
export interface GlobalsScenarioPreviewDto {
  /** The template with every `{{macro}}` resolved (unknown → empty). */
  resolved: string;
  /** The macro table used: latest recorded + computed (`participants`). */
  macros: Record<string, string>;
  /** ISO timestamp of the turn the recorded table came from, when any. */
  updated_at?: string | null;
  /** Present when the preview degrades (no recorded macros yet, empty
   *  template) or carries a caveat (participants includes every NPC). */
  note?: string;
}

export const globalsApi = {
  /** The effective global scenario template + default. */
  scenario: () => getJson<GlobalsScenarioDto>(`${UI_API}/globals/scenario`),
  /** Save the global scenario template. */
  saveScenario: (body: GlobalsScenarioSaveRequest) =>
    putJson<GlobalsScenarioDto>(`${UI_API}/globals/scenario`, body),
  /** Resolve a (draft) template through the latest recorded macro table. */
  previewScenario: (body: GlobalsScenarioPreviewRequest) =>
    postJson<GlobalsScenarioPreviewDto>(
      `${UI_API}/globals/scenario/preview`,
      body,
    ),
};
