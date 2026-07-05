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

/** One dynamic-scenario variant: user config + fixed catalog facts. */
export interface GlobalsScenarioVariantDto {
  /** Stable id ("companion", "traveling", …) — also the condition selector. */
  id: string;
  /** Display label ("Companion, sneaking"). */
  label: string;
  /** Read-only description of the gamestate that triggers this variant. */
  condition_hint: string;
  enabled: boolean;
  /** Selection priority (higher wins; ties break by catalog order). */
  priority: number;
  /** The variant's template ("" = fall through to the next match). */
  template: string;
  /** The shipped template (per-variant reset). */
  default_template: string;
  /** The shipped priority. */
  default_priority: number;
}

/** The per-variant config sent back on save (catalog facts stay server-side). */
export interface GlobalsScenarioVariantConfig {
  id: string;
  enabled: boolean;
  priority: number;
  template: string;
}

/** The gamestate flags of the preview state-picker (all default false). */
export interface GlobalsScenarioPreviewState {
  teammate?: boolean;
  following?: boolean;
  waiting?: boolean;
  sneaking?: boolean;
  player_sneaking?: boolean;
  weapon_drawn?: boolean;
  player_weapon_drawn?: boolean;
  sitting?: boolean;
  player_swimming?: boolean;
  traveling?: boolean;
}

/** The effective global scenario template + dynamic variants. */
export interface GlobalsScenarioDto {
  /** The DEFAULT variant's template (saved value — may be "" = disabled —
   *  else the built-in default). */
  template: string;
  /** True when no override is saved (the built-in default is in effect). */
  is_default: boolean;
  /** The built-in default template (for "reset to default"). */
  default_template: string;
  /** The dynamic-scenario variants, in catalog order. */
  variants: GlobalsScenarioVariantDto[];
}

/** Request body for saving the template (and, optionally, the variants). */
export interface GlobalsScenarioSaveRequest {
  /** The template text ("" allowed = omit the scenario from prompts). */
  template: string;
  /** When present, replaces the stored variant config wholesale. */
  variants?: GlobalsScenarioVariantConfig[];
}

/** Request body for the resolved preview. */
export interface GlobalsScenarioPreviewRequest {
  /** The draft default template to preview; omitted → the saved one. */
  template?: string;
  /** Draft variant configs (the editor state), used with `state`. */
  variants?: GlobalsScenarioVariantConfig[];
  /** State-picker flags: when present the backend runs real variant
   *  selection against them and previews the winner. */
  state?: GlobalsScenarioPreviewState;
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
  /** The variant the state-picker selection matched (id + label); only
   *  present when the request carried a `state`. */
  variant_id?: string;
  variant_label?: string;
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
