// Gamestate domain of the UI API — the Macros page.
//
// Two endpoints (crates/chasm-web/src/ui/gamestate.rs):
//   - GET  /api/ui/v1/gamestate       the LATEST recorded macro table (the
//     `metadata.macros` map the mod extracted on the most recent in-game turn)
//     plus its timestamp, for the read-only table at the top of the page.
//   - POST /api/ui/v1/gamestate/test  the substitution proof: resolve a
//     `{{macro}}` template (against an optional override table, else the latest
//     recorded one), run ONE minimal system+user generation, and return the
//     resolved prompt + the model's reply.
//
// IMPORTANT: the UI must NOT call the game/bridge contract (`/api/headless/*`,
// `/api/game/*`). All gamestate data is exposed under `/api/ui/v1/gamestate*`.

import { getJson, postJson, UI_API } from "./http";

/** The latest recorded gamestate macro table. */
export interface GamestateViewDto {
  /** The live chat the table came from (absent before any chat exists). */
  live_chat_id?: string | null;
  /** ISO timestamp of the turn that recorded the table (absent before the
   *  first macros-bearing turn). */
  updated_at?: string | null;
  /** Flat `{ key: value }` macro table; `{}` before the first turn. */
  macros: Record<string, string>;
}

/** Request body for the substitution tester. */
export interface GamestateTestRequest {
  /** The system-prompt template containing `{{macro}}` placeholders. */
  template: string;
  /** The user turn sent with the resolved system prompt (default "Hello."). */
  user_message?: string;
  /** OPTIONAL override table; omitted → the latest recorded table is used. */
  macros?: Record<string, string>;
}

/** The substitution + generation proof for one template. */
export interface GamestateTestDto {
  /** The template with every `{{macro}}` substituted (unknown → empty). */
  resolved_prompt: string;
  /** The model's reply to the resolved prompt. */
  reply: string;
  /** The macro table the resolution actually used. */
  macros: Record<string, string>;
  /** Present when the table was empty, explaining why macros resolved empty. */
  note?: string;
}

export const gamestateApi = {
  /** The latest recorded macro table + timestamp. */
  view: () => getJson<GamestateViewDto>(`${UI_API}/gamestate`),
  /** Resolve a template and run one minimal test generation with it. */
  test: (body: GamestateTestRequest) =>
    postJson<GamestateTestDto>(`${UI_API}/gamestate/test`, body),
};
