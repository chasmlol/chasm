// Triggers domain — witness memory + event-trigger reactions.
//
// GET  /api/ui/v1/triggers       → settings + the full event-type catalog
// POST /api/ui/v1/triggers/save  → persist settings, returns the fresh view

import { getJson, postJson, UI_API } from "./http";

export interface TriggerCatalogEntryDto {
  /** Event type key, e.g. "item", "theft", "combat". */
  type: string;
  /** True = a witnessed event of this type fires an immediate reaction. */
  enabled: boolean;
  /** Probability (0–100) that an eligible event actually fires. */
  chancePercent: number;
  /** Cooldown between reactions of THIS type (seconds; per type, not global). */
  cooldownSecs: number;
  /** Sight gate: NPCs the player is hidden from don't witness this at all. */
  requireSight: boolean;
  /** True for types discovered from the store (not in the static catalog). */
  dynamic: boolean;
  /** True for types excluded from witness fan-out entirely (conversation). */
  excluded: boolean;
}

export interface TriggersViewDto {
  /** Master switch for the whole witness system (memory + triggers). */
  enabled: boolean;
  /** Restrict witnessing to companions only. */
  companionsOnly: boolean;
  /** Optional cooldown shared across ALL trigger types. */
  globalCooldownEnabled: boolean;
  globalCooldownSecs: number;
  catalog: TriggerCatalogEntryDto[];
}

export interface TriggerRuleSave {
  type: string;
  enabled: boolean;
  chancePercent: number;
  cooldownSecs: number;
  requireSight: boolean;
}

export interface TriggersSaveBody {
  enabled: boolean;
  companionsOnly: boolean;
  globalCooldownEnabled: boolean;
  globalCooldownSecs: number;
  /** Every rule to persist (types not listed become memory-only defaults). */
  triggers: TriggerRuleSave[];
}

export const triggersApi = {
  view: () => getJson<TriggersViewDto>(`${UI_API}/triggers`),
  save: (body: TriggersSaveBody) =>
    postJson<TriggersViewDto>(`${UI_API}/triggers/save`, body),
};
