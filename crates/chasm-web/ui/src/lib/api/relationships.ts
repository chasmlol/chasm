// Relationships domain — the Gamemaster's directional character→target ledger.
//
// GET  /api/ui/v1/relationships       → the full grouped ledger
// POST /api/ui/v1/relationships/save  → edit/clear ONE pair (empty text clears;
//                                       ids ride in the body — they have spaces)

import { getJson, postJson, UI_API } from "./http";

export interface RelationshipEntryDto {
  targetId: string;
  targetName: string;
  /** "player" | "npc" */
  targetKind: string;
  text: string;
  createdAt?: string;
  updatedAt?: string;
}

export interface RelationshipCharacterDto {
  characterId: string;
  characterName: string;
  entries: RelationshipEntryDto[];
}

export interface RelationshipsViewDto {
  characters: RelationshipCharacterDto[];
  lastPassAt?: string;
  passInFlight: boolean;
}

export const relationshipsApi = {
  list: () => getJson<RelationshipsViewDto>(`${UI_API}/relationships`),
  save: (characterId: string, targetId: string, text: string) =>
    postJson<RelationshipsViewDto>(`${UI_API}/relationships/save`, {
      characterId,
      targetId,
      text,
    }),
};
