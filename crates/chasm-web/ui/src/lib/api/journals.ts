// Journals domain — the read-only per-NPC private journals written by the
// self-improving-NPC journal pass after every save.
//
// GET /api/ui/v1/journals → every NPC's append-only journal, grouped per character

import { getJson, postJson, UI_API } from "./http";

export interface JournalEntryDto {
  createdAt: string;
  gameTime?: string;
  gameDay?: number;
  text: string;
}

export interface JournalCharacterDto {
  characterId: string;
  characterName: string;
  entries: JournalEntryDto[];
}

export interface JournalsViewDto {
  characters: JournalCharacterDto[];
  lastPassAt?: string;
  passInFlight: boolean;
}

export const journalsApi = {
  list: () => getJson<JournalsViewDto>(`${UI_API}/journals`),
  deleteEntry: (characterId: string, createdAt: string) =>
    postJson<JournalsViewDto>(`${UI_API}/journals/delete-entry`, {
      characterId,
      createdAt,
    }),
};
