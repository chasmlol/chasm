// Companions domain of the UI API — authored FNV followers.
//
// Mirrors crates/chasm-web/src/ui/companions.rs: pool status (the plugin's
// registry merged with card/voice state), create (card + voice clip + plugin
// command), and per-slot ops relayed to the game.

import { getJson, postJson, UI_API } from "./http";

export interface CompanionSlotDto {
  slot: number;
  claimed: boolean;
  npcKey: string;
  name: string;
  characterName: string;
  voice: string;
  /** Body-variant id (declared by the game profile, reported by the plugin). */
  body: string;
  faceDesigned: boolean;
  waiting: boolean;
  status: string; // unclaimed | claimed | spawned | dismissed
  appearanceSaved: boolean;
  hasCard: boolean;
  voiceStatus: string; // none | reference | cloning | cloned | failed
}

export interface CompanionAckDto {
  requestId: string;
  ok: boolean;
  error: string;
  op: string;
  slot: number;
  npcKey: string;
}

export interface CompanionBodyDto {
  id: string;
  label: string;
  slots: number;
  free: number;
}

export interface CompanionsViewDto {
  /** False when the active game profile declares no companion support. */
  enabled: boolean;
  inGameFaceDesign: boolean;
  faceDesignHint: string;
  voiceHint: string;
  bodies: CompanionBodyDto[];
  registryFound: boolean;
  registryRev: number;
  slots: CompanionSlotDto[];
  acks: CompanionAckDto[];
}

export interface CreateCompanionDto {
  name: string;
  description: string;
  personality: string;
  firstMessage: string;
  exampleDialogue: string;
  systemPrompt: string;
  /** Body-variant id from the view's `bodies`. */
  body: string;
  faceDesign: boolean;
  /** Voice clip bytes as base64 (WAV/FLAC/OGG recommended, ~10–20s). */
  voiceBase64: string;
}

export interface CreateCompanionResponseDto {
  requestId: string;
  cardId: string;
  voiceSaved: boolean;
  cloneStarted: boolean;
}

export type CompanionOp =
  | "summon"
  | "dismiss"
  | "despawn"
  | "release"
  | "face_design"
  | "rename";

export const companionsApi = {
  view: () => getJson<CompanionsViewDto>(`${UI_API}/companions`),
  create: (body: CreateCompanionDto) =>
    postJson<CreateCompanionResponseDto>(`${UI_API}/companions`, body),
  op: (slot: number, op: CompanionOp, name?: string) =>
    postJson<{ requestId: string }>(`${UI_API}/companions/${slot}/op`, {
      op,
      name: name ?? "",
    }),
};
