// TTS-specific UI API — hosted-API voice cloning.
//
// The LOCAL voice-cloning flow lives in system.ts (voiceCloneStatus /
// voiceCloneStart, backed by /api/voices/clone) and clones each character's
// in-game voice for the managed local engine. THIS module is the API-provider
// equivalent: it clones a character's recorded reference clip into the ACTIVE
// hosted-TTS provider (ElevenLabs / Cartesia / Inworld) and stores the voice id
// the provider hands back, so subsequent synthesis for that character uses it.
//
// Field names mirror the backend JSON verbatim (snake_case: voice_id) so the
// round-trip is lossless.

import { getJson, postJson, UI_API } from "./http";

/** Result of `POST /tts/clone` — the cloned voice id, or a readable error. */
export interface CloneApiVoiceResult {
  ok: boolean;
  /** The provider-side voice id created for this character (on success). */
  voice_id?: string;
  /** A readable failure reason (e.g. "record a reference first", "no API key"). */
  error?: string;
}

/**
 * `GET /tts/api-voices` — which characters already have a cloned voice id for
 * the ACTIVE hosted-TTS provider. `voices` maps character name → provider voice
 * id.
 */
export interface ApiVoicesView {
  /** The active hosted-TTS provider id these voice ids belong to. */
  provider: string;
  /** Map of character name → cloned provider voice id. */
  voices: Record<string, string>;
}

export const ttsApi = {
  /** Clone one character's recorded reference clip into the active API provider. */
  cloneApiVoice: (character: string) =>
    postJson<CloneApiVoiceResult>(`${UI_API}/tts/clone`, { character }),
  /** Which characters already have a cloned voice for the active API provider. */
  listApiVoices: () => getJson<ApiVoicesView>(`${UI_API}/tts/api-voices`),
};
