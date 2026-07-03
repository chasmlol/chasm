// Provider domain of the UI API — the per-capability provider picker + config.
//
// Each of LLM / STT / TTS picks a PROVIDER: "local" (the managed runtime) or a
// hosted API (OpenAI-compatible, ElevenLabs, …). Selecting an API provider
// reveals its config (key / model / base URL / voice). Backed by
// crates/chasm-web/src/ui/providers.rs.
//
// Field names mirror the backend JSON verbatim (snake_case: local_runtime,
// needs_base_url, default_base_url, api_key) so the round-trip is lossless.

import { getJson, postJson, UI_API } from "./http";

/** The three capabilities that pick a provider. */
export type ProviderCapability = "llm" | "stt" | "tts";

/** A provider is either the managed local runtime or a hosted API. */
export type ProviderKind = "local" | "api";

/** A voice option offered by a hosted TTS provider. */
export interface ProviderVoice {
  id: string;
  label: string;
}

/** The saved per-provider config (all optional; unset fields come back empty). */
export interface ProviderConfig {
  api_key: string;
  model: string;
  base_url: string;
  voice: string;
  /** OpenRouter routing preference (`speed`/`balanced`/`price`); empty elsewhere. */
  routing: string;
}

/** One provider card in the picker. */
export interface ProviderDto {
  id: string;
  name: string;
  kind: ProviderKind;
  blurb: string;
  /** Suggested model ids (datalist hints; editable — hosted ids rotate). */
  models: string[];
  /** Suggested voices (TTS providers). */
  voices: ProviderVoice[];
  needs_base_url: boolean;
  needs_voice: boolean;
  default_base_url: string;
  default_model: string;
  /** OpenRouter routing choices (empty for other providers) → Price/Balanced/Speed. */
  routing_options: ProviderVoice[];
  config: ProviderConfig;
}

/** The managed local runtime backing the "local" provider for this capability. */
export interface LocalRuntimeDto {
  name: string;
  installed: boolean;
  hint: string;
}

/** The `GET /providers/:capability` payload. */
export interface ProvidersView {
  capability: ProviderCapability;
  /** The selected provider id. First provider is always "local". */
  selected: string;
  local_runtime: LocalRuntimeDto;
  providers: ProviderDto[];
}

/** The editable config posted to `.../config` (omit a field to leave it unchanged). */
export interface ProviderConfigForm {
  provider: string;
  apiKey?: string;
  model?: string;
  baseUrl?: string;
  voice?: string;
  routing?: string;
}

export const providersApi = {
  get: (capability: ProviderCapability) =>
    getJson<ProvidersView>(`${UI_API}/providers/${capability}`),
  select: (capability: ProviderCapability, provider: string) =>
    postJson<ProvidersView>(`${UI_API}/providers/${capability}/select`, {
      provider,
    }),
  saveConfig: (capability: ProviderCapability, form: ProviderConfigForm) =>
    postJson<ProvidersView>(`${UI_API}/providers/${capability}/config`, form),
};
