// Per-engine CONFIGURATION domain of the UI API (LLM sampling, TTS tuning +
// volumes, STT params, Retrieval tuning).
//
// Distinct from models.ts (which owns the model PICKER). These endpoints surface
// the per-engine knobs the legacy settings pages exposed (temperature, top-p,
// voice volumes, min-score, …). Backed by crates/chasm-web/src/ui/config.rs,
// which reuses the legacy apply/normalize path so saved values round-trip exactly.

import { getJson, postJson, UI_API } from "./http";

/** LLM generation sampling. Mirrors `LlmConfig` (ui/config.rs). */
export interface LlmConfig {
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repeat_penalty: number;
  max_tokens: number;
  n_ctx: number;
  seed: number;
}

/** TTS volumes (percent, 100 = unity) + synthesis tuning. Mirrors `TtsConfig`. */
export interface TtsConfig {
  npc_volume_pct: number;
  admin_volume_pct: number;
  lead_in_ms: number;
  trailing_ms: number;
  sentence_gap_ms: number;
  gain_db: number;
  temperature: number;
  lsd_decode_steps: number;
  eos_threshold: number;
  noise_clamp: number;
  max_tokens: number;
  frames_after_eos: number;
}

/** STT language / prompt / timeout + word-boosting. Mirrors `SttConfig`. */
export interface SttConfig {
  language: string;
  prompt: string;
  timeout_ms: number;
  /** Master switch for word boosting (Parakeet only). */
  boost_vocab: boolean;
  /** Include character-book names in the boost vocabulary. */
  boost_characters: boolean;
  /** Include lorebook entry names + keys in the boost vocabulary. */
  boost_lore: boolean;
  /** Read-only: proper nouns actually being boosted (respects the toggles). */
  boosted_word_count?: number;
  /** Read-only: distinct character-derived terms available. */
  boosted_character_count?: number;
  /** Read-only: distinct lore-derived terms available. */
  boosted_lore_count?: number;
  /** Read-only: a small preview of the boosted words. */
  boost_sample?: string[];
}

/** Retrieval toggles / tiers / limits / scores. Mirrors `RetrievalConfig`. */
export interface RetrievalConfig {
  enabled: boolean;
  chat_memory_enabled: boolean;
  lore_semantic_enabled: boolean;
  action_semantic_enabled: boolean;
  quest_semantic_enabled: boolean;
  reranker_enabled: boolean;
  reranker_tier: string;
  execution: string;
  top_k: number;
  candidates: number;
  min_score: number;
  action_min_score: number;
  chat_memory_limit: number;
  lore_limit: number;
  quest_limit: number;
}

/** The config envelope: exactly one field is populated per domain. */
export interface ConfigDto {
  llm?: LlmConfig;
  tts?: TtsConfig;
  stt?: SttConfig;
  retrieval?: RetrievalConfig;
}

export const configApi = {
  get: (domain: "llm" | "tts" | "stt" | "retrieval") =>
    getJson<ConfigDto>(`${UI_API}/config/${domain}`),
  saveLlm: (config: LlmConfig) =>
    postJson<ConfigDto>(`${UI_API}/config/llm`, config),
  saveTts: (config: TtsConfig) =>
    postJson<ConfigDto>(`${UI_API}/config/tts`, config),
  saveStt: (config: SttConfig) =>
    postJson<ConfigDto>(`${UI_API}/config/stt`, config),
  saveRetrieval: (config: RetrievalConfig) =>
    postJson<ConfigDto>(`${UI_API}/config/retrieval`, config),
};
