// AI-models domain of the UI API (LLM / TTS / STT / Retrieval settings).
//
// STUB. The four AI settings screens render their model lists via the shared
// <ModelPicker> component (src/components/ModelPicker.tsx). The endpoints below
// return empty/placeholder JSON until a fill agent implements them in
// crates/chasm-web/src/ui/models.rs (reusing the existing core view
// builders: llm_models_panel_view, stt_panel_view, retrieval_panel_view, …).
//
// Fill agents: shape each screen's payload to what your ModelPicker mapping
// needs; keep them per-domain so LLM/TTS/STT/Retrieval stay isolated.

import { getJson, postJson, UI_API } from "./http";

/** The status-pill tone, mirrored from the React `StatusTone`. */
export type ModelStatusTone = "ok" | "warn" | "error" | "busy" | "idle";

/** An explicit status pill (download/active/running) for one model card. */
export interface ModelStatusDto {
  tone: ModelStatusTone;
  label: string;
}

/** A model as surfaced to the ModelPicker (one card). Backend-shaped subset. */
export interface ModelDto {
  id: string;
  name: string;
  description?: string;
  installed: boolean;
  recommended?: boolean;
  /** Free-form meta chips (size / VRAM / params). */
  meta?: { label: string; value: string }[];
  /** Explicit status pill; when omitted the picker derives one. */
  status?: ModelStatusDto;
}

/** A model-settings payload: the catalog + the selected id + folder path. */
export interface ModelSettingsDto {
  models: ModelDto[];
  selected_id?: string;
  /** The "drop files here" folder for this category. */
  folder?: string;
}

/** The AI settings domains served by ModelPicker (runtime = the LLM runtime picker). */
export type ModelDomain =
  | "llm"
  | "tts"
  | "stt"
  | "retrieval"
  | "runtime"
  | "music";

export const modelsApi = {
  get: (domain: ModelDomain) =>
    getJson<ModelSettingsDto>(`${UI_API}/models/${domain}`),
  select: (domain: ModelDomain, id: string) =>
    postJson<ModelSettingsDto>(`${UI_API}/models/${domain}/select`, { id }),
  download: (domain: ModelDomain, id: string) =>
    postJson<ModelSettingsDto>(`${UI_API}/models/${domain}/download`, { id }),
};
