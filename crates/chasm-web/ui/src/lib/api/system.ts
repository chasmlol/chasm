// System + settings domain of the UI API.
//
// Owns: the Interface settings round-trip (the one fully-wired screen), the
// settings nav, and the shared connection status. Other settings categories
// (Profiles / Bridge / Tracing) add their endpoints + types HERE as they're
// filled in.

import { getJson, postJson, UI_API } from "./http";

export interface SettingsNavItem {
  key: string;
  label: string;
  active: boolean;
}

export interface SettingsNavGroup {
  label: string;
  items: SettingsNavItem[];
}

export interface ThemeOption {
  id: string;
  label: string;
  selected: boolean;
  bg: string;
  panel: string;
}

export interface AccentOption {
  value: string;
  label: string;
  selected: boolean;
}

export interface SelectOption {
  value: string;
  label: string;
  selected: boolean;
}

/** Mirrors `InterfacePanelView` (crates/chasm-core/src/settings.rs). */
export interface InterfacePanel {
  themes: ThemeOption[];
  accent: string;
  accents: AccentOption[];
  densities: SelectOption[];
  density: string;
  font_scale: number;
  font_scale_min: number;
  font_scale_max: number;
  font_scale_step: number;
  reduce_motion: boolean;
  show_timestamps: boolean;
  show_prompt_panel: boolean;
}

/** The subset of `SettingsPageView` the React Settings shell consumes. */
export interface SettingsPage {
  category: string;
  nav_groups: SettingsNavGroup[];
  settings_path: string;
  interface: InterfacePanel;
}

/** The editable Interface settings posted back to the save endpoint. */
export interface InterfaceForm {
  theme: string;
  accent: string;
  density: string;
  font_scale: number;
  reduce_motion: boolean;
  show_timestamps: boolean;
  show_prompt_panel: boolean;
}

/** Mirrors `GET /api/app/version` — the Settings → Updates check. */
export interface AppVersion {
  current: string;
  latest: string | null;
  update_available: boolean;
  download_url: string | null;
  release_url: string | null;
  /** "nightly" (commit-tracked CI build) or "release" (semver fallback). */
  channel: "nightly" | "release" | string;
  current_commit: string | null;
  latest_commit: string | null;
}

/** Mirrors `POST /api/app/update/install` — the one-click self-update trigger. */
export interface AppUpdateResult {
  started: boolean;
  error?: string;
  /** Whether the updater will also refresh the NVBridge mod in MO2. */
  bridge_update?: boolean;
}

/** Mirrors `GET /connection/status`. */
export interface ConnectionStatus {
  connected: boolean;
  phase: "disconnected" | "starting" | "connected" | "stopping" | string;
  last_seen_secs: number | null;
}

// --- Profiles (GET /api/ui/v1/profiles, POST .../profiles/select) ----------

/** One profile card. Mirrors `UiProfile` (crates/.../ui/profiles.rs). */
export interface UiProfile {
  id: string;
  name: string;
  description: string;
  initials: string;
  active: boolean;
  character_count: number;
  lorebook_count: number;
  quest_count: number;
  action_count: number;
}

/** Mirrors `UiProfilesView`. */
export interface ProfilesView {
  active_id: string;
  profiles_dir: string;
  profiles: UiProfile[];
}

// --- Bridge (GET /api/ui/v1/settings/bridge, POST .../bridge/save) ---------

/** The editable bridge config fields. Mirrors `BridgeConfig`. */
export interface BridgeConfig {
  helper_config: string;
  helper_script: string;
  helper_node: string;
  helper_cwd: string;
  trace_dir: string;
}

/** The read-only connection projection. Mirrors `BridgeConnection`. */
export interface BridgeConnection {
  connected: boolean;
  phase: string;
  last_seen_secs: number | null;
}

/** Mirrors `UiBridgeView`. */
export interface BridgeView {
  settings_path: string;
  config: BridgeConfig;
  connection: BridgeConnection;
}

// --- Tracing (GET /api/ui/v1/traces, GET .../traces/:id) -------------------

/** One row in the recent-traces list. Mirrors `TraceListEntry`. */
export interface TraceListEntry {
  request_id: string;
  started_at: string;
  total_ms: number;
  stage_count: number;
}

/** Mirrors the `GET /traces` envelope. */
export interface TracesList {
  traceDir: string;
  traces: TraceListEntry[];
}

/** One parsed stage. Mirrors `TraceStage`. */
export interface TraceStage {
  index: number;
  name: string;
  at: string;
  elapsed_ms: number;
  duration_ms: number;
  group: string;
  is_error: boolean;
  fields: [string, string][];
}

/** One summary metric. Mirrors `TraceMetric`. */
export interface TraceMetric {
  label: string;
  value: string;
  primary: boolean;
}

/** Mirrors the `GET /traces/:id` detail. */
export interface TraceDetail {
  requestId: string;
  startedAt: string;
  totalMs: number;
  stageCount: number;
  stages: TraceStage[];
  summary: { metrics: TraceMetric[] };
  llm: unknown | null;
}

/** Per-service status for the sidebar model lights. Values are StatusTone-ish:
 * "ok" = up/loaded, "idle" = down/not loaded. */
export interface StackStatus {
  llm: string;
  stt: string;
  tts: string;
  embedder: string;
  reranker: string;
}

/** One character's voice-clone status for the selected engine. */
export interface VoiceCloneCharacter {
  name: string;
  status: string; // cloned | cloning | failed | pending
  status_label: string;
}

/** Voice-clone panel state, scoped to the currently-selected TTS engine. */
export interface VoiceCloneView {
  has_profile: boolean;
  profile_id: string;
  profile_name: string;
  engine_id: string;
  engine_label: string;
  characters: VoiceCloneCharacter[];
  any_cloning: boolean;
  cloned_count: number;
}

export const systemApi = {
  settings: (category: string) =>
    getJson<SettingsPage>(`${UI_API}/settings/${category}`),
  saveInterface: (form: InterfaceForm) =>
    postJson<SettingsPage>(`${UI_API}/settings/interface/save`, form),
  /** Shared with the backend/desktop shell; NOT under /api/ui. */
  connectionStatus: () => getJson<ConnectionStatus>(`/connection/status`),
  /** Per-service model lights. NOT under /api/ui; top-level router. */
  stackStatus: () => getJson<StackStatus>(`/api/stack/status`),
  /** Manually start the whole model stack (LLM+STT, TTS, retriever). */
  startStack: () => postJson<{ started: boolean }>(`/api/stack/start`, {}),
  /** Per-character voice-clone status for the selected TTS engine. */
  voiceCloneStatus: () => getJson<VoiceCloneView>(`/api/voices/clone`),
  /** Start cloning the active profile's voices with the selected engine. */
  voiceCloneStart: () => postJson<VoiceCloneView>(`/api/voices/clone`, {}),
  /** The update check. NOT under /api/ui; served by the top-level router. */
  appVersion: () => getJson<AppVersion>(`/api/app/version`),
  /** One-click self-update: backend downloads the installer, runs it silently, relaunches. */
  installUpdate: () =>
    postJson<AppUpdateResult>(`/api/app/update/install`, {}),

  // Profiles
  profiles: () => getJson<ProfilesView>(`${UI_API}/profiles`),
  selectProfile: (id: string) =>
    postJson<ProfilesView>(`${UI_API}/profiles/select`, { id }),

  // Bridge
  bridge: () => getJson<BridgeView>(`${UI_API}/settings/bridge`),
  saveBridge: (config: BridgeConfig) =>
    postJson<BridgeView>(`${UI_API}/settings/bridge/save`, config),

  // Tracing (read-only)
  traces: () => getJson<TracesList>(`${UI_API}/traces`),
  trace: (id: string) => getJson<TraceDetail>(`${UI_API}/traces/${id}`),
};
