import { create } from "zustand";

// Ephemeral, client-only UI state (Zustand). Navigation now lives in the URL
// (react-router) and server data lives in TanStack Query, so this store is for
// transient view state only (e.g. a screen's open panel, a draft filter).
//
// It's intentionally empty for now — fill agents add their own slices here when
// a screen needs cross-component ephemeral state. Kept so the seam exists.
interface UiState {
  /** Placeholder so the store type isn't empty; remove when a real slice lands. */
  _ready: true;
}

export const useUiStore = create<UiState>(() => ({
  _ready: true,
}));
