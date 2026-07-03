import type { ComponentType } from "react";
import {
  MessagesSquare,
  Users,
  HeartHandshake,
  UsersRound,
  BookText,
  ScrollText,
  Swords,
  Variable,
  UserRound,
  Globe,
  SlidersHorizontal,
  Layers,
  Cpu,
  AudioLines,
  Mic,
  Megaphone,
  Music,
  Database,
  Server,
  Cable,
  Keyboard,
  Activity,
  Download,
} from "lucide-react";

// ===========================================================================
// Navigation config — the single source of truth for the persistent sidebar.
// The shell renders these groups; each item routes to its screen in the content
// pane (via react-router). Adding/moving a screen is a one-line change here.
//
// `path` is RELATIVE to the app basename (/app). The router (App.tsx) maps the
// same paths to screen components.
// ===========================================================================

export interface NavItem {
  /** Stable key (also the settings `category` for settings items). */
  key: string;
  label: string;
  /** Route path relative to /app (e.g. "chat", "settings/llm"). */
  path: string;
  icon: ComponentType<{ className?: string; strokeWidth?: number }>;
}

export interface NavGroup {
  label: string;
  items: NavItem[];
}

export const NAV_GROUPS: NavGroup[] = [
  {
    label: "Main",
    items: [
      { key: "chat", label: "Chat", path: "chat", icon: MessagesSquare },
      {
        key: "characters",
        label: "Characters Book",
        path: "characters",
        icon: Users,
      },
      {
        key: "companions",
        label: "Companions",
        path: "companions",
        icon: UsersRound,
      },
      { key: "lore", label: "Lore Book", path: "lore", icon: BookText },
      { key: "quest", label: "Quest Book", path: "quest", icon: ScrollText },
      { key: "action", label: "Action Book", path: "action", icon: Swords },
      {
        key: "relationships",
        label: "Relationships",
        path: "relationships",
        icon: HeartHandshake,
      },
      {
        key: "gamestate",
        label: "Gamestate",
        path: "gamestate",
        icon: Variable,
      },
      {
        key: "persona",
        label: "Persona",
        path: "persona",
        icon: UserRound,
      },
    ],
  },
  {
    label: "Globals",
    items: [
      {
        key: "globals-scenario",
        label: "Scenario",
        path: "globals/scenario",
        icon: Globe,
      },
    ],
  },
  {
    label: "Settings",
    items: [
      {
        key: "interface",
        label: "Interface",
        path: "settings/interface",
        icon: SlidersHorizontal,
      },
      {
        key: "profiles",
        label: "Profiles",
        path: "settings/profiles",
        icon: Layers,
      },
      { key: "llm", label: "LLM", path: "settings/llm", icon: Cpu },
      { key: "tts", label: "TTS", path: "settings/tts", icon: AudioLines },
      { key: "stt", label: "STT", path: "settings/stt", icon: Mic },
      { key: "music", label: "Music", path: "settings/music", icon: Music },
      {
        key: "stt-boost",
        label: "Word Boosting",
        path: "settings/stt-boost",
        icon: Megaphone,
      },
      {
        key: "retrieval",
        label: "Retrieval",
        path: "settings/retrieval",
        icon: Database,
      },
      {
        key: "runtimes",
        label: "Runtimes",
        path: "settings/runtimes",
        icon: Server,
      },
      { key: "bridge", label: "Bridge", path: "settings/bridge", icon: Cable },
      {
        key: "hotkeys",
        label: "Hotkeys",
        path: "settings/hotkeys",
        icon: Keyboard,
      },
      {
        key: "tracing",
        label: "Tracing",
        path: "settings/tracing",
        icon: Activity,
      },
      {
        key: "updates",
        label: "Updates",
        path: "settings/updates",
        icon: Download,
      },
    ],
  },
];
