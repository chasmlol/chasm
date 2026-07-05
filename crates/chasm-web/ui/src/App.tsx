import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
} from "react-router-dom";

import { AppShell } from "@/components/AppShell";
import { InterfaceSettings } from "@/screens/InterfaceSettings";
import { Chat } from "@/screens/Chat";
import { CharactersBook } from "@/screens/books/CharactersBook";
import { Companions } from "@/screens/Companions";
import { LoreBook } from "@/screens/books/LoreBook";
import { QuestBook } from "@/screens/books/QuestBook";
import { ActionBook } from "@/screens/books/ActionBook";
import { Gamestate } from "@/screens/Gamestate";
import { Schedule } from "@/screens/Schedule";
import { Travel } from "@/screens/Travel";
import { Relationships } from "@/screens/Relationships";
import { Events } from "@/screens/Events";
import { Persona } from "@/screens/Persona";
import { Globals } from "@/screens/Globals";
import { Profiles } from "@/screens/settings/Profiles";
import { Llm } from "@/screens/settings/Llm";
import { Tts } from "@/screens/settings/Tts";
import { Stt } from "@/screens/settings/Stt";
import { SttBoost } from "@/screens/settings/SttBoost";
import { Music } from "@/screens/settings/Music";
import { Retrieval } from "@/screens/settings/Retrieval";
import { Runtimes } from "@/screens/settings/Runtimes";
import { Bridge } from "@/screens/settings/Bridge";
import { Hotkeys } from "@/screens/settings/Hotkeys";
import { Tracing } from "@/screens/settings/Tracing";
import { Updates } from "@/screens/settings/Updates";

// ===========================================================================
// App routing. The SPA is served under /app, so the router uses
// `basename="/app"`. The single AppShell route renders the persistent sidebar +
// an <Outlet/>; every child route swaps ONLY the content pane. The path map
// mirrors NAV_GROUPS (src/lib/nav.tsx) one-to-one.
//
// Fill agents: your screen is already routed here. Replace the screen's body —
// do NOT add routes or navigation; the sidebar (NAV_GROUPS) is the only nav.
// ===========================================================================
export function App() {
  return (
    <BrowserRouter basename="/app">
      <Routes>
        <Route element={<AppShell />}>
          {/* Default → Chat */}
          <Route index element={<Navigate to="chat" replace />} />

          {/* Main */}
          <Route path="chat" element={<Chat />} />
          <Route path="characters" element={<CharactersBook />} />
          <Route path="companions" element={<Companions />} />
          <Route path="lore" element={<LoreBook />} />
          <Route path="quest" element={<QuestBook />} />
          <Route path="action" element={<ActionBook />} />
          <Route path="relationships" element={<Relationships />} />
          <Route path="events" element={<Events />} />
          <Route path="gamestate" element={<Gamestate />} />
          <Route path="schedule" element={<Schedule />} />
          <Route path="travel" element={<Travel />} />
          <Route path="persona" element={<Persona />} />

          {/* Globals */}
          <Route path="globals">
            <Route index element={<Navigate to="scenario" replace />} />
            <Route path="scenario" element={<Globals />} />
          </Route>

          {/* Settings */}
          <Route path="settings">
            <Route index element={<Navigate to="interface" replace />} />
            <Route path="interface" element={<InterfaceSettings />} />
            <Route path="profiles" element={<Profiles />} />
            <Route path="llm" element={<Llm />} />
            <Route path="tts" element={<Tts />} />
            <Route path="stt" element={<Stt />} />
            <Route path="stt-boost" element={<SttBoost />} />
            <Route path="music" element={<Music />} />
            <Route path="retrieval" element={<Retrieval />} />
            <Route path="runtimes" element={<Runtimes />} />
            <Route path="bridge" element={<Bridge />} />
            <Route path="hotkeys" element={<Hotkeys />} />
            <Route path="tracing" element={<Tracing />} />
            <Route path="updates" element={<Updates />} />
          </Route>

          {/* Unknown → Chat */}
          <Route path="*" element={<Navigate to="chat" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
