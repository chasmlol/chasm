# chasm — Agentic NPC Engine

> A local-first, **game-agnostic** AI backend for agentic game characters. They talk, they listen, they **remember what they personally saw** — and they **act** in the world on their own. Bring any game (or build a new one) and connect it to chasm.

**chasm is the backend, not a game.** It turns game characters into AI agents: walk up to a character, speak to them with your own voice, and they reply — in *their* own cloned voice, generated live by a local LLM that knows who they are, where they are, and what's happening around them. Then they go further and **do things in the world**: follow you, hand over an item, loot a body, draw on someone, walk across town to meet you at a set time, play a song. Over a playthrough they build memories of what they witnessed, form opinions, and — genuinely unscripted — start developing **habits of their own**.

It runs entirely on your machine by default. No cloud, no API keys, no account.

**chasm is a universal connector.** It doesn't ship a game — it connects to one. It exposes a headless HTTP API, and to use it you (or anyone) write a small **bridge mod** for a game: something that tells chasm what's happening in-world and plays back what chasm returns. That same API can just as easily power an **entirely new game** built around AI characters. Any game, any engine — if it can speak HTTP, it can plug into chasm.

The reference bridge is **[Chasm Bridge FNV](https://github.com/chasmlol/chasm-bridge-fnv)** — a Fallout: New Vegas mod that drops live AI dialogue, voice, and autonomous actions into a real, playable game. Everything below is real today through that bridge.

```
   YOUR GAME  ──►  YOUR BRIDGE MOD  ──►  chasm  ──►  LLM · TTS · STT · music · retrieval
  (any engine)     (you write this)      (backend)       (local models, offline)
```

---

## What makes the characters *agentic*

Most "AI NPC" demos stop at conversation. chasm treats every spoken line as an agent turn — and keeps working between turns, too:

1. **Hear** — the player's microphone audio is transcribed locally (STT); chasm works out *which* nearby character is being addressed.
2. **Think** — it pulls that character's card, relevant lore, active quests, live gamestate, their personal memories, and the actions they're allowed to take, assembles a prompt, and asks the local LLM for a reply.
3. **Speak** — the reply is synthesized in the character's *own* cloned voice (streaming TTS), sentence by sentence, so they start talking almost immediately.
4. **Act** — alongside the line, the NPC can **discover and perform actions**: search for an ability it needs, follow, travel somewhere, loot/give/take items, start (or stop) a fight, gesture, play a song — or schedule any of that for later.
5. **Remember & reflect** — after the turn (and after every save), NPCs privately record what they witnessed, update their relationships, and can decide to change how they behave going forward.

Character cards, lore, quests, voices, and the catalogue of actions a character may take live in a **game profile** — a drop-in folder. One profile per game; without a profile loaded, the app is empty by design.

---

## Local-first, private & fast

Everything runs locally out of the box. Each capability picks a **provider** — the managed local runtime (default, fully offline) or a hosted API — and the local engines **start and stop automatically with the game**, so nothing is running when you're not playing.

| Capability | Local (managed, default) | Hosted APIs | Notes |
| --- | --- | --- | --- |
| **LLM** | llama.cpp `llama-server` (OpenAI-compatible, `:5001`) | OpenAI, Anthropic, Google Gemini, OpenRouter, any OpenAI-compatible base URL | Pick which placed GGUF is active in Settings → LLM (guided download + drag-drop to add more), or set an API key. |
| **TTS** | faster-qwen3-tts (streaming, `:5002`) | ElevenLabs, Cartesia, Inworld | Voice-cloned per character — locally, or through a hosted provider's cloning API. |
| **STT** | Parakeet TDT 0.6B v3 (dedicated GPU server, `:5003`) | OpenAI, Groq, Deepgram, AssemblyAI, OpenRouter | Local microphone transcription with no LLM contention, or a hosted key. Optional word-boosting from your character + lore names. |
| **Music** | ACE-Step (dedicated service, `:5004`) | — | Generates original songs on demand so an NPC can actually perform music in-world. |
| **Retrieval** | ONNX embeddings + reranker | — | Two-stage lore/memory/action retrieval, CPU-friendly. Place the embedder ONNX from Settings → Retrieval. |

Local runtimes/engines auto-install with one click on Settings → Runtimes; **model files** are placed manually (a guided flow opens the model's download page, shows the exact folder, and accepts a drag-drop / chosen file). Or skip local entirely and point any capability at a hosted API.

Speed is a first-class goal, not an afterthought:

- **Warm model workers** — the TTS model loads **once** and caches per-character speaker states, so synthesis stays low-latency instead of reloading per line.
- **A hand-rolled CUDA-graph TTS fast path** — `StaticCache` + TF32 + `inference_mode` + batched forward + reduce-overhead graph capture, to push real-time-factor down.
- **Sentence-level TTS streaming** — audio comes back sentence by sentence, so a character starts speaking before the whole line is synthesized.
- **GPU retrieval + prompt-cache preservation + a pooled LLM client** — the heavy AI path is tuned to keep the model's KV cache warm across turns.
- **Scales with your box** — sane small / CPU / ONNX defaults on low-end hardware (RTX 2060 / CPU-only), scaling up to high-end GPUs.

---

## Features

### Conversation & voice
- **Talk to any mapped NPC with your own voice.** Push-to-talk records your mic; chasm transcribes it, decides who you're addressing among the nearby, audible characters, and replies in-character.
- **Streaming voice output, per-character cloned.** Every NPC speaks in their own voice, synthesized locally and streamed sentence-by-sentence.
- **Character cards, lorebooks & world-info, personas.** A portable character-card + World-Info data model drives persona, description, first message, example dialogue, and keyword/vector-activated lore.
- **A player persona built from real game data.** Every save, the bridge sends a pure data snapshot (stats, appearance, gear); chasm writes a natural-language persona of *you* that every NPC is aware of.
- **Dynamic scenarios.** The scene-setting prompt wording changes with live gamestate (following you, waiting, sneaking together, traveling, in combat, …) so NPCs read the situation correctly.
- **Gamestate macros.** `{{player_name}}`, `{{major_location}}`, `{{time_of_day}}`, `{{health}}`, `{{quests}}`, and more resolve live into scenario templates.
- **A relationships ledger.** After play, a persona-less "Gamemaster" pass updates how each NPC feels about you and others — durable, evolving, and shown in the UI.
- **Dead NPCs stay silent.** Killed characters are gated out of speech and presence, mid-turn if necessary.

### Autonomous actions
- **A freeform action loop.** Instead of a fixed menu, an NPC **semantically discovers** the action it needs (`find_action`), and only the actions it has surfaced are offered to the model — so the grammar stays tight while the vocabulary is open.
- **Real item handling.** Loot a container, take items, or hand something over — `give`/`take` use real inventory transfers (the NPC actually walks over and makes the exchange), not teleport hacks.
- **Combat, on command.** An NPC can start a fight with the player *or another NPC*, and stop fighting — chasm-ordered NPC-vs-NPC combat included. In-combat turns get a last-word directive so NPCs react to a live fight.
- **Gestures & performance.** Wave, dance, do pushups, and other animations; NPCs can also **play a generated song** (guitar/rap) that ACE-Step composes on the spot.
- **Errand chains.** "Loot the body then bring it to me" parses into a sequenced multi-step task.

### Movement, schedules & companions
- **A travel/movement engine.** NPCs walk to real places, can depart early, and **arrive on time** — offscreen legs are simulated with timed waypoint teleports so travel works even through unloaded cells.
- **Timed schedules ("cronjobs").** Tell an NPC to do something *at* a game time or *when* a condition is met — "travel to the saloon at 1am, then wave" — and chasm parses, persists, and fires it on an in-game clock.
- **Companions with designed faces.** Create a character in chasm, design their face in-game with the vanilla character creator, and they spawn as a named, voiced follower — no third-party mod dependencies.

### Events & memory
- **A game event log.** The bridge streams what happens (shots, deaths, thefts, trades, location changes, level-ups, …) into a save-aware event store, browsable on the Events page.
- **NPC witness memory.** Each event records *who was in range to see it*; those NPCs get a durable, personal memory of what they witnessed — and it **rolls back with your saves**.
- **Event-triggered reactions.** Checked event types make witnessing NPCs react unprompted, rate-limited so they don't spam.
- **Reliable weapon-fire detection.** A dedicated engine hook emits an immediate, per-shot signal that both reactions and skills can trigger on.

### Save-aware by construction
Chat history, events, schedules, movement, relationships, journals, and skills all **checkpoint on save and restore on load**, keyed to the exact save. Quick-load rewinds NPC memory to match — a habit or a witnessed event from a discarded branch simply vanishes.

### Web control panel
A modern React control panel (React 19 + Vite + Tailwind), served by the backend — a persistent sidebar with a swappable content pane:

- **Main** — Chat (with an inline per-message strip showing which lore/actions were injected and which actually executed), Characters Book, Companions, Lore Book, Quest Book, Action Book, **Relationships**, **Events**, **Journals**, **Skills**, **Triggers**, **Gamestate**, **Schedule**, **Travel**, and **Persona**.
- **Globals** — the Scenario template with a gamestate-driven variant editor.
- **Settings** — Interface, Profiles, LLM, TTS, STT, Music, Word Boosting, Retrieval, Runtimes, Bridge, Hotkeys, request **Tracing**, and Updates.
- **Live theming** — accent, theme preset, density, and font scale apply instantly.

### Headless API + game bridges
chasm exposes a headless HTTP API for everything a turn needs — generation, speech (synthesize / recognize), lore / quest / action injection, and save-sync. **That API is the entire integration surface.** To bring chasm to a game you write a thin **bridge**: a mod (or external tool) that watches in-game state — who's nearby, who the player is addressing, their microphone — calls chasm, then speaks the returned audio and executes the returned actions in-world. The same API can drive a brand-new game built around AI characters from scratch. See **[docs/building-a-bridge-mod.md](docs/building-a-bridge-mod.md)** for the full contract.

### Connection-driven AI stack
chasm runs as a passive backend: it watches for the game's bridge to connect (the in-game plugin reports its process id), shows **Connected**, and automatically brings the local AI stack up while the game is running and tears it back down when the game exits — no manual start/stop.

---

## Self-improving NPCs

This is the part that isn't a chatbot. Over a playthrough, chasm's characters **learn your patterns, form durable habits, and develop their own reactive behaviors — with no hand-authored scripting.**

It works as a two-part inner life that runs quietly after every save (or on demand via a rebindable **reflect** hotkey):

1. **Each NPC keeps a private journal.** After a save, every character who was present since they last wrote makes a single LLM call **with their own character card in context** and appends one entry to an **append-only** journal — in *their* voice, noting patterns they've spotted in how you or the world behave, and what their personality inclines them to do about it. Nothing is ever rewritten; the journal is a genuine, growing record of their point of view.

2. **A separate skill-creator reads everyone's journals.** A persona-less curator — the same relationship the Gamemaster has to the relationships ledger, an independent intelligence with no character of its own — reads what each NPC privately wrote and decides whether anyone should **start, change, or stop** an automatic behavior. A "skill" is a habit: an owner, one **real game-event trigger**, and the owner's first-person intention. It's deliberately conservative — it acts only on a clearly repeated, settled intention, not a one-off mood — and it's grammar-constrained so a trigger is always a real event and an action is always one the NPC can actually take.

3. **Skills fire on their own.** When the event log ingests a fresh batch, each new event is matched against enabled skills — a purely mechanical check, with no LLM deciding whether to act. A skill fires only if its owner **actually witnessed** that event (per the witness list), with a per-skill cooldown so a burst can't flood. Firing doesn't run a canned animation — it plants the owner's intention as a private impulse and nudges them to take a real, freeform turn, so the habit adapts to the moment.

The loop, concretely:

> **Observe → journal → form a skill → it fires.** You keep drawing your weapon before every fight while a cautious companion is watching. On the next save she notices the pattern in her journal — *"the Courier reaches for that gun the second trouble's near; I'd best be ready too."* The skill-creator reads it and gives her a skill: **on `weapon_fire`, ready herself.** From then on, the moment you fire near her, she reacts on her own — no dialogue, no prompt, no script. Another companion, reading the same event, might journal that he's had enough of the shooting and decide to hang back instead. Same trigger, opposite habit — each true to who they are.

Because habits are grown from journaled *intentions*, they **evolve** as the NPC's thinking changes, and — like every other memory in chasm — they **checkpoint and roll back with your saves**. Load an earlier save and a companion un-learns a habit she hadn't formed yet. Nothing here is authored per-NPC: the personality is baked in at the moment the skill is created, and the runtime that fires it is deterministic and cheap. You can watch the whole thing happen on the **Journals** and **Skills** pages.

---

## Goals / vision

- **A universal connector.** chasm is the backend; games are pluggable. Write a bridge for an existing game, or build a new game on top of the API — chasm provides the brains either way.
- **Anyone can make a profile.** Author a profile for a game, share the folder, drop it in, and play.
- **Truly agentic characters** — they don't just chat, they *act*, *remember*, and *change*.
- **Local-first, private, and fast** — your models, your machine, no cloud dependency, low latency from low-end to high-end hardware.

---

## Status

**Experimental.** A personal project under active development — APIs and data formats may change. The backend (generation, speech, music, lore / quests / actions, retrieval, per-character voice cloning, the autonomous action loop, events + witness memory, schedules + movement, companions, self-improving NPCs, and the web control panel) is built and runs locally today. The **Fallout: New Vegas** reference bridge is a real, playable mod that exercises all of it.

---

## Architecture

chasm is a Rust workspace (`axum` HTTP API + a React control panel), split by role:

| Layer | Responsibility |
| --- | --- |
| Backend binary | Binds the listen address, spawns the warm TTS worker, serves the API + UI. |
| Core (`chasm-core`) | Config, settings, **profiles**, view models. |
| Data readers (`chasm-st-compat`) | Read the on-disk data model (character-card PNGs, lorebooks, action / quest books, world state) plus the journal/skill stores. |
| Prompt (`chasm-prompt`) | Ordered prompt components mirroring the generation path; also drives the prompt panel. |
| Embeddings (`chasm-embed`) | ONNX two-stage embeddings + reranker. |
| FNV bridge (`chasm-fnv-bridge`) | The in-process Fallout: New Vegas bridge (file/HTTP transport, NPC/action resolution). |
| Web (`chasm-web`) | The `axum` router: the game-turn API (`/api/game/v1/turn`) + headless API, the React control panel (served at `/app`), the `/speech/*` + generation endpoints, and the in-process bridge lifecycle. Home of generation, the action loop, scheduler/movement, event log, witness memory, relationships, journals, and skills. |

Supporting directories: `profiles/` (game profiles), `engines/<id>/` (per-engine virtualenvs), `voices/` (extracted source audio + per-engine clones), `scripts/` (the warm TTS worker and tooling), and `apps/chasm-desktop/` (the Tauri window + tray shell).

---

## Running it

**Install (Windows).** Grab a build from the Releases page:

- **`chasm_<version>_x64-setup.exe`** — the desktop app installer. Run it and chasm opens as a **window** (no console, no browser).
- **`chasm-<version>-windows-x64.zip`** — a portable build. Extract and run **`Start chasm.bat`**; the control panel opens at <http://127.0.0.1:7341/app>.

On first run the app is empty by design. Open **Settings → LLM / TTS / STT** and download a model for each (nothing is pre-selected), then add a game profile. For Fallout: New Vegas, install the **[Chasm Bridge FNV](https://github.com/chasmlol/chasm-bridge-fnv)** mod — it ships its profile and connects automatically when you launch the game.

**Dev.** Build and run the Rust workspace from source:

```powershell
cargo build
cargo fmt --all --check
cargo check
cargo test
```

chasm stores its data + profiles under a per-user home (override with `CHASM_ROOT` / `CHASM_DATA_ROOT`); `/health` returns JSON, and with no data the page shows a short setup hint.

---

## License

AGPL-3.0 (see `Cargo.toml`). Game data, character cards, and extracted voices are **not** included and are not covered by this license.
