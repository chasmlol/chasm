# chasm — Agentic NPC Engine

> A local-first, **game-agnostic** AI backend for agentic game characters. They talk, they listen, they remember — and they **act** in the world. Bring any game (or build a new one) and connect it to chasm.

**chasm is the backend, not a game.** It turns game characters into AI agents: walk up to a character, speak to them with your own voice, and they reply — in *their* own cloned voice, generated live by a local LLM that knows who they are, where they are, and what's happening around them. Then they go a step further and **take actions in the world**: follow you, hand over an item, start a quest, lead you somewhere.

It runs entirely on your machine by default. No cloud, no API keys, no account.

**chasm is a universal connector.** It doesn't ship a game — it connects to one. It exposes a headless API, and to use it you (or anyone) write a small **mod / bridge** for a game: something that tells chasm what's happening in-world and plays back what chasm returns. That same API can just as easily power an **entirely new game** built around AI characters. Any game, any engine — if it can speak HTTP, it can plug into chasm.

```
   YOUR GAME  ──►  YOUR MOD / BRIDGE  ──►  chasm  ──►  LLM · TTS · STT · retrieval
  (any engine)      (you write this)      (backend)        (local models)
```

---

## What makes the characters *agentic*

Most "AI NPC" demos stop at conversation. chasm treats every spoken line as an agent turn:

1. **Hear** — the player's microphone audio is transcribed locally (STT).
2. **Think** — chasm works out which character is being addressed, pulls their character card, relevant lore, active quests, and nearby world state, assembles a prompt, and asks the local LLM for a reply.
3. **Speak** — that reply is synthesized in the character's *own* voice (TTS), cloned from real audio.
4. **Act** — alongside the spoken line, the model can emit structured **actions** from the game's *action book*, which the bridge executes in-world.

Character cards, lore, quests, voices, and the catalogue of actions a character is allowed to take all live in a **game profile** — a drop-in folder. One profile per game; without a profile loaded, the app is empty by design.

---

## Heavily inspired by SillyTavern

chasm is, unapologetically, built on ideas from [SillyTavern](https://github.com/SillyTavern/SillyTavern). It borrows:

- **Character cards** — the same persona / first-message / description model.
- **World Info / lorebooks** — keyword- and vector-activated lore injected into the prompt.
- **Prompt assembly** — the ordered, layered prompt-construction model (system blocks → character → lore → history → pending turn).

…and then adds the parts a *game* needs that a chat UI doesn't: an **agentic action layer**, a **game-bridge API** so a running game can drive generation instead of a human typing, **per-character voice cloning**, and a **local-first runtime** engineered for low latency. It reads SillyTavern-compatible data formats, so existing cards and World Info carry straight over.

---

## Local-first, private & fast

Everything runs locally out of the box:

| Component | Engine | Notes |
| --- | --- | --- |
| **LLM** | koboldcpp (OpenAI-compatible local server) | Download a GGUF from Settings → LLM; small models recommended on low-end hardware. |
| **TTS** | faster-qwen3-tts / PocketTTS (streaming) | Voice-cloned per character; install + pick either in Settings → TTS. Both stream on `:5002`. |
| **STT** | Whisper (via koboldcpp) | Local microphone transcription; download the Whisper model in Settings → STT. Served by koboldcpp on the same `:5001` port as the LLM. |
| **Retrieval** | ONNX embeddings + reranker | Two-stage lore/memory retrieval, CPU-friendly. |

Nothing is pre-selected — you download and choose each model (LLM, TTS, STT) from the control panel, so you only ever pull the ones you want.

Speed is a first-class goal, not an afterthought:

- **Warm model workers** — the TTS model loads **once** and caches per-character speaker states, so synthesis stays low-latency instead of reloading per line.
- **A hand-rolled CUDA-graph TTS fast path** — `StaticCache` + TF32 + `inference_mode` + batched forward + reduce-overhead graph capture, to push real-time-factor down.
- **Sentence-level TTS streaming** — audio comes back sentence by sentence, so a character can start speaking before the whole line is synthesized.
- **Scales with your box** — sane small / CPU / ONNX defaults on low-end hardware (RTX 2060 / CPU-only), scaling up to high-end GPUs.

---

## Features

### Game profiles (drop-in)
A profile is a folder. It bundles everything that's specific to one game:

- **Characters** — the character cards the profile owns.
- **Lorebook** — World Info entries for setting / faction / character lore.
- **Quest book** — quest state and gated, retrievable quest context.
- **Action book** — the catalogue of in-world actions characters may take.
- **Voices** — how to find and extract each character's source audio for cloning.

Drop a profile in and the engine works for that game. The engine itself is game-agnostic — the profile is the only thing that ties it to a particular world.

### Web control panel
A modern React control panel (React 19 + Vite + Tailwind), served by the backend — a persistent sidebar with a swappable content pane:

- **Live chat** — the conversation with the in-scene NPCs, with an inline per-message strip showing which lore entries and actions were **injected** and which actions actually **executed** (highlighted green) — no clicking required. A searchable conversation list (busiest first) switches between characters.
- **Content books** — aligned Characters / Lore / Quest / Action editors built on one shared component; each entry expands to edit, and character system prompts save back into the card.
- **AI settings** — LLM / TTS / STT / Retrieval, each with a model picker (download + recommended/GPU-fit + status) plus the full per-engine configuration (sampling, voices/tuning, language, retrieval knobs).
- **System** — Profiles (the active drop-in game profile), Bridge (connection config + status), and a read-only request Tracing waterfall.
- **Live theming** — accent, theme preset, density, and font scale apply instantly.

### Headless API + game bridges
chasm exposes a headless HTTP API for everything a turn needs — generation, speech (synthesize / recognize), lore / quest / action injection, and save-sync. **That API is the entire integration surface.** To bring chasm to a game, you write a thin **bridge**: a mod (or external tool) that watches in-game state — who's nearby, who the player is addressing, their microphone — calls chasm, then speaks the returned audio and executes the returned actions in-world. The same API can drive a brand-new game built around AI characters from scratch. chasm doesn't care what's on the other side of the seam: any game, any engine, any bridge that can speak HTTP.

### Connection-driven AI stack
chasm runs as a passive backend: it watches for the game's bridge to connect (the in-game plugin reports its process id), shows **Connected**, and automatically brings the local AI stack (LLM + TTS + STT) up while the game is running and tears it back down when the game exits — no manual start/stop.

---

## Goals / vision

- **A universal connector.** chasm is the backend; games are pluggable. Write a bridge for an existing game, or build a new game on top of the API — chasm provides the brains either way.
- **Anyone can make a profile.** Author a profile for a game, share the folder, drop it in, and play.
- **Truly agentic characters** — they don't just chat, they *do things* in the world.
- **Local-first, private, and fast** — your models, your machine, no cloud dependency, low latency from low-end to high-end hardware.

---

## Status

**Experimental.** A personal project under active development — APIs and data formats may change. The core backend (generation, speech, lore / quests / actions, retrieval, per-character voice cloning, and the web control panel) is built and runs locally today; reference game integrations are in progress.

---

## Architecture

chasm is a Rust workspace (`axum` HTTP API + a React control panel), split by role:

| Layer | Responsibility |
| --- | --- |
| Backend binary | Binds the listen address, spawns the warm TTS worker, serves the API + UI. |
| Core | Config, settings, **profiles**, view models. |
| Data readers | Read SillyTavern-compatible on-disk data (character cards, lorebooks, action / quest books, world state). |
| Prompt assembler | Ordered prompt components mirroring the generation path; also drives the prompt panel. |
| Retrieval | ONNX two-stage embeddings + reranker. |
| Web | The `axum` router: the headless + game-turn API (`/api/game/v1/turn`), the React control panel (served at `/app`), the `/speech/*` + generation endpoints, and the legacy server-rendered UI (kept as a fallback). |

Supporting directories: `profiles/` (game profiles), `engines/<id>/` (per-TTS-engine virtualenvs), `voices/` (extracted source audio + per-engine clones), `templates/` + `static/` (the UI), `scripts/` (the warm TTS worker and tooling).

---

## Running it

**Install (Windows).** Grab a build from the Releases page:

- **`chasm_<version>_x64-setup.exe`** — the desktop app installer. Run it and chasm opens as a **window** (no console, no browser).
- **`chasm-<version>-windows-x64.zip`** — a portable build. Extract and run **`Start chasm.bat`**; the control panel opens at <http://127.0.0.1:7341/app>.

On first run the app is empty by design. Open **Settings → LLM / TTS / STT** and download a model for each (nothing is pre-selected), then add a game profile. For Fallout: New Vegas, install the **Chasm Bridge FNV** mod — it ships its profile and connects automatically when you launch the game.

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

AGPL-3.0 (see `Cargo.toml`), consistent with its SillyTavern heritage. Game data, character cards, and extracted voices are **not** included and are not covered by this license.
