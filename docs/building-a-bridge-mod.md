# Building a Bridge Mod for chasm

> **Audience:** engineers (and coding agents) building a *bridge mod* that connects a
> game to the **chasm** backend, so in-game NPCs are driven by a local LLM with
> streaming text-to-speech, speech-to-text, retrieval-augmented lore, and structured
> in-world actions.
>
> **This document is game-agnostic.** chasm ships with a Fallout: New Vegas bridge as
> the reference implementation, but nothing in the HTTP contract is FNV-specific.
> Wherever a concrete example helps, FNV is used purely as *the worked example* — the
> same shapes apply to any game. If you are porting to a new game, read
> [§9 Porting to a new game](#9-porting-to-a-new-game) last; it ties the rest together.

---

## Table of contents

1. [The big picture](#1-the-big-picture)
2. [How the two halves find each other (connection model)](#2-how-the-two-halves-find-each-other-connection-model)
3. [Two ways to integrate](#3-two-ways-to-integrate)
4. [The one-call path: `POST /api/game/v1/turn`](#4-the-one-call-path-post-apigamev1turn)
5. [The granular path: `/api/headless/v1/*`](#5-the-granular-path-apiheadlessv1)
6. [Speech: TTS and STT](#6-speech-tts-and-stt)
7. [The profile bundle (game-specific data)](#7-the-profile-bundle-game-specific-data)
8. [Actions, lore, and save-sync](#8-actions-lore-and-save-sync)
9. [Porting to a new game](#9-porting-to-a-new-game)
10. [Endpoint quick reference](#10-endpoint-quick-reference)

---

## 1. The big picture

A chasm integration has **two halves**:

```
┌─────────────────────────┐         HTTP (localhost)          ┌──────────────────────────┐
│  GAME-SIDE BRIDGE MOD    │  ───────────────────────────────▶ │  chasm BACKEND (this repo) │
│  (a plugin / script mod  │                                    │  axum server on 127.0.0.1  │
│   inside the game)       │  ◀─────────────────────────────── │  :7341                     │
│                          │       NDJSON / JSON responses      │                            │
│  • detects the player    │                                    │  • LLM (koboldcpp)         │
│    talking to an NPC     │                                    │  • TTS (streaming)         │
│  • gathers who/where     │                                    │  • STT (whisper/parakeet)  │
│  • plays back audio      │                                    │  • retrieval (lore/actions)│
│  • executes actions      │                                    │  • per-game profile        │
└─────────────────────────┘                                    └──────────────────────────┘
```

- **The game-side bridge mod** is *yours to build*. It lives inside the game (a native
  plugin, a script extender module, a Lua/Papyrus mod — whatever the game supports). Its
  job is to notice an interaction, describe it to chasm over HTTP, then apply what comes
  back: play the streamed audio, show the subtitle, and fire any action the NPC chose.
- **The chasm backend** is a local HTTP server (`127.0.0.1:7341` by default) that owns
  everything AI: prompt assembly, the language model, streaming voice synthesis, speech
  recognition, retrieval of lore/actions, per-character memory, and save/load
  continuity. You do **not** re-implement any of that — you call it.

Everything is **local**. No cloud calls are required for a turn; the models run on the
player's machine and chasm brokers them.

The backend base URL is `http://127.0.0.1:7341`. Two namespaces matter to a mod:

| Namespace | Purpose |
|-----------|---------|
| `/api/game/v1/*` | The **high-level** game transport. One streaming call per NPC turn. Start here. |
| `/api/headless/v1/*` | The **granular** contract (live-chats, presence, generate, speech, save-sync). Use when you need fine control. |
| `/connection/status`, `/health`, `/api/app/version` | Liveness / status probes. |
| `/voices/*` | Serves synthesized/cloned voice clips for the active profile. |

> `/api/ui/v1/*` and `/app` belong to the desktop UI. **Mods never call those.**

---

## 2. How the two halves find each other (connection model)

chasm runs as a desktop app the player launches; your mod runs inside the game. They
rendezvous two ways, and a good mod uses both:

### 2a. The HTTP endpoint

The backend listens on `http://127.0.0.1:7341`. That is the address your mod POSTs to.
A turn is just an HTTP request. If chasm isn't running, the connection is refused —
your mod should degrade gracefully (fall back to the game's normal dialogue).

### 2b. The heartbeat / rendezvous directory

chasm and the mod also share a small on-disk **rendezvous directory** so each side can
tell whether the other is alive without a request in flight. The mod writes a heartbeat
file that chasm polls; chasm exposes the result at `GET /connection/status`:

```jsonc
// GET /connection/status  →  200 application/json
{
  "connected": true,          // heartbeat is fresh (≤5s) OR the stack is starting/up
  "phase": "connected",       // "disconnected" | "starting" | "connected" | "stopping"
  "last_seen_secs": 0.123     // seconds since the mod's heartbeat file changed (null = never)
}
```

- The mod should rewrite its heartbeat file frequently while the game is running (the FNV
  plugin does ~every 100 ms). chasm treats a heartbeat ≤ 5 seconds old as "connected",
  and tolerates a stale heartbeat if the game process is still alive (paused / alt-tabbed).
- `phase` reflects chasm's own model-stack lifecycle, so the UI can show "starting the
  models…" before the first turn can succeed.

You do not have to use the heartbeat to make turns work — it exists so both sides can
show accurate connection state. The rendezvous directory is also where the mod **stages
its profile bundle** for chasm to import (see [§7](#7-the-profile-bundle-game-specific-data)).

### 2c. Liveness probes

```jsonc
// GET /health → 200 (always succeeds)
{ "ok": true, "app": "chasm-rs", "data_root": "C:\\…\\data", "live_chats": 3 }

// GET /api/app/version → 200 (never errors; null latest on network failure)
{
  "current": "0.2.18",
  "latest": "0.2.18",
  "update_available": false,
  "download_url": "https://github.com/…/chasm_0.2.18_x64-setup.exe",
  "release_url": "https://github.com/…/releases/tag/v0.2.18"
}
```

---

## 3. Two ways to integrate

There are two levels of contract. **Most new mods should use the one-call path** and
only reach for the granular path when they need something it doesn't cover.

| | **One-call** (`/api/game/v1/turn`) | **Granular** (`/api/headless/v1/*`) |
|---|---|---|
| Calls per NPC turn | 1 | several (create → presence → generate/stream → synthesize) |
| STT → LLM → TTS → action | all in one streaming response | you orchestrate each step |
| Streaming | yes (NDJSON) | yes (generate/stream is NDJSON; synth stream is NDJSON) |
| Who fires the action | your mod (the turn hands you the classified action) | your mod |
| Best for | new integrations, simplest correct client | fine-grained control, custom flows, debugging |

The one-call path is literally a server-side orchestration of the granular path, so the
two are consistent — same models, same profile, same prompt assembly.

---

## 4. The one-call path: `POST /api/game/v1/turn`

This is the recommended entry point. One request runs a whole NPC turn — optional speech
recognition, prompt assembly + LLM, streaming TTS, and action classification — and
streams every output back as **NDJSON** (one JSON object per line) in production order,
so audio starts playing before the full reply is generated.

### 4a. Request

`POST /api/game/v1/turn`  ·  `Content-Type: application/json`

```jsonc
{
  "request_id": "turn_1234",      // optional; echoed on every event. Auto-generated if omitted.
  "npc_key":    "easy_pete",      // the NPC's stable native key (used for mapping + voice)
  "npc_name":   "Easy Pete",      // display name
  "player_text": "Where's the saloon?", // the player's words. If empty + audio_base64 set → STT first.
  "want_tts":    true,            // synthesize voice for the reply (default true)
  "location": {                   // optional; all fields optional strings
    "cell":       "GoodspringsSaloon",
    "worldspace": "MojaveWasteland",
    "region":     "Goodsprings",
    "major":      "Goodsprings",
    "minor":      "Prospector Saloon"
  },
  "metadata": {                   // free-form; distance, nearby NPCs, voice flags, etc.
    "targeting": {
      "nearby_npcs": [
        { "npc_key": "easy_pete", "npc_name": "Easy Pete",
          "characterId": "Easy Pete", "distance_m": 2.0, "under_crosshair": true }
      ]
    }
  },
  "audio_base64": ""              // optional base64 WAV for push-to-talk. Used only when player_text is empty.
}
```

Field notes:

- **`npc_key` / `npc_name`** identify the speaking NPC. `npc_key` is the stable key your
  game exposes; chasm maps it to a character card (by explicit map, or by name against
  the active profile — see [§7](#7-the-profile-bundle-game-specific-data)).
- **`metadata.targeting.nearby_npcs[]`** is how you tell chasm who is present and how
  far. An entry that already carries a `characterId` resolves a participant with no
  mapping needed. `distance_m` and `under_crosshair` help pick who the player is
  addressing in a crowd.
- **`want_tts: false`** returns text-only (no `audio.chunk` events) — useful for subtitle
  previews or text-only NPCs.
- **Voice input:** set `player_text: ""` and `audio_base64` to a base64 WAV. chasm
  transcribes it first (STT) and proceeds on the resulting text.

### 4b. Response — NDJSON event stream

`200 OK` · `Content-Type: application/x-ndjson`. One JSON object per line. Read line by
line; each line has a `type`. Events arrive in this order:

| `type` | When | Key fields |
|--------|------|-----------|
| `speech.delta` | subtitle for the next audio chunk (fires just before it) | `requestId`, `index`, `text` |
| `audio.chunk` | a synthesized audio slice (base64 WAV), streamed as produced | `requestId`, `index`, `audio.data`, `mimeType`, `text`, `npcKey`, `npcName`, `captionMaxChars?`, `metadata?` |
| `action` | the turn classified a triggerable action | `requestId`, `action`, `actionId`, `confidence`, `shouldTrigger`, `reason`, `actor{…}`, `queued:false` |
| `reply` | the structured NPC reply (after audio) | `requestId`, `status`, `npcKey`, `npcName`, `text`, `audioFilename`, `playerText`, `error`, `gameMaster{action,confidence,shouldTrigger}` |
| `turn.completed` | terminal success marker | — |
| `turn.error` | terminal failure *mid-stream* (pre-stream failures are a non-200) | `requestId`, `error` |

Concrete stream (whitespace added for clarity — real output is one object per line):

```jsonc
{"type":"speech.delta","requestId":"turn_1234","index":0,"text":"Well now,"}
{"type":"audio.chunk","requestId":"turn_1234","index":0,
 "audio":{"data":"UklGR…(base64 WAV)…"},"mimeType":"audio/wav",
 "text":"Well now,","npcKey":"easy_pete","npcName":"Easy Pete","captionMaxChars":80}
{"type":"speech.delta","requestId":"turn_1234","index":1,"text":"the saloon's right up the road."}
{"type":"audio.chunk","requestId":"turn_1234","index":1,"audio":{"data":"UklGR…"},
 "mimeType":"audio/wav","text":"the saloon's right up the road.","npcKey":"easy_pete","npcName":"Easy Pete"}
{"type":"action","requestId":"turn_1234","action":"POINT","actionId":"gesture.point",
 "confidence":"0.81","shouldTrigger":true,"reason":"Directed the player up the road.",
 "actor":{"npcKey":"easy_pete","npcName":"Easy Pete","characterName":"Easy Pete","characterId":"Easy Pete"},
 "queued":false}
{"type":"reply","requestId":"turn_1234","status":"1","npcKey":"easy_pete","npcName":"Easy Pete",
 "text":"Well now, the saloon's right up the road.","audioFilename":"turn_1234_0.wav",
 "playerText":"Where's the saloon?","error":"",
 "gameMaster":{"action":"POINT","confidence":"0.81","shouldTrigger":true}}
{"type":"turn.completed"}
```

### 4c. What the mod does with each event

1. **`speech.delta`** → render/append the subtitle. It always immediately precedes the
   matching `audio.chunk` (same `index`), so you can show the caption exactly when its
   audio plays.
2. **`audio.chunk`** → base64-decode `audio.data` (a complete, self-contained WAV with
   headers) and play it. `index` is the play order. `metadata` may carry flags like
   `admin_voice` / `non_positional` (play as 2D/UI audio rather than positional).
3. **`action`** → **your mod fires this action in-game**, exactly once, if
   `shouldTrigger` is true. Over the HTTP transport chasm reports `queued: false`,
   meaning it did *not* enqueue the action for you — the HTTP client owns triggering it.
   Use `actionId` (canonical id) to look up how to perform it; `actor` tells you which
   NPC acts. (Contrast with the file bridge, which has a durable server-side action
   queue.)
4. **`reply`** → the authoritative final text + status. `status:"1"` means success;
   a non-empty `error` means the turn failed. `gameMaster` restates the chosen action.
5. **`turn.completed`** → done; tear down your reader.
6. **`turn.error`** → surface/log and stop.

> **Robustness:** treat the connection dropping as "abandon the turn". chasm finishes the
> turn server-side regardless of whether you keep reading, so a player skipping dialogue
> is safe.

---

## 5. The granular path: `/api/headless/v1/*`

When you need finer control than one call gives — custom multi-NPC choreography, your own
audio pipeline, replaying history — drive the primitives directly. A live-chat is a
persistent, per-*scene* conversation with presence and memory. The one-call path is
exactly this sequence run server-side.

### Typical flow

```
1. POST /api/headless/v1/live-chats                       (create the scene, once)
2. POST /api/headless/v1/live-chats/:id/presence          (who is present / audible / how far)
3. POST /api/headless/v1/live-chats/:id/generate/stream   (run a turn → NDJSON of text)
4. POST /api/headless/v1/speech/synthesize/stream         (voice the reply → NDJSON of audio)
5. POST /api/headless/v1/save-sync/events                 (on game save/load)
```

### 5a. `POST /api/headless/v1/live-chats` — create a scene

Request (only `id` is required):

```jsonc
{
  "id": "fnv-goodsprings",          // required, unique scene id
  "groupId": "fnv-goodsprings",     // optional; defaults to id
  "title": "Goodsprings",           // optional; defaults to id
  "location": "Prospector Saloon",  // optional
  "participants": [ /* same shape as presence, below; optional initial presence */ ]
}
```

Response (also returned by `GET /api/headless/v1/live-chats/:id`):

```jsonc
{
  "id": "fnv-goodsprings",
  "title": "Goodsprings",
  "groupId": "fnv-goodsprings",
  "currentSegmentId": "…",
  "activeParticipantIds": ["player", "npc:easy_pete"],
  "segments": [ { "id": "…", "title": "…", "location": "…", "sessionId": "…" } ],
  "createdAt": "2026-07-01T12:00:00Z",
  "updatedAt": "2026-07-01T12:00:00Z"
}
```

### 5b. `POST /api/headless/v1/live-chats/:id/presence` — update who's around

```jsonc
{
  "replace": true,                  // (alias replacePresence) mark NPCs not listed as absent
  "participants": [
    {
      "participantId": "npc:easy_pete",  // required, unique
      "type": "npc",                     // default "npc"
      "characterId": "Easy Pete",        // character card reference
      "name": "Easy Pete",
      "present": true,
      "audible": true,
      "distance": 2.0,
      "metadata": { }
    }
  ]
}
```

Returns the same live-chat object. **`present` / `audible` matter for memory scoping:**
who *heard* a line determines whose history it belongs to. Keep presence honest — an NPC
listed as audible on a line will consider it part of the conversation.

### 5c. `POST /api/headless/v1/live-chats/:id/generate[/stream]` — run a turn

Request:

```jsonc
{
  "message": "Where's the saloon?",   // the player's input
  "participantId": "player",          // default "player"
  "responseFormat": "structured",     // "structured" → JSON with speech + actions; else plain text
  "extraContext": "",                 // optional string appended to the system prompt
  "metadata": { },                    // persisted on the message
  "actionBookScopes": ["global"],     // (alias action_book_scopes) which action scopes are allowed this turn
  "forceParticipantId": "",           // optional: force a specific NPC to speak
  "forceCharacterId": ""              // optional: force a specific character card to speak
}
```

`/generate` returns the whole turn buffered. `/generate/stream` streams **NDJSON**:

| `type` | fields |
|--------|--------|
| `live.start` | `liveChatId` |
| `speaker.start` | `speaker{ participantId, characterId, name, queueIndex, reason }` |
| `speech.delta` | `text`, `speaker{…}` |
| `live.error` | `error{ message }` |
| `live.completed` | `turn{ … }` — the full buffered turn (below) |

The buffered turn / `live.completed.turn` shape:

```jsonc
{
  "liveChatId": "fnv-goodsprings",
  "segmentId": "…",
  "speaker": { "participantId": "npc:easy_pete", "characterId": "Easy Pete",
               "name": "Easy Pete", "queueIndex": 0, "reason": "addressed" },
  "message": { "role": "assistant", "content": "Well now, the saloon's up the road.", "name": "Easy Pete" },
  "metadata": { "live": { … }, "structured": { … }, "activatedActions": [ … ] },
  "structured": {                      // present only when responseFormat was "structured"
    "speech": "Well now, the saloon's up the road.",
    "actions": [ { "id": "gesture.point", "target": "player", "parameters": {}, "reason": "…" } ]
  }
}
```

> Multi-NPC scenes return `speakers[]` / `messages[]` / `turns[]` (one per speaker chosen
> that turn) alongside the singular fields. Use the arrays when more than one NPC replies.

### 5d. Admin / single-character generation (`/api/headless/v1/generate`)

A non-live, single-character generation (chasm's "admin" path) for out-of-scene text —
e.g. a narrator or a console character. Same idea, keyed by `characterId` +
base64url `sessionId` for history, with a `generationOptions` object for
`temperature` / `max_tokens`. `/generate/stream` here is **SSE**
(`event: <type>\ndata: <json>`) with `run.started` → `token` … → `run.completed`.
Most game mods do not need this; the live-chat path is the norm.

---

## 6. Speech: TTS and STT

### 6a. `POST /api/headless/v1/speech/synthesize` — buffered TTS

```jsonc
// request
{
  "text": "Well now, the saloon's up the road.",
  "characterName": "Easy Pete",     // (alias character) selects the NPC voice
  "nonPositional": false,           // true → UI/admin voice volume, not positional NPC volume
  "tuning": { "gain_db": 0.0, "temperature": 0.7 }  // optional per-request overrides
}
// response
{ "audio": { "data": "UklGR…(base64 WAV, 24kHz mono int16)…" }, "mimeType": "audio/wav" }
```

### 6b. `POST /api/headless/v1/speech/synthesize/stream` — streaming TTS (NDJSON)

Same request. Streams the reply as it's synthesized, first slice ~200 ms then growing:

```jsonc
{"type":"audio.chunk","index":0,"audio":{"data":"UklGR…"},"mimeType":"audio/wav",
 "text":"Well now,","captionMaxChars":80}          // text present on the first chunk
{"type":"audio.chunk","index":1,"audio":{"data":"UklGR…"},"mimeType":"audio/wav","text":""}
{"type":"speech.error","error":{"message":"…"}}     // on failure
```

Each chunk is a complete, independently-playable WAV. `captionMaxChars` is a subtitle
display hint.

### 6c. `POST /api/headless/v1/speech/recognize` — STT

```jsonc
// request — base64 WAV in an `audio` object (flat aliases audioBase64 / data also accepted)
{
  "audio": { "data": "UklGR…", "encoding": "base64", "format": "wav", "mimeType": "audio/wav" },
  "language": "en",                 // optional
  "model": "",                      // optional (alias modelId)
  "timeoutMs": 30000                // optional (alias timeout_ms)
}
// response
{
  "provider": "…", "text": "where's the saloon",
  "audio": { "format": "wav", "encoding": "base64", "mimeType": "audio/wav", "byteLength": 12345 },
  "metadata": { "model": null, "language": null, "task": "transcribe", "timeoutMs": 30000, "durationMs": 2450 }
}
```

Audio shorter than ~2 s is padded with trailing silence automatically. The one-call
`/api/game/v1/turn` uses this internally when you pass `audio_base64`.

### 6d. `GET /voices/*path` — serve voice clips

Serves audio/metadata from the **active profile's** voices directory (content type by
extension: `.wav`→`audio/wav`, `.json`→JSON, etc.). Paths are sanitized (no `..`, drive
letters, or shell metacharacters). Switching profiles repoints this with no restart. This
is how a mod fetches cloned/pre-baked NPC voice clips it references by name.

---

## 7. The profile bundle (game-specific data)

The HTTP contract is generic; the **game-specific knowledge** lives in a *profile bundle*
your mod ships and chasm imports. A bundle is a self-contained folder of authored content —
who the characters are, what the world's lore is, what actions exist, how to voice NPCs.
chasm imports it on connect and makes it the *active profile*.

### 7a. Bundle layout

```
<mod-root>/chasm-profile/<game-id>/
├── profile.json                         # manifest (REQUIRED — the only file that must exist)
├── characters/
│   ├── Easy Pete.png                    # one card per NPC; filename (no ext) = character id
│   └── Sunny Smiles.png
├── worlds/
│   └── <Game> Lore.json                 # lorebook(s): baseline world facts (see §8)
├── groups/
│   └── <group-id>.json                  # optional multi-NPC scene rosters
├── headless/
│   ├── action-books/
│   │   └── <Game> Action Book.json      # structured actions NPCs can take (see §8)
│   ├── action-catalogs/
│   │   ├── <game-id>.entities.json      # (optional) spawnable NPCs/creatures
│   │   └── <game-id>.items.json         # (optional) spawnable items
│   └── quest-books/
│       └── <Game> Quests.json           # (optional) quest-scoped context
└── extract_voices.py                    # (optional) game-specific voice extractor
```

`<game-id>` is a slug (letters, digits, `-`, `_`) — e.g. `fallout-new-vegas`. It is the
global identifier for the game and appears in `profile.json`, in action scopes
(`game:<game-id>`), and in catalog filenames.

### 7b. `profile.json` — the manifest

The only required file. It declares the game, a `bundleVersion` for update logic, the
character roster, and optional voice-extraction metadata:

```jsonc
{
  "id": "fallout-new-vegas",        // required slug; the profile / game id
  "name": "Fallout: New Vegas",     // display name
  "description": "Goodsprings starting area…",
  "bundleVersion": 3,               // integer; bump when you ship new content (see import rules)
  "game": "fallout-new-vegas",      // game id used in action scoping
  "characters": [
    { "name": "Easy Pete", "edid": "GSEasyPete" },      // name must match a card filename
    { "name": "Goodsprings settler", "voicetype": "MaleAdult03" }  // generic/unnamed NPC voice
  ],
  "voice": {                         // optional — drives in-game voice cloning
    "extractor": "extract_voices.py",
    "steam_app": "Fallout New Vegas",
    "plugin": "FalloutNV.esm",
    "bsa_relative": "Data\\Fallout - Voices1.bsa",
    "voice_root": "sound\\voice\\falloutnv.esm"
  }
}
```

- **`characters[].name`** must exactly match a card filename in `characters/` (without the
  extension). `edid` (an engine editor-id) and `voicetype` are optional game hints used for
  native resolution and generic-NPC voices.
- **`bundleVersion`** governs updates (below). Increment it whenever you change shipped content.

### 7c. Character cards

`characters/<Name>.<img>` — one portrait per NPC. The filename **is** the character id and
must match a `profile.json` entry. (chasm ships PNG cards; the card image is the identity —
the character's persona/voice come from the profile + lore + voice metadata.)

### 7d. How chasm imports a bundle

On connect, your mod stages the bundle into the shared rendezvous directory and chasm
imports it:

1. **Stage** — the mod copies `chasm-profile/<game-id>/` into
   `%LOCALAPPDATA%\chasm\bridge\chasm-profile\<game-id>\` (the rendezvous dir; see §2b and
   the config below).
2. **Discover** — chasm scans that folder for any subdirectory containing a valid
   `profile.json`.
3. **Validate** — the `id` must be a safe slug (no `/`, `\`, `..`, empty).
4. **Version-gate** — if a profile with that `id` already exists, chasm compares
   `bundleVersion`. **Incoming > installed → replace; incoming ≤ installed → skip** (so a
   reinstall never clobbers a newer/edited profile). New id → install fresh.
5. **Allowlist copy** — chasm copies **only authored content** into `profiles/<id>/`:
   `profile.json`, `characters/`, `worlds/`, `groups/`, `headless/action-books/`,
   `headless/quest-books/`, `headless/action-catalogs/`, and `extract_voices.py`. Everything
   else is ignored. Per-user *runtime* state is **never** imported — `chats/`, `group chats/`,
   `voices/`, `embed-cache/`, `vectors/`, `headless/live-chats.json`,
   `headless/save-sync/`, `headless/world-state.json`. This keeps a mod update from wiping a
   player's conversations or cloned voices.
6. **Atomic swap** — the copy lands in a temp dir, then replaces `profiles/<id>/` in one move.
7. **Activate** — on first import (no active profile yet), chasm sets this as the active
   profile. Players can switch profiles later from the UI.

> **Practical rule for mod authors:** ship only authored content, and **bump
> `bundleVersion`** every time you change it, or players won't get the update.

---

## 8. Actions, lore, and save-sync

### 8a. Lore / world entries

`worlds/*.json` is a lorebook: a set of entries, each a fact chasm may inject into the
prompt. An entry is injected when it *activates*:

```jsonc
{
  "name": "Fallout New Vegas",
  "entries": {
    "0": {
      "uid": 0,
      "key": ["Goodsprings", "town", "here"],   // keywords that activate this entry
      "keysecondary": [],
      "content": "Goodsprings is a small, quiet settlement…",  // the injected text
      "constant": true,     // ALWAYS injected
      "vectorized": false
    },
    "1": {
      "uid": 1,
      "key": ["Mojave Wasteland", "Mojave", "the wastes"],
      "content": "The Mojave Wasteland is the stretch of post-War Nevada desert…",
      "constant": false,    // conditional
      "vectorized": true    // also findable by meaning, not just exact keywords
    }
  }
}
```

**Activation model (the same three gates apply to lore *and* actions):**

| Gate | Fires when | Use for |
|------|-----------|---------|
| **constant** (`constant:true`) | every turn | a handful of always-true baseline facts (where you are, the setting) |
| **keyword** | a `key` string matches the scan text | topic facts ("the Legion", "New Vegas") |
| **vector** (`vectorized:true`) | the turn is *semantically* similar, even without the exact keyword | fuzzy topics phrased many ways |

- **Keyword matching is regex/substring**, case-insensitive, so `key: ["follow"]` also
  matches "following". Keys don't need to be surgically precise — they gate *availability*,
  not the final choice.
- **What is scanned:** lore/quest activation scans the **player's message** (not game state),
  so lore triggers on what the player actually says. Keep the constant set tiny; let keyword
  and vector do the rest.
- **Lore content should be timeless, pre-game baseline world facts** (places, factions,
  history) — *not* in-game events, quest state, or opinions. Per-character personality and
  memory belong on the character, not in world lore.

### 8b. Actions

`headless/action-books/*.json` defines the structured actions an NPC can take instead of
just talking — follow, point, attack, spawn, sit, etc. Each entry maps a model-facing
`actionId` to activation keys, permission scopes, and (game-specific) execution details:

```jsonc
{
  "id": "Fallout New Vegas Action Book",
  "entries": {
    "4": {
      "uid": 4,
      "key": ["follow", "follow me", "come with", "escort"],  // availability keywords
      "keysecondary": ["player", "travel"],
      "content": "Request a loaded NPC to follow or escort the player.",
      "constant": false,
      "vectorized": true,                        // also offered on semantic match
      "actionId": "movement.follow_target",      // the canonical id the model emits
      "riskTier": "medium",
      "targetGame": "fallout-new-vegas",
      "scopes": ["global", "game:fallout-new-vegas"],  // who may use it (see below)
      "parametersSchema": { "target": "player", "confidence": "number 0..1" },
      "preconditions": ["actor is loaded and mapped", "actor within the nearby radius"],
      "effects": ["actor begins following the player"],
      "vectorSearchTexts": ["follow me", "can you follow me", "walk with me"],
      "execution": {                             // GAME-SPECIFIC — how the mod performs it
        "language": "geck/xnvse",
        "script": "ref rActor\n…"
      },
      "tags": ["movement", "ai-package", "native-supported"]
    }
  }
}
```

**Availability vs. triggering — two separate stages:**

1. **Availability** (server-side): each turn, chasm builds the set of actions *offered* to
   the model using the same constant / keyword / vector gates as lore, **filtered by scope**.
   Keys here are broad on purpose — "follow" also making "stop following" available is fine.
2. **Triggering** (the model + your mod): the model picks at most one action and returns it.
   On the one-call path it arrives as an `action` event with `actionId`, `confidence`,
   `shouldTrigger`, and the acting NPC; **your mod performs it** (using the entry's
   `execution` / whatever your engine needs) exactly once.

**Scopes** gate *who* may use an action so a normal NPC turn can't fire admin/world-editing
actions:

| Scope | Meaning |
|-------|---------|
| `global` | any NPC turn may use it |
| `game:<game-id>` | restricted to that game's profile |
| `admin` | only the admin/"god" character (a console/narrator persona) |
| `godmode` | world-mutating (spawn/despawn) — admin-only |

A normal player turn requests scopes like `["global", "game:<game-id>"]`; the admin path
requests `["admin", …]`. An action is only offered if its `scopes` intersect the turn's
requested scopes. **Every action should be `vectorized:true`** so it can be offered on
meaning; the semantic threshold (a chasm retrieval setting) is tuned so vague inputs like
"What?" don't flood the model with irrelevant actions.

**Action catalogs** (`headless/action-catalogs/<game-id>.entities.json` / `.items.json`) are
optional generated lists of spawnable NPCs/creatures/items, referenced by spawn-type actions
so the model can pick a real target by name (each entry carries the engine `formId`/editor-id
in `metadata`). Generate these by scraping your game's data files.

### 8c. Save-sync (checkpoint / restore on save & load)

So NPC memory stays consistent with the player's saves, the mod tells chasm when the game
saves or loads. chasm checkpoints the conversation/world state on save and restores it on
load, so quick-loading rewinds NPC memory to match:

`POST /api/headless/v1/save-sync/events`

```jsonc
// on game SAVE  (event ∈ save|saved|checkpoint|autosave|quicksave)
{
  "event": "save",
  "gameId": "fallout-new-vegas",
  "saveId": "Save42",               // stable id for this save slot
  "saveName": "Goodsprings 03:14",
  "saveFile": "…/Save42.fos",
  "saveFingerprint": "…"            // e.g. mtime/hash, to detect the exact save
}
// → { "status": "checkpoint_created", "checkpoint": { … }, "counts": { … } }

// on game LOAD  (event ∈ load|loaded|restore|reload)
{ "event": "load", "gameId": "fallout-new-vegas", "saveId": "Save42" }
// → { "status": "restored", "restored": true, "checkpoint": { … }, "counts": { … } }
```

- The checkpoint id is derived from `gameId` + `saveId` (stable hash), so save and load pair
  up automatically. Pass `dryRun: true` on a load to preview without applying.
- Statuses include `checkpoint_created` / `checkpoint_updated`, `restored` /
  `restore_preview`, `snapshot_missing` (loaded a save chasm never checkpointed), and
  `disabled` / `ignored` (save-sync off). An unknown `event` is a `400`.

---

## 9. Porting to a new game

Nothing in chasm is tied to a specific game. To bring up a new game, you build the
game-side bridge mod and author a profile bundle; the backend is unchanged.

### 9a. The rendezvous directory and bridge config

The mod and chasm meet at a fixed path, identical on both sides so they connect even when
installed separately:

```
%LOCALAPPDATA%\chasm\bridge\          (override with the CHASM_BRIDGE_ROOT env var)
├── chasm-profile/<game-id>/          # your staged bundle (chasm imports from here)
├── nvbridge.config.json              # bridge config (optional; all fields have defaults)
├── runtime_heartbeat.json            # your plugin rewrites this frequently while the game runs
└── traces/                           # request/response traces (debugging)
```

The optional `nvbridge.config.json` tunes how the bridge behaves. Keys are **camelCase**.
The knobs a port usually sets:

| Key | Default | Meaning |
|-----|---------|---------|
| `apiBase` | chasm backend URL | Where the mod sends turns (point at the chasm backend, `http://127.0.0.1:7341/api/headless/v1`). |
| `ttsApiBase` / `sttApiBase` | = `apiBase` | Optional split endpoints for speech. |
| `requestTimeoutMs` | `180000` | Per-request timeout. |
| `liveChatId` / `groupId` | scene id | The scene/conversation identity. |
| `participantId` | `player` | The player's participant id. |
| `npcCharacterMap` (`characterMap` / `npcCharacters`) | `{}` | Maps a native NPC key → a character card id. Merged; later keys win. |
| `nativeMaxDistanceMeters` | `10.0` | How close an NPC must be to be interactable. |
| `gameStateRadiusMeters` | `30.0` | Radius that counts as "nearby" for context. |
| `enableActionBooks` | `true` | Allow structured actions. |
| `actionBookTargetGame` | `<game-id>` | Which game's action books to load. |
| `actionBookIds` | book name(s) | Which action book(s) to enable. |
| `nativeActionConfidence` | `0.65` | Confidence threshold to fire a native action. |
| `adminCharacterId` / `adminCharacterName` | admin persona | The "god"/console character for admin-scoped actions. |
| `ttsOverrides` / `speechRecognition` | `{}` | Extra fields merged into the TTS / STT request bodies. |

If you use the one-call `/api/game/v1/turn` path, most of this is handled server-side and a
port needs little more than the bundle plus a name-based NPC mapping (chasm maps nearby NPCs
to characters by name when no explicit map is given).

### 9b. Checklist

1. **Pick a `<game-id>` slug** (e.g. `elder-scrolls-oblivion`). Use it everywhere.
2. **Author the profile bundle** (§7): `profile.json`, character cards, a small constant-heavy
   `worlds/` lorebook of baseline facts, and an `action-books/` file mapping each `actionId`
   to your engine's execution. Optionally generate entity/item catalogs.
3. **Build the game-side mod**: detect the player addressing an NPC, gather `npc_key`,
   `npc_name`, nearby NPCs + distances, and location, then call `/api/game/v1/turn`. Play the
   streamed `audio.chunk`s, show `speech.delta` subtitles, and perform the `action` event.
4. **Stage on connect**: copy the bundle to `%LOCALAPPDATA%\chasm\bridge\chasm-profile\<game-id>\`,
   write `nvbridge.config.json`, and start rewriting `runtime_heartbeat.json`.
5. **Wire save/load** to `POST /api/headless/v1/save-sync/events` (§8c).
6. **Voices** (optional): ship an `extract_voices.py` that pulls per-NPC reference clips so
   chasm can clone in-game voices; fetch clips back via `/voices/*`.
7. **Test**: launch the game, confirm `GET /connection/status` shows `connected`, hold a
   conversation, verify actions fire and lore stays relevant. Bump `bundleVersion` on every
   content change.

### 9c. Common pitfalls

- **Bundle didn't update** → you forgot to bump `bundleVersion` (chasm skips equal/older).
- **NPC won't resolve** → the `characters[].name` doesn't match the card filename, or the
  native key isn't in `npcCharacterMap` (and name-matching failed).
- **Admin actions firing on normal turns** → an action's `scopes` is missing `admin`/`godmode`
  gating, or the turn requested admin scopes.
- **Lore over-injecting** → too many `constant:true` entries, or keys too broad *and* the entry
  is large; prefer keyword/vector activation and keep constants to a few baseline facts.
- **Config ignored** → keys must be camelCase; the file lives in the rendezvous dir, not the
  mod folder.

---

## 10. Endpoint quick reference

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/game/v1/turn` | **One-call NPC turn** → NDJSON (speech.delta, audio.chunk, action, reply, turn.completed) |
| POST | `/api/headless/v1/live-chats` | Create a scene (live-chat) |
| GET  | `/api/headless/v1/live-chats/:id` | Fetch a scene |
| POST | `/api/headless/v1/live-chats/:id/presence` | Update presence/audibility/distance |
| POST | `/api/headless/v1/live-chats/:id/generate` | Run a turn (buffered) |
| POST | `/api/headless/v1/live-chats/:id/generate/stream` | Run a turn (NDJSON: live.start, speaker.start, speech.delta, live.completed) |
| POST | `/api/headless/v1/generate` · `/generate/stream` | Admin/single-character generation (SSE stream) |
| POST | `/api/headless/v1/speech/synthesize` | Buffered TTS → base64 WAV |
| POST | `/api/headless/v1/speech/synthesize/stream` | Streaming TTS → NDJSON audio.chunk |
| POST | `/api/headless/v1/speech/recognize` | STT → transcript |
| POST | `/api/headless/v1/save-sync/events` | Checkpoint/restore on game save/load |
| GET  | `/connection/status` | Heartbeat/lifecycle state |
| GET  | `/health` | Liveness |
| GET  | `/api/app/version` | Running vs latest version |
| GET  | `/voices/*path` | Serve a voice clip from the active profile |

---

*Generated for chasm. The reference bridge implementation is
[chasm-bridge-fnv](https://github.com/chasmlol/chasm-bridge-fnv).*
