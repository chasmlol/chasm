# FNV Bridge → Rust Port + Latency Wins — Phased Plan

> **⚠️ Legacy / historical.** This is the completed porting plan that folded the old
> standalone Node bridge into chasm as the in-process `chasm-fnv-bridge` crate. The
> port is done; this document is retained only as a record of how it was carried out.
> See [retired-node-bridge.md](retired-node-bridge.md) for the end state.

> Fold the Node bridge (`nvbridge-helper.mjs`, ~4,283 lines) into chasm as native
> Rust, retire the separate Node process, move the bridge out of the legacy upstream
> fork — **and** pick up the free latency wins along the way. Built section by
> section so the game stays playable and **you test each slice in-game before the
> next one starts.**

---

## 0. Background (read once)

Three components, talking over files + localhost HTTP:

```
Fallout: New Vegas
  └─ C++ NVSE plugin (Chasm-FNV/native/nvse-plugin/main.cpp)
       writes request files  ┐                      ┌ reads response / audio-chunk / command files
                             ▼                      ▲
  Node helper (Chasm/tools/fnv/nvbridge-helper.mjs)  ← THE THING WE'RE PORTING
       polls request dirs, resolves NPCs/actions/targets, writes the plugin's files,
       and calls chasm over HTTP for the actual AI
                             │ HTTP (localhost)
                             ▼
  chasm (chasm-rs, Rust/axum :7341) — LLM, prompt, RAG, TTS, STT, save-sync
       ALL the heavy AI already lives here.
```

The helper is **FNV glue**, not AI. It does: poll/parse the game's request files →
NPC-key→character mapping + gamestate → call chasm `/generate`, `/speech/*`,
`/save-sync/events` → resolve structured actions to native FormID/refid command
files → write audio-chunk + response files in the plugin's exact byte format →
handle Windows file-write races, request supersede/cancel, locks.

**End state:** a Rust crate `chasm-fnv-bridge` doing all of that, ultimately
**in-process inside chasm** (no Node, no localhost HTTP hop, one process, one build,
compiler-checked). The bridge code leaves the legacy upstream fork.

### Why Rust here is low-risk
- No exotic Rust (no `unsafe`, FFI, complex lifetimes) — just `tokio` async + `serde`
  + file I/O + `regex`, all of which chasm already uses.
- chasm already owns every hard primitive and the AI itself — this is **translation,
  not invention**.
- The whole HTTP-client layer in the helper (~200 lines) **deletes** once we fold in.
- The compiler catches exactly the bug class the Node code defends against with
  `a || b || c` field-spelling soup and `|| ''` null-guards (and it kills the
  stale-bridge class of bug — Node doesn't hot-reload, which already bit us once).

### Honest expectations on "speed"
The things that make an NPC feel slow are the **LLM** and **TTS model** — unchanged
by this. Real wins from the port, in order of how much you'll feel them:
1. **Kill the poll delay** — the helper polls every **750 ms** (`DEFAULT_POLL_MS`),
   so up to ¾ s of dead time before it even notices you spoke. Watch the dir
   instead → near-instant pickup. **Biggest perceptible win, and free in Section 0.**
2. **Remove the Node↔chasm HTTP hop** — once in-process, `/generate`, TTS chunks,
   and STT become direct calls. Modest (tens of ms/turn, more on chunky TTS streams),
   lands in the final fold (Section 7).
3. One less runtime/process (memory, no second GC). Marginal.

Per-turn CPU (JSON, fuzzy match, string ops on tiny payloads) is microseconds in
either language — **not** a bottleneck, so "Rust is fast" is true but irrelevant here.

---

## 1. Architecture for the port

Build the bridge as a **library crate with a thin standalone binary**, so we get
fast dev iteration first and the full in-process fold last:

```
crates/chasm-fnv-bridge/
  src/lib.rs     # run(BridgeConfig, impl ChasmClient) -> the whole bridge loop
  src/main.rs    # standalone bin: parse the SAME --config nvbridge.config.json,
                 # build an HttpChasmClient, call run(). Drop-in for the Node helper.
```

- **`ChasmClient` trait** = the seam between "FNV glue" and "talk to chasm":
  `ensure_live_chat`, `presence`, `generate_npc_turn_stream`, `generate_admin`,
  `synthesize_stream`, `recognize`, `save_sync_event`.
  - **`HttpChasmClient`** (reqwest → chasm :7341): used by the standalone bin for
    Sections 1–6. Lets us run the Rust bridge **instead of** Node with zero changes
    to chasm, and roll back instantly.
  - **`InProcessChasmClient`** (direct calls into chasm-web): swapped in at Section 7
    for the fold + HTTP-hop removal.

- Deps: `tokio`, `serde`/`serde_json`, `reqwest` (http client), `notify` (file
  watcher), `regex`, `base64`, `anyhow`, `tracing`. Reuse `chasm-core` config
  types where they overlap.

- **The standalone bin reads the exact same `nvbridge.config.json`** the Node helper
  uses, so testing a section = stop Node, run the Rust bin, same everything.

### The golden rule: rollback is always one command
At any point, if the Rust slice misbehaves in-game:
```
# stop the Rust bridge, restart the Node helper — you're back to known-good
node "C:\Users\user\Documents\Chasm\tools\fnv\nvbridge-helper.mjs" --config "C:\Users\user\Documents\Chasm\.codex\runtime\nvbridge.config.json"
```
Keep Node helper installed and working until Section 7's cutover is signed off.

### Parity harness (build it in Section 1, use it every section)
The plugin reads the bridge's files **byte-for-byte**; a silent format mismatch
breaks things with no error. So:
- Archived requests already exist under each native root's `processed/` and `traces/`.
- Add a `--replay <dir>` mode to the bin: feed captured request files through the
  Rust pipeline and **diff the produced command/response/chunk files against the
  Node helper's archived output**. Any diff = a parity bug caught *before* you load
  the game. This is the safety net that makes the whole port tractable.

---

## How to build & run (the loop you'll repeat)

```bash
# 1. Build chasm / the bridge bin (kill the server first so the .exe isn't locked)
#    (PowerShell) Get-Process chasm,chasm-fnv-bridge -EA SilentlyContinue | Stop-Process -Force
cargo build --release --bin chasm-fnv-bridge   # (and --bin chasm when chasm changed)

# 2. Stop the Node helper (so two bridges don't fight over the same inbox)
#    (PowerShell) kill the node PID running nvbridge-helper.mjs

# 3. Run the Rust bridge instead (same config the Node helper used)
target\release\chasm-fnv-bridge.exe --config "C:\Users\user\Documents\Chasm\.codex\runtime\nvbridge.config.json"

# 4. Play FNV, run the section's in-game test.
# 5. Broken? -> rollback command above. Working? -> tell Claude, next section.
```

**Only ONE bridge (Node or Rust) may watch the inbox at a time** — they'd
double-process otherwise.

---

## SECTION 0 — Free latency win (Node helper, no Rust yet)

**Goal:** make NPCs *notice you faster* today, before any Rust exists. Pure win,
pure confidence-builder.

**Why:** the helper polls every 750 ms. That's the single most perceptible lag that
isn't the AI itself.

**Changes** — `Chasm/tools/fnv/nvbridge-helper.mjs`:
1. Add an `fs.watch` on each native inbox dir (`nativeInbox(root)` and the save-state
   event dir) that triggers an immediate `runPollCycle()` (debounced ~15 ms). Keep a
   **slow safety poll** (e.g. 1000 ms) as a backstop, because Windows `fs.watch` can
   miss events and can fire mid-write — the existing audio stabilization already
   guards partial reads.
2. As a one-liner fallback if the watcher is flaky: drop `DEFAULT_POLL_MS` 750 → 150.

**In-game test:** talk to an NPC; response should *start* noticeably sooner (the
gap before "thinking" shrinks). Spam a few requests; confirm no double-processing,
no missed requests, no regressions in TTS/actions.

**Rollback:** revert the helper edit; it's isolated to the poll loop.

**Done when:** pickup feels instant and nothing regressed. (This change is also a
free win even if we never finished the port.)

---

## SECTION 1 — Rust skeleton + protocol round-trip

**Goal:** prove a Rust process can see a request and write a response the plugin
**accepts**, byte-for-byte. No AI yet.

**Why:** the file protocol is the riskiest parity surface. Nail it first, in
isolation, before wiring anything intelligent.

**Changes:**
- New crate `crates/chasm-fnv-bridge` (lib + bin), added to the workspace.
- Config parse: read the same `nvbridge.config.json` (dataRoots, nativeBridgeRoots,
  apiBase, character map, etc.).
- Watcher + safety poll over the native inbox roots (`notify`).
- Port the request **parser** (`parseNativeTextRequest`: line 0–9 fixed fields +
  `key=value` metadata from line 10) and **archiver**.
- Port the response **writer** (`writeNativeResponse` exact format) + audio-chunk
  writer format (`writeNativeAudioChunk`) — even though we won't fill them yet.
- For this slice: on a request, write a **stub response that echoes `player_text`**
  (or a fixed line) so the plugin shows *something*.
- Build the `--replay` parity harness; diff stub response format vs a Node archive.
- Lock-file handling (`acquireHelperLock`) so it won't run alongside Node by accident.

**In-game test:** stop Node, run the Rust bin, talk to an NPC. You should see your
own words (or the fixed line) echoed back as the NPC "reply" with no error. That
proves: watcher fires, request parses, response file is byte-correct, archive works.

**Rollback:** run the Node helper.

**Done when:** `--replay` shows **zero diff** on response formatting for a batch of
captured requests, and the in-game echo round-trips cleanly.

---

## SECTION 2 — NPC text turn (real reply + cloned-voice TTS)

**Goal:** a real conversation with a regular NPC, through the Rust bridge.

**Why:** this is the core 80% path; everything else hangs off it.

**Changes (`ChasmClient` = HTTP):**
- Port NPC mapping + candidate normalization (`getNpcMappingEntry`,
  `normalizeNpcCandidate`, prefix/fuzzy fallbacks), gamestate build
  (`buildNativeGamestate`), nearby-NPC selection + attention target.
- `ensure_live_chat` + `presence`.
- `generate_npc_turn_stream` → consume the NDJSON (`speaker.start`, `speech.delta`,
  `live.completed`), port **early-segment TTS** (synthesize sentence 1 while the rest
  generates) + `takeSpeechSegment`.
- `synthesize_stream` → write audio-chunk files (incl. the `caption_max_chars` +
  `nonPositional` plumbing we just added) + buffered fallback.
- Speaker-prefix stripping + audio-tag stripping for captions.
- Write the final response with the spoken text + sound path.

**In-game test:** walk up to e.g. Sunny/Easy Pete, speak (typed for now), hear a
real, in-character, cloned-voice reply with captions. Check first-audio latency feels
on par with Node.

**Rollback:** Node helper.

**Done when:** `--replay` matches Node on response + chunk files for NPC turns, and
a live conversation works with correct voice + captions.

---

## SECTION 3 — Voice input (STT)

**Goal:** press-to-talk to an NPC.

**Changes:**
- Port the STT audio-file path: `waitForNativeSpeechAudio` (wait for the game to
  finish writing the WAV), `isNativeSpeechAudioReady` (size/mtime settle),
  `retryNativeFileOperation` (EBUSY/EPERM retries), `stabilizeNativeSpeechAudio`,
  archive.
- `recognize` via chasm; feed transcript into the Section-2 turn path.
- `isNativeVoiceRequest` detection + sidecar-path candidates.

**In-game test:** hold your push-to-talk key, speak to an NPC, confirm the transcript
is right and the reply makes sense. Test a few times for the file-race retries.

**Rollback:** Node helper.

**Done when:** voice turns work reliably (no "empty audio"/locked-file flakes) and
`--replay` matches on STT-driven requests.

---

## SECTION 4 — Actions (attack / follow / stop + Action-Book/spawn relay)

**Goal:** NPCs act on commands.

**Why:** the action-resolution engine is the most intricate ~600 lines — isolate it
in its own gate.

**Changes:**
- Port structured/activated action collection (`collectStructuredActions`,
  `getActivatedActionMap`, `getTrustedActivatedExecution`).
- The trusted-execution arg resolver: `resolveTrustedExecutionArgument(s)`,
  `findScopedCatalogItem` (catalog-metadata → FormID), `normalizeTrustedNativeArgValue`
  (`ref:`/`refid:`/`form:`/`number:`/`string:` encoding), fuzzy target resolution
  (`levenshtein`, `fuzzyResolveNpcTarget`), repeat-count expansion.
- `getNativeGameMasterAction` (structured action → ATTACK/FOLLOW/STOP_FOLLOW/ACTION_BOOK).
- Native command-file writer (`buildNativeActionCommandLines` — the
  `NVBRIDGE_ACTION_V2` format with base64 script + comma-joined args + repeat
  sequences), written to every `nativeBridgeRoots/control/actions`.

**In-game test:** "follow me" → NPC follows; "stop following" → stops; "attack X" →
combat. Confirm targeting hits the *right* NPC. (Spawn is exercised in Section 5 with
Todd, but the engine lands here.)

**Rollback:** Node helper.

**Done when:** `--replay` produces byte-identical command files to Node for a batch
of action requests, and follow/attack/stop fire in-game.

---

## SECTION 5 — Admin / Todd (god voice + spawns)

**Goal:** Todd works — non-positional voice + spawning entities/items.

**Changes:**
- `isAdminRequest`, `generateAdminTurn` (admin scopes, extraContext, one-sentence
  rule), admin response path.
- `resolveNativeActorForAdmin` (5-strategy actor match: exact → fuzzy → text-scan →
  crosshair) + admin actor for `world.*` actions.
- Non-positional audio metadata (`admin_voice=1`, `non_positional_audio=1`,
  `nonPositional` → admin volume — already wired in chasm).
- Spawn catalog candidate flow (entity/item via vector-matched candidates).

**In-game test:** talk to Todd (god voice, straight into your ear), "spawn a deathclaw
on me", "give me 5 stimpaks". Confirm spawns fire and the admin volume slider affects
only Todd.

**Rollback:** Node helper.

**Done when:** Todd spawns + speaks correctly; `--replay` matches on admin requests.

---

## SECTION 6 — Multi-NPC group + save-sync + edge cases

**Goal:** the long tail.

**Changes:**
- Multi-line/group turns (multiple speakers in one turn → per-speaker chunks +
  per-speaker command files), the group co-speaker behavior.
- Save-sync: both paths (`processNativeSaveSyncRequest` and the save-state event
  files `pollNativeSaveStateEvents`), identity resolution, ack files.
- Request **supersede/cancel**: re-read the request file mid-flight, cancel the
  in-flight generate/TTS when a newer request appears (`tokio` `CancellationToken`).
- Distance gating, "too far", empty-text, silence responses, error responses.

**In-game test:** group conversation ("hey you two"), save + reload (checkpoint sync),
talk-then-immediately-talk-again (supersede), walk away mid-sentence.

**Rollback:** Node helper.

**Done when:** group + save/load + cancellation behave like Node; `--replay` clean
across the full captured corpus.

---

## SECTION 7 — The fold: in-process + kill the HTTP hop + retire Node

**Goal:** one process. No Node. No localhost HTTP between bridge and chasm. Bridge
code out of the legacy upstream fork.

**Changes:**
- Extract chasm-web's `/generate`, `/speech/*`, `/save-sync/events` handler bodies
  into **callable functions** (the handlers become thin wrappers). This is the main
  work — bounded and mechanical.
- Implement `InProcessChasmClient` over those functions; swap it in.
- Launch `run()` as a `tokio` task from `chasm-web::serve` behind a flag
  (`CHASM_FNV_BRIDGE`, default ON once proven), sharing chasm's runtime.
- Stop launching the Node helper in the stack launcher.
- Move the bridge out of `Chasm` (the legacy upstream fork); delete `nvbridge-helper.mjs` once
  signed off. Update [[node-bridge-architecture]] + [[tts-volume-control]] memory.

**In-game test:** full playthrough on the single-process backend — conversation,
voice, follow/attack, Todd spawns, group, save/load. Confirm first-audio is at least
as snappy (HTTP hop gone) and pickup is instant (watcher).

**Rollback:** flip the flag off + restart the Node helper (keep it around one more
session before deleting).

**Done when:** everything works in-process, Node is retired, and the bridge lives in
chasm.

---

## Risks & mitigations
- **Byte-for-byte protocol parity with the C++ plugin** → the `--replay` golden-file
  harness, every section.
- **Windows file-write races** (game writing while we read) → port the existing
  stabilization/retry faithfully (Section 3); don't "simplify" it.
- **Two bridges fighting the inbox** → lock file + the discipline of stopping Node
  before running Rust.
- **Section 7 handler extraction** touching chasm's HTTP surface → keep the axum
  handlers as wrappers so `/api/headless/v1` shapes are unchanged (mod still works
  even if pointed at HTTP).

## Project constraints (carry into every section)
- Commit/push **only when asked**. Branch policy = this repo's solo direct-to-main.
- Don't change `/api/headless/v1` response shapes (the C++ plugin + mod depend on them).
- Kill `chasm.exe` (and the bridge bin) before `cargo build` (file lock).
- Books/profiles are **not** in git — back up before editing; never commit `profiles/`
  or `scripts/__pycache__/*.pyc`.
- The Node helper edit (Section 0) commits to the **legacy upstream fork** (`chasmlol/chasm`),
  not chasm.

## Free wins summary (what speeds things up, and when)
| Win | Where | Lands in |
|-----|-------|----------|
| Watch inbox instead of 750 ms poll | request pickup latency (biggest) | Section 0 (Node) + Section 1 (Rust) |
| Remove Node↔chasm HTTP hop | per-turn + per-TTS-chunk | Section 7 (in-process fold) |
| One runtime (no Node process/GC) | memory, marginal CPU | Section 7 |
| Compiler-checked glue, no stale-bridge bug | correctness/ops, not latency | throughout |
