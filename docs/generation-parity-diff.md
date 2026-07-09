# Generation parity diff — Rust vs the legacy reference stack

> **⚠️ Legacy / historical.** A one-time parity audit from the initial port: the Rust
> generation path was diffed against the legacy upstream chat stack (the reference
> implementation chasm was ported away from) to reach byte-parity on prompt assembly.
> The gaps below were since closed. Retained as a record of that work.

Same prompt ("Howdy there. Who are you…", Easy Pete, `fnv-goodsprings`, structured,
action+quest books on, `actionBookLimit:12`) run through **both** backends, each
captured at the llama.cpp hop via `scripts/llm_capture_proxy.py`. The legacy reference
stack is the baseline (`scripts/captures/st/`), Rust is the candidate (`scripts/captures/`).

## STATUS — over-activation fixed (P0 done)

The three over-activation bugs below were root-caused and fixed. System prompt for
the same scenario went **18,523 → 4,859 chars**:

1. **Scan text was the whole conversation history.** `assemble_prompt` keyword-scanned
   all 40 history messages, so action/lore/quest keywords matched everywhere. Fixed:
   the live path now scans only the current turn (`message + gamestate + extraContext`),
   matching `generation.js`'s `activationText`. → action over-activation (7.3k) gone.
2. **Constant quest with no giver gate.** The one FNV quest (`constant:true`, given by
   Sunny Smiles) injected even when talking to Easy Pete. Fixed: a quest with named
   givers only activates when the speaker is a giver (the legacy stack's `quest-books.js` rule).
   → quest over-activation (2.7k) gone.
3. **Structured quest/action instructions emitted on the *enable* flag**, not on actual
   activation. Fixed: gated on whether the block activated (the legacy stack behavior). → orphan
   1.7k action instruction gone.

**Remaining gap is vector retrieval only** (see §"What diverges" 3–4): lore now selects
a *different* (keyword-matched) set than the legacy stack's semantic set, and chat-vector recall is
absent. Both need the embeddings backend. Audio Tags stay intentionally omitted.

## Headline

| | The legacy reference stack | Rust |
|---|---|---|
| system prompt | **7,540 chars / 21 blocks** | **18,523 chars / 23 blocks** |
| messages | 41 | 42 |
| model id sent | `gemma-4-26b-a4b-it` | full `...\gemma-4-26B-A4B-it-...gguf` path |

The character-card half is **identical** (same labels + byte lengths: System prompt
399, Description 715, Personality 257, Scenario 269, Example dialogue 531). All the
divergence is in the **dynamic context** sections.

## What matches ✅
- `Character:` / `System prompt:` / `Description:` / `Personality:` / `Scenario:` /
  `Example dialogue:` — byte-identical.
- `Activated lore:` header (439) and `Structured response fields:` (369) — identical.
- `Gamestate:` — equivalent (177 vs 179).

## What diverges ❌ (ordered by impact)

1. **Action books over-activate in Rust (the dominant gap).**
   For "who are you", the legacy stack activated **zero** action entries (no action block at all);
   Rust emitted a **7,341-char `Activated Action Book entries:`** block. the legacy stack gates
   actions by *relevance* (vector/intent) — actions only appear when the player's
   line implies one (verified: the legacy stack scenario 2 "follow me" *did* include actions and
   returned `movement.follow_target`). Rust's keyword/constant activation fires them
   on nearly every turn. **This alone is ~7.3k of the ~11k overage.**

2. **Quest books over-activate in Rust.** the legacy stack: no quest block for this turn. Rust:
   **2,742-char `Activated Quest Book entries:`**. Same root cause as actions.

3. **Lore selects *different entries*.** Both emit ~9 entries of similar total size,
   but different ones. the legacy stack (vector/semantic): Ghost Town Gunfight, Run Goodsprings Run,
   Securitrons, Bottle caps, The Great War, the Courier ambush. Rust (keyword/constant):
   `Opinion (Easy Pete)` entries, Doc Mitchell, Sunny Smiles, Joe Cobb, "Easy Pete
   retired". → semantic vs keyword retrieval picks a different relevant set.

4. **Chat-vector memory missing in Rust.** the legacy stack has `Relevant past chat context:` (302,
   vector recall of earlier conversation); Rust has none (no embeddings path).

5. **`Response instructions:` (431) present in the legacy stack, absent in Rust.**

6. **Audio Tags block: the legacy stack includes it (1,136 chars); Rust omits it.**
   Note: this is the audio-tags instruction you want *gone* — so Rust omitting it is
   arguably the desired end state, not a bug. Flag, don't "fix".

7. **Structured-instruction wording + ordering differ.** the legacy stack appends one
   `STRUCTURED_OUTPUT_INSTRUCTION`; Rust appends its own structured + per-section quest
   / action instructions (662 + 1,743), and places `Gamestate:` near the end rather
   than mid-prompt. The *fields* block matches; the surrounding instruction prose and
   section order do not.

8. **Model id**: Rust forwards the gguf file path as `model`; the legacy stack forwards the short
   alias. Cosmetic for llama.cpp, but should match for clean parity.

## Verdict / priorities before wiring generation in-game

- **P0 — Gate action/quest activation by relevance.** Today Rust injects the full
  action/quest catalog most turns. That bloats the prompt ~2.5× *and* will make NPCs
  action-happy. Needs intent/keyword gating that matches the legacy stack's behavior (and ideally
  the vector path).
- **P1 — Lore parity.** Port semantic retrieval, or accept the keyword set as an
  approximation and document the divergence.
- **P1 — Chat-vector recall** (`Relevant past chat context`) — needs the embeddings
  runtime; currently absent.
- **P2 — Section ordering + instruction wording**, `Response instructions`, model id.
- **Decision needed — Audio Tags**: keep omitted (kills the tags) or re-add for strict
  parity. Recommend: keep omitted, since removing them was a goal.

The card/persona core already has exact parity — the work left is entirely in the
dynamic retrieval (action/quest/lore gating + chat vectors), which is where the legacy stack's
vector machinery does the heavy lifting.
