# Retired Node FNV bridge — where it went + how to roll back

> **⚠️ Legacy / historical.** Records the retirement of the old standalone Node bridge,
> now fully replaced by the in-process `chasm-fnv-bridge` crate. Kept as a rollback
> reference only.

The Fallout: New Vegas bridge used to be a Node helper (`nvbridge-helper.mjs`, ~4,283
lines) that ran as its own process and talked to chasm over `127.0.0.1:7341`. It has
been **fully ported to native Rust and folded in-process into chasm** (the
`chasm-fnv-bridge` crate, run as a tokio task inside `chasm-web::serve`).
No Node process, no localhost hop.

## Where the Node stuff is now (archived, not deleted)

Moved out of the legacy upstream fork (`Chasm\tools\fnv\`) to:

```
C:\Users\user\Documents\nvbridge-node-archive\
  ├─ nvbridge-helper.mjs            # THE bridge — replaced by crates/chasm-fnv-bridge
  ├─ fnv-plugin-action-catalog.mjs  # dev tool: generates the plugin's action catalog
  ├─ seed-goodsprings-content.mjs   # one-off: seeds Goodsprings lore/content
  └─ README.md                      # short pointer back to this file
```

The `Chasm\tools\fnv\` folder is now empty. (That folder lives in the legacy upstream
fork repo `chasmlol/chasm`, so that repo's working tree shows these three files as
deleted — uncommitted. Commit that separately there if/when you want it cleaned up.)

The config the bridge reads is **NOT** archived — it's still live and used by the
in-process bridge: `C:\Users\user\Documents\Chasm\.codex\runtime\nvbridge.config.json`.

## What runs now

- chasm spawns the in-process bridge when the env flag `CHASM_FNV_BRIDGE` is set
  (`1`/`true`/`on`). `start-chasm.bat` now sets it, so the normal launch runs the Rust
  bridge. The launcher (`POST /launch`) skips the Node helper entirely when the flag is on.
- The bridge crate also ships a **standalone binary** (`chasm-fnv-bridge.exe`) that
  speaks HTTP to chasm — same as the old Node helper did. It's still buildable and is the
  drop-in equivalent if you ever want the out-of-process form back.

## Roll back to the Node helper (if needed)

1. Move the three `.mjs` files back: `nvbridge-node-archive\*.mjs` → `Chasm\tools\fnv\`.
2. Disable the in-process bridge: delete the `set "CHASM_FNV_BRIDGE=1"` line in
   `start-chasm.bat` (or `set ...=0`). With the flag off, the launcher spawns Node on Play.
3. Restart chasm and hit Play. Or run it directly:
   ```
   node "C:\Users\user\Documents\Chasm\tools\fnv\nvbridge-helper.mjs" --config "C:\Users\user\Documents\Chasm\.codex\runtime\nvbridge.config.json"
   ```
   (Only ONE bridge — Node or the in-process Rust one — may watch the inbox at a time.)

## Known caveat

There's one **unresolved intermittent stall** in the in-process bridge (froze ~60s once,
self-recovered, never reproduced) — diagnosed down to an unbounded TTS/STT call upstream
of the LLM, made catastrophic by the fold dropping the old HTTP-client timeout + coupling
save-sync acks to the turn loop. Details + candidate fixes are in the memory note
`fnv-inprocess-stall`. If it bites in practice, the rollback above puts you back on the
proven Node path while it's fixed.
