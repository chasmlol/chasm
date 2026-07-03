# chasm desktop shell

A Tauri v2 native window + system-tray around chasm's existing axum backend —
**Path 1: wrap, don't rewrite.** No frontend code, Askama template, or web
handler is touched; this crate only spawns the existing server and points a
webview at it.

## What it does

- On launch it spawns `chasm_web::serve(config)` on Tauri's tokio-backed
  runtime. That's the same entry point the `chasm` bin uses, so reusing it
  gives the **in-process FNV bridge** and the **connection-driven AI-stack
  lifecycle** for free — both are `tokio::spawn`ed by `router()` and gated on
  (1) a current tokio runtime and (2) `CHASM_FNV_BRIDGE` being set.
- Waits until `127.0.0.1:7341` accepts connections, then reveals the window
  pointed at `http://127.0.0.1:7341` (never a blank "connection refused" page).
- **Minimize → tray** and **close (X) → tray**: the window hides instead of
  minimizing/closing, so the axum server + bridge + lifecycle keep running and
  the game stays connected. Only the tray **Quit** terminates.
- **System tray**: left-click (or "Show chasm") restores/focuses the window;
  the tooltip reflects the live connection phase (polls `/connection/status`
  every 2s: *Not connected* / *Starting…* / *Connected*).
- **Single instance**: a second launch focuses the existing window instead of
  starting a second server (which would double-bind `:7341` and run two
  bridges/lifecycles).
- **Quit** tears down the llama.cpp + TTS stack chasm started (so the runtimes
  aren't orphaned), then exits.

## Environment

The shell sets the same env `start-chasm.bat` uses, but only when not already
set (so an outer launcher can override):

- `CHASM_DATA_ROOT=C:\Users\user\Documents\Chasm\data\default-user`
- `CHASM_FNV_BRIDGE=1`  (enables the in-process bridge + lifecycle)

Other `CHASM_*` vars (e.g. `CHASM_BIND_ADDR`) are honored by
`AppConfig::from_env()` as usual.

## Run

Dev (console attached, so you see the backend + bridge + lifecycle logs):

```
cargo tauri dev          # from apps/chasm-desktop
# or
cargo run -p chasm-desktop
```

Release build + installer:

```
cargo tauri build        # from apps/chasm-desktop
```

Artifacts:

- `target/release/chasm-desktop.exe` — standalone exe
- `target/release/bundle/nsis/chasm_0.1.0_x64-setup.exe` — installer

> Only one chasm backend can own `:7341`. Quit any `start-chasm.bat` /
> `chasm.exe` instance before launching the desktop shell, or they'll
> fight over the port and the NVBridge inbox.

## Icon

`app-icon.png` (1024²) is the source; regenerate the icon set with
`cargo tauri icon apps/chasm-desktop/app-icon.png -o apps/chasm-desktop/icons`.
