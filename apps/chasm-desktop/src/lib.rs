//! chasm desktop shell (Path 1: wrap, don't rewrite).
//!
//! This is a thin Tauri v2 window + system-tray around chasm's existing axum
//! backend. On launch it spawns `chasm_web::serve(config)` on Tauri's
//! tokio-backed runtime — which is what makes the in-process FNV bridge AND the
//! connection-driven AI-stack lifecycle start (both are `tokio::spawn`ed by
//! `router()` and gated on a current tokio runtime + `CHASM_FNV_BRIDGE`).
//! The native window just points at `http://127.0.0.1:7341` and renders the
//! existing server-rendered UI; no frontend code is touched.
//!
//! Crucially the server task + its bridge/lifecycle are owned by the app and are
//! independent of window visibility: minimizing or closing the window only HIDES
//! it to the tray, so the game stays connected and the AI stack stays managed.
//! Only the tray "Quit" item terminates the process (and tears down the stack).

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use chasm_core::AppConfig;
use tauri::{
    menu::{MenuBuilder, MenuItemBuilder},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager, WindowEvent,
};
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// The single window's label, shared by the tray + window-event handlers.
const MAIN_WINDOW: &str = "main";

/// Set to `true` once the tray "Quit" item fires, so the `CloseRequested`
/// handler lets the process actually exit instead of hiding to tray.
static QUITTING: AtomicBool = AtomicBool::new(false);

/// Mirrors the environment `start-chasm.bat` sets, so the shell runs the
/// in-process bridge + lifecycle. Only sets a var that isn't already present, so
/// an outer launcher/env can still override. Note: the packaging-specific paths
/// (`CHASM_ROOT` / `CHASM_DATA_ROOT`) are NOT set here — they need the
/// Tauri app handle to resolve, so they're set in [`apply_packaging_paths`].
fn apply_default_env() {
    let defaults = [
        // Enables the in-process FNV bridge AND the connection-driven AI-stack
        // lifecycle (both spawned by chasm_web::router under our runtime).
        ("CHASM_FNV_BRIDGE", "1"),
        // Disable the HuggingFace xet backend: it symlinks weights (blobs->snapshots)
        // and bypasses hf_hub's symlink-support fallback, so on Windows without admin
        // / Developer Mode snapshot_download hard-fails (WinError 1314). Plain hf_hub
        // copies instead. Applies to every child that downloads models.
        ("HF_HUB_DISABLE_XET", "1"),
        ("HF_HUB_DISABLE_SYMLINKS_WARNING", "1"),
    ];
    for (key, value) in defaults {
        if std::env::var_os(key).is_none() {
            std::env::set_var(key, value);
        }
    }
}

/// Resolves the packaging-specific paths from the Tauri app handle, so an
/// installed copy finds its bundled web assets and writes per-user data to a
/// writable location. Both are only-if-unset, so a `cargo run` from the repo (or
/// an outer launcher env) can still override:
///   * `CHASM_ROOT` ← `resource_dir()` — the server resolves `<root>/static`
///     and `<root>/crates/chasm-web/ui/dist` from here; those dirs are
///     bundled into the resource dir via `bundle.resources` in tauri.conf.json.
///   * `CHASM_DATA_ROOT` ← `app_data_dir()/data/default-user` — a writable
///     per-user location (the install dir is read-only on a normal install).
fn apply_packaging_paths(app: &tauri::AppHandle) {
    use std::path::PathBuf;

    // CHASM_ROOT ← resource_dir. Strip the `\\?\` verbatim prefix Tauri returns,
    // else spawned tools (powershell -File, koboldcpp) can't load from it.
    if std::env::var_os("CHASM_ROOT").is_none() {
        if let Ok(resource_dir) = app.path().resource_dir() {
            std::env::set_var("CHASM_ROOT", chasm_core::strip_verbatim_prefix(resource_dir));
        } else {
            warn!("could not resolve resource_dir; falling back to CHASM_ROOT discovery");
        }
    }

    // data_root ← CHASM_DATA_ROOT override, else app_data_dir/data/default-user.
    let data_root: Option<PathBuf> = std::env::var_os("CHASM_DATA_ROOT")
        .map(PathBuf::from)
        .or_else(|| {
            app.path()
                .app_data_dir()
                .ok()
                .map(|dir| dir.join("data").join("default-user"))
        });
    let Some(data_root) = data_root else {
        warn!("could not resolve app_data_dir; model paths left at defaults");
        return;
    };
    if std::env::var_os("CHASM_DATA_ROOT").is_none() {
        std::env::set_var("CHASM_DATA_ROOT", &data_root);
    }

    // Consolidate EVERY model/runtime directory under ONE writable folder,
    // `<data_root>/models`, so the (possibly read-only) install dir stays clean and
    // the models survive updates/uninstall. Each var is only-if-unset so an outer
    // launcher/env can still override.
    let models = data_root.join("models");
    let set_if_unset = |key: &str, value: PathBuf| {
        if std::env::var_os(key).is_none() {
            std::env::set_var(key, value);
        }
    };
    // Capture the OLD HuggingFace hub BEFORE we repoint HF_HOME, so the migration
    // can move already-downloaded weights instead of re-fetching them.
    let old_hf_hub = huggingface_hub_dir();
    set_if_unset("CHASM_LLM_MODELS_DIR", models.join("llm"));
    set_if_unset("CHASM_ENGINES_DIR", models.join("tts"));
    set_if_unset("CHASM_WHISPER_MODELS_DIR", models.join("stt"));
    set_if_unset(
        "CHASM_KOBOLDCPP_EXE",
        models.join("koboldcpp").join("koboldcpp.exe"),
    );
    // Retrieval (fastembed) downloads via HuggingFace, which honors HF_HOME over
    // fastembed's own cache_dir — so point CHASM_EMBED_DIR at the SAME `hf` folder,
    // else `models_present` checks the wrong dir and retrieval reads "not installed".
    set_if_unset("HF_HOME", models.join("hf"));
    set_if_unset("CHASM_EMBED_DIR", models.join("hf"));

    migrate_scattered_models(&models, old_hf_hub.as_deref());
}

/// The HuggingFace hub cache dir the process would use RIGHT NOW: `HF_HUB_CACHE`,
/// else `HF_HOME/hub`, else `~/.cache/huggingface/hub`. Read before we override
/// `HF_HOME` so the migration knows where old weights actually live.
fn huggingface_hub_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Some(cache) = std::env::var_os("HF_HUB_CACHE") {
        return Some(PathBuf::from(cache));
    }
    if let Some(home) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(home).join("hub"));
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    Some(
        PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub"),
    )
}

/// One-time, best-effort move of models sitting in old scattered locations into the
/// consolidated `<data_root>/models` layout. Same-drive renames only, only when the
/// target is absent, never destructive on failure. The heavy movable bits are the
/// koboldcpp runtime and the HF TTS weights; the LLM already lives under
/// `<data_root>/models/llm`, and TTS engine venvs aren't relocatable so they simply
/// reinstall into `models/tts` on demand.
fn migrate_scattered_models(models: &std::path::Path, old_hf_hub: Option<&std::path::Path>) {
    let move_dir = |src: std::path::PathBuf, dst: std::path::PathBuf| {
        if !src.exists() || dst.exists() {
            return;
        }
        if let Some(parent) = dst.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::rename(&src, &dst) {
            Ok(()) => info!("consolidated {} -> {}", src.display(), dst.display()),
            Err(error) => warn!(
                "could not consolidate {} -> {}: {error}",
                src.display(),
                dst.display()
            ),
        }
    };

    // koboldcpp runtime: <CHASM_ROOT>/koboldcpp -> models/koboldcpp (only if it
    // actually holds the exe, so an empty/marker-only dir never wins).
    if let Some(root) = std::env::var_os("CHASM_ROOT") {
        let old = std::path::PathBuf::from(&root).join("koboldcpp");
        if old.join("koboldcpp.exe").exists() {
            move_dir(old, models.join("koboldcpp"));
        }
    }

    // HF TTS weights: <old hub>/models--… -> models/hf/hub/models--…
    if let Some(hub) = old_hf_hub {
        let new_hub = models.join("hf").join("hub");
        for repo in [
            "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base",
            "models--Qwen--Qwen3-TTS-Tokenizer-12Hz",
            "models--kyutai--pocket-tts",
        ] {
            move_dir(hub.join(repo), new_hub.join(repo));
        }
    }
}

/// Reveals the main window: un-hide, un-minimize, and focus it. Used by the tray
/// left-click, the "Show chasm" menu item, and the single-instance hook.
fn show_main_window(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window(MAIN_WINDOW) {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// The `http://host:port` origin the window points at, derived from the same
/// `bind_addr` the server binds, so the two never drift.
fn web_origin(config: &AppConfig) -> String {
    format!("http://{}", config.bind_addr)
}

/// Polls the bind address until it accepts a TCP connection (or a generous
/// deadline elapses), so the window is only shown once the server is actually
/// serving — never a blank "connection refused" page. Returns `true` if the port
/// came up.
async fn wait_for_server(bind_addr: &str) -> bool {
    // ~30s ceiling: first run loads the host-detect + repository, well under this.
    for attempt in 0..300u32 {
        if tokio::net::TcpStream::connect(bind_addr).await.is_ok() {
            info!("backend is accepting connections on {bind_addr}");
            return true;
        }
        if attempt == 0 {
            info!("waiting for backend to bind {bind_addr}…");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    warn!("backend did not come up on {bind_addr} within the timeout");
    false
}

/// Background loop: every ~2s read `GET /connection/status` and reflect the
/// AI-stack lifecycle `phase` in the tray tooltip (works whether the window is
/// shown or hidden, since the lifecycle runs inside the always-on server).
async fn poll_connection_status(app: tauri::AppHandle, origin: String) {
    let url = format!("{origin}/connection/status");
    let client = reqwest::Client::new();
    let mut last: Option<String> = None;
    loop {
        let tooltip = match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => resp
                .json::<serde_json::Value>()
                .await
                .ok()
                .map(|body| tooltip_for_phase(&body))
                .unwrap_or_else(|| "chasm".to_string()),
            _ => "chasm — backend unreachable".to_string(),
        };
        if last.as_deref() != Some(tooltip.as_str()) {
            if let Some(tray) = app.tray_by_id(MAIN_WINDOW) {
                let _ = tray.set_tooltip(Some(&tooltip));
            }
            last = Some(tooltip);
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

/// Maps a `/connection/status` body to a human tray tooltip using its `phase`
/// (`disconnected` / `starting` / `connected` / `stopping`).
fn tooltip_for_phase(body: &serde_json::Value) -> String {
    match body.get("phase").and_then(|p| p.as_str()) {
        Some("starting") => "chasm — Starting… (loading models)".to_string(),
        Some("connected") => "chasm — Connected".to_string(),
        Some("stopping") => "chasm — Stopping…".to_string(),
        _ => "chasm — Not connected".to_string(),
    }
}

/// App entry point shared by the desktop bin (and any future mobile target).
pub fn run() {
    // Logs go to the attached console in dev (and are why we can confirm the
    // bridge + lifecycle started). In a release/windows-subsystem build there's
    // no console; the tray is the UI.
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "chasm=info,tower_http=info,chasm_desktop=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    apply_default_env();

    tauri::Builder::default()
        // MUST be the first plugin: focus the existing window instead of starting
        // a second process (which would double-bind :7341, run two in-process
        // bridges over the same file inbox, and two stack lifecycles fighting over
        // koboldcpp/TTS).
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            info!("second instance launched; focusing the existing window");
            show_main_window(app);
        }))
        .setup(move |app| {
            // 0) Resolve packaging paths (resource dir → CHASM_ROOT, app
            //    data dir → CHASM_DATA_ROOT) from the app handle BEFORE the
            //    config is built, so an installed copy finds its bundled web
            //    assets and a writable per-user data dir. Then build the config —
            //    it must run here (not before the builder) so it observes those.
            apply_packaging_paths(app.handle());
            let config = AppConfig::from_env();
            let bind_addr = config.bind_addr.clone();
            let origin = web_origin(&config);

            // 1) Spawn the existing axum server on Tauri's tokio runtime. This is
            //    the reusable entry point — router() it calls spawns the FNV
            //    bridge + AI-stack lifecycle because a tokio runtime is current
            //    here and CHASM_FNV_BRIDGE is set. Owned by the app, so it
            //    keeps running regardless of window visibility.
            let server_config = config.clone();
            tauri::async_runtime::spawn(async move {
                if let Err(error) = chasm_web::serve(server_config).await {
                    error!("chasm backend exited with error: {error:#}");
                }
            });

            // 2) System tray: left-click / "Show chasm" reveal the window; "Quit"
            //    tears down the AI stack and exits.
            let show_item = MenuItemBuilder::with_id("show", "Show chasm").build(app)?;
            let quit_item = MenuItemBuilder::with_id("quit", "Quit chasm").build(app)?;
            let menu = MenuBuilder::new(app)
                .items(&[&show_item, &quit_item])
                .build()?;

            TrayIconBuilder::with_id(MAIN_WINDOW)
                .icon(app.default_window_icon().expect("bundled icon").clone())
                .tooltip("chasm — starting…")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "show" => show_main_window(app),
                    "quit" => quit(app),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show_main_window(tray.app_handle());
                    }
                })
                .build(app)?;

            // 3) Once the port is accepting connections, navigate the (hidden)
            //    window to the live UI and reveal it — never a blank page. Then
            //    keep the tray tooltip in sync with the connection phase.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                wait_for_server(&bind_addr).await;
                if let Some(window) = handle.get_webview_window(MAIN_WINDOW) {
                    // Land on the new React UI served under /app/.
                    let target = format!("{}/app/", origin.trim_end_matches('/'));
                    if let Ok(url) = target.parse() {
                        let _ = window.navigate(url);
                    }
                    let _ = window.show();
                    let _ = window.set_focus();
                }
                poll_connection_status(handle, origin).await;
            });

            Ok(())
        })
        .on_window_event(|window, event| match event {
            // Minimize → tray. On Windows a minimize arrives as a Resized event
            // with the window in the minimized state; intercept it, un-minimize,
            // and hide so it leaves the taskbar and lives in the tray instead.
            WindowEvent::Resized(_) => {
                if window.is_minimized().unwrap_or(false) {
                    let _ = window.unminimize();
                    let _ = window.hide();
                }
            }
            // The (X) close button → tray as well, so an accidental close can't
            // kill the backend mid-game. Only the tray "Quit" truly exits.
            WindowEvent::CloseRequested { api, .. } => {
                if !QUITTING.load(Ordering::SeqCst) {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
            _ => {}
        })
        .build(tauri::generate_context!())
        .expect("failed to build chasm desktop app")
        .run(|_app, _event| {});
}

/// Tears down the AI stack chasm started, then exits the process. The lifecycle
/// brings koboldcpp + TTS up on game connect; if we exit while they're up they'd
/// be orphaned, so we stop them first. Reuses the same env-derived config the
/// server ran with (env is still set), so it resolves the same endpoints/ports.
fn quit(app: &tauri::AppHandle) {
    QUITTING.store(true, Ordering::SeqCst);
    info!("Quit requested — tearing down the AI stack and exiting");
    // Best-effort + bounded: stop_ai_stack shells out to taskkill/netstat, so run
    // it off the main thread and don't let a hang block exit forever.
    let handle = std::thread::spawn(|| {
        chasm_web::shutdown_ai_stack(AppConfig::from_env());
    });
    let _ = handle.join();
    app.exit(0);
}
