//! Local-runtime launch helpers (model swaps + warm-ups).
//!
//! chasm is a passive backend now — it does NOT launch the game (the user starts
//! Fallout: New Vegas through Mod Organizer 2 themselves). What remains here is the
//! plumbing that flips the live LLM / Whisper / TTS model by rewriting the helper
//! config and respawning the local runtime, plus the retrieval warm-up the router
//! fires at startup, and the helper-config-driven `RuntimeSpec` resolution they
//! share.

use std::{net::ToSocketAddrs, path::Path, sync::Arc, time::Duration};

use chasm_core::{AppSettings, LauncherConfig, LauncherSettings};

use crate::AppState;

/// TCP connect timeout for the worker/endpoint reachability checks. Short so a
/// model-swap never hangs when a service is down.
const PORT_TIMEOUT: Duration = Duration::from_millis(1000);

/// Default llama.cpp / koboldcpp host/port (the helper config's `llm.host`/
/// `llm.port` / `llm.endpoint` override these). koboldcpp serves the LLM AND
/// Whisper STT on this one port.
const DEFAULT_LLAMA_HOST: &str = "127.0.0.1";
const DEFAULT_LLAMA_PORT: u16 = 8080;

/// Builds the launch command preview string for display on the Game settings page:
/// the MO2 exe path followed by the quoted `moshortcut://` argument. chasm no
/// longer runs this — it's shown so the user can copy it into MO2 / a shortcut.
pub fn launch_command_string(cfg: &LauncherConfig) -> String {
    format!("{} \"{}\"", cfg.mo2_exe.display(), cfg.moshortcut_arg())
}

/// A resolved, runnable local-runtime command (koboldcpp / llama.cpp / TTS),
/// built from the helper config JSON. `env` entries are merged into the child's
/// environment; `cwd` (when set) is the working directory.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeSpec {
    program: String,
    args: Vec<String>,
    cwd: Option<String>,
    env: Vec<(String, String)>,
}


// ---------------------------------------------------------------------------
// Helper config JSON → runtime launch specs
// ---------------------------------------------------------------------------

/// Reads + parses the helper config JSON at `path`. Returns `None` when the file
/// is missing or not valid JSON, so a misconfigured machine launches the game
/// only instead of failing.
fn load_helper_config(path: &str) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// The `localRuntimes.<key>` sub-object (`"llm"` / `"stt"`) of the helper config.
fn runtime_config<'a>(config: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
    config.get("localRuntimes")?.get(key)
}

/// A JSON value as a trimmed, non-empty owned string (objects/arrays excluded).
fn json_str(value: Option<&serde_json::Value>) -> Option<String> {
    let s = value?.as_str()?.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// A JSON number/numeric-string as a `u64` (for ports, ctx sizes, etc.).
fn json_u64(value: Option<&serde_json::Value>) -> Option<u64> {
    match value? {
        serde_json::Value::Number(n) => n.as_u64(),
        serde_json::Value::String(s) => s.trim().parse::<u64>().ok(),
        _ => None,
    }
}

/// First present key among `keys` on `obj` (mirrors the JS `a ?? b ?? c` reads,
/// e.g. `gpuLayers ?? nGpuLayers ?? n_gpu_layers`).
fn first_present<'a>(obj: &'a serde_json::Value, keys: &[&str]) -> Option<&'a serde_json::Value> {
    keys.iter().find_map(|k| obj.get(*k))
}

/// Builds the llama.cpp [`RuntimeSpec`] from `localRuntimes.llm`, mirroring the
/// reference `getLlamaCppSpawnArgs` in `src/endpoints/fnv-bridge.js`.
///
/// `program` = `llm.command`; `cwd` = `llm.cwd` (else the command's parent dir).
/// If `llm.args` is an array it is used verbatim; otherwise the args are built
/// from `modelPath`/`host`/`port`/`gpuLayers`/`contextSize`/`parallel`/
/// `noWarmup`/`reasoning*`. Returns `None` when no `command` is configured.
fn build_llm_spec(config: &serde_json::Value) -> Option<RuntimeSpec> {
    let llm = runtime_config(config, "llm")?;
    let program = json_str(first_present(llm, &["command", "executable", "path"]))?;

    let cwd = json_str(llm.get("cwd")).or_else(|| {
        Path::new(&program)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.display().to_string())
    });

    // `args` array passthrough wins over the built args.
    let args = if let Some(arr) = llm.get("args").and_then(|v| v.as_array()) {
        arr.iter().map(json_value_to_arg).collect::<Vec<String>>()
    } else {
        build_llm_args(llm)
    };

    Some(RuntimeSpec {
        program,
        args,
        cwd,
        env: Vec::new(),
    })
}

/// Builds llama-server CLI args from `localRuntimes.llm` (the non-`args` path),
/// applying the exact rules from `getLlamaCppSpawnArgs`.
fn build_llm_args(llm: &serde_json::Value) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();

    if let Some(model) = json_str(first_present(llm, &["modelPath", "model"])) {
        args.push("--model".to_string());
        args.push(model);
    }

    let host = json_str(llm.get("host")).unwrap_or_else(|| DEFAULT_LLAMA_HOST.to_string());
    args.push("--host".to_string());
    args.push(host);

    let port = llm_port(llm).unwrap_or(DEFAULT_LLAMA_PORT);
    args.push("--port".to_string());
    args.push(port.to_string());

    // `--n-gpu-layers`: include unless the backend is cpu AND gpuLayers is unset.
    let backend = json_str(first_present(llm, &["backend", "acceleration", "device"]))
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let gpu_layers = json_u64(first_present(
        llm,
        &["gpuLayers", "nGpuLayers", "n_gpu_layers"],
    ));
    if backend != "cpu" || gpu_layers.is_some() {
        args.push("--n-gpu-layers".to_string());
        args.push(gpu_layers.unwrap_or(999).to_string());
    }

    if let Some(ctx) = json_u64(first_present(llm, &["contextSize", "ctxSize", "ctx_size"])) {
        args.push("--ctx-size".to_string());
        args.push(ctx.to_string());
    }

    if let Some(parallel) = json_u64(first_present(llm, &["parallel", "nParallel", "n_parallel"])) {
        args.push("--parallel".to_string());
        args.push(parallel.to_string());
    }

    let no_warmup = first_present(llm, &["noWarmup", "no_warmup"])
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if no_warmup {
        args.push("--no-warmup".to_string());
    }

    if let Some(reasoning) = json_str(first_present(llm, &["reasoning", "thinking"]))
        .map(|s| s.to_ascii_lowercase())
        .filter(|s| matches!(s.as_str(), "on" | "off" | "auto"))
    {
        args.push("--reasoning".to_string());
        args.push(reasoning);
    }

    if let Some(fmt) = json_str(first_present(llm, &["reasoningFormat", "reasoning_format"]))
        .map(|s| s.to_ascii_lowercase())
        .filter(|s| matches!(s.as_str(), "none" | "deepseek" | "deepseek-legacy"))
    {
        args.push("--reasoning-format".to_string());
        args.push(fmt);
    }

    // reasoning-budget: any present, non-empty value (stringified).
    if let Some(budget) = first_present(llm, &["reasoningBudget", "reasoning_budget"]) {
        let s = json_value_to_arg(budget);
        if !s.is_empty() {
            args.push("--reasoning-budget".to_string());
            args.push(s);
        }
    }

    args
}

/// Builds the faster-qwen3-tts [`RuntimeSpec`] from `localRuntimes.tts`: `program`
/// = `tts.command`; `args` = `tts.args` (verbatim string array); `cwd` = `tts.cwd`;
/// `env` = the `tts.env` map merged into the child env. Returns `None` when no
/// `command` is configured. The TTS service is a Python OpenAI-compatible server
/// (`qwen3_tts_server.py --voices voices.json …`), so it maps to a
/// command/args/cwd/env shape.
fn build_tts_spec(config: &serde_json::Value) -> Option<RuntimeSpec> {
    let tts = runtime_config(config, "tts")?;
    let program = json_str(first_present(tts, &["command", "executable", "path"]))?;
    let args = tts
        .get("args")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().map(json_value_to_arg).collect::<Vec<String>>())
        .unwrap_or_default();
    let cwd = json_str(tts.get("cwd"));
    let env = tts
        .get("env")
        .and_then(|v| v.as_object())
        .map(|map| {
            map.iter()
                .map(|(k, v)| (k.clone(), json_value_to_arg(v)))
                .collect::<Vec<(String, String)>>()
        })
        .unwrap_or_default();
    Some(RuntimeSpec {
        program,
        args,
        cwd,
        env,
    })
}

// ---------------------------------------------------------------------------
// Picker-authoritative TTS engine resolution
// ---------------------------------------------------------------------------

/// The port parsed from a `host:port` authority (default 5002, the TTS port, for
/// unparseable input). Used for both the TTS port and — when killing koboldcpp to
/// reload Whisper — the LLM/STT port (whose authority always carries a port).
fn port_from_addr(addr: &str) -> u16 {
    addr.rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(5002)
}

/// The helper config path from settings (blank → built-in default).
fn helper_config_path(launcher: &LauncherSettings) -> String {
    // Blank = no helper config (the default for a fresh install). Every reader of
    // this path treats an unreadable/empty path as "use built-in defaults", so we
    // no longer point at any developer-specific path.
    launcher.helper_config.trim().to_string()
}

/// Builds the spawn spec for the SELECTED local TTS engine — the bridge between
/// the Settings → TTS picker and what actually serves :5002. Both engines now spawn
/// from a chasm-managed `engines/<id>` venv:
///   * `pockettts`        → [`build_pockettts_spec`]
///   * `faster-qwen3-tts` → [`build_qwen3_spec`]
/// A developer who still has a helper config (`localRuntimes.tts`) keeps that path
/// for faster-qwen3-tts: it takes precedence ([`build_tts_spec`]) so an existing
/// hand-configured install is never overridden. `None` ⇒ engine not installed (or
/// `""`/unknown engine ⇒ none selected). The empty string never falls through to a
/// default — nothing is auto-selected for a public release.
fn tts_spec_for_engine(
    engine: &str,
    config: Option<&serde_json::Value>,
    state: &Arc<AppState>,
    port: u16,
) -> Option<RuntimeSpec> {
    match engine {
        "pockettts" => build_pockettts_spec(state, port),
        "faster-qwen3-tts" => {
            // A developer helper config's `localRuntimes.tts` wins (keeps an existing
            // dev install working); otherwise the chasm-managed venv.
            config
                .and_then(build_tts_spec)
                .or_else(|| build_qwen3_spec(state, port))
        }
        // "" (none selected) or any unknown engine → no spec, never a default.
        _ => None,
    }
}

/// Builds the faster-qwen3-tts [`RuntimeSpec`]: the `engines/faster-qwen3-tts` venv
/// python running `scripts/qwen3_tts_server.py` on the TTS port, pointed at the
/// active profile's voices dir (the server auto-builds its voice map from the
/// `<name>/reference.wav` layout via `--voices-dir`). Mirrors [`build_pockettts_spec`]
/// exactly. Returns `None` when the venv or script is missing (engine not installed),
/// so Play reports "not installed" instead of spawning a broken engine.
fn build_qwen3_spec(state: &Arc<AppState>, port: u16) -> Option<RuntimeSpec> {
    let python = state
        .config
        .engines_dir
        .join("faster-qwen3-tts")
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    if !python.exists() {
        return None;
    }
    let script = state
        .config
        .workspace_root
        .join("scripts")
        .join("qwen3_tts_server.py");
    if !script.exists() {
        return None;
    }
    let voices_dir = crate::active_voices_dir(&state.config);
    Some(RuntimeSpec {
        program: python.display().to_string(),
        args: vec![
            script.display().to_string(),
            "--voices-dir".to_string(),
            voices_dir.display().to_string(),
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            port.to_string(),
        ],
        cwd: Some(state.config.workspace_root.display().to_string()),
        env: Vec::new(),
    })
}

/// Builds the PocketTTS [`RuntimeSpec`]: the `engines/pockettts` venv python running
/// `scripts/pockettts_server.py` on the TTS port, pointed at the active profile's
/// voices dir. Returns `None` when the venv or script is missing (engine not
/// installed), so Play reports "not installed" instead of spawning a broken engine.
fn build_pockettts_spec(state: &Arc<AppState>, port: u16) -> Option<RuntimeSpec> {
    let python = state
        .config
        .engines_dir
        .join("pockettts")
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    if !python.exists() {
        return None;
    }
    let script = state
        .config
        .workspace_root
        .join("scripts")
        .join("pockettts_server.py");
    if !script.exists() {
        return None;
    }
    let voices_dir = crate::active_voices_dir(&state.config);
    Some(RuntimeSpec {
        program: python.display().to_string(),
        args: vec![
            script.display().to_string(),
            "--voices-dir".to_string(),
            voices_dir.display().to_string(),
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            port.to_string(),
        ],
        cwd: Some(state.config.workspace_root.display().to_string()),
        env: Vec::new(),
    })
}

/// Whether the faster-qwen3-tts engine is set up. Two ways it can be installed:
///   1. The chasm-managed `engines/faster-qwen3-tts` venv (the public-release path):
///      its `.venv/Scripts/python.exe` + `scripts/qwen3_tts_server.py` both exist.
///   2. A developer's helper config `localRuntimes.tts` resolving to an existing
///      python (+ server script when the first arg looks like a path).
/// Mirrors [`build_pockettts_spec`]'s install check for case 1. Used by the
/// Settings → TTS picker so faster-qwen3-tts reflects its real state.
pub(crate) fn faster_qwen3_tts_installed(settings: &AppSettings, config: &chasm_core::AppConfig) -> bool {
    // 1) Managed venv install.
    let managed_python = config
        .engines_dir
        .join("faster-qwen3-tts")
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    let server_script = config
        .workspace_root
        .join("scripts")
        .join("qwen3_tts_server.py");
    if managed_python.exists() && server_script.exists() {
        return true;
    }

    // 2) Developer helper config.
    let Some(helper) = load_helper_config(&helper_config_path(&settings.launcher)) else {
        return false;
    };
    let Some(spec) = build_tts_spec(&helper) else {
        return false;
    };
    if !Path::new(&spec.program).exists() {
        return false;
    }
    match spec.args.first() {
        Some(first) if !first.starts_with('-') => Path::new(first).exists(),
        _ => true,
    }
}

/// Path to the marker recording which engine currently serves :5002, so a Play /
/// Save can tell whether the running service matches the selected engine (and skip
/// a needless model reload when it does).
fn active_tts_engine_marker(state: &Arc<AppState>) -> std::path::PathBuf {
    state.config.engines_dir.join(".active-tts-engine")
}

/// The engine id currently serving :5002 (per the marker), or `None` if unknown.
fn read_active_tts_engine(state: &Arc<AppState>) -> Option<String> {
    let text = std::fs::read_to_string(active_tts_engine_marker(state)).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Records the engine now serving :5002 (best-effort).
fn write_active_tts_engine(state: &Arc<AppState>, engine: &str) {
    let path = active_tts_engine_marker(state);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(error) = std::fs::write(&path, engine) {
        tracing::debug!("could not write active TTS engine marker: {error}");
    }
}

/// Kills every running TTS engine server — faster-qwen3-tts (`qwen3_tts_server.py`)
/// and PocketTTS (`pockettts_server.py`) — by command-line match, so switching
/// engines fully unloads the previous model from VRAM. Belt-and-suspenders beyond
/// the port-based kill, which only frees `:5002` and can miss an orphaned/duplicate
/// process not bound to the port. Best-effort; a no-op when nothing matches.
#[cfg(windows)]
fn kill_tts_servers() {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let _ = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | \
             Where-Object { $_.CommandLine -like '*qwen3_tts_server.py*' -or $_.CommandLine -like '*pockettts_server.py*' } | \
             ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output();
}

#[cfg(not(windows))]
fn kill_tts_servers() {}

/// The engine currently serving :5002 for the picker's "Running" badge: the
/// active-engine marker, but only when the TTS port is actually reachable (so a
/// stale marker from a dead service doesn't claim to be running). A closed
/// localhost port refuses instantly, so this never stalls the settings page.
pub(crate) fn tts_running_engine(state: &Arc<AppState>) -> Option<String> {
    let addr = authority_from_url(&state.config.tts_endpoint)?;
    if !tcp_reachable(&addr) {
        return None;
    }
    read_active_tts_engine(state)
}

/// Whether koboldcpp (LLM + Whisper STT — one process, one port) is reachable
/// right now. Mirrors the exact address [`start_ai_stack`] targets (helper config
/// authority, else the managed default), so the model-status lights agree with
/// what the launcher spawns. A closed localhost port refuses instantly.
pub(crate) fn koboldcpp_running(state: &Arc<AppState>) -> bool {
    let settings = AppSettings::load(&state.config.settings_path);
    let config = load_helper_config(&helper_config_path(&settings.launcher));
    let llm_addr = config
        .as_ref()
        .and_then(llm_authority_from_config)
        .unwrap_or_else(|| DEFAULT_STACK_LLM_ADDR.to_string());
    tcp_reachable(&llm_addr)
}

/// Applies the currently-selected TTS engine to :5002 right now: kill whatever is
/// serving the port, then spawn the selected engine. Called when the picker changes
/// on Save so the swap takes effect for the in-settings voice Test and the next
/// in-game line without waiting for a Play. Best-effort; reads settings fresh.
/// Blocking (sleeps briefly between kill + spawn) — run it off the async path.
pub(crate) fn apply_selected_tts_engine(state: &Arc<AppState>) {
    let settings = AppSettings::load(&state.config.settings_path);
    let engine = chasm_core::normalize_local_engine(&settings.tts.local_engine);
    let config = load_helper_config(&helper_config_path(&settings.launcher));
    let tts_addr = authority_from_url(&state.config.tts_endpoint)
        .unwrap_or_else(|| "127.0.0.1:5002".to_string());
    let port = port_from_addr(&tts_addr);
    let Some(spec) = tts_spec_for_engine(&engine, config.as_ref(), state, port) else {
        tracing::info!("TTS engine '{engine}' not installed; not applying on save");
        return;
    };
    // Unload any TTS engine already in memory (the active one + any orphan) before
    // loading the selected one, so two models never share VRAM.
    kill_tts_servers();
    if tcp_reachable(&tts_addr) {
        crate::kill_process_on_port(port);
    }
    std::thread::sleep(Duration::from_millis(500));
    match spawn_runtime(&spec) {
        Ok(()) => {
            write_active_tts_engine(state, &engine);
            tracing::info!("applied TTS engine '{engine}' on save ({tts_addr})");
        }
        Err(error) => tracing::warn!("could not apply TTS engine '{engine}' on save: {error}"),
    }
}

// ---------------------------------------------------------------------------
// Connection-driven AI stack lifecycle (start the whole stack when the game
// connects, tear it down when it leaves). See [`crate::stack_lifecycle`].
// ---------------------------------------------------------------------------

/// Default koboldcpp (LLM + STT) authority used when the helper config can't be
/// read or carries no port. koboldcpp serves the LLM and Whisper STT on one port.
const DEFAULT_STACK_LLM_ADDR: &str = "127.0.0.1:5001";
/// Default TTS authority used when the configured `tts_endpoint` is unparseable.
const DEFAULT_STACK_TTS_ADDR: &str = "127.0.0.1:5002";

/// Spawns the FULL AI stack (koboldcpp for LLM + Whisper STT, and the selected
/// TTS engine) from the helper config, the same source the model-swap paths use.
/// This is the un-gated launch (not the change-gated swap): each runtime is only
/// spawned when nothing is already listening on its port, so calling this when a
/// service is already up is a cheap no-op (never a double-spawn / two models in
/// VRAM). Best-effort + blocking (reachability probes + spawn), so the lifecycle
/// task runs it via `spawn_blocking`. Reads settings + config fresh.
pub(crate) fn start_ai_stack(state: &Arc<AppState>) {
    let settings = AppSettings::load(&state.config.settings_path);
    let config = load_helper_config(&helper_config_path(&settings.launcher));

    // --- koboldcpp (LLM + Whisper STT) ---
    let llm_addr = config
        .as_ref()
        .and_then(llm_authority_from_config)
        .unwrap_or_else(|| DEFAULT_STACK_LLM_ADDR.to_string());
    // Helper-config spec FIRST (a developer's hand-configured install must keep
    // working), then the chasm-MANAGED spec (the public-release path: downloaded
    // koboldcpp + the selected/downloaded LLM, no helper config needed).
    let llm_spec = config
        .as_ref()
        .and_then(build_llm_spec)
        .or_else(|| build_managed_koboldcpp_spec(&settings, &state.config));
    if tcp_reachable(&llm_addr) {
        tracing::debug!("AI stack: koboldcpp already up on {llm_addr}; not spawning");
    } else if let Some(spec) = llm_spec {
        match spawn_runtime(&spec) {
            Ok(()) => tracing::info!("AI stack: spawned koboldcpp (LLM+STT) on {llm_addr}"),
            Err(error) => tracing::warn!("AI stack: could not spawn koboldcpp: {error}"),
        }
    } else {
        // No helper config AND nothing selected/downloaded. Surface clearly (one
        // line) instead of silently hanging on "starting".
        tracing::info!(
            "AI stack: LLM not selected/downloaded (or koboldcpp not installed); LLM/STT not starting"
        );
    }

    // --- TTS (the selected local engine) ---
    let tts_addr = authority_from_url(&state.config.tts_endpoint)
        .unwrap_or_else(|| DEFAULT_STACK_TTS_ADDR.to_string());
    if tcp_reachable(&tts_addr) {
        tracing::debug!("AI stack: TTS already up on {tts_addr}; not spawning");
    } else {
        let engine = chasm_core::normalize_local_engine(&settings.tts.local_engine);
        let port = port_from_addr(&tts_addr);
        match tts_spec_for_engine(&engine, config.as_ref(), state, port) {
            Some(spec) => match spawn_runtime(&spec) {
                Ok(()) => {
                    write_active_tts_engine(state, &engine);
                    tracing::info!("AI stack: spawned TTS engine '{engine}' on {tts_addr}");
                }
                Err(error) => {
                    tracing::warn!("AI stack: could not spawn TTS engine '{engine}': {error}")
                }
            },
            None => tracing::warn!("AI stack: TTS engine '{engine}' not installed; TTS not started"),
        }
    }
}

/// Tears the FULL AI stack down: kills every koboldcpp process (LLM + STT) and
/// every TTS engine server (faster-qwen3-tts + PocketTTS), freeing their VRAM.
/// We kill by the known koboldcpp image name + the TTS server command-line match
/// (belt) and by the LLM/TTS ports (suspenders) — both scoped to chasm's own
/// stack, so an unrelated process is never touched. Best-effort; a no-op when
/// nothing is running.
pub(crate) fn stop_ai_stack(state: &Arc<AppState>) {
    let settings = AppSettings::load(&state.config.settings_path);
    let config = load_helper_config(&helper_config_path(&settings.launcher));

    // koboldcpp: by image name, then free its port.
    kill_llm_servers();
    let llm_addr = config
        .as_ref()
        .and_then(llm_authority_from_config)
        .unwrap_or_else(|| DEFAULT_STACK_LLM_ADDR.to_string());
    if tcp_reachable(&llm_addr) {
        crate::kill_process_on_port(port_from_addr(&llm_addr));
    }

    // TTS: by server command-line match, then free its port.
    kill_tts_servers();
    let tts_addr = authority_from_url(&state.config.tts_endpoint)
        .unwrap_or_else(|| DEFAULT_STACK_TTS_ADDR.to_string());
    if tcp_reachable(&tts_addr) {
        crate::kill_process_on_port(port_from_addr(&tts_addr));
    }
    tracing::info!("AI stack: stopped (koboldcpp + TTS killed)");
}

/// Stringifies a JSON arg value: strings verbatim, numbers/bools via `to_string`,
/// everything else (null/array/object) as empty.
fn json_value_to_arg(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        _ => String::new(),
    }
}

/// The configured llama.cpp port: `llm.port`, else the port in `llm.endpoint`.
fn llm_port(llm: &serde_json::Value) -> Option<u16> {
    if let Some(p) = json_u64(llm.get("port")) {
        return u16::try_from(p).ok();
    }
    let endpoint = json_str(first_present(llm, &["endpoint", "apiUrl", "url"]))?;
    authority_from_url(&endpoint)
        .as_deref()
        .and_then(|a| a.rsplit(':').next())
        .and_then(|p| p.parse::<u16>().ok())
}

/// The koboldcpp/llama.cpp reachability authority (`host:port`) derived from the
/// config (also the Whisper STT port — koboldcpp serves both on one port).
fn llm_authority_from_config(config: &serde_json::Value) -> Option<String> {
    let llm = runtime_config(config, "llm")?;
    let host = json_str(llm.get("host")).unwrap_or_else(|| DEFAULT_LLAMA_HOST.to_string());
    let port = llm_port(llm).unwrap_or(DEFAULT_LLAMA_PORT);
    Some(format!("{host}:{port}"))
}

// ---------------------------------------------------------------------------
// Whisper model (koboldcpp `--whispermodel`) resolution + swap
// ---------------------------------------------------------------------------

/// The Whisper models directory: where koboldcpp's `.bin` Whisper builds live.
/// Resolved from the helper config's `localRuntimes.llm` `--whispermodel` arg's
/// parent dir (so it tracks the real koboldcpp install), falling back to the
/// koboldcpp dir's `models` subfolder next to its `command`/`cwd`, then to a
/// sensible default. Env override: `CHASM_WHISPER_MODELS_DIR`.
pub(crate) fn whisper_models_dir(settings: &AppSettings) -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("CHASM_WHISPER_MODELS_DIR") {
        return std::path::PathBuf::from(dir);
    }
    let config = load_helper_config(&helper_config_path(&settings.launcher));
    // 1) Parent of the configured --whispermodel path.
    if let Some(path) = config.as_ref().and_then(whisper_path_from_config) {
        if let Some(parent) = Path::new(&path).parent() {
            if !parent.as_os_str().is_empty() {
                return parent.to_path_buf();
            }
        }
    }
    // 2) <koboldcpp cwd or command dir>/models.
    if let Some(llm) = config.as_ref().and_then(|c| runtime_config(c, "llm")) {
        let base = json_str(llm.get("cwd")).or_else(|| {
            json_str(first_present(llm, &["command", "executable", "path"]))
                .and_then(|cmd| Path::new(&cmd).parent().map(|p| p.display().to_string()))
        });
        if let Some(base) = base {
            return Path::new(&base).join("models");
        }
    }
    // 3) Last-ditch default: a managed per-user dir under chasm's home, so STT can
    //    download and find its Whisper `.bin` with no helper config at all.
    chasm_core::chasm_home().join("models").join("stt")
}

/// Extracts the configured `--whispermodel` path from the helper config's
/// `localRuntimes.llm.args` (koboldcpp loads Whisper from this flag at launch).
fn whisper_path_from_config(config: &serde_json::Value) -> Option<String> {
    let llm = runtime_config(config, "llm")?;
    let args = llm.get("args")?.as_array()?;
    let pos = args
        .iter()
        .position(|v| v.as_str() == Some("--whispermodel"))?;
    json_str(args.get(pos + 1))
}

// ---------------------------------------------------------------------------
// LLM model swap (picker-authoritative; relaunch koboldcpp with the new --model)
// ---------------------------------------------------------------------------

/// Rewrites the `--model <path>` argument inside `localRuntimes.llm.args` of the
/// helper config JSON at `path` to `model_path`, preserving every other key +
/// arg (pretty-printed back). koboldcpp loads its weights from this `--model`
/// flag at launch, so this is what makes a model swap stick across the relaunch.
///
/// Only rewrites when the value actually changes (so repeated saves don't churn
/// the file). Returns `Ok(true)` when the file was changed, `Ok(false)` when it
/// was already correct or there was no `--model` arg to update. A missing config
/// file is `Ok(false)` (nothing to sync).
fn set_llm_model_arg(path: &str, model_path: &str) -> std::io::Result<bool> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    let mut value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;

    // Walk to localRuntimes.llm.args (a string array), bailing cleanly if the
    // shape isn't what we expect rather than rewriting something unrelated.
    let Some(args) = value
        .get_mut("localRuntimes")
        .and_then(|v| v.get_mut("llm"))
        .and_then(|v| v.get_mut("args"))
        .and_then(|v| v.as_array_mut())
    else {
        return Ok(false);
    };

    // Find the value slot following the `--model` flag.
    let model_idx = args
        .iter()
        .position(|a| a.as_str() == Some("--model"))
        .map(|i| i + 1)
        .filter(|&i| i < args.len());
    let Some(model_idx) = model_idx else {
        return Ok(false);
    };
    if args[model_idx].as_str() == Some(model_path) {
        return Ok(false); // already correct - skip the write
    }
    args[model_idx] = serde_json::Value::String(model_path.to_string());

    let mut json = serde_json::to_string_pretty(&value)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    json.push('\n');
    std::fs::write(path, json)?;
    Ok(true)
}

/// Kills every running koboldcpp process by image name, so switching the active
/// LLM fully unloads the previous model from VRAM. Belt-and-suspenders beyond the
/// port-based kill (which only frees `:5001` and can miss a process that hasn't
/// bound the port yet, e.g. still loading weights). Best-effort; a no-op when
/// nothing matches.
#[cfg(windows)]
fn kill_llm_servers() {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let _ = Command::new("taskkill")
        .args(["/F", "/IM", "koboldcpp.exe"])
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

#[cfg(not(windows))]
fn kill_llm_servers() {}

/// Applies the currently-selected LLM model right now: point the helper config's
/// koboldcpp `--model` at the selected GGUF, then unload the running model
/// (kill koboldcpp + free `:5001`) and relaunch koboldcpp on the new weights.
/// koboldcpp loads `--model` only at launch, so a full reload is expected here -
/// the previous model IS unloaded before the new one loads (never two in VRAM).
///
/// Called from the LLM settings save when the model selection changed, so the
/// swap takes effect for the next in-game line without waiting for a Play.
/// Best-effort + blocking (it sleeps briefly between kill + spawn), so the caller
/// runs it off the async path. Reads settings + config fresh.
pub(crate) fn apply_selected_llm_model(state: &Arc<AppState>) {
    let settings = AppSettings::load(&state.config.settings_path);
    let model_status = llm_model_statuses_for(&state.config.llm_models_dir);
    let selected =
        chasm_core::selected_llm_model_id(&settings.llm.model, &model_status);
    if selected.is_empty() {
        tracing::info!("no LLM model selected/downloaded; not applying on save");
        return;
    }
    let Some(gguf) = chasm_core::llm_model_gguf_path(&state.config.llm_models_dir, &selected)
    else {
        return;
    };
    if !gguf.exists() {
        tracing::info!(
            "selected LLM '{selected}' not downloaded yet ({}); not relaunching",
            gguf.display()
        );
        return;
    }

    // Point koboldcpp's --model at the selected GGUF in the helper config so the
    // relaunch (and every future Play) loads it.
    let config_path = helper_config_path(&settings.launcher);
    match set_llm_model_arg(&config_path, &gguf.display().to_string()) {
        Ok(_) => {}
        Err(error) => {
            tracing::warn!("could not set koboldcpp --model in helper config: {error}");
            return;
        }
    }

    // Build the (now-updated) spawn spec + reachability address. Helper-config spec
    // FIRST (developer install), then the chasm-MANAGED spec (public-release path).
    // No spec at all means koboldcpp isn't downloaded yet — nothing to relaunch.
    let config = load_helper_config(&config_path);
    let Some(spec) = config
        .as_ref()
        .and_then(build_llm_spec)
        .or_else(|| build_managed_koboldcpp_spec(&settings, &state.config))
    else {
        tracing::info!(
            "LLM --model set to {} but koboldcpp not installed / no spawnable command; \
             it will load on the next start",
            gguf.display()
        );
        return;
    };
    let llm_addr = config
        .as_ref()
        .and_then(llm_authority_from_config)
        .unwrap_or_else(|| DEFAULT_STACK_LLM_ADDR.to_string());
    let port = llm_addr
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(DEFAULT_LLAMA_PORT);

    // Unload the running model (by name + by port) before loading the new one, so
    // two models never share VRAM.
    kill_llm_servers();
    if tcp_reachable(&llm_addr) {
        crate::kill_process_on_port(port);
    }
    std::thread::sleep(Duration::from_millis(800));
    match spawn_runtime(&spec) {
        Ok(()) => tracing::info!(
            "applied LLM model '{selected}' on save: relaunched koboldcpp ({llm_addr}) on {}",
            gguf.display()
        ),
        Err(error) => tracing::warn!("could not relaunch koboldcpp for '{selected}': {error}"),
    }
}

/// The download status of each LLM model keyed by id, read from the on-disk GGUFs
/// in `models_dir` (mirrors the web layer's `llm_model_statuses`, duplicated here
/// so the launcher's model-swap path doesn't depend on it). Only the `downloaded`
/// state is load-bearing for the swap (a stem-matching `*.gguf` is present).
fn llm_model_statuses_for(
    models_dir: &Path,
) -> std::collections::HashMap<String, String> {
    let gguf_names: Vec<String> = std::fs::read_dir(models_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_lowercase();
            name.ends_with(".gguf").then_some(name)
        })
        .collect();
    chasm_core::LLM_MODELS
        .iter()
        .filter_map(|model| {
            let stem = chasm_core::llm_model_match_stem(model);
            gguf_names
                .iter()
                .any(|name| name.contains(&stem))
                .then(|| (model.id.to_string(), "downloaded".to_string()))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Whisper model swap (rewrite --whispermodel + force a koboldcpp relaunch)
// ---------------------------------------------------------------------------

/// Applies the selected Whisper model so the NEXT koboldcpp launch uses it AND
/// the previously-loaded model is guaranteed gone:
///   1. Rewrite `--whispermodel <path>` in the helper config's `localRuntimes.llm.args`.
///   2. Best-effort rewrite of the `--whispermodel "<path>"` line in
///      `koboldcpp/start_kobold.bat` (so the manual start script matches).
///   3. Kill koboldcpp on the LLM/STT port so the model unloads now; the next Play
///      relaunches koboldcpp with the new `--whispermodel`.
///
/// TRADEOFF (documented for the user): koboldcpp loads `--whispermodel` only at
/// process start and has no per-slot hot-swap for the Whisper model, so the only
/// way to GUARANTEE the old model is freed from VRAM is to restart the process -
/// which also reloads the (large) LLM. We accept that one-time reload cost in
/// exchange for a correct unload. `model_file` is a Whisper `.bin` filename
/// (a [`chasm_core::WhisperModel::file`]); it is resolved against
/// [`whisper_models_dir`]. Best-effort + blocking - run it off the async path.
pub(crate) fn apply_selected_whisper_model(state: &Arc<AppState>, model_file: &str) {
    let settings = AppSettings::load(&state.config.settings_path);
    let models_dir = whisper_models_dir(&settings);
    let model_path = models_dir.join(model_file);
    let model_path_str = model_path.display().to_string();

    // 1) Helper config JSON.
    let config_path = helper_config_path(&settings.launcher);
    if let Err(error) = write_whisper_model_in_config(&config_path, &model_path_str) {
        tracing::warn!("could not set --whispermodel in helper config: {error}");
    }

    // 2) start_kobold.bat (next to the koboldcpp command/cwd), best-effort.
    if let Some(bat) = kobold_start_bat(&settings) {
        if let Err(error) = rewrite_whispermodel_in_bat(&bat, &model_path_str) {
            tracing::debug!("could not update start_kobold.bat (ignored): {error}");
        }
    }

    // 3) Kill koboldcpp so the old whisper model is freed; next Play relaunches it
    //    with the new model. This unloads the LLM too (koboldcpp is one process).
    let llm_addr = authority_from_url(&state.config.llm_endpoint)
        .or_else(|| authority_from_url(&state.config.stt_endpoint));
    if let Some(addr) = llm_addr {
        if tcp_reachable(&addr) {
            let port = port_from_addr(&addr);
            tracing::info!(
                "Whisper model -> {model_file}; stopping koboldcpp on :{port} so it reloads with the new --whispermodel"
            );
            crate::kill_process_on_port(port);
        }
    }
}

/// Rewrites (or inserts) `--whispermodel <path>` in the helper config's
/// `localRuntimes.llm.args`, preserving every other key. Only writes when the
/// value actually changes. Missing file / no `llm` section means Ok (nothing to do).
fn write_whisper_model_in_config(path: &str, model_path: &str) -> std::io::Result<()> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    let mut value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let Some(args) = value
        .get_mut("localRuntimes")
        .and_then(|v| v.get_mut("llm"))
        .and_then(|v| v.get_mut("args"))
        .and_then(|v| v.as_array_mut())
    else {
        return Ok(()); // no args array to update
    };

    // Find the existing flag's value slot, else append the flag + value.
    if let Some(pos) = args.iter().position(|v| v.as_str() == Some("--whispermodel")) {
        if let Some(slot) = args.get_mut(pos + 1) {
            if slot.as_str() == Some(model_path) {
                return Ok(()); // already correct
            }
            *slot = serde_json::Value::String(model_path.to_string());
        } else {
            args.push(serde_json::Value::String(model_path.to_string()));
        }
    } else {
        args.push(serde_json::Value::String("--whispermodel".to_string()));
        args.push(serde_json::Value::String(model_path.to_string()));
    }

    let mut json = serde_json::to_string_pretty(&value)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    json.push('\n');
    std::fs::write(path, json)
}

/// Resolves `koboldcpp/start_kobold.bat` from the helper config's koboldcpp cwd /
/// command dir, returning it only if it exists.
fn kobold_start_bat(settings: &AppSettings) -> Option<std::path::PathBuf> {
    let config = load_helper_config(&helper_config_path(&settings.launcher))?;
    let llm = runtime_config(&config, "llm")?;
    let base = json_str(llm.get("cwd")).or_else(|| {
        json_str(first_present(llm, &["command", "executable", "path"]))
            .and_then(|cmd| Path::new(&cmd).parent().map(|p| p.display().to_string()))
    })?;
    let bat = Path::new(&base).join("start_kobold.bat");
    bat.exists().then_some(bat)
}

// ---------------------------------------------------------------------------
// koboldcpp runtime auto-download (the exe that runs the LLM + Whisper STT)
// ---------------------------------------------------------------------------

/// Install status of the koboldcpp runtime (the exe koboldcpp serves the LLM AND
/// Whisper STT from). `Installed` once the exe exists at the resolved path;
/// `Downloading` while the `koboldcpp.downloading` marker is present; `Missing`
/// otherwise (so a model download can kick off the runtime fetch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KoboldcppStatus {
    Installed,
    Downloading,
    Missing,
}

impl KoboldcppStatus {
    /// The status string the UI keys its pill class + label off (matching the
    /// `install-pill is-…` classes already used for engines/models).
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            KoboldcppStatus::Installed => "installed",
            KoboldcppStatus::Downloading => "downloading",
            KoboldcppStatus::Missing => "missing",
        }
    }
}

/// Resolves the koboldcpp exe path the launcher should use, preferring the path
/// the helper config already points its `localRuntimes.llm.command` at (so existing
/// users keep their install), then a managed default under the workspace
/// (`<workspace>/koboldcpp/koboldcpp.exe`) where the auto-download lands. Env
/// override: `CHASM_KOBOLDCPP_EXE`.
pub(crate) fn koboldcpp_exe_path(
    settings: &AppSettings,
    config: &chasm_core::AppConfig,
) -> std::path::PathBuf {
    if let Some(env) = std::env::var_os("CHASM_KOBOLDCPP_EXE") {
        return std::path::PathBuf::from(env);
    }
    // 1) The koboldcpp command the helper config already launches, when it looks
    //    like a koboldcpp exe (so we never clobber a hand-configured install).
    if let Some(helper) = load_helper_config(&helper_config_path(&settings.launcher)) {
        if let Some(llm) = runtime_config(&helper, "llm") {
            if let Some(cmd) = json_str(first_present(llm, &["command", "executable", "path"])) {
                let lower = cmd.to_ascii_lowercase();
                if lower.contains("koboldcpp") && lower.ends_with(".exe") {
                    return std::path::PathBuf::from(cmd);
                }
            }
        }
    }
    // 2) Managed default: chasm downloads koboldcpp here when no install exists.
    koboldcpp_managed_default(config)
}

/// The chasm-managed koboldcpp install path (`<workspace>/koboldcpp/koboldcpp.exe`),
/// where the auto-download writes the exe + its `.downloading`/`.done`/`.failed`
/// markers. Kept separate so the download endpoint + the status check agree on the
/// location without re-reading the helper config.
pub(crate) fn koboldcpp_managed_default(
    config: &chasm_core::AppConfig,
) -> std::path::PathBuf {
    config
        .workspace_root
        .join("koboldcpp")
        .join("koboldcpp.exe")
}

/// Builds the chasm-MANAGED koboldcpp [`RuntimeSpec`] for a public-release install
/// that has NO developer helper config. This is the fallback `start_ai_stack` uses
/// when the helper-config spec is absent, and it is what makes the local LLM + STT
/// run for a downloaded-from-the-panel user.
///
/// Resolution (all from settings + on-disk state, no helper config):
///   * program = [`koboldcpp_exe_path`]'s managed default, but only if the exe
///     exists (koboldcpp downloaded). Missing exe ⇒ `None` (don't launch).
///   * `--model` = the SELECTED + downloaded LLM GGUF (settings `llm.model` resolved
///     via [`selected_llm_model_id`] + [`llm_model_gguf_path`]). No selection / not
///     downloaded ⇒ `None` (the caller logs "LLM not selected/downloaded").
///   * `--whispermodel` = the SELECTED + present Whisper `.bin` (settings `stt.model`
///     under [`whisper_models_dir`]) — included ONLY when selected + present; STT is
///     optional, so its absence never blocks the LLM from starting.
///   * args mirror the real koboldcpp config: `--usecublas --model <gguf>
///     --gpulayers 999 --contextsize 8192 [--whispermodel <bin>] --port 5001
///     --host 127.0.0.1`. cwd = the exe's dir.
fn build_managed_koboldcpp_spec(
    settings: &AppSettings,
    config: &chasm_core::AppConfig,
) -> Option<RuntimeSpec> {
    // koboldcpp must be downloaded.
    let exe = koboldcpp_exe_path(settings, config);
    if !exe.exists() {
        return None;
    }

    // The LLM must be selected AND its GGUF present — there is no default selection.
    let model_status = llm_model_statuses_for(&config.llm_models_dir);
    let selected = chasm_core::selected_llm_model_id(&settings.llm.model, &model_status);
    if selected.is_empty() {
        return None;
    }
    let gguf = chasm_core::llm_model_gguf_path(&config.llm_models_dir, &selected)?;
    if !gguf.exists() {
        return None;
    }

    let cwd = exe
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.display().to_string());

    let mut args: Vec<String> = vec![
        "--usecublas".to_string(),
        "--model".to_string(),
        gguf.display().to_string(),
        "--gpulayers".to_string(),
        "999".to_string(),
        "--contextsize".to_string(),
        "8192".to_string(),
    ];

    // Whisper STT is optional: add --whispermodel only when a model is selected AND
    // its .bin is on disk, so STT never blocks the LLM from starting.
    let whisper_file = chasm_core::stt_effective_model(&settings.stt);
    if !whisper_file.is_empty() {
        let whisper_path = whisper_models_dir(settings).join(&whisper_file);
        if whisper_path.exists() {
            args.push("--whispermodel".to_string());
            args.push(whisper_path.display().to_string());
        } else {
            tracing::info!(
                "managed koboldcpp: Whisper model '{whisper_file}' selected but not downloaded ({}); starting LLM without STT",
                whisper_path.display()
            );
        }
    }

    args.push("--port".to_string());
    args.push("5001".to_string());
    args.push("--host".to_string());
    args.push("127.0.0.1".to_string());

    Some(RuntimeSpec {
        program: exe.display().to_string(),
        args,
        cwd,
        env: Vec::new(),
    })
}

/// The koboldcpp runtime status: `Installed` when the resolved exe exists,
/// `Downloading` when a `koboldcpp.downloading` marker sits beside it (an
/// in-flight auto-download), else `Missing`. A present exe always wins (existing
/// users are never asked to re-download).
pub(crate) fn koboldcpp_status(
    settings: &AppSettings,
    config: &chasm_core::AppConfig,
) -> KoboldcppStatus {
    let exe = koboldcpp_exe_path(settings, config);
    if exe.exists() {
        return KoboldcppStatus::Installed;
    }
    let Some(dir) = exe.parent() else {
        return KoboldcppStatus::Missing;
    };
    let marker = dir.join("koboldcpp.downloading");
    // Flip a stalled download marker to failed (progress = koboldcpp.log + the
    // .exe.part curl is writing), so a dead spawn can't show "Downloading" forever.
    crate::flip_marker_if_stale(
        &marker,
        &dir.join("koboldcpp.failed"),
        &[dir.join("koboldcpp.log"), exe.with_extension("exe.part")],
    );
    if marker.exists() {
        KoboldcppStatus::Downloading
    } else {
        KoboldcppStatus::Missing
    }
}

/// Rewrites the `--whispermodel "<old>"` argument in a koboldcpp `.bat` launch
/// script to point at `model_path`, preserving the rest of the line. Matches the
/// flag whether or not its value is quoted; writes the new value quoted. A no-op
/// (Ok) when the flag isn't present.
fn rewrite_whispermodel_in_bat(bat: &Path, model_path: &str) -> std::io::Result<()> {
    let text = std::fs::read_to_string(bat)?;
    let mut out_lines: Vec<String> = Vec::with_capacity(text.lines().count());
    let mut changed = false;
    for line in text.lines() {
        if let Some(idx) = line.find("--whispermodel") {
            let (prefix, rest) = line.split_at(idx + "--whispermodel".len());
            // `rest` begins with the separator + the (maybe quoted) path, possibly
            // followed by more args / a `^` continuation. Replace just the path token.
            let trimmed = rest.trim_start();
            let lead_ws = &rest[..rest.len() - trimmed.len()];
            let (_, tail) = split_first_token(trimmed);
            let new_line = format!("{prefix}{lead_ws}\"{model_path}\"{tail}");
            out_lines.push(new_line);
            changed = true;
        } else {
            out_lines.push(line.to_string());
        }
    }
    if !changed {
        return Ok(());
    }
    let mut joined = out_lines.join("\r\n");
    if text.ends_with('\n') {
        joined.push_str("\r\n");
    }
    std::fs::write(bat, joined)
}

/// Splits off the first whitespace-delimited token, honouring a leading
/// double-quoted span (so a quoted path with spaces is one token). Returns
/// `(token, remainder)` where `remainder` keeps its leading separator/whitespace.
fn split_first_token(s: &str) -> (&str, &str) {
    if let Some(stripped) = s.strip_prefix('"') {
        if let Some(end) = stripped.find('"') {
            // token spans the quotes; remainder is everything after the closing quote
            return (&s[..end + 2], &s[end + 2..]);
        }
    }
    match s.find(|c: char| c.is_whitespace()) {
        Some(idx) => (&s[..idx], &s[idx..]),
        None => (s, ""),
    }
}

// NOTE: the retrieval warm-up (embedder + reranker + catalog vectors) moved to
// `crate::warmup::spawn_stack_warmup`, which folds it into the full connect-time
// stack warm-up (LLM KV prefix, Whisper, TTS first-inference) with one summary
// log line.

/// Spawns a configured local runtime (koboldcpp / TTS service) detached +
/// hidden, with its cwd + merged env applied and stdio nulled.
#[cfg(windows)]
fn spawn_runtime(spec: &RuntimeSpec) -> std::io::Result<()> {
    use std::os::windows::process::CommandExt;
    use std::process::{Command, Stdio};
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let mut cmd = Command::new(&spec.program);
    cmd.args(&spec.args);
    if let Some(cwd) = &spec.cwd {
        if !cwd.is_empty() {
            cmd.current_dir(cwd);
        }
    }
    for (key, val) in &spec.env {
        cmd.env(key, val);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .map(|_child| ())
}

#[cfg(not(windows))]
fn spawn_runtime(spec: &RuntimeSpec) -> std::io::Result<()> {
    use std::process::{Command, Stdio};
    let mut cmd = Command::new(&spec.program);
    cmd.args(&spec.args);
    if let Some(cwd) = &spec.cwd {
        if !cwd.is_empty() {
            cmd.current_dir(cwd);
        }
    }
    for (key, val) in &spec.env {
        cmd.env(key, val);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|_child| ())
}

/// Whether a TCP connect to `addr` (`host:port`) succeeds within [`PORT_TIMEOUT`].
/// Used for the soft worker/endpoint reachability warnings.
fn tcp_reachable(addr: &str) -> bool {
    let Ok(mut addrs) = addr.to_socket_addrs() else {
        return false;
    };
    addrs.any(|socket| std::net::TcpStream::connect_timeout(&socket, PORT_TIMEOUT).is_ok())
}

/// Extracts a `host:port` authority from a base URL like
/// `http://127.0.0.1:8080` or `https://example.com/v1`. Defaults the port from
/// the scheme (80/443) when the URL omits it. Returns `None` for unparseable
/// input.
fn authority_from_url(url: &str) -> Option<String> {
    let trimmed = url.trim();
    let (scheme, rest) = match trimmed.split_once("://") {
        Some((scheme, rest)) => (scheme.to_ascii_lowercase(), rest),
        None => (String::new(), trimmed),
    };
    // Authority ends at the first '/', '?', or '#'.
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("")
        .trim_end_matches('.');
    if authority.is_empty() {
        return None;
    }
    // Strip any userinfo.
    let host_port = authority.rsplit('@').next().unwrap_or(authority);
    if host_port.contains(':') {
        Some(host_port.to_string())
    } else {
        let port = if scheme == "https" { 443 } else { 80 };
        Some(format!("{host_port}:{port}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_defaults_port_from_scheme() {
        assert_eq!(
            authority_from_url("http://127.0.0.1:8080").as_deref(),
            Some("127.0.0.1:8080")
        );
        assert_eq!(
            authority_from_url("http://127.0.0.1:8080/v1/chat").as_deref(),
            Some("127.0.0.1:8080")
        );
        assert_eq!(
            authority_from_url("http://localhost").as_deref(),
            Some("localhost:80")
        );
        assert_eq!(
            authority_from_url("https://api.example.com/v1").as_deref(),
            Some("api.example.com:443")
        );
        assert_eq!(authority_from_url("").as_deref(), None);
    }

    #[test]
    fn unreachable_port_is_false_fast() {
        // Nothing listens on this port → false (and returns within the timeout).
        assert!(!tcp_reachable("127.0.0.1:1"));
    }

    #[test]
    fn set_llm_model_arg_rewrites_only_the_model_value() {
        let dir = std::env::temp_dir().join(format!("sb-llm-arg-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("nvbridge.config.json");
        // Mirror the real config shape: koboldcpp args with --model + siblings.
        let config = serde_json::json!({
            "liveChatStreaming": true,
            "localRuntimes": {
                "llm": {
                    "command": "C:\\kobold\\koboldcpp.exe",
                    "args": [
                        "--usecublas",
                        "--model", "C:\\old\\a.gguf",
                        "--gpulayers", "999",
                        "--port", "5001"
                    ]
                }
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
        let p = path.display().to_string();

        // First write changes the value and returns true.
        assert!(set_llm_model_arg(&p, "C:\\new\\b.gguf").unwrap());
        let after: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let args = after["localRuntimes"]["llm"]["args"].as_array().unwrap();
        let model_idx = args.iter().position(|a| a == "--model").unwrap();
        assert_eq!(args[model_idx + 1], "C:\\new\\b.gguf");
        // Every other arg is preserved verbatim.
        assert_eq!(args[0], "--usecublas");
        assert_eq!(args.last().unwrap(), "5001");
        // Unrelated keys preserved.
        assert_eq!(after["liveChatStreaming"], serde_json::json!(true));

        // Idempotent: the same value is a no-op write (returns false).
        assert!(!set_llm_model_arg(&p, "C:\\new\\b.gguf").unwrap());

        // Missing file → Ok(false) (nothing to sync), never an error.
        let missing = dir.join("nope.json").display().to_string();
        assert!(!set_llm_model_arg(&missing, "x").unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_llm_model_arg_bails_without_model_flag() {
        let dir = std::env::temp_dir().join(format!("sb-llm-arg2-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("c.json");
        // No --model in args → nothing to rewrite, returns false, file untouched.
        let config = serde_json::json!({
            "localRuntimes": { "llm": { "args": ["--port", "5001"] } }
        });
        let text = serde_json::to_string_pretty(&config).unwrap();
        std::fs::write(&path, &text).unwrap();
        let p = path.display().to_string();
        assert!(!set_llm_model_arg(&p, "x.gguf").unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A sample `localRuntimes` for the llama.cpp arg-builder tests (the non-`args`
    /// path, where args are derived from fields like modelPath/host/port/…).
    fn sample_config() -> serde_json::Value {
        serde_json::json!({
            "liveChatStreaming": false,
            "localRuntimes": {
                "llm": {
                    "runner": "llama.cpp",
                    "endpoint": "http://127.0.0.1:8080",
                    "host": "127.0.0.1",
                    "port": 8080,
                    "command": "C:\\llama\\llama-server.exe",
                    "cwd": "C:\\llama",
                    "backend": "cuda",
                    "gpuLayers": 999,
                    "contextSize": 8192,
                    "parallel": 1,
                    "noWarmup": true,
                    "reasoning": "off",
                    "reasoningFormat": "none",
                    "reasoningBudget": 0,
                    "modelPath": "C:\\models\\gemma.gguf"
                }
            }
        })
    }

    /// A sample config mirroring the real koboldcpp helper config: the `llm`
    /// section uses an `args` array (passed verbatim) that includes
    /// `--whispermodel`, since koboldcpp serves LLM + Whisper STT in one process.
    fn kobold_config() -> serde_json::Value {
        serde_json::json!({
            "localRuntimes": {
                "llm": {
                    "runner": "koboldcpp",
                    "endpoint": "http://127.0.0.1:5001",
                    "host": "127.0.0.1",
                    "port": 5001,
                    "command": "C:\\kobold\\koboldcpp.exe",
                    "cwd": "C:\\kobold",
                    "args": [
                        "--usecublas",
                        "--model", "C:\\models\\gemma.gguf",
                        "--gpulayers", "999",
                        "--contextsize", "8192",
                        "--whispermodel", "C:\\kobold\\models\\whisper-small-q5_1.bin",
                        "--port", "5001"
                    ]
                }
            }
        })
    }

    #[test]
    fn llm_spec_builds_args_in_reference_order() {
        let config = sample_config();
        let spec = build_llm_spec(&config).expect("llm spec");
        assert_eq!(spec.program, "C:\\llama\\llama-server.exe");
        assert_eq!(spec.cwd.as_deref(), Some("C:\\llama"));
        assert!(spec.env.is_empty());
        // Matches getLlamaCppSpawnArgs: model, host, port, n-gpu-layers, ctx-size,
        // parallel, no-warmup, reasoning, reasoning-format. (reasoning-budget 0
        // stringifies to "0" — present and non-empty — so it is included.)
        assert_eq!(
            spec.args,
            vec![
                "--model",
                "C:\\models\\gemma.gguf",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--n-gpu-layers",
                "999",
                "--ctx-size",
                "8192",
                "--parallel",
                "1",
                "--no-warmup",
                "--reasoning",
                "off",
                "--reasoning-format",
                "none",
                "--reasoning-budget",
                "0",
            ]
        );
    }

    #[test]
    fn llm_spec_uses_args_array_verbatim_when_present() {
        let config = serde_json::json!({
            "localRuntimes": {
                "llm": {
                    "command": "/usr/bin/llama-server",
                    "args": ["--model", "/m/x.gguf", "--port", "9090", "--flash-attn"]
                }
            }
        });
        let spec = build_llm_spec(&config).expect("llm spec");
        assert_eq!(
            spec.args,
            vec!["--model", "/m/x.gguf", "--port", "9090", "--flash-attn"]
        );
        // cwd falls back to the command's parent dir when `cwd` is absent.
        assert_eq!(spec.cwd.as_deref(), Some("/usr/bin"));
    }

    #[test]
    fn llm_spec_omits_gpu_layers_for_cpu_backend_without_override() {
        let config = serde_json::json!({
            "localRuntimes": { "llm": {
                "command": "llama-server",
                "modelPath": "/m/x.gguf",
                "backend": "cpu"
            }}
        });
        let spec = build_llm_spec(&config).expect("llm spec");
        assert!(
            !spec.args.iter().any(|a| a == "--n-gpu-layers"),
            "cpu + no gpuLayers should omit --n-gpu-layers: {:?}",
            spec.args
        );
        // But an explicit gpuLayers forces it back in even on cpu.
        let config_forced = serde_json::json!({
            "localRuntimes": { "llm": {
                "command": "llama-server",
                "modelPath": "/m/x.gguf",
                "backend": "cpu",
                "gpuLayers": 0
            }}
        });
        let spec = build_llm_spec(&config_forced).expect("llm spec");
        let idx = spec.args.iter().position(|a| a == "--n-gpu-layers");
        assert_eq!(idx.map(|i| spec.args[i + 1].as_str()), Some("0"));
    }

    #[test]
    fn whisper_path_parsed_from_kobold_config() {
        // koboldcpp serves STT via --whispermodel in the llm args; the parser pulls
        // the value, and the models dir is its parent.
        let config = kobold_config();
        assert_eq!(
            whisper_path_from_config(&config).as_deref(),
            Some("C:\\kobold\\models\\whisper-small-q5_1.bin")
        );
        // Missing flag → None.
        let no_flag = serde_json::json!({
            "localRuntimes": { "llm": { "args": ["--model", "x.gguf", "--port", "5001"] } }
        });
        assert!(whisper_path_from_config(&no_flag).is_none());
    }

    #[test]
    fn tts_spec_maps_program_args_cwd_env_from_json() {
        let config = serde_json::json!({
            "localRuntimes": { "tts": {
                "command": "C:\\venv\\python.exe",
                "args": ["examples\\openai_server.py", "--voices", "voices.json", "--port", "5002"],
                "cwd": "C:\\faster-qwen3-tts",
                "env": { "HF_HUB_OFFLINE": "0" }
            }}
        });
        let spec = build_tts_spec(&config).expect("tts spec");
        assert_eq!(spec.program, "C:\\venv\\python.exe");
        assert_eq!(
            spec.args,
            vec![
                "examples\\openai_server.py",
                "--voices",
                "voices.json",
                "--port",
                "5002"
            ]
        );
        assert_eq!(spec.cwd.as_deref(), Some("C:\\faster-qwen3-tts"));
        let env: std::collections::HashMap<_, _> = spec.env.iter().cloned().collect();
        assert_eq!(env.get("HF_HUB_OFFLINE").map(String::as_str), Some("0"));
    }

    #[test]
    fn port_derivation_from_config_and_overrides() {
        let config = sample_config();
        assert_eq!(
            llm_authority_from_config(&config).as_deref(),
            Some("127.0.0.1:8080")
        );
        // koboldcpp config: the LLM/STT authority is the single :5001 port.
        assert_eq!(
            llm_authority_from_config(&kobold_config()).as_deref(),
            Some("127.0.0.1:5001")
        );

        // LLM port derives from the endpoint URL when `port` is absent.
        let config3 = serde_json::json!({
            "localRuntimes": { "llm": { "endpoint": "http://127.0.0.1:8085" } }
        });
        assert_eq!(
            llm_authority_from_config(&config3).as_deref(),
            Some("127.0.0.1:8085")
        );
    }

    #[test]
    fn missing_runtime_sections_yield_no_spec() {
        let config = serde_json::json!({ "localRuntimes": {} });
        assert!(build_llm_spec(&config).is_none());
        assert!(build_tts_spec(&config).is_none());
        // No command → no spec even if the section exists.
        let config2 = serde_json::json!({ "localRuntimes": { "llm": { "modelPath": "x" } } });
        assert!(build_llm_spec(&config2).is_none());
    }

    #[test]
    fn write_whisper_model_in_config_updates_arg_in_place() {
        let dir = std::env::temp_dir().join(format!("sb-whisper-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nvbridge.config.json");
        let path_str = path.to_string_lossy().to_string();
        // Seed with a koboldcpp-shaped config (args array incl. --whispermodel).
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&kobold_config()).unwrap(),
        )
        .unwrap();

        write_whisper_model_in_config(&path_str, "D:\\w\\ggml-large-v3-turbo.bin").unwrap();
        let after: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let args = after["localRuntimes"]["llm"]["args"].as_array().unwrap();
        let pos = args
            .iter()
            .position(|v| v.as_str() == Some("--whispermodel"))
            .unwrap();
        assert_eq!(
            args[pos + 1].as_str(),
            Some("D:\\w\\ggml-large-v3-turbo.bin")
        );
        // Other args preserved (still has --model + --port).
        assert!(args.iter().any(|v| v.as_str() == Some("--model")));
        assert!(args.iter().any(|v| v.as_str() == Some("--port")));

        // Inserts the flag when absent.
        let no_flag = serde_json::json!({
            "localRuntimes": { "llm": { "args": ["--model", "x.gguf", "--port", "5001"] } }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&no_flag).unwrap()).unwrap();
        write_whisper_model_in_config(&path_str, "D:\\w\\ggml-base.bin").unwrap();
        let after2: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let args2 = after2["localRuntimes"]["llm"]["args"].as_array().unwrap();
        let pos2 = args2
            .iter()
            .position(|v| v.as_str() == Some("--whispermodel"))
            .unwrap();
        assert_eq!(args2[pos2 + 1].as_str(), Some("D:\\w\\ggml-base.bin"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rewrite_whispermodel_in_bat_replaces_quoted_path() {
        let dir = std::env::temp_dir().join(format!("sb-whisper-bat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bat = dir.join("start_kobold.bat");
        // A realistic multi-line .bat with a quoted --whispermodel + trailing ^.
        let original = "@echo off\r\n\"C:\\kobold\\koboldcpp.exe\" ^\r\n  --model \"C:\\m\\g.gguf\" ^\r\n  --whispermodel \"C:\\kobold\\models\\whisper-small-q5_1.bin\" ^\r\n  --port 5001\r\n";
        std::fs::write(&bat, original).unwrap();

        rewrite_whispermodel_in_bat(&bat, "D:\\w\\ggml-large-v3-turbo.bin").unwrap();
        let after = std::fs::read_to_string(&bat).unwrap();
        assert!(after.contains("--whispermodel \"D:\\w\\ggml-large-v3-turbo.bin\" ^"));
        // The old path is gone; the rest of the line (the ^ continuation) survives.
        assert!(!after.contains("whisper-small-q5_1.bin"));
        assert!(after.contains("--model \"C:\\m\\g.gguf\""));
        assert!(after.contains("--port 5001"));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Builds a throwaway `AppConfig` rooted at `workspace` for the koboldcpp
    /// resolution tests (only `workspace_root` is read by those paths).
    fn test_config(workspace: &Path) -> chasm_core::AppConfig {
        chasm_core::AppConfig {
            bind_addr: "127.0.0.1:0".to_string(),
            data_root: workspace.to_path_buf(),
            workspace_root: workspace.to_path_buf(),
            settings_path: workspace.join("settings.json"),
            engines_dir: workspace.join("engines"),
            profiles_dir: workspace.join("profiles"),
            voices_dir: workspace.join("voices"),
            llm_models_dir: workspace.join("models").join("llm"),
            stt_endpoint: "http://127.0.0.1:5001".to_string(),
            llm_endpoint: "http://127.0.0.1:5001".to_string(),
            tts_endpoint: "http://127.0.0.1:5002".to_string(),
        }
    }

    #[test]
    fn koboldcpp_status_tracks_exe_and_marker() {
        let dir = std::env::temp_dir().join(format!("sb-kobold-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // No helper config / no managed exe → Missing. Point helper_config at a
        // missing file so the resolver skips any real config on the host and falls
        // through to the managed default under our temp workspace.
        let mut settings = AppSettings::default();
        settings.launcher.helper_config = dir.join("no-such-config.json").display().to_string();
        let config = test_config(&dir);
        // Make sure no env override leaks in from the host.
        std::env::remove_var("CHASM_KOBOLDCPP_EXE");
        let exe = koboldcpp_exe_path(&settings, &config);
        assert_eq!(exe, dir.join("koboldcpp").join("koboldcpp.exe"));
        assert_eq!(koboldcpp_status(&settings, &config), KoboldcppStatus::Missing);

        // A `.downloading` marker beside the exe → Downloading.
        std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
        std::fs::write(dir.join("koboldcpp").join("koboldcpp.downloading"), "").unwrap();
        assert_eq!(
            koboldcpp_status(&settings, &config),
            KoboldcppStatus::Downloading
        );

        // The exe present → Installed (wins over the marker).
        std::fs::write(&exe, "binary").unwrap();
        assert_eq!(
            koboldcpp_status(&settings, &config),
            KoboldcppStatus::Installed
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn managed_koboldcpp_spec_needs_exe_and_selected_downloaded_llm() {
        let dir = std::env::temp_dir().join(format!("sb-kobold-mgd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::env::remove_var("CHASM_KOBOLDCPP_EXE");
        // Pin the Whisper dir into our temp workspace so the test never reads/writes
        // the host's real chasm home and stays deterministic.
        let whisper_dir = dir.join("whisper");
        std::env::set_var("CHASM_WHISPER_MODELS_DIR", &whisper_dir);

        let mut settings = AppSettings::default();
        // No real helper config on the host.
        settings.launcher.helper_config = dir.join("no-such-config.json").display().to_string();
        let config = test_config(&dir);

        // 1) koboldcpp not installed → None.
        assert!(build_managed_koboldcpp_spec(&settings, &config).is_none());

        // Install a fake koboldcpp exe.
        let exe = config.workspace_root.join("koboldcpp").join("koboldcpp.exe");
        std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
        std::fs::write(&exe, "binary").unwrap();

        // 2) Exe present but no LLM selected/downloaded → None (no default selection).
        assert!(build_managed_koboldcpp_spec(&settings, &config).is_none());

        // Download a fake GGUF for the first registry model + select it.
        let model = &chasm_core::LLM_MODELS[0];
        let gguf =
            chasm_core::llm_model_gguf_path(&config.llm_models_dir, model.id).unwrap();
        std::fs::create_dir_all(gguf.parent().unwrap()).unwrap();
        std::fs::write(&gguf, "weights").unwrap();
        settings.llm.model = model.id.to_string();

        // 3) Exe + selected + downloaded LLM, no Whisper → spec WITHOUT --whispermodel.
        let spec = build_managed_koboldcpp_spec(&settings, &config).expect("managed spec");
        assert_eq!(spec.program, exe.display().to_string());
        assert_eq!(spec.cwd.as_deref(), exe.parent().map(|p| p.to_str().unwrap()));
        assert!(spec.args.iter().any(|a| a == "--usecublas"));
        let m = spec.args.iter().position(|a| a == "--model").unwrap();
        assert_eq!(spec.args[m + 1], gguf.display().to_string());
        assert!(spec.args.iter().any(|a| a == "--gpulayers"));
        assert!(spec.args.iter().any(|a| a == "--contextsize"));
        let p = spec.args.iter().position(|a| a == "--port").unwrap();
        assert_eq!(spec.args[p + 1], "5001");
        assert!(!spec.args.iter().any(|a| a == "--whispermodel"));

        // 4) Select + download a Whisper .bin → spec gains --whispermodel.
        let whisper = &chasm_core::WHISPER_MODELS[0];
        std::fs::create_dir_all(&whisper_dir).unwrap();
        std::fs::write(whisper_dir.join(whisper.file), "ggml").unwrap();
        settings.stt.model = whisper.file.to_string();
        let spec = build_managed_koboldcpp_spec(&settings, &config).expect("managed spec");
        let w = spec.args.iter().position(|a| a == "--whispermodel").unwrap();
        assert_eq!(
            spec.args[w + 1],
            whisper_dir.join(whisper.file).display().to_string()
        );

        std::env::remove_var("CHASM_WHISPER_MODELS_DIR");
        std::fs::remove_dir_all(&dir).ok();
    }
}
