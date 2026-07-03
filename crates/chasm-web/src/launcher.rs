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

/// Default llama.cpp host/port (the helper config's `llm.host`/
/// `llm.port` / `llm.endpoint` override these). llama.cpp serves the LLM on this
/// port; STT is the dedicated Parakeet server on its own port.
const DEFAULT_LLAMA_HOST: &str = "127.0.0.1";
const DEFAULT_LLAMA_PORT: u16 = 8080;

/// Builds the launch command preview string for display on the Game settings page:
/// the MO2 exe path followed by the quoted `moshortcut://` argument. chasm no
/// longer runs this — it's shown so the user can copy it into MO2 / a shortcut.
pub fn launch_command_string(cfg: &LauncherConfig) -> String {
    format!("{} \"{}\"", cfg.mo2_exe.display(), cfg.moshortcut_arg())
}

/// A resolved, runnable local-runtime command (llama.cpp / Parakeet STT / TTS),
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
/// unparseable input). Used for both the TTS port and — when killing the LLM
/// runtime to reload a model — the LLM port (whose authority always carries a port).
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

// ---------------------------------------------------------------------------
// Parakeet STT engine (dedicated OpenAI-compatible ASR server on its own port)
// ---------------------------------------------------------------------------

/// Whether the Parakeet STT engine is installed: the chasm-managed
/// `engines/parakeet` venv python + `scripts/parakeet_stt_server.py` both exist
/// (the same install shape as the TTS engines). Cheap fs checks, safe per request.
pub(crate) fn parakeet_installed(config: &chasm_core::AppConfig) -> bool {
    let python = config
        .engines_dir
        .join(chasm_core::PARAKEET_ENGINE_ID)
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    let script = config
        .workspace_root
        .join("scripts")
        .join("parakeet_stt_server.py");
    python.exists() && script.exists()
}

/// Whether the effective local STT path is the Parakeet server: the STT provider
/// is the managed-local option AND the engine is installed. Parakeet is now the
/// ONLY managed local STT (the legacy Whisper-in-koboldcpp path was removed); a hosted-API STT
/// provider doesn't use this server at all.
pub(crate) fn stt_uses_parakeet(
    settings: &AppSettings,
    config: &chasm_core::AppConfig,
) -> bool {
    chasm_core::normalize_stt_provider(&settings.stt.provider) == chasm_core::PROVIDER_LOCAL
        && parakeet_installed(config)
}

/// The Parakeet server's reachability authority (`host:port`), from the
/// configured transcription endpoint (default `127.0.0.1:5003`).
fn parakeet_addr(config: &chasm_core::AppConfig) -> String {
    authority_from_url(&config.parakeet_stt_endpoint)
        .unwrap_or_else(|| "127.0.0.1:5003".to_string())
}

/// Whether the Parakeet STT server is reachable right now (for the stack lights).
pub(crate) fn parakeet_running(state: &Arc<AppState>) -> bool {
    tcp_reachable(&parakeet_addr(&state.config))
}

/// Builds the Parakeet STT [`RuntimeSpec`]: the `engines/parakeet` venv python
/// running `scripts/parakeet_stt_server.py` on the Parakeet port. Mirrors
/// [`build_pockettts_spec`]. Returns `None` when the venv or script is missing
/// (engine not installed) so the caller logs + falls back to Whisper instead of
/// spawning a broken server.
fn build_parakeet_spec(state: &Arc<AppState>, port: u16) -> Option<RuntimeSpec> {
    let python = state
        .config
        .engines_dir
        .join(chasm_core::PARAKEET_ENGINE_ID)
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
        .join("parakeet_stt_server.py");
    if !script.exists() {
        return None;
    }
    Some(RuntimeSpec {
        program: python.display().to_string(),
        args: vec![
            script.display().to_string(),
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            port.to_string(),
            "--model".to_string(),
            chasm_core::PARAKEET_HF_REPO.to_string(),
        ],
        cwd: Some(state.config.workspace_root.display().to_string()),
        env: Vec::new(),
    })
}

/// Kills every running Parakeet STT server by command-line match (mirrors
/// [`kill_tts_servers`]), so deselecting Parakeet fully unloads its model from
/// VRAM. Best-effort; a no-op when nothing matches.
#[cfg(windows)]
fn kill_parakeet_servers() {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let _ = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | \
             Where-Object { $_.CommandLine -like '*parakeet_stt_server.py*' } | \
             ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output();
}

#[cfg(not(windows))]
fn kill_parakeet_servers() {}

/// Applies the currently-selected STT provider right now (called when the STT
/// picker changes on save, mirroring [`apply_selected_tts_engine`]):
///   * `local`  → spawn the managed Parakeet server if it isn't already up
///     (installed check inside; not installed = log so the Runtimes page nudge is
///     the fix).
///   * an API provider → kill any running Parakeet server (frees its VRAM); the
///     transcription request is served by the hosted API instead.
/// Best-effort + blocking — run it off the async path.
pub(crate) fn apply_selected_stt_provider(state: &Arc<AppState>) {
    let settings = AppSettings::load(&state.config.settings_path);
    let provider = chasm_core::normalize_stt_provider(&settings.stt.provider);
    let addr = parakeet_addr(&state.config);
    if provider == chasm_core::PROVIDER_LOCAL {
        if tcp_reachable(&addr) {
            tracing::info!("STT provider -> local Parakeet (already serving {addr})");
            return;
        }
        let port = port_from_addr(&addr);
        match build_parakeet_spec(state, port) {
            Some(spec) => match spawn_runtime(&spec) {
                Ok(()) => tracing::info!("STT provider -> local Parakeet: spawned server on {addr}"),
                Err(error) => {
                    tracing::warn!("STT provider -> local Parakeet: could not spawn server: {error}")
                }
            },
            None => tracing::warn!(
                "STT provider -> local selected but the Parakeet engine is not installed \
                 (Settings -> Runtimes)"
            ),
        }
    } else {
        // A hosted API provider is selected: unload the local Parakeet server.
        kill_parakeet_servers();
        if tcp_reachable(&addr) {
            crate::kill_process_on_port(port_from_addr(&addr));
        }
        tracing::info!("STT provider -> {provider} API; local Parakeet server stopped if it ran");
    }
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

/// Stops the managed local TTS engine (kills the server + frees :5002), so
/// switching the TTS provider to a hosted API frees the local model's VRAM.
/// Best-effort + blocking — run it off the async path.
pub(crate) fn stop_tts_engines(state: &Arc<AppState>) {
    kill_tts_servers();
    let tts_addr = authority_from_url(&state.config.tts_endpoint)
        .unwrap_or_else(|| DEFAULT_STACK_TTS_ADDR.to_string());
    if tcp_reachable(&tts_addr) {
        crate::kill_process_on_port(port_from_addr(&tts_addr));
    }
}

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

/// Whether the managed local LLM runtime (llama.cpp `llama-server` on :5001) is
/// reachable right now. Mirrors the exact address [`start_ai_stack`] targets
/// (helper config authority, else the managed default), so the model-status
/// lights agree with what the launcher spawns. A closed localhost port refuses
/// instantly.
pub(crate) fn llm_runtime_running(state: &Arc<AppState>) -> bool {
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

/// Default managed-LLM authority (llama.cpp `llama-server` on :5001) used when the
/// helper config can't be read or carries no port.
const DEFAULT_STACK_LLM_ADDR: &str = "127.0.0.1:5001";
/// Default TTS authority used when the configured `tts_endpoint` is unparseable.
const DEFAULT_STACK_TTS_ADDR: &str = "127.0.0.1:5002";

/// Spawns the FULL managed AI stack (llama.cpp for the LLM, the selected TTS
/// engine, and the Parakeet STT server) from the helper config, the same source
/// the model-swap paths use. This is the un-gated launch (not the change-gated
/// swap): each runtime is only spawned when nothing is already listening on its
/// port, so calling this when a service is already up is a cheap no-op (never a
/// double-spawn / two models in VRAM). Any capability whose provider is a hosted
/// API is SKIPPED — its requests go straight to the API, so no local server is
/// needed. Best-effort + blocking, so the lifecycle task runs it via
/// `spawn_blocking`. Reads settings + config fresh.
pub(crate) fn start_ai_stack(state: &Arc<AppState>) {
    let settings = AppSettings::load(&state.config.settings_path);
    let config = load_helper_config(&helper_config_path(&settings.launcher));

    // --- LLM runtime (managed llama.cpp) — skipped when a hosted API is chosen ---
    let llm_provider = chasm_core::normalize_llm_provider(&settings.llm.provider);
    let llm_addr = config
        .as_ref()
        .and_then(llm_authority_from_config)
        .unwrap_or_else(|| DEFAULT_STACK_LLM_ADDR.to_string());
    let llm_spec = if llm_provider == chasm_core::PROVIDER_LOCAL {
        managed_llm_runtime_spec(&settings, &state.config, config.as_ref())
    } else {
        None
    };
    if llm_provider != chasm_core::PROVIDER_LOCAL {
        tracing::info!("AI stack: LLM provider is {llm_provider} API; not starting a local runtime");
    } else if tcp_reachable(&llm_addr) {
        tracing::debug!("AI stack: LLM runtime already up on {llm_addr}; not spawning");
    } else if let Some(spec) = llm_spec {
        let runtime_name = Path::new(&spec.program)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "llm".to_string());
        match spawn_runtime(&spec) {
            Ok(()) => tracing::info!("AI stack: spawned {runtime_name} (LLM) on {llm_addr}"),
            Err(error) => tracing::warn!("AI stack: could not spawn {runtime_name}: {error}"),
        }
    } else {
        // No helper config AND nothing selected/downloaded. Surface clearly (one
        // line) instead of silently hanging on "starting".
        tracing::info!(
            "AI stack: LLM model not selected/downloaded (or llama.cpp not installed); LLM not starting"
        );
    }

    // --- TTS (the selected local engine) — skipped when a hosted API is chosen ---
    let tts_provider = chasm_core::normalize_tts_provider(&settings.tts.provider);
    let tts_addr = authority_from_url(&state.config.tts_endpoint)
        .unwrap_or_else(|| DEFAULT_STACK_TTS_ADDR.to_string());
    if tts_provider != chasm_core::PROVIDER_LOCAL {
        tracing::info!("AI stack: TTS provider is {tts_provider} API; not starting a local engine");
    } else if tcp_reachable(&tts_addr) {
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

    // --- STT: the managed local Parakeet server (skipped when a hosted API is chosen) ---
    let stt_provider = chasm_core::normalize_stt_provider(&settings.stt.provider);
    if stt_provider == chasm_core::PROVIDER_LOCAL {
        let stt_addr = parakeet_addr(&state.config);
        if tcp_reachable(&stt_addr) {
            tracing::debug!("AI stack: Parakeet STT already up on {stt_addr}; not spawning");
        } else {
            match build_parakeet_spec(state, port_from_addr(&stt_addr)) {
                Some(spec) => match spawn_runtime(&spec) {
                    Ok(()) => {
                        tracing::info!("AI stack: spawned Parakeet STT server on {stt_addr}")
                    }
                    Err(error) => {
                        tracing::warn!("AI stack: could not spawn Parakeet STT server: {error}")
                    }
                },
                None => tracing::warn!(
                    "AI stack: local STT selected but the Parakeet engine is not installed \
                     (Settings -> Runtimes); voice input will error until it is"
                ),
            }
        }
    } else {
        tracing::info!("AI stack: STT provider is {stt_provider} API; not starting local Parakeet");
    }
}

/// Tears the FULL AI stack down: kills the llama.cpp LLM server, the Parakeet STT
/// server, and every TTS engine server (faster-qwen3-tts + PocketTTS), freeing
/// their VRAM. We kill by each server's command-line match (belt) and by the
/// LLM/TTS/STT ports (suspenders) — both scoped to chasm's own stack, so an
/// unrelated process is never touched. Best-effort; a no-op when nothing is
/// running.
pub(crate) fn stop_ai_stack(state: &Arc<AppState>) {
    let settings = AppSettings::load(&state.config.settings_path);
    let config = load_helper_config(&helper_config_path(&settings.launcher));

    // LLM (llama.cpp): by command line, then free its port.
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

    // Parakeet STT: by server command-line match, then free its port.
    kill_parakeet_servers();
    let stt_addr = parakeet_addr(&state.config);
    if tcp_reachable(&stt_addr) {
        crate::kill_process_on_port(port_from_addr(&stt_addr));
    }
    tracing::info!("AI stack: stopped (LLM runtime + TTS + Parakeet killed)");
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

/// The llama.cpp reachability authority (`host:port`) derived from the config.
/// STT is a separate Parakeet server on its own port, not this one.
fn llm_authority_from_config(config: &serde_json::Value) -> Option<String> {
    let llm = runtime_config(config, "llm")?;
    let host = json_str(llm.get("host")).unwrap_or_else(|| DEFAULT_LLAMA_HOST.to_string());
    let port = llm_port(llm).unwrap_or(DEFAULT_LLAMA_PORT);
    Some(format!("{host}:{port}"))
}

// ---------------------------------------------------------------------------
// LLM model swap (picker-authoritative; relaunch llama.cpp with the new --model)
// ---------------------------------------------------------------------------

/// Rewrites the `--model <path>` argument inside `localRuntimes.llm.args` of the
/// helper config JSON at `path` to `model_path`, preserving every other key +
/// arg (pretty-printed back). A developer helper-config runtime loads its weights
/// from this `--model` flag at launch, so this is what makes a model swap stick
/// across the relaunch.
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

/// Kills chasm's running llama.cpp `llama-server` (the managed LLM runtime on
/// :5001), so switching the active LLM fully unloads the previous model from VRAM.
/// Belt-and-suspenders beyond the port-based kill (which only frees `:5001` and
/// can miss a process that hasn't bound the port yet, e.g. still loading weights).
/// Best-effort; a no-op when nothing matches.
#[cfg(windows)]
fn kill_llm_servers() {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    // llama-server: only chasm's own instance (the one serving :5001), never an
    // unrelated llama-server the user runs — match the command line, not the name.
    let _ = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | \
             Where-Object { $_.CommandLine -like '*--port 5001*' } | \
             ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output();
}

#[cfg(not(windows))]
fn kill_llm_servers() {}

/// Applies the currently-selected LLM model right now: point the LLM runtime's
/// model arg at the selected GGUF, then unload the running model
/// (kill the LLM server + free `:5001`) and relaunch it on the new weights.
/// llama-server loads its model only at launch, so a full reload is expected here -
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

    // Point the helper config's `--model` at the selected GGUF so a developer
    // helper-config install's relaunch (and every future Play) loads it. The
    // managed llama.cpp spec reads the selection from settings directly.
    let config_path = helper_config_path(&settings.launcher);
    match set_llm_model_arg(&config_path, &gguf.display().to_string()) {
        Ok(_) => {}
        Err(error) => {
            tracing::warn!("could not set llama.cpp --model in helper config: {error}");
            return;
        }
    }

    // Build the (now-updated) spawn spec + reachability address for the SELECTED
    // runtime (helper-config spec FIRST, then the managed llama-server spec). No
    // spec at all means the runtime isn't downloaded yet — nothing to relaunch.
    let config = load_helper_config(&config_path);
    let Some(spec) = managed_llm_runtime_spec(&settings, &state.config, config.as_ref()) else {
        tracing::info!(
            "LLM --model set to {} but the LLM runtime is not installed / no spawnable \
             command; it will load on the next start",
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
            "applied LLM model '{selected}' on save: relaunched the LLM runtime ({llm_addr}) on {}",
            gguf.display()
        ),
        Err(error) => {
            tracing::warn!("could not relaunch the LLM runtime for '{selected}': {error}")
        }
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
// Managed LLM runtime status (llama.cpp llama-server auto-download)
// ---------------------------------------------------------------------------

/// Install status of the managed local LLM runtime (llama.cpp `llama-server`).
/// `Installed` once the exe exists at the resolved path; `Downloading` while the
/// `.downloading` marker is present; `Missing` otherwise (so a model download can
/// kick off the runtime fetch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeStatus {
    Installed,
    Downloading,
    Missing,
}

impl RuntimeStatus {
    /// The status string the UI keys its pill class + label off (matching the
    /// `install-pill is-…` classes already used for engines/models).
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            RuntimeStatus::Installed => "installed",
            RuntimeStatus::Downloading => "downloading",
            RuntimeStatus::Missing => "missing",
        }
    }
}

/// Finds a vision projector (`*mmproj*.gguf`) in the LLM models dir for the
/// selected model. Preference: the candidate sharing the longest common prefix
/// with the model file's name (case-insensitive) — so `gemma-4-12b-it-mmproj-F16`
/// pairs with `gemma-4-12b-it-UD-Q4_K_XL` even when several projectors exist.
fn find_vision_projector(models_dir: &Path, model_gguf: &Path) -> Option<std::path::PathBuf> {
    let model_name = model_gguf.file_name()?.to_string_lossy().to_lowercase();
    let mut best: Option<(usize, std::path::PathBuf)> = None;
    for entry in std::fs::read_dir(models_dir).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().to_lowercase();
        if !name.ends_with(".gguf") || !name.contains("mmproj") {
            continue;
        }
        let shared = model_name
            .bytes()
            .zip(name.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        if best.as_ref().map(|(len, _)| shared > *len).unwrap_or(true) {
            best = Some((shared, entry.path()));
        }
    }
    best.map(|(_, path)| path)
}

// ---------------------------------------------------------------------------
// llama.cpp (llama-server) — the managed local LLM runtime
// ---------------------------------------------------------------------------

/// Resolves the llama-server exe path: `CHASM_LLAMACPP_EXE` override, else the
/// managed download location under the data tree (`<data>/models/llamacpp/`,
/// beside the LLM/STT model dirs the desktop app already uses).
pub(crate) fn llamacpp_exe_path(config: &chasm_core::AppConfig) -> std::path::PathBuf {
    if let Some(env) = std::env::var_os("CHASM_LLAMACPP_EXE") {
        return std::path::PathBuf::from(env);
    }
    llamacpp_managed_default(config)
}

/// The chasm-managed llama.cpp install path (`<data>/models/llamacpp/llama-server.exe`),
/// where `scripts/download-llamacpp.ps1` extracts the release build + writes its
/// `.downloading`/`.done`/`.failed` markers.
pub(crate) fn llamacpp_managed_default(config: &chasm_core::AppConfig) -> std::path::PathBuf {
    config
        .data_root
        .join("models")
        .join("llamacpp")
        .join("llama-server.exe")
}

/// The managed llama.cpp runtime status (installed / downloading / missing) off
/// the `llamacpp.*` markers beside the `llama-server.exe`.
pub(crate) fn llamacpp_status(config: &chasm_core::AppConfig) -> RuntimeStatus {
    let exe = llamacpp_exe_path(config);
    if exe.exists() {
        return RuntimeStatus::Installed;
    }
    let Some(dir) = exe.parent() else {
        return RuntimeStatus::Missing;
    };
    let marker = dir.join("llamacpp.downloading");
    crate::flip_marker_if_stale(
        &marker,
        &dir.join("llamacpp.failed"),
        &[dir.join("llamacpp.log")],
    );
    if marker.exists() {
        RuntimeStatus::Downloading
    } else {
        RuntimeStatus::Missing
    }
}

/// Builds the chasm-MANAGED llama-server [`RuntimeSpec`] for the selected model,
/// on port 5001 — the port the whole stack (endpoints, bridge, warmup) expects.
///
/// Notable flags:
///   * `--parallel 2` = two server slots, each keeping its OWN prompt cache. In a
///     group scene, speaker A's system prompt + history stay cached in slot 0
///     while speaker B generates in slot 1 (llama-server routes each request to
///     the slot with the most similar cached prompt), so a speaker swap costs a
///     delta prefill instead of a full ~0.4-0.6 s reprocess of the prompt.
///   * `--ctx-size 16384` — the total KV budget is split across slots, so 16384
///     gives 8192 context PER SLOT.
///   * `--cache-reuse 256` — chunked KV reuse for partially-matching prompts
///     (history windows shift as the conversation grows).
///   * `--reasoning-budget 0` — no hidden thinking preamble delaying first audio.
///   * `--jinja` uses the model's own chat template for /v1/chat/completions.
/// Missing exe / no selected+downloaded model ⇒ `None` (caller logs + falls back
/// to the helper-config spec — never breaks the stack).
fn build_managed_llamacpp_spec(
    settings: &AppSettings,
    config: &chasm_core::AppConfig,
) -> Option<RuntimeSpec> {
    let exe = llamacpp_exe_path(config);
    if !exe.exists() {
        return None;
    }

    // Model resolution: the SELECTED + downloaded GGUF, no default selection.
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
        "-m".to_string(),
        gguf.display().to_string(),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        "5001".to_string(),
        "-ngl".to_string(),
        "999".to_string(),
        // 2 slots × 8192 ctx each (--ctx-size is the TOTAL budget, split across
        // slots) = 8192 per-conversation context per slot.
        "-c".to_string(),
        "16384".to_string(),
        "-np".to_string(),
        "2".to_string(),
        // Chunked KV reuse for shifted prefixes (min chunk 256 tokens).
        "--cache-reuse".to_string(),
        "256".to_string(),
        // No thinking preamble delaying first audio.
        "--reasoning-budget".to_string(),
        "0".to_string(),
        // Use the GGUF's own chat template.
        "--jinja".to_string(),
        // Explicit flash attention: measured ~8% generation speedup on the
        // 5090 vs leaving it to auto.
        "-fa".to_string(),
        "on".to_string(),
    ];

    // Vision projector: auto-attach the projector GGUF via --mmproj.
    if let Some(mmproj) = find_vision_projector(&config.llm_models_dir, &gguf) {
        tracing::info!("llama-server: attaching vision projector {}", mmproj.display());
        args.push("--mmproj".to_string());
        args.push(mmproj.display().to_string());
    }

    Some(RuntimeSpec {
        program: exe.display().to_string(),
        args,
        cwd,
        env: Vec::new(),
    })
}

/// The managed LLM runtime spec. llama.cpp `llama-server` is the only managed
/// runtime now, so this prefers a developer helper-config spec (a hand-configured
/// install must keep working) and otherwise builds the managed llama-server spec.
/// `None` when nothing is spawnable (runtime exe or model not present), so the
/// caller logs rather than spawning a broken process.
fn managed_llm_runtime_spec(
    settings: &AppSettings,
    config: &chasm_core::AppConfig,
    helper: Option<&serde_json::Value>,
) -> Option<RuntimeSpec> {
    helper
        .and_then(build_llm_spec)
        .or_else(|| build_managed_llamacpp_spec(settings, config))
}

// NOTE: the retrieval warm-up (embedder + reranker + catalog vectors) moved to
// `crate::warmup::spawn_stack_warmup`, which folds it into the full connect-time
// stack warm-up (LLM KV prefix, Parakeet STT, TTS first-inference) with one
// summary log line.

/// Spawns a configured local runtime (llama.cpp / TTS service) detached +
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

    /// A sample developer helper config whose `llm` section runs a llama.cpp
    /// `llama-server` on :5001 (an `args` array passed verbatim).
    fn llm_helper_config() -> serde_json::Value {
        serde_json::json!({
            "localRuntimes": {
                "llm": {
                    "runner": "llamacpp",
                    "endpoint": "http://127.0.0.1:5001",
                    "host": "127.0.0.1",
                    "port": 5001,
                    "command": "C:\\llama\\llama-server.exe",
                    "cwd": "C:\\llama",
                    "args": [
                        "--model", "C:\\models\\gemma.gguf",
                        "--host", "127.0.0.1",
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
        // Developer helper config: the LLM authority is the configured :5001 port.
        assert_eq!(
            llm_authority_from_config(&llm_helper_config()).as_deref(),
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

    /// Builds a throwaway `AppConfig` rooted at `workspace` for the runtime
    /// resolution tests (only `workspace_root` / `data_root` are read by those paths).
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
            parakeet_stt_endpoint: "http://127.0.0.1:5003/v1/audio/transcriptions".to_string(),
            llm_endpoint: "http://127.0.0.1:5001".to_string(),
            tts_endpoint: "http://127.0.0.1:5002".to_string(),
        }
    }

    #[test]
    fn llamacpp_status_tracks_exe_and_marker() {
        let dir = std::env::temp_dir().join(format!("sb-llamacpp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::env::remove_var("CHASM_LLAMACPP_EXE");
        let config = test_config(&dir);
        let exe = llamacpp_exe_path(&config);
        assert_eq!(
            exe,
            dir.join("models").join("llamacpp").join("llama-server.exe")
        );
        assert_eq!(llamacpp_status(&config), RuntimeStatus::Missing);

        // A `.downloading` marker beside the exe → Downloading.
        std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
        std::fs::write(exe.parent().unwrap().join("llamacpp.downloading"), "").unwrap();
        assert_eq!(llamacpp_status(&config), RuntimeStatus::Downloading);

        // The exe present → Installed (wins over the marker).
        std::fs::write(&exe, "binary").unwrap();
        assert_eq!(llamacpp_status(&config), RuntimeStatus::Installed);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn managed_llamacpp_spec_needs_exe_and_selected_downloaded_llm() {
        let dir = std::env::temp_dir().join(format!("sb-llamacpp-mgd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::env::remove_var("CHASM_LLAMACPP_EXE");

        let mut settings = AppSettings::default();
        let config = test_config(&dir);

        // 1) llama-server not installed → None.
        assert!(build_managed_llamacpp_spec(&settings, &config).is_none());

        // Install a fake llama-server exe at the managed default.
        let exe = llamacpp_managed_default(&config);
        std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
        std::fs::write(&exe, "binary").unwrap();

        // 2) Exe present but no LLM selected/downloaded → None (no default selection).
        assert!(build_managed_llamacpp_spec(&settings, &config).is_none());

        // Download a fake GGUF for the first registry model + select it.
        let model = &chasm_core::LLM_MODELS[0];
        let gguf = chasm_core::llm_model_gguf_path(&config.llm_models_dir, model.id).unwrap();
        std::fs::create_dir_all(gguf.parent().unwrap()).unwrap();
        std::fs::write(&gguf, "weights").unwrap();
        settings.llm.model = model.id.to_string();

        // 3) Exe + selected + downloaded LLM → a valid llama-server spec.
        let spec = build_managed_llamacpp_spec(&settings, &config).expect("managed spec");
        assert_eq!(spec.program, exe.display().to_string());
        let m = spec.args.iter().position(|a| a == "-m").unwrap();
        assert_eq!(spec.args[m + 1], gguf.display().to_string());
        let p = spec.args.iter().position(|a| a == "--port").unwrap();
        assert_eq!(spec.args[p + 1], "5001");
        // Two prompt-cache slots (the whole point of llama-server).
        let np = spec.args.iter().position(|a| a == "-np").unwrap();
        assert_eq!(spec.args[np + 1], "2");

        std::fs::remove_dir_all(&dir).ok();
    }
}
