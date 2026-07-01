//! Standalone FNV bridge binary. Reads the same `nvbridge.config.json` as the
//! Node helper, so running it is a drop-in replacement: stop Node, run this.
//!
//!   chasm-fnv-bridge --config <nvbridge.config.json> [--force]
//!   chasm-fnv-bridge --replay  <native-root-or-request-dir>

use std::path::PathBuf;

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "chasm_fnv_bridge=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let mut config: Option<PathBuf> = None;
    let mut replay: Option<PathBuf> = None;
    let mut force = false;
    let mut turn_selftest: Option<String> = None;
    let mut stt_selftest: Option<PathBuf> = None;
    let mut admin_selftest: Option<String> = None;
    let mut npc = String::from("easy_pete");

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => config = args.next().map(PathBuf::from),
            "--replay" => replay = args.next().map(PathBuf::from),
            "--force" => force = true,
            // Offline smoke test: run one full turn (resolve→generate→TTS) against
            // the live backend and print the result. Needs --config.
            "--turn-selftest" => turn_selftest = args.next(),
            "--stt-selftest" => stt_selftest = args.next().map(PathBuf::from),
            "--admin-selftest" => admin_selftest = args.next(),
            "--npc" => {
                if let Some(value) = args.next() {
                    npc = value;
                }
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage:\n  chasm-fnv-bridge --config <nvbridge.config.json> [--force]\n  chasm-fnv-bridge --replay <native-root-or-request-dir>\n  chasm-fnv-bridge --config <cfg> --turn-selftest \"<message>\" [--npc <npc_key>]"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    if let Some(dir) = replay {
        return chasm_fnv_bridge::replay::run_replay(&dir);
    }

    let config_path = config.ok_or_else(|| {
        anyhow::anyhow!("missing --config <nvbridge.config.json> (or use --replay <dir>)")
    })?;
    let config = chasm_fnv_bridge::load_config(&config_path)?;

    if let Some(message) = turn_selftest {
        return chasm_fnv_bridge::turn_selftest(&config, &npc, &message).await;
    }
    if let Some(wav) = stt_selftest {
        return chasm_fnv_bridge::stt_selftest(&config, &wav).await;
    }
    if let Some(message) = admin_selftest {
        return chasm_fnv_bridge::admin_selftest(&config, &message).await;
    }
    chasm_fnv_bridge::run(config, force).await
}
