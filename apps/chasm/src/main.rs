use chasm_core::AppConfig;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "chasm=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Hidden CLI subcommand used by `scripts/download-embed-model.ps1` to force a
    // retrieval model's weights to download (reusing the embed crate's loaders).
    // Usage: `chasm download-embed-model <id>`.
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() == Some("download-embed-model") {
        let id = args
            .next()
            .ok_or_else(|| anyhow::anyhow!("usage: download-embed-model <id>"))?;
        return chasm_embed::download_model(&id);
    }

    let config = AppConfig::from_env();
    chasm_web::serve(config).await
}
