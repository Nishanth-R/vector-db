//! Thin entrypoint over the `mara-api` library — mirrors `marad`'s own
//! `main.rs`, since the two are peer processes over the same
//! `mara-daemon` composition root, just different transports.

use clap::Parser;
use mara_daemon::Config;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "mara-api", version, about = "mara vector database HTTP API")]
struct Cli {
    /// Path to the TOML config file. Must set `[http] listen`.
    #[arg(long, value_name = "PATH", default_value = "mara.toml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    let config = Config::load(&cli.config)?;

    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| config.log.level.clone().into());
    tracing_subscriber::fmt().with_env_filter(filter).init();

    mara_api::run(config).await
}
