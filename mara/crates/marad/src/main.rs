//! Thin daemon entrypoint over the `mara-daemon` library — the explicit,
//! installable form (`systemd`/`launchd` target this binary directly).
//! `mara serve` (CLI autostart) runs the exact same library code via a
//! different, eight-line `main`; see the master plan's *Why both `marad`
//! and `mara serve`*.

use clap::Parser;
use mara_daemon::Config;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "marad", version, about = "mara vector database daemon")]
struct Cli {
    /// Path to the TOML config file.
    #[arg(long, value_name = "PATH", default_value = "mara.toml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    let config = Config::load(&cli.config)?;

    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| config.log.level.clone().into());
    tracing_subscriber::fmt().with_env_filter(filter).init();

    mara_daemon::run(config).await
}
