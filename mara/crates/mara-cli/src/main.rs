//! `mara-cli` (binary name `mara`): endpoint resolution, autostart,
//! `--embedded`, `mara serve`, and `mara daemon status|stop|logs` — see
//! the master plan's *Autostart & embedded*. Basic storage commands
//! (`create-collection`/`put`/`get`/`delete`) exercise the same
//! `Request`/`Response` path whether they run over a socket or, with the
//! `embedded` feature, in-process.

#[cfg(feature = "embedded")]
mod autostart;
mod daemon_ctl;
#[cfg(feature = "embedded")]
mod embedded;
mod endpoint;
mod repl;
mod request;

use clap::{CommandFactory, Parser, Subcommand};
use mara_client::MaraClient;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "mara", version, about = "mara vector database CLI")]
pub struct Cli {
    /// Path to the daemon's Unix socket. Resolution order: this flag ->
    /// `MARA_ENDPOINT` -> a discovered `mara.toml` -> `<data_dir>/mara.sock`.
    #[arg(long, global = true, value_name = "PATH")]
    endpoint: Option<PathBuf>,

    /// Data directory to use for autostart/--embedded/`daemon` subcommands
    /// when no config file is found. Defaults to `~/.mara`.
    #[arg(long, global = true, value_name = "PATH")]
    data_dir: Option<PathBuf>,

    /// Run this command in-process against `--data-dir`, no daemon socket
    /// involved. Fails if a daemon already owns that data dir.
    #[cfg(feature = "embedded")]
    #[arg(long, global = true)]
    embedded: bool,

    /// Fail immediately instead of autostarting a daemon when nothing
    /// answers at the resolved endpoint.
    #[arg(long, global = true)]
    no_autostart: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Create a collection.
    CreateCollection {
        name: String,
        #[arg(long, default_value_t = 384)]
        dim: usize,
        /// cosine | l2 | dot
        #[arg(long, default_value = "cosine")]
        metric: String,
    },
    /// Insert or update a row. `--vector` is a comma-separated float list
    /// (local embedding from raw text isn't wired into the daemon yet).
    Put {
        coll: String,
        key: String,
        #[arg(long, value_name = "F,F,F,...", allow_hyphen_values = true)]
        vector: String,
    },
    /// Fetch one row by key.
    Get { coll: String, key: String },
    /// Delete one row by key.
    Delete { coll: String, key: String },
    /// Split a file's text into chunks, embed each with the daemon's
    /// configured local model, and commit the whole document as one
    /// transaction. Fails if `--doc-key` already exists in `coll`.
    InsertDocument {
        coll: String,
        path: PathBuf,
        /// Defaults to `path`'s file name.
        #[arg(long)]
        doc_key: Option<String>,
        /// characters | markdown | sentences | tokens:<tokenizer-id>
        #[arg(long, default_value = "markdown")]
        chunk_strategy: String,
        #[arg(long)]
        max_tokens: Option<usize>,
        #[arg(long)]
        overlap_tokens: Option<usize>,
    },
    /// Search a collection. `--vector` is a comma-separated float list;
    /// `--text` embeds locally first (`vector` mode) or is matched
    /// directly (`bm25` mode). Exactly one of `--vector`/`--text` is
    /// required.
    Search {
        coll: String,
        #[arg(long, value_name = "F,F,F,...", allow_hyphen_values = true)]
        vector: Option<String>,
        #[arg(long)]
        text: Option<String>,
        /// vector | bm25 | hybrid — `hybrid` requires `--text` (its BM25
        /// arm always needs it) and fuses both arms server-side via
        /// reciprocal rank fusion.
        #[arg(long, default_value = "vector")]
        mode: String,
        #[arg(short, long, default_value_t = 10)]
        k: usize,
    },
    /// Builds or rebuilds an index for a collection from current storage
    /// state. `bm25` also backfills a collection this daemon process
    /// never saw created (see `mara_proto::WireIndexKind::Bm25`).
    Reindex {
        coll: String,
        /// flat | ivf-pq | lsh | bm25
        #[arg(long, default_value = "flat")]
        kind: String,
    },
    /// Run the daemon in the foreground — what autostart spawns, and what
    /// `marad` itself wraps. Requires the `embedded` feature (it links the
    /// same engine `--embedded` does).
    #[cfg(feature = "embedded")]
    Serve {
        #[arg(long, value_name = "PATH")]
        data_dir: Option<PathBuf>,
        /// The exact config file to load — what autostart passes,
        /// carrying forward the same file `endpoint::resolve` already
        /// found, rather than re-deriving `<data_dir>/mara.toml` (which
        /// silently misses the config whenever `[server] data_dir` isn't
        /// the same directory the config file itself lives in — a
        /// perfectly normal layout, e.g. `data_dir = "./data"` relative
        /// to the config). Takes priority over `--data-dir` when given.
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
        /// Accepted for compatibility with how autostart invokes this
        /// command; `serve` always runs in the foreground of its own
        /// (already-detached) process either way.
        #[arg(long)]
        detach: bool,
    },
    /// Manage a running daemon by its pid file.
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },
    /// Print a shell completion script to stdout — e.g.
    /// `mara completions zsh > ~/.zfunc/_mara`.
    Completions {
        shell: clap_complete::Shell,
    },
    /// Interactive shell: one connection, reused across every line, each
    /// parsed with the exact same command grammar as the non-interactive
    /// CLI (`get docs a`, `search docs --text "..." -k 5`, ...). `exit`
    /// or Ctrl-D leaves. Not available with `--embedded`.
    Repl,
}

#[derive(Subcommand)]
pub enum DaemonAction {
    /// Report whether a daemon is running against the resolved data dir.
    Status,
    /// Send SIGTERM to the daemon's pid and wait for it to exit.
    Stop,
    /// Print the daemon's log file.
    Logs,
}

async fn run(cli: Cli) -> Result<(), String> {
    #[cfg(feature = "embedded")]
    if let Commands::Serve { data_dir, config: config_arg, detach: _ } = &cli.command {
        // `--config`, when given (always, from autostart — see
        // `endpoint::resolve` and `autostart::autostart`), is the exact
        // file already discovered and must be used as-is: re-deriving
        // `<data_dir>/mara.toml` here instead would silently miss it
        // whenever `[server] data_dir` isn't the same directory the
        // config file itself lives in. Falls back to that derivation only
        // for someone running `mara serve --data-dir X` by hand with no
        // `--config`.
        let config = if let Some(config_path) = config_arg {
            mara_daemon::Config::load(config_path).map_err(|e| e.to_string())?
        } else {
            let data_dir = data_dir.clone().or_else(|| cli.data_dir.clone()).unwrap_or_else(endpoint::default_data_dir);
            let config_path = data_dir.join("mara.toml");
            if config_path.exists() {
                mara_daemon::Config::load(&config_path).map_err(|e| e.to_string())?
            } else {
                mara_daemon::Config::default_for(&data_dir)
            }
        };
        // Autostart redirects this process's stdout/stderr to
        // `<data_dir>/logs/daemon.log` before spawning it — without a
        // subscriber, every `tracing::info!`/`warn!` call in mara-daemon
        // goes nowhere and that log file stays empty.
        let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| config.log.level.clone().into());
        tracing_subscriber::fmt().with_env_filter(filter).init();
        return mara_daemon::run(config).await.map_err(|e| e.to_string());
    }

    if let Commands::Completions { shell } = &cli.command {
        clap_complete::generate(*shell, &mut Cli::command(), "mara", &mut std::io::stdout());
        return Ok(());
    }

    if let Commands::Daemon { action } = &cli.command {
        let ep = endpoint::resolve(cli.endpoint.clone(), cli.data_dir.clone());
        return match action {
            DaemonAction::Status => {
                println!("{}", daemon_ctl::status(&ep.data_dir));
                Ok(())
            }
            DaemonAction::Stop => {
                println!("{}", daemon_ctl::stop(&ep.data_dir, Duration::from_secs(10))?);
                Ok(())
            }
            DaemonAction::Logs => {
                print!("{}", daemon_ctl::logs(&ep.data_dir)?);
                Ok(())
            }
        };
    }

    #[cfg(feature = "embedded")]
    if cli.embedded && matches!(cli.command, Commands::Repl) {
        return Err("repl is not available with --embedded (it needs its own daemon connection to reuse across lines)".into());
    }

    let ep = endpoint::resolve(cli.endpoint.clone(), cli.data_dir.clone());

    #[cfg(feature = "embedded")]
    if cli.embedded {
        return embedded::run(&ep.data_dir, ep.config.as_deref(), &cli.command).await;
    }

    if !endpoint::socket_answers(&ep.socket_path).await {
        if cli.no_autostart {
            return Err(format!(
                "no daemon answering at {} and --no-autostart was given; start one with `mara serve` or `marad`",
                ep.socket_path.display()
            ));
        }
        #[cfg(feature = "embedded")]
        {
            // A discovered config file's own `[daemon]` section governs
            // autostart when present — including `autostart = false`,
            // which behaves like `--no-autostart` was passed.
            let (autostart_enabled, timeout_ms) = match &ep.config {
                Some(path) => match mara_daemon::Config::load(path) {
                    Ok(config) => (config.daemon.autostart, config.daemon.autostart_timeout_ms),
                    Err(_) => (true, 15_000),
                },
                None => (true, 15_000),
            };
            if !autostart_enabled {
                return Err(format!(
                    "no daemon answering at {} and daemon.autostart = false in {}",
                    ep.socket_path.display(),
                    ep.config.as_ref().expect("autostart_enabled came from a loaded config").display()
                ));
            }
            autostart::autostart(&ep.data_dir, &ep.socket_path, ep.config.as_deref(), Duration::from_millis(timeout_ms)).await?;
        }
        #[cfg(not(feature = "embedded"))]
        {
            return Err(format!(
                "no daemon answering at {} — this is a slim client build with no autostart; start `marad` separately",
                ep.socket_path.display()
            ));
        }
    }

    let client = MaraClient::connect_unix(&ep.socket_path, "mara-cli").map_err(|e| e.to_string())?;

    if matches!(cli.command, Commands::Repl) {
        return repl::run(&client).await;
    }

    let Some(req) = request::command_to_request(&cli.command)? else {
        return Err("internal error: this command has no request form".into());
    };
    let response = client.call(req).await.map_err(|e| e.to_string())?;
    request::print_response(&cli.command, response)
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
