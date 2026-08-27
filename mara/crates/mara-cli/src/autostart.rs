//! Autostart (master plan Layer 3, *Autostart & embedded*): nothing
//! answers at the resolved socket, so take an exclusive spawn lock (so ten
//! concurrent CLI invocations start exactly one daemon), spawn
//! `std::env::current_exe() serve --detach`, and poll the socket until
//! ready or `daemon.autostart_timeout_ms`.

use crate::endpoint::socket_answers;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

fn spawn_lock_path(data_dir: &Path) -> PathBuf {
    data_dir.join(".mara.spawn.lock")
}

/// Spawns a detached `mara serve` targeting `data_dir`, then polls
/// `socket_path` until it answers or `timeout` elapses. If another
/// concurrent invocation already holds the spawn lock, this one doesn't
/// spawn a second daemon — it just polls, same as if it had. `config`, if
/// `endpoint::resolve` found one, is passed straight through via `mara
/// serve --config` so the spawned daemon loads the *exact* file that was
/// already discovered — see `Commands::Serve`'s doc comment for why
/// re-deriving it from `data_dir` alone isn't safe.
pub async fn autostart(data_dir: &Path, socket_path: &Path, config: Option<&Path>, timeout: Duration) -> Result<(), String> {
    std::fs::create_dir_all(data_dir).map_err(|e| e.to_string())?;
    let lock_path = spawn_lock_path(data_dir);
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)
        .map_err(|e| e.to_string())?;
    let mut fd_lock = fd_lock::RwLock::new(lock_file);

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());

    if let Ok(_guard) = fd_lock.try_write() {
        spinner.set_message("starting mara daemon (first run may download an embedding model)...");
        spawn_detached(data_dir, config)?;
        wait_for_socket(socket_path, timeout, &spinner).await?;
        spinner.finish_with_message("mara daemon is ready");
    } else {
        // Someone else is already spawning it — just wait.
        spinner.set_message("waiting for another process's mara daemon to finish starting...");
        wait_for_socket(socket_path, timeout, &spinner).await?;
        spinner.finish_with_message("mara daemon is ready");
    }
    Ok(())
}

fn spawn_detached(data_dir: &Path, config: Option<&Path>) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("could not locate the current executable to autostart: {e}"))?;
    let log_dir = data_dir.join("logs");
    std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_dir.join("daemon.log"))
        .map_err(|e| e.to_string())?;
    let log_file2 = log_file.try_clone().map_err(|e| e.to_string())?;

    let mut cmd = std::process::Command::new(exe);
    cmd.arg("serve").arg("--data-dir").arg(data_dir);
    if let Some(config_path) = config {
        cmd.arg("--config").arg(config_path);
    }
    cmd.arg("--detach")
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file2));

    // Detaches into its own process group so it survives the invoking
    // shell/terminal closing — the practical core of "detached spawn"
    // without a full Unix double-fork daemonization ceremony (no `setsid`,
    // no re-chdir to `/`, no closing every inherited fd). Good enough for
    // autostart; a real installed `marad` under systemd/launchd doesn't
    // need this at all, since the service manager already owns the
    // process's lifecycle.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    cmd.spawn().map_err(|e| format!("failed to spawn mara daemon: {e}"))?;
    Ok(())
}

async fn wait_for_socket(socket_path: &Path, timeout: Duration, spinner: &ProgressBar) -> Result<(), String> {
    let start = Instant::now();
    loop {
        if socket_answers(socket_path).await {
            return Ok(());
        }
        if start.elapsed() >= timeout {
            spinner.finish_and_clear();
            return Err(format!(
                "mara daemon did not become ready within {}ms (socket: {}). Check its log under <data_dir>/logs/daemon.log.",
                timeout.as_millis(),
                socket_path.display()
            ));
        }
        spinner.tick();
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
