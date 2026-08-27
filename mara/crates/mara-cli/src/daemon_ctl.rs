//! `mara daemon status|stop|logs` (master plan Layer 3, *Autostart &
//! embedded* — *Reaping*): lifecycle management for a `marad`/`mara serve`
//! process via its pid file and data-dir lock, without needing a live
//! socket connection.

use std::path::Path;
use std::time::{Duration, Instant};

fn pid_file(data_dir: &Path) -> std::path::PathBuf {
    data_dir.join("marad.pid")
}

fn read_pid(data_dir: &Path) -> Option<u32> {
    std::fs::read_to_string(pid_file(data_dir)).ok()?.trim().parse().ok()
}

#[cfg(unix)]
fn process_is_alive(pid: u32) -> bool {
    // `kill(pid, 0)` sends no signal — it only checks whether the process
    // exists and is signalable by us. Returning `true` on any negative
    // errno *except* ESRCH deliberately errs toward "alive" (e.g. EPERM
    // means a real process is there, just owned by someone else).
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH) }
}

#[cfg(not(unix))]
fn process_is_alive(_pid: u32) -> bool {
    false
}

pub fn status(data_dir: &Path) -> String {
    match read_pid(data_dir) {
        Some(pid) if process_is_alive(pid) => format!("running (pid {pid}, data_dir {})", data_dir.display()),
        Some(pid) => format!("not running (stale pid file names pid {pid}, which is no longer alive)"),
        None => format!("not running (no pid file under {})", data_dir.display()),
    }
}

pub fn stop(data_dir: &Path, timeout: Duration) -> Result<String, String> {
    let Some(pid) = read_pid(data_dir) else {
        return Err(format!("no pid file under {} — is a daemon running?", data_dir.display()));
    };
    if !process_is_alive(pid) {
        return Ok(format!("pid {pid} was already not running"));
    }
    #[cfg(unix)]
    {
        // SAFETY: `pid` is a plain integer read from our own pid file;
        // `kill` with SIGTERM is the standard, safe-to-call-repeatedly
        // request for graceful termination.
        let ret = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
        if ret != 0 {
            return Err(format!("failed to signal pid {pid}: {}", std::io::Error::last_os_error()));
        }
    }
    #[cfg(not(unix))]
    {
        return Err("stopping a daemon by pid is only implemented on unix".into());
    }

    let start = Instant::now();
    while process_is_alive(pid) {
        if start.elapsed() >= timeout {
            return Err(format!("pid {pid} did not exit within {}ms after SIGTERM", timeout.as_millis()));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Ok(format!("stopped (pid {pid})"))
}

pub fn logs(data_dir: &Path) -> Result<String, String> {
    let path = data_dir.join("logs").join("daemon.log");
    std::fs::read_to_string(&path).map_err(|e| format!("failed to read {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_reports_not_running_with_no_pid_file() {
        let dir = tempfile::tempdir().unwrap();
        assert!(status(dir.path()).contains("not running"));
    }

    #[test]
    fn status_reports_running_for_the_current_process() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(pid_file(dir.path()), std::process::id().to_string()).unwrap();
        assert!(status(dir.path()).contains("running"));
    }

    #[test]
    fn status_reports_stale_for_an_unlikely_pid() {
        let dir = tempfile::tempdir().unwrap();
        // A pid essentially guaranteed not to be alive.
        std::fs::write(pid_file(dir.path()), "999999").unwrap();
        let s = status(dir.path());
        assert!(s.contains("not running"), "expected a stale-pid report, got: {s}");
    }

    #[test]
    fn stop_without_a_pid_file_errors_clearly() {
        let dir = tempfile::tempdir().unwrap();
        assert!(stop(dir.path(), Duration::from_millis(100)).is_err());
    }

    #[test]
    fn logs_without_a_log_file_errors_clearly() {
        let dir = tempfile::tempdir().unwrap();
        assert!(logs(dir.path()).is_err());
    }
}
