//! Endpoint resolution (master plan Layer 3, *Autostart & embedded*): the
//! CLI resolves an endpoint in a fixed order — `--endpoint` -> `MARA_ENDPOINT`
//! -> config file -> default UDS at `<data_dir>/mara.sock`.

use std::path::{Path, PathBuf};
use tokio::net::UnixStream;

/// `true` if something is listening (a plain connect attempt, no `Hello`
/// handshake — cheap enough to poll with). Always available, even in a
/// slim (no-`embedded`) build, since a plain client still needs to know
/// whether to even try connecting.
pub async fn socket_answers(socket_path: &Path) -> bool {
    UnixStream::connect(socket_path).await.is_ok()
}

pub struct Endpoint {
    pub socket_path: PathBuf,
    pub data_dir: PathBuf,
    /// Whether a config file was found and actually drove this resolution
    /// — if so, its own `[daemon]` settings (autostart, timeout) apply;
    /// otherwise the built-in defaults do.
    #[cfg_attr(not(feature = "embedded"), allow(dead_code))]
    pub config: Option<PathBuf>,
}

pub fn default_data_dir() -> PathBuf {
    dirs::home_dir().map(|h| h.join(".mara")).unwrap_or_else(|| PathBuf::from(".mara"))
}

#[cfg(feature = "embedded")]
fn config_candidates(explicit_data_dir: Option<&PathBuf>) -> Vec<PathBuf> {
    let mut candidates = vec![PathBuf::from("mara.toml")];
    if let Some(d) = explicit_data_dir {
        candidates.push(d.join("mara.toml"));
    }
    candidates.push(default_data_dir().join("mara.toml"));
    candidates
}

pub fn resolve(cli_endpoint: Option<PathBuf>, cli_data_dir: Option<PathBuf>) -> Endpoint {
    if let Some(socket_path) = cli_endpoint.or_else(|| std::env::var("MARA_ENDPOINT").ok().map(PathBuf::from)) {
        let data_dir = cli_data_dir
            .or_else(|| socket_path.parent().map(PathBuf::from))
            .unwrap_or_else(default_data_dir);
        return Endpoint { socket_path, data_dir, config: None };
    }

    #[cfg(feature = "embedded")]
    for candidate in config_candidates(cli_data_dir.as_ref()) {
        if candidate.exists() {
            if let Ok(config) = mara_daemon::Config::load(&candidate) {
                return Endpoint {
                    socket_path: config.unix_socket_path(),
                    data_dir: config.server.data_dir.clone(),
                    config: Some(candidate),
                };
            }
        }
    }

    let data_dir = cli_data_dir.unwrap_or_else(default_data_dir);
    Endpoint {
        socket_path: data_dir.join("mara.sock"),
        data_dir,
        config: None,
    }
}
