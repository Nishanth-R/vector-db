//! Daemon config (master plan Layer 3, *Config file*). One TOML file,
//! sections owned by their respective layers. Only the sections `marad`
//! v0 actually reads are implemented — `[embedding]`/`[index]`/`[bm25]`/
//! `[fusion]`/`[chunking]`/`[replication]` join this struct as the layers
//! that consume them land.

use serde::Deserialize;
use std::path::{Path, PathBuf};

fn default_max_connections() -> usize {
    256
}
fn default_true() -> bool {
    true
}
fn default_wal_segment_mb() -> u32 {
    128
}
fn default_txn_history_limit() -> usize {
    10_000
}
fn default_audit_dir() -> String {
    "audit".into()
}
fn default_audit_retention_days() -> i64 {
    90
}
fn default_audit_max_stall_ms() -> u64 {
    5000
}
fn default_audit_rotate_mb() -> u64 {
    256
}
fn default_log_level() -> String {
    "info".into()
}
fn default_embedding_batch_size() -> usize {
    32
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub data_dir: PathBuf,
    /// Relative to `data_dir` unless absolute. Default `<data_dir>/mara.sock`.
    pub unix_socket: Option<PathBuf>,
    /// `host:port`, e.g. `"127.0.0.1:7700"`. `None` disables the TCP listener.
    pub tcp_listen: Option<String>,
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
}

/// `mara-api`'s bind address — a separate, optional process from
/// `marad`/`mara serve` (see the master plan's *`mara-api` (axum)*), never
/// started by `mara-daemon::server::run` itself. `listen: None` is the
/// default because most deployments only ever run the native UDS/TCP
/// protocol; `mara-api` requires it be set explicitly, the same way
/// `server.tcp_listen` opts a deployment into the TCP transport.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct HttpConfig {
    #[serde(default)]
    pub listen: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_true")]
    pub allow_local_unauthenticated: bool,
}

// A derived `Default` would set every field to its type's default (`bool`
// -> `false`), silently discarding the `#[serde(default = "default_true")]`
// on `allow_local_unauthenticated` — that attribute only fires when the
// `[auth]` table is present but missing the key, not when the whole
// section (and therefore `#[serde(default)]` on `Config::auth`) is absent.
impl Default for AuthConfig {
    fn default() -> Self {
        AuthConfig {
            enabled: false,
            allow_local_unauthenticated: true,
        }
    }
}

fn default_audit_mode(auth_enabled: bool) -> &'static str {
    if auth_enabled {
        "strict"
    } else {
        "lossy"
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AuditConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_audit_dir")]
    pub dir: String,
    /// `"strict"` or `"lossy"`; resolved against `auth.enabled` at load time
    /// if left unset in the file (see `Config::load`).
    pub mode: Option<String>,
    #[serde(default = "default_audit_max_stall_ms")]
    pub max_stall_ms: u64,
    #[serde(default = "default_audit_retention_days")]
    pub retention_days: i64,
    #[serde(default = "default_audit_rotate_mb")]
    pub rotate_size_mb: u64,
}

impl Default for AuditConfig {
    fn default() -> Self {
        AuditConfig {
            enabled: true,
            dir: default_audit_dir(),
            mode: None,
            max_stall_ms: default_audit_max_stall_ms(),
            retention_days: default_audit_retention_days(),
            rotate_size_mb: default_audit_rotate_mb(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WalConfig {
    #[serde(default = "default_wal_segment_mb")]
    pub segment_size_mb: u32,
    /// `"always"`, `"interval"`, or `"never"`.
    #[serde(default = "default_fsync_policy")]
    pub fsync_policy: String,
}

fn default_fsync_policy() -> String {
    "always".into()
}

impl Default for WalConfig {
    fn default() -> Self {
        WalConfig {
            segment_size_mb: default_wal_segment_mb(),
            fsync_policy: default_fsync_policy(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct UndoConfig {
    #[serde(default = "default_txn_history_limit")]
    pub txn_history_limit: usize,
}

// Same reasoning as `AuthConfig`'s manual `Default` impl above: a derived
// one would silently give `txn_history_limit: 0` when `[undo]` is absent
// entirely, not the documented default of 10,000.
impl Default for UndoConfig {
    fn default() -> Self {
        UndoConfig {
            txn_history_limit: default_txn_history_limit(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LogConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
}

impl Default for LogConfig {
    fn default() -> Self {
        LogConfig { level: default_log_level() }
    }
}

/// Local text embedding is opt-in (master plan *Embedding model
/// selection*): `model = None` means `Put`/`PutBatch`/`PutDocument`
/// requests carrying `text` but no `vector` are rejected with a clear
/// `embedding_not_configured` error rather than the daemon silently
/// downloading a model on first use. A real deployment sets
/// `[embedding] model = "sentence-transformers/all-MiniLM-L6-v2"`
/// explicitly — see `mara-embed::known_model_ids`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    pub model: Option<String>,
    /// Relative to `data_dir` unless absolute. Default `<data_dir>/models`.
    pub cache_dir: Option<PathBuf>,
    #[serde(default = "default_embedding_batch_size")]
    pub batch_size: usize,
}

// Same reasoning as `AuthConfig`/`UndoConfig` above: a derived `Default`
// would give `batch_size: 0` (and would happen to get `model`/`cache_dir`
// right only because `Option`'s zero value already is `None`) when
// `[embedding]` is absent entirely — spelled out manually so every field's
// documented default actually applies.
impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            model: None,
            cache_dir: None,
            batch_size: default_embedding_batch_size(),
        }
    }
}

fn default_autostart_timeout_ms() -> u64 {
    15_000
}

#[derive(Debug, Clone, Deserialize)]
pub struct DaemonConfig {
    #[serde(default = "default_true")]
    pub autostart: bool,
    #[serde(default = "default_autostart_timeout_ms")]
    pub autostart_timeout_ms: u64,
    /// `0` (the default for an explicit `marad`) means never. An
    /// autostarted daemon sets this to a real value (3600s) itself — see
    /// `mara-cli`'s autostart path, not this struct's own default, since
    /// the *same* `[daemon]` section is shared by both entrypoints and
    /// `marad` should not silently reap itself.
    #[serde(default)]
    pub idle_shutdown_secs: u64,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        DaemonConfig {
            autostart: true,
            autostart_timeout_ms: default_autostart_timeout_ms(),
            idle_shutdown_secs: 0,
        }
    }
}

fn default_replication_role() -> String {
    "none".into()
}
fn default_replication_poll_ms() -> u64 {
    200
}
fn default_replication_reconnect_backoff_ms() -> u64 {
    1_000
}
fn default_replication_max_reconnect_backoff_ms() -> u64 {
    30_000
}

/// Leader-follower, async replication (master plan *Replication*) — see
/// `mara_daemon::replication`. `role = "none"` (the default) runs neither
/// side; a real deployment sets `role = "leader"` with `listen` on the
/// node that owns writes, and `role = "follower"` with `leader_addr` +
/// `auth_token` (a `Role::Replica` token minted on the leader) on every
/// read-only replica.
#[derive(Debug, Clone, Deserialize)]
pub struct ReplicationConfig {
    #[serde(default = "default_replication_role")]
    pub role: String,
    /// Leader only: bind address for the dedicated replication listener
    /// (deliberately separate from `server.tcp_listen` — see the master
    /// plan's *Replication*), e.g. `"0.0.0.0:7702"`.
    pub listen: Option<String>,
    /// Follower only: the leader's replication listener address.
    pub leader_addr: Option<String>,
    /// Follower only: a `Role::Replica` token minted on the leader.
    pub auth_token: Option<String>,
    /// Leader only: how often its replication task checks each
    /// collection's WAL for new lines to push. Async replication has no
    /// ack/quorum concept to trigger a push sooner — see the master
    /// plan's explicit justification for why that's an accepted trade,
    /// not a gap.
    #[serde(default = "default_replication_poll_ms")]
    pub poll_interval_ms: u64,
    /// Follower only: initial reconnect delay after a dropped leader
    /// connection, doubling on every consecutive failure up to
    /// `max_reconnect_backoff_ms`.
    #[serde(default = "default_replication_reconnect_backoff_ms")]
    pub reconnect_backoff_ms: u64,
    #[serde(default = "default_replication_max_reconnect_backoff_ms")]
    pub max_reconnect_backoff_ms: u64,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        ReplicationConfig {
            role: default_replication_role(),
            listen: None,
            leader_addr: None,
            auth_token: None,
            poll_interval_ms: default_replication_poll_ms(),
            reconnect_backoff_ms: default_replication_reconnect_backoff_ms(),
            max_reconnect_backoff_ms: default_replication_max_reconnect_backoff_ms(),
        }
    }
}

impl ReplicationConfig {
    pub fn role_is_leader(&self) -> bool {
        self.role == "leader"
    }
    pub fn role_is_follower(&self) -> bool {
        self.role == "follower"
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    #[serde(default)]
    pub http: HttpConfig,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default)]
    pub audit: AuditConfig,
    #[serde(default)]
    pub wal: WalConfig,
    #[serde(default)]
    pub undo: UndoConfig,
    #[serde(default)]
    pub log: LogConfig,
    #[serde(default)]
    pub daemon: DaemonConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub replication: ReplicationConfig,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file {path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("failed to parse config file {path}: {source}")]
    Parse { path: PathBuf, source: toml::de::Error },
    #[error("wal.segment_size_mb ({0} MiB) exceeds the 4 GiB LSN byte-offset ceiling")]
    SegmentTooLarge(u32),
    #[error("invalid wal.fsync_policy {0:?}; expected always, interval, or never")]
    InvalidFsyncPolicy(String),
    #[error("invalid audit.mode {0:?}; expected strict or lossy")]
    InvalidAuditMode(String),
    #[error("invalid replication.role {0:?}; expected none, leader, or follower")]
    InvalidReplicationRole(String),
    #[error("replication.role = \"follower\" requires both replication.leader_addr and replication.auth_token to be set")]
    FollowerMissingLeaderInfo,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io { path: path.to_path_buf(), source })?;
        let mut config: Config = toml::from_str(&text).map_err(|source| ConfigError::Parse { path: path.to_path_buf(), source })?;
        config.validate()?;
        Ok(config)
    }

    /// An in-memory config with every section at its documented default,
    /// rooted at `data_dir` — no TOML file required. What `mara serve`
    /// (autostart's spawn target) and `--embedded` use when no config file
    /// was ever given; a real config file, when one exists, always wins
    /// (see `mara-cli`'s endpoint/config resolution).
    pub fn default_for(data_dir: impl Into<PathBuf>) -> Self {
        let mut config = Config {
            server: ServerConfig {
                data_dir: data_dir.into(),
                unix_socket: None,
                tcp_listen: None,
                max_connections: default_max_connections(),
            },
            http: HttpConfig::default(),
            auth: AuthConfig::default(),
            audit: AuditConfig::default(),
            wal: WalConfig::default(),
            undo: UndoConfig::default(),
            log: LogConfig::default(),
            daemon: DaemonConfig::default(),
            embedding: EmbeddingConfig::default(),
            replication: ReplicationConfig::default(),
        };
        // All-default fields always pass validation — this just resolves
        // `audit.mode`'s auth-dependent default, the same way `load()`
        // does, so the two construction paths never disagree about it.
        config.validate().expect("an all-default Config always validates");
        config
    }

    fn validate(&mut self) -> Result<(), ConfigError> {
        // A Postgres-style Lsn packs the byte offset into a u32 — see
        // mara_proto::Lsn — so a segment must never be able to grow past
        // that, checked here rather than discovered as silent wraparound
        // deep in the WAL writer.
        if (self.wal.segment_size_mb as u64) * 1024 * 1024 > u32::MAX as u64 {
            return Err(ConfigError::SegmentTooLarge(self.wal.segment_size_mb));
        }
        if !matches!(self.wal.fsync_policy.as_str(), "always" | "interval" | "never") {
            return Err(ConfigError::InvalidFsyncPolicy(self.wal.fsync_policy.clone()));
        }
        if let Some(mode) = &self.audit.mode {
            if !matches!(mode.as_str(), "strict" | "lossy") {
                return Err(ConfigError::InvalidAuditMode(mode.clone()));
            }
        } else {
            self.audit.mode = Some(default_audit_mode(self.auth.enabled).to_string());
        }
        if !matches!(self.replication.role.as_str(), "none" | "leader" | "follower") {
            return Err(ConfigError::InvalidReplicationRole(self.replication.role.clone()));
        }
        if self.replication.role_is_follower() && (self.replication.leader_addr.is_none() || self.replication.auth_token.is_none()) {
            return Err(ConfigError::FollowerMissingLeaderInfo);
        }
        Ok(())
    }

    pub fn unix_socket_path(&self) -> PathBuf {
        match &self.server.unix_socket {
            Some(p) if p.is_absolute() => p.clone(),
            Some(p) => self.server.data_dir.join(p),
            None => self.server.data_dir.join("mara.sock"),
        }
    }

    pub fn wal_segment_size_bytes(&self) -> u32 {
        self.wal.segment_size_mb * 1024 * 1024
    }

    pub fn audit_dir(&self) -> PathBuf {
        self.server.data_dir.join(&self.audit.dir)
    }

    pub fn embedding_cache_dir(&self) -> PathBuf {
        match &self.embedding.cache_dir {
            Some(p) if p.is_absolute() => p.clone(),
            Some(p) => self.server.data_dir.join(p),
            None => self.server.data_dir.join("models"),
        }
    }

    pub fn audit_mode_is_strict(&self) -> bool {
        self.audit.mode.as_deref() == Some("strict")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_config_loads_with_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mara.toml");
        std::fs::write(&path, format!("[server]\ndata_dir = {:?}\n", dir.path())).unwrap();

        let config = Config::load(&path).unwrap();
        assert_eq!(config.server.max_connections, 256);
        assert!(!config.auth.enabled);
        assert!(config.auth.allow_local_unauthenticated);
        assert_eq!(config.wal.segment_size_mb, 128);
        assert_eq!(config.audit.mode.as_deref(), Some("lossy"), "auth disabled -> audit defaults to lossy");
    }

    #[test]
    fn audit_mode_defaults_to_strict_when_auth_is_enabled() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mara.toml");
        std::fs::write(&path, format!("[server]\ndata_dir = {:?}\n[auth]\nenabled = true\n", dir.path())).unwrap();
        let config = Config::load(&path).unwrap();
        assert_eq!(config.audit.mode.as_deref(), Some("strict"));
    }

    #[test]
    fn oversized_wal_segment_is_rejected_at_load_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mara.toml");
        std::fs::write(&path, format!("[server]\ndata_dir = {:?}\n[wal]\nsegment_size_mb = 5000\n", dir.path())).unwrap();
        let err = Config::load(&path).unwrap_err();
        assert!(matches!(err, ConfigError::SegmentTooLarge(5000)));
    }

    #[test]
    fn embedding_is_disabled_by_default_and_batch_size_still_gets_its_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mara.toml");
        std::fs::write(&path, format!("[server]\ndata_dir = {:?}\n", dir.path())).unwrap();
        let config = Config::load(&path).unwrap();
        assert!(config.embedding.model.is_none());
        assert_eq!(config.embedding.batch_size, 32);
        assert_eq!(config.embedding_cache_dir(), dir.path().join("models"));
    }

    #[test]
    fn embedding_model_can_be_configured_explicitly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mara.toml");
        std::fs::write(
            &path,
            format!(
                "[server]\ndata_dir = {:?}\n[embedding]\nmodel = \"sentence-transformers/all-MiniLM-L6-v2\"\n",
                dir.path()
            ),
        )
        .unwrap();
        let config = Config::load(&path).unwrap();
        assert_eq!(config.embedding.model.as_deref(), Some("sentence-transformers/all-MiniLM-L6-v2"));
    }

    #[test]
    fn http_listen_defaults_to_disabled_when_the_section_is_absent_entirely() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mara.toml");
        std::fs::write(&path, format!("[server]\ndata_dir = {:?}\n", dir.path())).unwrap();
        let config = Config::load(&path).unwrap();
        assert_eq!(config.http.listen, None);
    }

    #[test]
    fn http_listen_can_be_configured_explicitly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mara.toml");
        std::fs::write(&path, format!("[server]\ndata_dir = {:?}\n[http]\nlisten = \"127.0.0.1:9000\"\n", dir.path())).unwrap();
        let config = Config::load(&path).unwrap();
        assert_eq!(config.http.listen.as_deref(), Some("127.0.0.1:9000"));
    }

    #[test]
    fn unix_socket_path_defaults_under_data_dir() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mara.toml");
        std::fs::write(&path, format!("[server]\ndata_dir = {:?}\n", dir.path())).unwrap();
        let config = Config::load(&path).unwrap();
        assert_eq!(config.unix_socket_path(), dir.path().join("mara.sock"));
    }
}
