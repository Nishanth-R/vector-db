//! Bootstraps the composition root (master plan Layer 3): config -> audit
//! sink -> auth token store -> durable `Storage` -> `Engine` -> UDS/TCP
//! listeners, run concurrently until either fails.

use crate::config::Config;
use crate::engine::EngineImpl;
use crate::listener::{serve_tcp, serve_unix, DaemonShared};
use crate::lock::DataDirLock;
use mara_auth::{AuditConfig, AuditFsync, AuditMode, AuditSink, JsonlAuditSink, TokenStore};
use mara_embed::EmbeddingBackend;
use mara_storage::{FsyncPolicy, Storage};
use std::sync::Arc;
use std::time::Duration;

pub type BootError = Box<dyn std::error::Error + Send + Sync>;

/// One of `run`'s concurrently-raced listener/replication tasks.
type ServerTask = std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), BootError>> + Send>>;

fn fsync_policy(name: &str) -> FsyncPolicy {
    match name {
        "always" => FsyncPolicy::Always,
        "never" => FsyncPolicy::Never,
        _ => FsyncPolicy::Interval(Duration::from_secs(1)),
    }
}

/// `None` when `[embedding] model` is unset — see `EngineImpl`'s doc
/// comment. A configured model that fails to load (bad id, no network on
/// a genuine first run) fails daemon startup outright rather than booting
/// into a half-working state that only fails later, on some client's
/// first `PutDocument`.
fn load_embedder(config: &Config) -> Result<Option<Arc<dyn EmbeddingBackend>>, BootError> {
    let Some(model_id) = &config.embedding.model else {
        return Ok(None);
    };
    std::fs::create_dir_all(config.embedding_cache_dir())?;
    tracing::info!(model = %model_id, "loading embedding model (first run may download it)");
    let backend = mara_embed::FastEmbedBackend::load(model_id, &config.embedding_cache_dir(), true)?;
    tracing::info!(model = %model_id, dim = backend.fingerprint().dim, "embedding model ready");
    Ok(Some(Arc::new(backend) as Arc<dyn EmbeddingBackend>))
}

/// `<data_dir>/marad.pid` — what `mara daemon status`/`stop` read. Not
/// removed on exit (there's no graceful-shutdown hook yet); `status`/`stop`
/// treat a pid file naming a dead process as "not running" rather than
/// trusting the file's mere existence.
fn write_pid_file(data_dir: &std::path::Path) -> std::io::Result<()> {
    std::fs::write(data_dir.join("marad.pid"), std::process::id().to_string())
}

/// The composition root shared by every entrypoint that speaks for a
/// `data_dir`: `marad`/`mara serve` (UDS+TCP, via [`run`]) and `mara-api`
/// (HTTP) both boot through this single function — one place that opens
/// storage, the audit sink, the token store, and the embedder, so the two
/// frontends can never disagree about how a `data_dir` comes up. Returns
/// the `DataDirLock` alongside `DaemonShared` because it must outlive
/// whatever the caller does next (dropping it releases the advisory lock)
/// — same reasoning as `run`'s own local `_data_dir_lock` before this was
/// extracted.
pub async fn boot(config: &Config) -> Result<(Arc<DaemonShared>, DataDirLock), BootError> {
    std::fs::create_dir_all(&config.server.data_dir)?;

    let data_dir_lock = DataDirLock::acquire(&config.server.data_dir)?;
    write_pid_file(&config.server.data_dir)?;

    let audit_sink: Arc<dyn AuditSink> = Arc::new(JsonlAuditSink::open(
        config.audit_dir(),
        AuditConfig {
            mode: if config.audit_mode_is_strict() { AuditMode::Strict } else { AuditMode::Lossy },
            fsync: AuditFsync::Interval(Duration::from_secs(1)),
            max_stall: Duration::from_millis(config.audit.max_stall_ms),
            channel_capacity: 4096,
            rotate_size_mb: config.audit.rotate_size_mb,
            retention_days: config.audit.retention_days,
        },
    )?);

    let token_store = if config.auth.enabled {
        Some(Arc::new(TokenStore::load_or_create(config.server.data_dir.join("auth/tokens.toml"))?))
    } else {
        None
    };

    let storage = Arc::new(Storage::open(&config.server.data_dir, config.wal_segment_size_bytes(), fsync_policy(&config.wal.fsync_policy)));
    let embedder = load_embedder(config)?;
    let mut engine_impl = EngineImpl::with_embedder(storage.clone(), audit_sink, embedder, config.embedding.batch_size);
    if config.replication.role_is_follower() {
        engine_impl = engine_impl.as_follower();
    }
    let engine = Arc::new(engine_impl);

    let shared = Arc::new(DaemonShared {
        engine,
        token_store,
        auth_enabled: config.auth.enabled,
        allow_local_unauthenticated: config.auth.allow_local_unauthenticated,
        storage: Some(storage),
    });

    Ok((shared, data_dir_lock))
}

/// Runs the daemon until a listener errors or the process is killed —
/// there is no graceful-shutdown path yet (`daemon.idle_shutdown_secs` and
/// friends are a later step). Also starts whichever side of replication
/// `[replication] role` names, alongside the normal UDS/TCP listeners —
/// one process, one `data_dir`, every transport sharing the same
/// `Engine`/`Storage`.
pub async fn run(config: Config) -> Result<(), BootError> {
    // Held for the rest of this function — `run()` only ever returns on
    // shutdown or fatal error, at which point the lock releases with it.
    let (shared, _data_dir_lock) = boot(&config).await?;

    let mut tasks: Vec<ServerTask> = Vec::new();

    let socket_path = config.unix_socket_path();
    let uds_shared = shared.clone();
    let max_conn = config.server.max_connections;
    tasks.push(Box::pin(async move { serve_unix(&socket_path, uds_shared, max_conn).await.map_err(BootError::from) }));

    if let Some(addr) = config.server.tcp_listen.clone() {
        let tcp_shared = shared.clone();
        tasks.push(Box::pin(async move { serve_tcp(&addr, tcp_shared, max_conn).await.map_err(BootError::from) }));
    }

    if config.replication.role_is_leader()
        && let Some(addr) = config.replication.listen.clone()
    {
        let rep_shared = shared.clone();
        let poll_interval = Duration::from_millis(config.replication.poll_interval_ms);
        tasks.push(Box::pin(async move { crate::replication::leader::serve(&addr, rep_shared, poll_interval).await.map_err(BootError::from) }));
    }

    if config.replication.role_is_follower() {
        // Validated at config-load time: `role = "follower"` always has
        // both `leader_addr` and `auth_token` set.
        let leader_addr = config.replication.leader_addr.clone().expect("validated at config load");
        let auth_token = config.replication.auth_token.clone().expect("validated at config load");
        let rep_shared = shared.clone();
        let initial_backoff = Duration::from_millis(config.replication.reconnect_backoff_ms);
        let max_backoff = Duration::from_millis(config.replication.max_reconnect_backoff_ms);
        tasks.push(Box::pin(async move {
            crate::replication::follower::run(leader_addr, auth_token, rep_shared, initial_backoff, max_backoff).await;
            Ok(())
        }));
    }

    let (result, _index, _remaining) = futures_util::future::select_all(tasks).await;
    result?;
    Ok(())
}
