//! `--embedded` (master plan Layer 3, *Autostart & embedded*): no socket —
//! construct the `Engine` in-process against `data_dir`, run one command,
//! exit. The lowest-latency path for one-shot scripting, and the closest
//! thing to embedded SQLite. Only compiled with the `embedded` feature —
//! it's what pulls `mara-daemon` (and, transitively, the rest of the
//! engine stack) into the CLI binary.

use crate::request::{command_to_request, print_response};
use crate::Commands;
use mara_auth::{AuditConfig, AuditFsync, AuditMode, AuditSink, JsonlAuditSink};
use mara_daemon::{Config, DataDirLock, EngineImpl};
use mara_embed::EmbeddingBackend;
use mara_proto::{Principal, PrincipalId, RequestCtx, Role, SessionId, Source};
use mara_storage::{FsyncPolicy, Storage};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

fn fsync_policy(name: &str) -> FsyncPolicy {
    match name {
        "always" => FsyncPolicy::Always,
        "never" => FsyncPolicy::Never,
        _ => FsyncPolicy::Interval(Duration::from_secs(1)),
    }
}

fn load_embedder(config: &Config) -> Result<Option<Arc<dyn EmbeddingBackend>>, String> {
    let Some(model_id) = &config.embedding.model else {
        return Ok(None);
    };
    std::fs::create_dir_all(config.embedding_cache_dir()).map_err(|e| e.to_string())?;
    let backend = mara_embed::FastEmbedBackend::load(model_id, &config.embedding_cache_dir(), true).map_err(|e| e.to_string())?;
    Ok(Some(Arc::new(backend) as Arc<dyn EmbeddingBackend>))
}

fn current_username() -> String {
    std::env::var("USER").or_else(|_| std::env::var("USERNAME")).unwrap_or_else(|_| "unknown".into())
}

pub async fn run(data_dir: &Path, config_path: Option<&Path>, cmd: &Commands) -> Result<(), String> {
    // Fails fast and clearly if `marad`/`mara serve` already owns this data
    // dir — matches the master plan's "embedded mode doesn't violate the
    // single-writer-daemon decision because the lock guarantees exactly
    // one owner of the WAL at all times."
    let _lock = DataDirLock::acquire(data_dir).map_err(|e| format!("{e} (a daemon is likely already running against this data dir — connect to its socket instead of using --embedded)"))?;

    // A discovered config file's `[embedding]` (and every other) section
    // applies here exactly as it would to a real daemon — without this,
    // `--embedded` would silently ignore it and every text-only
    // `put`/`insert-document` would fail with `embedding_not_configured`
    // even when a config file nearby says otherwise.
    let config = match config_path {
        Some(path) => Config::load(path).map_err(|e| e.to_string())?,
        None => Config::default_for(data_dir),
    };
    let audit_sink: Arc<dyn AuditSink> = Arc::new(
        JsonlAuditSink::open(
            config.audit_dir(),
            AuditConfig {
                mode: if config.audit_mode_is_strict() { AuditMode::Strict } else { AuditMode::Lossy },
                fsync: AuditFsync::Interval(Duration::from_secs(1)),
                max_stall: Duration::from_millis(config.audit.max_stall_ms),
                channel_capacity: 4096,
                rotate_size_mb: config.audit.rotate_size_mb,
                retention_days: config.audit.retention_days,
            },
        )
        .map_err(|e| e.to_string())?,
    );

    let storage = Arc::new(Storage::open(data_dir, config.wal_segment_size_bytes(), fsync_policy(&config.wal.fsync_policy)));
    let embedder = load_embedder(&config)?;
    let engine = EngineImpl::with_embedder(storage, audit_sink, embedder, config.embedding.batch_size);

    let username = current_username();
    let ctx = RequestCtx::new(
        SessionId("embedded".into()),
        Principal {
            id: PrincipalId(format!("local:{username}")),
            name: format!("local:{username}"),
            role: Role::Admin,
        },
        Source::Embedded,
    );

    let Some(req) = command_to_request(cmd)? else {
        return Err("this command has no --embedded form".into());
    };
    let response = mara_daemon::Engine::handle(&engine, &ctx, req).await;
    print_response(cmd, response)
}
