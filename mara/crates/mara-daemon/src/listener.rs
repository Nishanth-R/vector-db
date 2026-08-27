//! UDS + TCP listeners, semaphore-bounded accept, and the per-connection
//! `Hello -> RequestCtx -> request loop` state machine (master plan Layer
//! 3, *Composition* / *ConnectionPool*'s server side).

use crate::engine::Engine;
use crate::identity::username_for_uid;
use futures_util::{SinkExt, StreamExt};
use mara_auth::TokenStore;
use mara_proto::{decode_request_body, encode_response, FrameKind, MaraCodec, Principal, PrincipalId, RequestCtx, Response, Role, Source};
use std::path::Path;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::Semaphore;
use tokio_util::codec::Framed;
use uuid::Uuid;

pub struct DaemonShared {
    pub engine: Arc<dyn Engine>,
    pub token_store: Option<Arc<TokenStore>>,
    pub auth_enabled: bool,
    pub allow_local_unauthenticated: bool,
    /// The concrete storage registry — `None` for anything built without
    /// `mara_daemon::server::boot` (e.g. `mara-api`'s own unit-test
    /// fixtures, which only exercise bearer-auth resolution and never
    /// touch storage). `Some` for every real daemon. Distinct from
    /// `engine`'s `Arc<dyn Engine>`, which only exposes `handle` — the
    /// replication leader task needs `Collection`-level access
    /// (`wal_lines_after`, `last_applied_lsn`, `schema_fields`) that
    /// `StorageApi`/`Engine` deliberately don't expose to ordinary
    /// request handling.
    pub storage: Option<Arc<mara_storage::Storage>>,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum HelloError {
    #[error("first frame on a connection must be Hello")]
    NotHello,
    #[error("connection closed before Hello")]
    ClosedBeforeHello,
    #[error("auth_token required but missing")]
    TokenRequired,
    #[error("auth_token did not resolve to an enabled principal")]
    InvalidToken,
    #[error("codec error: {0}")]
    Codec(#[from] mara_proto::CodecError),
    #[error("failed to decode Hello body: {0}")]
    Proto(#[from] mara_proto::ProtoError),
}

/// Resolves the identity for one connection from its `Hello` frame — see
/// the master plan's *Auth toggle and the local case*. On success, returns
/// the established `RequestCtx` plus the `Hello` frame's `request_id` (so
/// the caller can send a correlated `HelloAck`); on failure, still returns
/// the offending frame's `request_id` when one was successfully read, so
/// the rejection response correlates to what the client actually sent
/// rather than a synthetic id.
pub(crate) async fn resolve_hello<S>(
    framed: &mut Framed<S, MaraCodec>,
    source: &Source,
    shared: &DaemonShared,
) -> Result<(RequestCtx, Uuid), (Option<Uuid>, HelloError)>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let frame = framed
        .next()
        .await
        .ok_or((None, HelloError::ClosedBeforeHello))?
        .map_err(|e| (None, HelloError::from(e)))?;
    let request_id = frame.request_id;
    let fail = |e: HelloError| (Some(request_id), e);

    if frame.kind != FrameKind::Request {
        return Err(fail(HelloError::NotHello));
    }
    let req = decode_request_body(&frame.body).map_err(|e| fail(HelloError::from(e)))?;
    let mara_proto::Request::Hello { session_id, auth_token, .. } = req else {
        return Err(fail(HelloError::NotHello));
    };

    let is_uds = matches!(source, Source::Uds { .. });
    let exempt_from_token = shared.allow_local_unauthenticated && is_uds;

    let principal = if !shared.auth_enabled {
        synthetic_principal(source)
    } else if let Some(token) = &auth_token {
        let store = shared.token_store.as_ref().ok_or_else(|| fail(HelloError::InvalidToken))?;
        store.authenticate(token).ok_or_else(|| fail(HelloError::InvalidToken))?
    } else if exempt_from_token {
        synthetic_principal(source)
    } else {
        return Err(fail(HelloError::TokenRequired));
    };

    let ctx = RequestCtx::new(session_id, principal, source.clone());
    Ok((ctx, request_id))
}

pub(crate) fn synthetic_principal(source: &Source) -> Principal {
    match source {
        Source::Uds { os_user, .. } => Principal {
            id: PrincipalId(format!("local:{os_user}")),
            name: format!("local:{os_user}"),
            role: Role::Admin,
        },
        Source::Tcp(addr) => Principal {
            id: PrincipalId(format!("tcp:{addr}")),
            name: format!("tcp:{addr}"),
            role: Role::Admin,
        },
        Source::Http(addr) => Principal {
            id: PrincipalId(format!("http:{addr}")),
            name: format!("http:{addr}"),
            role: Role::Admin,
        },
        Source::Embedded => Principal {
            id: PrincipalId("local:embedded".into()),
            name: "local:embedded".into(),
            role: Role::Admin,
        },
    }
}

pub(crate) async fn send_error<S>(framed: &mut Framed<S, MaraCodec>, request_id: Uuid, code: &str, message: String)
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    if let Ok(bytes) = encode_response(
        request_id,
        &Response::Error {
            code: code.into(),
            message,
        },
    ) {
        let _ = framed.send(bytes).await;
    }
}

async fn handle_connection<S>(stream: S, source: Source, shared: Arc<DaemonShared>)
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut framed = Framed::new(stream, MaraCodec);

    let (ctx, hello_request_id) = match resolve_hello(&mut framed, &source, &shared).await {
        Ok(v) => v,
        Err((request_id, e)) => {
            tracing::warn!("connection rejected during Hello: {e}");
            // Correlate to the offending frame's own id when we managed to
            // read one; a client that never sent anything parseable gets
            // the nil id, since there's nothing to correlate to.
            send_error(&mut framed, request_id.unwrap_or(Uuid::nil()), "auth_failed", e.to_string()).await;
            return;
        }
    };

    let ack = encode_response(
        hello_request_id,
        &Response::HelloAck {
            session_id: ctx.session.clone(),
            server_version: env!("CARGO_PKG_VERSION").to_string(),
        },
    );
    match ack {
        Ok(bytes) => {
            if framed.send(bytes).await.is_err() {
                return;
            }
        }
        Err(_) => return,
    }

    loop {
        let frame = match framed.next().await {
            None => return,
            Some(Ok(f)) => f,
            Some(Err(e)) => {
                tracing::warn!(session = %ctx.session, "codec error, closing connection: {e}");
                return;
            }
        };
        if frame.kind != FrameKind::Request {
            continue;
        }
        let request_id = frame.request_id;
        let req = match decode_request_body(&frame.body) {
            Ok(r) => r,
            Err(e) => {
                send_error(&mut framed, request_id, "bad_request", e.to_string()).await;
                continue;
            }
        };

        let response = shared.engine.handle(&ctx, req).await;
        match encode_response(request_id, &response) {
            Ok(bytes) => {
                if framed.send(bytes).await.is_err() {
                    return;
                }
            }
            Err(_) => return,
        }
    }
}

/// Accepts UDS connections behind `server.max_connections` semaphore-bounded
/// concurrency, resolving each peer's real OS user via `SO_PEERCRED`
/// (`peer_cred()`) — used to attribute the connection even when
/// `auth.enabled = false`.
pub async fn serve_unix(path: &Path, shared: Arc<DaemonShared>, max_connections: usize) -> std::io::Result<()> {
    if path.exists() {
        let _ = std::fs::remove_file(path);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let listener = UnixListener::bind(path)?;
    let semaphore = Arc::new(Semaphore::new(max_connections));
    tracing::info!(socket = %path.display(), "listening on unix socket");

    loop {
        let permit = semaphore.clone().acquire_owned().await.expect("semaphore is never closed");
        let (stream, _addr) = listener.accept().await?;
        let cred = stream.peer_cred();
        let shared = shared.clone();
        tokio::spawn(async move {
            let source = match cred {
                Ok(cred) => {
                    let uid = cred.uid();
                    Source::Uds {
                        os_uid: uid,
                        os_user: username_for_uid(uid),
                    }
                }
                Err(e) => {
                    tracing::warn!("failed to read UDS peer credentials: {e}");
                    Source::Uds {
                        os_uid: u32::MAX,
                        os_user: "unknown".into(),
                    }
                }
            };
            handle_connection(stream, source, shared).await;
            drop(permit);
        });
    }
}

/// Accepts TCP connections the same way. Per the master plan, TCP always
/// requires a token when `auth.enabled` — `allow_local_unauthenticated`
/// only ever exempts UDS.
pub async fn serve_tcp(addr: &str, shared: Arc<DaemonShared>, max_connections: usize) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    let semaphore = Arc::new(Semaphore::new(max_connections));
    tracing::info!(addr, "listening on tcp");

    loop {
        let permit = semaphore.clone().acquire_owned().await.expect("semaphore is never closed");
        let (stream, peer_addr) = listener.accept().await?;
        let shared = shared.clone();
        tokio::spawn(async move {
            handle_connection(stream, Source::Tcp(peer_addr), shared).await;
            drop(permit);
        });
    }
}

