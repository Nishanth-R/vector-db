//! Leader-follower replication (master plan *Replication*): async, no
//! ack/quorum tracking — the leader's own local WAL+fsync is already the
//! real durability boundary, and quorum complexity isn't earned at this
//! project's stated scale. A dedicated TCP listener, separate from the
//! normal client-facing UDS/TCP ports, authenticates a `Role::Replica`
//! principal via the same `Hello`/`TokenStore` machinery every other
//! connection uses, then — after one `Request::ReplicaHello` — streams
//! WAL tail data continuously with no further `Request` needed.
//!
//! Because replicated records are literally WAL lines, a follower's apply
//! path (`Collection::apply_replicated_lines`) is the exact same function
//! used for local crash recovery.
//!
//! **v0 scope, stated honestly**: no snapshot-bootstrap fallback for a
//! follower that's very far behind — WAL segments are never pruned by
//! this codebase (see `mara_storage::snapshot`'s own doc comment: undo
//! could target any retained transaction, so deleting old segments on a
//! separate, uncoordinated schedule risks breaking it), so a WAL-tail
//! replay from the very first record always works, just less efficiently
//! than a fresh snapshot transfer would for a very long history. Reindex
//! WAL-record propagation (so a follower retrains its own derived indexes
//! on the same trigger the leader did) isn't implemented either — the
//! current `Reindex` handler doesn't append a WAL record for any node to
//! replay in the first place, leader or follower; building indexes on a
//! follower today means calling `Reindex` against it directly, which
//! `EngineImpl`'s follower-mode write gate deliberately still allows
//! (rebuilding a *derived* index from rows this follower already has
//! never touches replicated row data). Both are real, bounded gaps, not
//! silently dropped correctness.

use crate::listener::{resolve_hello, send_error, DaemonShared};
use futures_util::{SinkExt, StreamExt};
use mara_auth::{require, Capability};
use mara_proto::{
    decode_request_body, decode_response_body, encode_request, encode_response, FrameKind, Lsn, MaraCodec, Principal, PrincipalId, ReplicaCollectionInfo,
    Request, RequestCtx, Response, Role, SessionId, Source, WireFieldType,
};
use mara_storage::{Collection, FieldType, PayloadSchema, Storage, StorageApi};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::codec::Framed;
use uuid::Uuid;

fn wire_field_type(t: FieldType) -> WireFieldType {
    match t {
        FieldType::Keyword => WireFieldType::Keyword,
        FieldType::KeywordList => WireFieldType::KeywordList,
        FieldType::I64 => WireFieldType::I64,
        FieldType::F64 => WireFieldType::F64,
        FieldType::DateTime => WireFieldType::DateTime,
        FieldType::Bool => WireFieldType::Bool,
        FieldType::Text => WireFieldType::Text,
    }
}

fn storage_field_type(t: WireFieldType) -> FieldType {
    match t {
        WireFieldType::Keyword => FieldType::Keyword,
        WireFieldType::KeywordList => FieldType::KeywordList,
        WireFieldType::I64 => FieldType::I64,
        WireFieldType::F64 => FieldType::F64,
        WireFieldType::DateTime => FieldType::DateTime,
        WireFieldType::Bool => FieldType::Bool,
        WireFieldType::Text => FieldType::Text,
    }
}

fn collection_info_for(name: &str, coll: &Collection) -> ReplicaCollectionInfo {
    ReplicaCollectionInfo {
        name: name.to_string(),
        dim: coll.dim,
        metric: coll.metric,
        schema: coll.schema_fields().into_iter().map(|(n, t)| (n, wire_field_type(t))).collect(),
    }
}

/// Attribution for WAL records a replication task writes/creates on its
/// own initiative (a follower auto-creating a mirrored collection) —
/// distinct from any real client `RequestCtx`, the same way `Source::
/// Embedded`'s synthetic principal is for `--embedded` mode.
fn replication_ctx(label: &str) -> RequestCtx {
    RequestCtx::new(
        SessionId(format!("replication-{label}")),
        Principal {
            id: PrincipalId(format!("replication:{label}")),
            name: format!("replication:{label}"),
            role: Role::Replica,
        },
        Source::Embedded,
    )
}

pub mod leader {
    use super::*;

    /// Accepts replication connections on `addr` until the listener
    /// errors. One task per follower connection, each independent — a
    /// slow or disconnected follower never blocks another, and never
    /// blocks the normal client-facing listeners either.
    pub async fn serve(addr: &str, shared: Arc<DaemonShared>, poll_interval: Duration) -> std::io::Result<()> {
        let listener = TcpListener::bind(addr).await?;
        tracing::info!(addr, "listening for replication followers");
        loop {
            let (stream, peer_addr) = listener.accept().await?;
            let shared = shared.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_follower(stream, Source::Tcp(peer_addr), shared, poll_interval).await {
                    tracing::warn!(%peer_addr, error = %e, "replication connection ended");
                }
            });
        }
    }

    async fn handle_follower(stream: TcpStream, source: Source, shared: Arc<DaemonShared>, poll_interval: Duration) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut framed = Framed::new(stream, MaraCodec);

        let (ctx, hello_id) = match resolve_hello(&mut framed, &source, &shared).await {
            Ok(v) => v,
            Err((request_id, e)) => {
                send_error(&mut framed, request_id.unwrap_or(Uuid::nil()), "auth_failed", e.to_string()).await;
                return Err(Box::new(e));
            }
        };
        framed.send(encode_response(hello_id, &Response::HelloAck { session_id: ctx.session.clone(), server_version: env!("CARGO_PKG_VERSION").to_string() })?).await?;

        if !require(ctx.principal.role, Capability::ReplicaHello) {
            send_error(&mut framed, Uuid::nil(), "forbidden", "principal's role is not permitted to stream replication".into()).await;
            return Ok(());
        }

        let Some(frame) = framed.next().await else { return Ok(()) };
        let frame = frame?;
        let request_id = frame.request_id;
        let Request::ReplicaHello { known_lsns } = decode_request_body(&frame.body)? else {
            send_error(&mut framed, request_id, "invalid_argument", "expected replica_hello as the first request on the replication listener".into()).await;
            return Ok(());
        };

        let Some(storage) = &shared.storage else {
            send_error(&mut framed, request_id, "storage_error", "this daemon has no directly reachable storage registry".into()).await;
            return Ok(());
        };

        let mut cursors: HashMap<String, Option<Lsn>> = known_lsns.into_iter().collect();
        let mut known: HashSet<String> = HashSet::new();
        let mut infos = Vec::new();
        for name in storage.list_collections() {
            if let Ok(coll) = storage.collection(&name) {
                infos.push(collection_info_for(&name, &coll));
                known.insert(name);
            }
        }
        framed.send(encode_response(request_id, &Response::ReplicaWelcome { collections: infos })?).await?;

        let mut interval = tokio::time::interval(poll_interval);
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    for name in storage.list_collections() {
                        if known.insert(name.clone())
                            && let Ok(coll) = storage.collection(&name) {
                                let info = collection_info_for(&name, &coll);
                                framed.send(encode_response(Uuid::new_v4(), &Response::ReplicaNewCollection { info })?).await?;
                                cursors.entry(name).or_insert(None);
                            }
                    }
                    let names: Vec<String> = known.iter().cloned().collect();
                    for name in names {
                        let Ok(coll) = storage.collection(&name) else { continue };
                        let cursor = cursors.get(&name).copied().flatten();
                        let (lines, new_cursor) = coll.wal_lines_after(cursor).unwrap_or_default();
                        if lines.is_empty() {
                            continue;
                        }
                        framed.send(encode_response(Uuid::new_v4(), &Response::ReplicaWalLines { coll: name.clone(), lines })?).await?;
                        cursors.insert(name, new_cursor);
                    }
                }
                frame = framed.next() => {
                    match frame {
                        None => return Ok(()),
                        Some(Err(e)) => return Err(Box::new(e)),
                        // A follower never sends anything after its one
                        // `ReplicaHello` — this is a pure push stream from
                        // here on. Ignore anything unexpected rather than
                        // erroring the whole connection over it.
                        Some(Ok(_)) => {}
                    }
                }
            }
        }
    }
}

pub mod follower {
    use super::*;

    /// Connects to `leader_addr`, authenticates with `auth_token` (a
    /// `Role::Replica` token minted on the leader), and applies streamed
    /// WAL lines forever — reconnecting with doubling backoff (capped at
    /// `max_backoff`) whenever the connection drops. Never returns except
    /// on a truly unrecoverable local error (e.g. `shared.storage` is
    /// `None`); a leader that's merely unreachable is retried
    /// indefinitely, since a follower losing its leader is an expected,
    /// recoverable operational event, not a fatal one.
    pub async fn run(leader_addr: String, auth_token: String, shared: Arc<DaemonShared>, initial_backoff: Duration, max_backoff: Duration) {
        let Some(storage) = shared.storage.clone() else {
            tracing::error!("replication follower started with no local storage registry — cannot run");
            return;
        };

        let mut backoff = initial_backoff;
        loop {
            match run_once(&leader_addr, &auth_token, &storage).await {
                Ok(()) => {
                    // A clean disconnect (leader closed the connection) —
                    // reset backoff and retry promptly.
                    backoff = initial_backoff;
                }
                Err(e) => {
                    tracing::warn!(leader = %leader_addr, error = %e, backoff_ms = backoff.as_millis(), "replication connection failed, retrying");
                }
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(max_backoff);
        }
    }

    async fn run_once(leader_addr: &str, auth_token: &str, storage: &Arc<Storage>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let stream = TcpStream::connect(leader_addr).await?;
        let mut framed = Framed::new(stream, MaraCodec);

        let hello_id = Uuid::new_v4();
        let session_id = SessionId(format!("replica-{}", &Uuid::new_v4().simple().to_string()[..8]));
        framed
            .send(encode_request(
                hello_id,
                &Request::Hello {
                    client_name: "mara-replica".into(),
                    session_id,
                    auth_token: Some(auth_token.to_string()),
                },
            )?)
            .await?;
        let ack = next_response(&mut framed).await?;
        if !matches!(ack, Response::HelloAck { .. }) {
            return Err(format!("expected HelloAck from leader, got {ack:?}").into());
        }

        let known_lsns: Vec<(String, Option<Lsn>)> = storage
            .list_collections()
            .into_iter()
            .filter_map(|name| storage.collection(&name).ok().map(|c| (name, c.last_applied_lsn())))
            .collect();
        framed.send(encode_request(Uuid::new_v4(), &Request::ReplicaHello { known_lsns })?).await?;

        let welcome = next_response(&mut framed).await?;
        let Response::ReplicaWelcome { collections } = welcome else {
            return Err(format!("expected ReplicaWelcome from leader, got {welcome:?}").into());
        };
        for info in collections {
            ensure_local_collection(storage, &info)?;
        }

        loop {
            let frame = match framed.next().await {
                None => return Ok(()),
                Some(Ok(f)) => f,
                Some(Err(e)) => return Err(Box::new(e)),
            };
            if frame.kind != FrameKind::Response {
                continue;
            }
            match decode_response_body(&frame.body)? {
                Response::ReplicaNewCollection { info } => ensure_local_collection(storage, &info)?,
                Response::ReplicaWalLines { coll, lines } => {
                    let collection = storage.collection(&coll)?;
                    collection.apply_replicated_lines(&lines)?;
                }
                other => {
                    tracing::warn!(?other, "unexpected frame on the replication stream, ignoring");
                }
            }
        }
    }

    fn ensure_local_collection(storage: &Arc<Storage>, info: &ReplicaCollectionInfo) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if storage.collection(&info.name).is_ok() {
            return Ok(());
        }
        let mut builder = PayloadSchema::builder();
        for (name, ty) in &info.schema {
            builder = builder.field(name.clone(), storage_field_type(*ty));
        }
        let ctx = replication_ctx(&info.name);
        match storage.create_collection(&ctx, &info.name, info.dim, info.metric, builder.build()) {
            Ok(()) | Err(mara_storage::StorageError::CollectionAlreadyExists(_)) => Ok(()),
            Err(e) => Err(Box::new(e)),
        }
    }

    async fn next_response<S>(framed: &mut Framed<S, MaraCodec>) -> Result<Response, Box<dyn std::error::Error + Send + Sync>>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        let frame = framed.next().await.ok_or("connection closed while waiting for a response")??;
        Ok(decode_response_body(&frame.body)?)
    }
}
