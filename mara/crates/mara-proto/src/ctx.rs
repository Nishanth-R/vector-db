use crate::ids::{PrincipalId, SessionId};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Instant;
use uuid::Uuid;

/// A principal's role. Checked against a single `require(ctx, cap)` table in
/// `Engine::handle` — one arm per request kind — so the authorization
/// surface can be audited by reading one function rather than scattered
/// checks throughout the engine.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Admin,
    Writer,
    Reader,
    Replica,
}

/// The resolved-identity view of a principal carried on every `RequestCtx`
/// and every audit/WAL entry. Deliberately lighter than `mara-auth`'s full
/// persisted principal record (which also carries `token_hash`,
/// `created_at`, `last_seen`, `disabled`) — that full record lives in
/// `mara-auth`, which depends on `mara-proto`, so `mara-proto` cannot depend
/// back on it without a cycle. `mara-auth::TokenStore` resolves a token down
/// to this view.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Principal {
    pub id: PrincipalId,
    pub name: String,
    pub role: Role,
}

/// Where a request came from — determines how identity is resolved when
/// `auth.enabled = false` (UDS peer credentials give a real OS user; TCP/HTTP
/// carry only the socket address; embedded mode is attributed to the local
/// OS user directly).
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Source {
    Uds { os_uid: u32, os_user: String },
    Tcp(SocketAddr),
    Http(SocketAddr),
    Embedded,
}

/// One identity threaded everywhere: every `Engine::handle` call and every
/// mutating `StorageApi` method takes `&RequestCtx`. Threaded from the start
/// rather than retrofitted, because retrofitting an actor field through a
/// WAL format and a storage API after the fact is exactly the kind of change
/// that ends up half-done.
///
/// Not itself serialized — it's a server-local, per-request construct built
/// from a `Hello` plus the transport's peer info, never sent as a blob over
/// the wire or written verbatim to the WAL (the WAL and audit log each carry
/// their own narrower attribution fields derived from it).
#[derive(Clone, Debug)]
pub struct RequestCtx {
    pub request_id: Uuid,
    pub session: SessionId,
    pub principal: Principal,
    pub source: Source,
    pub received_at: Instant,
}

impl RequestCtx {
    pub fn new(session: SessionId, principal: Principal, source: Source) -> Self {
        RequestCtx {
            request_id: Uuid::new_v4(),
            session,
            principal,
            source,
            received_at: Instant::now(),
        }
    }
}
