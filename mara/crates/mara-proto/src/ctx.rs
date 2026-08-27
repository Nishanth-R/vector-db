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
    /// Full administrative access, including principal and cluster management.
    Admin,
    /// Read and write access to collection data.
    Writer,
    /// Read-only access to collection data.
    Reader,
    /// A replica peer, permitted to stream replication traffic.
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
    /// Unique identifier of the principal.
    pub id: PrincipalId,
    /// Human-readable name of the principal.
    pub name: String,
    /// The principal's authorization role.
    pub role: Role,
}

/// Where a request came from — determines how identity is resolved when
/// `auth.enabled = false` (UDS peer credentials give a real OS user; TCP/HTTP
/// carry only the socket address; embedded mode is attributed to the local
/// OS user directly).
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Source {
    /// A Unix domain socket connection, with the peer's OS credentials.
    Uds {
        /// The connecting peer's OS user id.
        os_uid: u32,
        /// The connecting peer's OS username.
        os_user: String,
    },
    /// A plain TCP connection, identified only by its socket address.
    Tcp(SocketAddr),
    /// An HTTP connection, identified only by its socket address.
    Http(SocketAddr),
    /// An in-process call with no transport, attributed to the local OS user.
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
    /// Unique identifier for this request.
    pub request_id: Uuid,
    /// The session this request belongs to.
    pub session: SessionId,
    /// The identity making the request.
    pub principal: Principal,
    /// Where the request came from.
    pub source: Source,
    /// When the server received the request.
    pub received_at: Instant,
}

impl RequestCtx {
    /// Builds a new context, generating a fresh request id and timestamping it now.
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
