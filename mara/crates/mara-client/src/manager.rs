//! The `deadpool::managed::Manager` impl — see the master plan's
//! `ConnectionPool` (client side): checkout/send/return, so the CLI and
//! RAG-app code aren't opening a fresh socket + handshake per query.
//! `create()` does the full `Hello` handshake once, up front, so every
//! pooled connection handed out by `Pool::get()` is already authenticated
//! and ready for application requests.

use crate::connection::Connection;
use crate::error::ClientError;
use deadpool::managed::{Metrics, RecycleResult};
use mara_proto::{MaraCodec, Request, Response, SessionId};
use std::path::PathBuf;
use tokio::net::UnixStream;
use tokio_util::codec::Framed;
use uuid::Uuid;

/// `deadpool` manager that dials and `Hello`s a fresh Unix socket connection on demand.
pub struct ConnectionManager {
    socket_path: PathBuf,
    client_name: String,
    auth_token: Option<String>,
}

impl ConnectionManager {
    /// Builds a manager that will connect to `socket_path`, identifying as `client_name`.
    pub fn new(socket_path: impl Into<PathBuf>, client_name: impl Into<String>, auth_token: Option<String>) -> Self {
        ConnectionManager {
            socket_path: socket_path.into(),
            client_name: client_name.into(),
            auth_token,
        }
    }
}

impl deadpool::managed::Manager for ConnectionManager {
    type Type = Connection;
    type Error = ClientError;

    async fn create(&self) -> Result<Connection, ClientError> {
        let stream = UnixStream::connect(&self.socket_path).await?;
        let mut conn = Connection { framed: Framed::new(stream, MaraCodec) };
        let session_id = SessionId(format!("cli-{}", &Uuid::new_v4().simple().to_string()[..8]));
        let response = conn
            .call(Request::Hello {
                client_name: self.client_name.clone(),
                session_id,
                auth_token: self.auth_token.clone(),
            })
            .await?;
        match response {
            Response::HelloAck { .. } => Ok(conn),
            Response::Error { code, message } => Err(ClientError::Server { code, message }),
            other => Err(ClientError::Unexpected(format!("{other:?}"))),
        }
    }

    /// Trust-until-fail: a pooled connection is handed out as-is; if a
    /// later `call()` on it fails (broken pipe, daemon restart), the
    /// caller sees that error directly rather than a recycle-time probe.
    /// Fine for v0 at this scale — a health-check ping before every
    /// checkout is the natural upgrade if a wedged connection in the pool
    /// ever proves to be a real problem.
    async fn recycle(&self, _conn: &mut Connection, _metrics: &Metrics) -> RecycleResult<ClientError> {
        Ok(())
    }
}
