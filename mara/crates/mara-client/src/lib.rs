#![deny(missing_docs)]
//! Pooled async client (master plan Layer 3, `ConnectionPool` — client
//! side): checkout/send/return, so `mara-cli`/`mara-sdk` aren't opening a
//! fresh socket + handshake per query.

pub mod client;
/// A single pooled request/response connection to the daemon.
pub mod connection;
/// Errors returned by the client.
pub mod error;
pub mod manager;

pub use client::{DocumentSummary, MaraClient, ReindexSummary, SearchOutcome};
pub use connection::Connection;
pub use error::ClientError;
pub use manager::ConnectionManager;
