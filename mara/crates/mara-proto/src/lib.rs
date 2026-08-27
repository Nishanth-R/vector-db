//! Shared wire types for `mara`: `Request`/`Response`, the framing codec,
//! `RowId`/`DocId`/`Lsn`/`DistanceMetric`, the `Filter` AST, `ChunkSpec`,
//! `UndoTarget`, and `RequestCtx`. Every other crate in the workspace
//! depends on this one; it depends on nothing else in the workspace.
#![deny(missing_docs)]

/// Async framed reader/writer for `Request`/`Response` over any `AsyncRead`/`AsyncWrite`.
pub mod async_codec;
/// Chunking strategies and specs for splitting documents into indexable pieces.
pub mod chunk;
/// Synchronous length-prefixed framing and (de)serialization of requests/responses.
pub mod codec;
/// Request context: principal, role, and source of an incoming call.
pub mod ctx;
/// Vector distance/similarity metrics.
pub mod distance;
/// Error and result types shared across the wire protocol.
pub mod error;
/// The filter expression AST used to restrict searches and scans.
pub mod filter;
/// Strongly-typed identifiers used throughout the protocol.
pub mod ids;
/// Log sequence number type for ordering write-ahead log entries.
pub mod lsn;
/// Embedding model fingerprinting.
pub mod model;
/// Payload row and value types attached to stored documents.
pub mod payload;
/// Request message types sent from clients to the daemon.
pub mod request;
/// Response message types returned from the daemon to clients.
pub mod response;
/// Undo/redo log types for reverting transactions.
pub mod undo;

pub use async_codec::{CodecError, MaraCodec};
pub use chunk::{ChunkSpec, ChunkStrategy};
pub use codec::{decode_request_body, decode_response_body, encode_request, encode_response, try_decode_frame, DecodedFrame, FrameKind};
pub use ctx::{Principal, RequestCtx, Role, Source};
pub use distance::DistanceMetric;
pub use error::{ProtoError, ProtoResult};
pub use filter::{Filter, Scalar};
pub use ids::{DocId, PrincipalId, RowId, SessionId, TxnId};
pub use lsn::Lsn;
pub use model::ModelFingerprint;
pub use payload::{ExtraPayload, PayloadRow, PayloadValue};
pub use request::{PutItem, Request, SearchMode, WireFieldType, WireFusionMethod, WireIndexKind, WireSearchParams};
pub use response::{ReplicaCollectionInfo, Response, Row, ScoredHit};
pub use undo::{Conflict, RevertPoint, TxnEntry, TxnSummary, UndoScope, UndoTarget};
