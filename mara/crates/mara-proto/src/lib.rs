//! Shared wire types for `mara`: `Request`/`Response`, the framing codec,
//! `RowId`/`DocId`/`Lsn`/`DistanceMetric`, the `Filter` AST, `ChunkSpec`,
//! `UndoTarget`, and `RequestCtx`. Every other crate in the workspace
//! depends on this one; it depends on nothing else in the workspace.

pub mod chunk;
pub mod codec;
pub mod ctx;
pub mod distance;
pub mod error;
pub mod filter;
pub mod ids;
pub mod lsn;
pub mod model;
pub mod payload;
pub mod request;
pub mod response;
pub mod undo;

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
pub use request::{PutItem, Request};
pub use response::{Response, Row};
pub use undo::{Conflict, RevertPoint, TxnEntry, TxnSummary, UndoScope, UndoTarget};
