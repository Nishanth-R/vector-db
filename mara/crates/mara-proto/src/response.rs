use crate::ids::{DocId, RowId, SessionId};
use crate::payload::{ExtraPayload, PayloadRow};
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Row {
    pub id: RowId,
    pub key: String,
    pub vector: Option<Vec<f32>>,
    pub fields: PayloadRow,
    pub extra: Option<ExtraPayload>,
    /// The row's source text, if any — always present on a document chunk,
    /// optional on a plain row. Kept separate from `fields` rather than a
    /// payload field of type `text`: BM25 and re-embedding both need it
    /// back verbatim, and it's present on every chunk unconditionally
    /// rather than being something a schema opts into.
    pub text: Option<String>,
    /// `None` for a standalone row never inserted through the document
    /// API; `Some` for a chunk, which is every row inserted via
    /// `put_document`/`replace_document`.
    pub doc_id: Option<DocId>,
    pub chunk_ord: Option<u32>,
}

/// Mirrors [`crate::request::Request`]'s current, deliberately minimal,
/// scope.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Response {
    Ok,
    HelloAck {
        session_id: SessionId,
        server_version: String,
    },
    Row(Row),
    Rows(Vec<Row>),
    Error {
        code: String,
        message: String,
    },
}
