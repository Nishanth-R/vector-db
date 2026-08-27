use crate::ids::{DocId, RowId, SessionId};
use crate::payload::{ExtraPayload, PayloadRow};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// A single stored row (or document chunk), returned in full rather than by reference.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, ToSchema)]
pub struct Row {
    /// Dense, per-collection row identifier.
    pub id: RowId,
    /// The row's key, unique within the collection.
    pub key: String,
    /// The row's embedding vector, if stored.
    pub vector: Option<Vec<f32>>,
    /// Typed, schema-checked payload fields.
    pub fields: PayloadRow,
    /// Un-indexed free-form JSON payload.
    #[schema(value_type = Object)]
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
    /// Position of this chunk within its document, if it is one.
    pub chunk_ord: Option<u32>,
}

/// One `Search` result — the wire mirror of `mara_index_vector::SearchHit`
/// / `mara_index_bm25::LexicalHit`, carrying the full `Row` rather than
/// just an id since a client asked a question and wants an answer, not a
/// second round trip to resolve it.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, ToSchema)]
pub struct ScoredHit {
    /// The matched row.
    pub row: Row,
    /// Higher is better. Not comparable across `SearchMode`s — a vector
    /// score and a BM25 score live on unrelated scales (see
    /// `mara_fusion`, the layer that reconciles them for hybrid search).
    pub score: f32,
    /// `true` for every BM25 hit and every vector hit that went through
    /// exact rerank (the default); `false` only for `IvfPqIndex`'s
    /// explicit `rerank_k: Some(0)` max-speed mode.
    pub exact: bool,
}

/// Mirrors [`crate::request::Request`]'s current, deliberately minimal,
/// scope.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Response {
    /// Generic success with no payload.
    Ok,
    /// Reply to a successful `Hello`.
    HelloAck {
        /// The session id now bound to this connection.
        session_id: SessionId,
        /// The daemon's version string.
        server_version: String,
    },
    /// A single row, e.g. in reply to `GetByKey`.
    Row(Row),
    /// Multiple rows.
    Rows(Vec<Row>),
    /// Confirms a document-lifecycle op (currently just `PutDocument`)
    /// without shipping every chunk's full `Row` back — the caller already
    /// has the text and payload it just sent.
    Document {
        /// Identifier assigned to the new document.
        doc_id: DocId,
        /// The document's key, as supplied by the caller.
        doc_key: String,
        /// Number of chunks the document was split into.
        chunk_count: u32,
        /// Document version after this write.
        version: u32,
    },
    /// `truncated_by_filter` mirrors `mara_index_vector::SearchResult` —
    /// see the master plan's *Filtered search*: `true` only when a
    /// filtered-ANN escalation hit its budget cap short of a full `k`,
    /// never when the collection genuinely has fewer than `k` matches.
    /// Always `false` for `SearchMode::Bm25Only`, which has no escalation
    /// concept.
    SearchResults {
        /// The matched hits, best first.
        hits: Vec<ScoredHit>,
        /// Whether a filtered-ANN escalation hit its budget before finding a full `k`.
        truncated_by_filter: bool,
    },
    /// Confirms a `Reindex` call — which index kind, and how many rows it
    /// now covers.
    Reindexed {
        /// The collection that was reindexed.
        coll: String,
        /// Which index kind was built.
        index_kind: crate::request::WireIndexKind,
        /// Number of rows now covered by the index.
        row_count: u64,
    },
    /// A request failed; carries a stable machine-readable code and a human message.
    Error {
        /// Stable, machine-readable error code.
        code: String,
        /// Human-readable description of the failure.
        message: String,
    },
    /// One entry per collection the leader currently knows about, sent
    /// once in reply to `Request::ReplicaHello` — a fresh follower
    /// `create_collection`s a local match for every entry it doesn't
    /// already have before any WAL streaming begins.
    ReplicaWelcome {
        /// Every collection the leader currently knows about.
        collections: Vec<ReplicaCollectionInfo>,
    },
    /// A collection created on the leader after the follower's initial
    /// `ReplicaWelcome` — pushed unprompted, the same way `ReplicaWalLines`
    /// is, so a follower that's been streaming for a while learns about it
    /// without reconnecting.
    ReplicaNewCollection {
        /// Identity of the newly created collection.
        info: ReplicaCollectionInfo,
    },
    /// Raw, already-checksummed WAL JSONL lines for `coll`, strictly after
    /// whatever this follower last applied — pushed unprompted by the
    /// leader's replication task, with no corresponding `Request` for any
    /// individual batch (only the one `ReplicaHello` that started the
    /// stream). Apply via the exact same function used for local crash
    /// recovery, per the master plan's *Replication*.
    ReplicaWalLines {
        /// The collection these WAL lines belong to.
        coll: String,
        /// Raw, checksummed WAL JSONL lines, in apply order.
        lines: Vec<String>,
    },
}

/// One collection's identity, as the replication leader states it — enough
/// for a follower to `create_collection` a matching local copy. Distinct
/// from `mara_storage::CollectionInfo` (which the protocol crate can't
/// depend on): this is the wire shape, carrying the wire `WireFieldType`
/// schema rather than storage's internal `FieldType`.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ReplicaCollectionInfo {
    /// Name of the collection.
    pub name: String,
    /// Vector dimensionality for the collection.
    pub dim: usize,
    /// Distance metric the collection searches under.
    pub metric: crate::distance::DistanceMetric,
    /// Payload field names and their types.
    pub schema: Vec<(String, crate::request::WireFieldType)>,
}
