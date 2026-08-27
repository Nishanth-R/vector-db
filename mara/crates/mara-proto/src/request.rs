use crate::chunk::ChunkSpec;
use crate::distance::DistanceMetric;
use crate::filter::Filter;
use crate::ids::SessionId;
use crate::payload::{ExtraPayload, PayloadRow};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Wire mirror of `mara_fusion::FusionMethod` (which the protocol crate
/// can't depend on).
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireFusionMethod {
    /// `k: None` resolves to the master plan's default, `60`.
    ReciprocalRankFusion {
        /// RRF constant; `None` resolves to the master plan's default.
        k: Option<u32>,
    },
    /// Fuses arms by a weighted sum of their normalized scores.
    WeightedSum {
        /// Weight applied to the vector arm's score.
        vector_weight: f32,
        /// Weight applied to the BM25 arm's score.
        bm25_weight: f32,
    },
}

/// Which *signal* backs a `Search` — vector, lexical, or both fused.
/// `Hybrid` runs both arms and fuses server-side (`mara-fusion`) in one
/// round trip — see `Request::Search`'s doc comment for what it requires.
/// Independent of `WireIndexKind`, which is about which *algorithm*
/// backs the vector arm specifically (`Flat`/`IvfPq`/`Lsh`), not which
/// signal a search uses.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// Search using only the vector (embedding) index.
    VectorOnly,
    /// Search using only the BM25 lexical index.
    Bm25Only,
    /// Search both arms and fuse their results server-side.
    Hybrid {
        /// `None` resolves to the master plan's default,
        /// `ReciprocalRankFusion { k: 60 }`.
        method: Option<WireFusionMethod>,
        /// How many hits each arm contributes to the fusion candidate
        /// pool before ranking. `None` resolves to the master plan's
        /// default, `max(k*4, 50)`.
        overfetch_k: Option<usize>,
    },
}

/// What `Reindex` builds or rebuilds for a collection. `Bm25` is a
/// one-time scan-backfill followed by (re-)attaching the live
/// `ChangeSubscriber` feed — everything after that point stays current on
/// its own. `Flat` needs no build step at all (it re-scans live storage
/// on every search) and is never wrapped for staleness; `IvfPq`/`Lsh`
/// both genuinely cache build-time state and are `LiveIndex`-wrapped, so
/// a write after `Reindex` is still immediately searchable via the delta
/// buffer rather than waiting on the next explicit `Reindex`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireIndexKind {
    /// Exact brute-force scan; no build step, always current.
    Flat,
    /// Inverted-file product-quantization approximate index.
    IvfPq,
    /// Locality-sensitive-hashing approximate index.
    Lsh,
    /// BM25 lexical index.
    Bm25,
}

/// Wire mirror of `mara_index_vector::SearchParams` (which the protocol
/// crate can't depend on) — every field optional so a client can override
/// just the knobs it cares about; `mara-daemon` fills in
/// `SearchParams::default()`'s values for anything left `None`.
#[derive(Clone, Copy, PartialEq, Debug, Default, Serialize, Deserialize, ToSchema)]
pub struct WireSearchParams {
    /// Caps how many chunks from the same document may appear in results.
    pub max_chunks_per_doc: Option<u32>,
    /// Number of IVF clusters to probe.
    pub nprobe: Option<usize>,
    /// Beam width used during approximate search.
    pub beam_width: Option<usize>,
    /// Number of candidates to exactly rerank after the approximate pass.
    pub rerank_k: Option<usize>,
    /// Below this candidate count, fall back to exact filtered scan.
    pub filter_exact_threshold: Option<usize>,
    /// Selectivity above which a filter is treated as "high selectivity".
    pub filter_selectivity_high: Option<f64>,
    /// Upper bound on `nprobe` when a filter is applied.
    pub filter_nprobe_max: Option<usize>,
}

/// One row to insert in a `PutBatch`.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, ToSchema)]
pub struct PutItem {
    /// The item's unique key within the collection.
    pub key: String,
    /// Text to embed, if the vector isn't supplied directly.
    pub text: Option<String>,
    /// Precomputed embedding vector, if not deriving it from `text`.
    pub vector: Option<Vec<f32>>,
    /// Typed, schema-checked payload fields.
    pub fields: PayloadRow,
    /// Un-indexed free-form JSON payload.
    #[schema(value_type = Object)]
    pub extra: Option<ExtraPayload>,
}

/// The wire-level mirror of `mara-storage::payload::FieldType`. Lives here
/// (not in `mara-storage`) because the protocol crate can't depend on
/// storage — `mara-daemon`, which depends on both, converts this into the
/// real `PayloadSchema` when handling `CreateCollection`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireFieldType {
    /// A single exact-match string.
    Keyword,
    /// A list of exact-match strings.
    KeywordList,
    /// A signed 64-bit integer.
    I64,
    /// A 64-bit floating point number.
    F64,
    /// A timestamp, in epoch milliseconds.
    DateTime,
    /// A boolean.
    Bool,
    /// A BM25-searchable text field.
    Text,
}

/// The wire protocol's request vocabulary. Deliberately minimal for now —
/// `Hello`/`CreateCollection`/`Put`/`PutBatch`/`GetByKey`/`Delete`/
/// `PutDocument` — matching what storage (steps 0-10), a testable daemon
/// (step 11), and server-side chunking (step 14) actually need; replace,
/// search, undo, and admin request kinds are added alongside the layers
/// that serve them.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Request {
    /// First message on a new connection: identifies the client and session.
    Hello {
        /// Human-readable name of the connecting client.
        client_name: String,
        /// Session id to attribute this and subsequent requests to.
        session_id: SessionId,
        /// Bearer auth token, required when `auth.enabled = true`.
        auth_token: Option<String>,
    },
    /// Creates a new collection with a fixed dimensionality, metric, and payload schema.
    CreateCollection {
        /// Name of the collection to create.
        name: String,
        /// Vector dimensionality for the collection.
        dim: usize,
        /// Distance metric the collection searches under.
        metric: DistanceMetric,
        /// Payload field names and their types.
        schema: Vec<(String, WireFieldType)>,
    },
    /// Inserts or replaces a single row.
    Put {
        /// Target collection name.
        coll: String,
        /// Row key, unique within the collection.
        key: String,
        /// Text to embed, if the vector isn't supplied directly.
        text: Option<String>,
        /// Precomputed embedding vector, if not deriving it from `text`.
        vector: Option<Vec<f32>>,
        /// Typed, schema-checked payload fields.
        fields: PayloadRow,
        /// Un-indexed free-form JSON payload.
        extra: Option<ExtraPayload>,
    },
    /// Inserts or replaces many rows in one transaction.
    PutBatch {
        /// Target collection name.
        coll: String,
        /// Rows to write.
        items: Vec<PutItem>,
    },
    /// Fetches a single row by its key.
    GetByKey {
        /// Target collection name.
        coll: String,
        /// Row key to fetch.
        key: String,
    },
    /// Deletes a single row by its key.
    Delete {
        /// Target collection name.
        coll: String,
        /// Row key to delete.
        key: String,
    },
    /// Inserts a brand-new document: the daemon splits `text` per
    /// `chunk_spec` (`mara-chunker`), embeds every chunk with its
    /// configured local model, and commits the whole document as one
    /// `mara-storage::put_document` transaction. Fails if `doc_key`
    /// already exists — no replace/upsert form yet.
    PutDocument {
        /// Target collection name.
        coll: String,
        /// Document key, unique within the collection.
        doc_key: String,
        /// Raw document text to chunk and embed.
        text: String,
        /// How to split `text` into chunks.
        #[serde(default)]
        chunk_spec: ChunkSpec,
        /// Typed, schema-checked payload fields, shared by every chunk.
        #[serde(default)]
        fields: PayloadRow,
        /// Optional provenance string (e.g. a file path or URL) for the document.
        source: Option<String>,
    },
    /// `mode: VectorOnly` embeds `query_text` locally (same as `Put`) when
    /// `query_vector` isn't already supplied; `mode: Bm25Only` requires
    /// `query_text` and ignores `query_vector`. `filter` is compiled by
    /// storage into a `FilterMask` before either index sees it, except a
    /// `Filter::TextMatch` leaf, which storage can't resolve itself (see
    /// `mara_storage::payload::FilterError::TextMatchUnsupported`) — the
    /// daemon resolves those against the BM25 index while compiling the
    /// rest of the tree.
    Search {
        /// Target collection name.
        coll: String,
        /// Query text to embed/search, if `query_vector` isn't supplied.
        query_text: Option<String>,
        /// Precomputed query embedding, if not deriving it from `query_text`.
        query_vector: Option<Vec<f32>>,
        /// Which signal(s) to search — vector, BM25, or both fused.
        mode: SearchMode,
        /// Number of top hits to return.
        k: usize,
        /// Predicate restricting which rows are eligible to match.
        filter: Option<Filter>,
        /// Tuning knobs overriding the daemon's default search parameters.
        #[serde(default)]
        params: WireSearchParams,
    },
    /// Builds or rebuilds `index_kind` for `coll` from current storage
    /// state. `CreateCollection` already attaches a live (empty) BM25
    /// index automatically, so `Reindex{index_kind: Bm25}` is really for
    /// backfilling a collection that existed before this daemon process
    /// did (rediscovered from the on-disk registry, never live-BM25-
    /// attached) — see `WireIndexKind::Bm25`'s doc comment.
    Reindex {
        /// Target collection name.
        coll: String,
        /// Which index to build or rebuild.
        index_kind: WireIndexKind,
    },
    /// Sent once, immediately after a successful `Hello` establishes a
    /// `Role::Replica` principal, over the leader's dedicated replication
    /// listener — never over the normal client-facing UDS/TCP listener.
    /// `known_lsns` is this follower's resume cursor per collection it
    /// already has locally; a collection missing from the map (or the
    /// follower's very first connection ever) gets everything from the
    /// start. Answered with exactly one `Response::ReplicaWelcome`,
    /// after which the leader streams `Response::ReplicaWalLines`/
    /// `Response::ReplicaNewCollection` continuously with no further
    /// `Request` needed — see the master plan's *Replication*.
    ReplicaHello {
        /// Per-collection resume cursor; a missing collection starts from the beginning.
        known_lsns: Vec<(String, Option<crate::lsn::Lsn>)>,
    },
}

impl Request {
    /// A short, stable label for audit/log entries — never the full
    /// request payload, which may contain raw vectors.
    pub fn action_name(&self) -> &'static str {
        match self {
            Request::Hello { .. } => "hello",
            Request::CreateCollection { .. } => "create_collection",
            Request::Put { .. } => "put",
            Request::PutBatch { .. } => "put_batch",
            Request::GetByKey { .. } => "get_by_key",
            Request::Delete { .. } => "delete",
            Request::PutDocument { .. } => "put_document",
            Request::Search { .. } => "search",
            Request::Reindex { .. } => "reindex",
            Request::ReplicaHello { .. } => "replica_hello",
        }
    }
}
