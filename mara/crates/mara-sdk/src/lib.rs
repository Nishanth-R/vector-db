#![deny(missing_docs)]
//! `mara-sdk`: a thin, single-collection ergonomic wrapper over the pooled
//! `mara-client` for Rust RAG apps (master plan Layer 5, *`mara-sdk` is a
//! thin ergonomic wrapper over the pooled `mara-client` for Rust RAG
//! apps*). Every non-Rust caller reaches `mara-api` over HTTP instead;
//! this crate exists for the Rust case specifically because it skips that
//! hop — same pooled Unix-socket connection, same `Engine`, lower latency.
//!
//! `MaraClient` already speaks every collection in a process; `MaraStore`
//! adds nothing to that surface except binding one client to one
//! collection name, since a RAG app almost always works against exactly
//! one (its document store) and passing `coll` to every call is pure
//! repetition once that's true. Reach through [`MaraStore::client`] for
//! anything this thin wrapper doesn't expose — a second collection on the
//! same pool, or the raw [`mara_client::MaraClient::call`] escape hatch.

pub use mara_client::{ClientError, DocumentSummary, MaraClient, ReindexSummary, SearchOutcome};
pub use mara_proto::{
    ChunkSpec, ChunkStrategy, DistanceMetric, ExtraPayload, Filter, PayloadRow, PayloadValue, PutItem, Row, Scalar, ScoredHit, SearchMode,
    WireFieldType, WireFusionMethod, WireIndexKind, WireSearchParams,
};

use std::path::PathBuf;

/// A [`MaraClient`] scoped to one collection — see the module docs for why.
pub struct MaraStore {
    client: MaraClient,
    coll: String,
}

impl MaraStore {
    /// Builds a pooled store for the Unix socket at `socket_path`, scoped to `coll`, with no auth token.
    pub fn connect_unix(socket_path: impl Into<PathBuf>, client_name: impl Into<String>, coll: impl Into<String>) -> Result<Self, ClientError> {
        Self::connect_unix_with_token(socket_path, client_name, coll, None)
    }

    /// Builds a pooled store for the Unix socket at `socket_path`, scoped to `coll`, authenticating with `auth_token`.
    pub fn connect_unix_with_token(
        socket_path: impl Into<PathBuf>,
        client_name: impl Into<String>,
        coll: impl Into<String>,
        auth_token: Option<String>,
    ) -> Result<Self, ClientError> {
        Ok(MaraStore {
            client: MaraClient::connect_unix_with_token(socket_path, client_name, auth_token)?,
            coll: coll.into(),
        })
    }

    /// Wraps an already-connected [`MaraClient`] rather than opening a new
    /// pool — for an app that already talks to other collections on the
    /// same connection and wants a `MaraStore`'s ergonomics for just one
    /// of them, without paying for a second pool.
    pub fn from_client(client: MaraClient, coll: impl Into<String>) -> Self {
        MaraStore { client, coll: coll.into() }
    }

    /// The collection name this store is scoped to.
    pub fn collection(&self) -> &str {
        &self.coll
    }

    /// The underlying pooled client — for anything this wrapper doesn't
    /// expose, or to reach a different collection on the same pool.
    pub fn client(&self) -> &MaraClient {
        &self.client
    }

    /// Creates the store's collection with a fixed dimensionality, metric, and payload schema.
    pub async fn create_collection(&self, dim: usize, metric: DistanceMetric, schema: Vec<(String, WireFieldType)>) -> Result<(), ClientError> {
        self.client.create_collection(&self.coll, dim, metric, schema).await
    }

    /// Chunk-and-insert: splits `text` per `chunk_spec` (falling back to
    /// [`ChunkSpec::default`] — markdown-aware, 512 tokens, 64 overlap —
    /// when `None`), embeds every chunk with the daemon's configured
    /// local model, and commits the whole document as one transaction.
    pub async fn add_document(
        &self,
        doc_key: &str,
        text: &str,
        chunk_spec: Option<ChunkSpec>,
        fields: PayloadRow,
        source: Option<String>,
    ) -> Result<DocumentSummary, ClientError> {
        self.client
            .put_document(&self.coll, doc_key, text, chunk_spec.unwrap_or_default(), fields, source)
            .await
    }

    /// Inserts or replaces a single row in the store's collection.
    pub async fn put(&self, key: &str, vector: Vec<f32>, fields: PayloadRow, extra: Option<ExtraPayload>) -> Result<Row, ClientError> {
        self.client.put(&self.coll, key, vector, fields, extra).await
    }

    /// Inserts or replaces many rows in the store's collection, in one transaction.
    pub async fn put_batch(&self, items: Vec<PutItem>) -> Result<Vec<Row>, ClientError> {
        self.client.put_batch(&self.coll, items).await
    }

    /// Fetches a single row by its key; `Ok(None)` if it doesn't exist.
    pub async fn get(&self, key: &str) -> Result<Option<Row>, ClientError> {
        self.client.get_by_key(&self.coll, key).await
    }

    /// Deletes a single row by its key.
    pub async fn delete(&self, key: &str) -> Result<(), ClientError> {
        self.client.delete(&self.coll, key).await
    }

    /// Hybrid search (BM25 + vector, fused via reciprocal rank fusion)
    /// with every advanced knob at its documented default — the entry
    /// point a RAG app reaches for first. Use [`MaraStore::search`] for
    /// vector-only/BM25-only/filtered/tuned control.
    pub async fn query(&self, query_text: &str, k: usize) -> Result<SearchOutcome, ClientError> {
        self.client
            .search(
                &self.coll,
                Some(query_text.to_string()),
                None,
                SearchMode::Hybrid { method: None, overfetch_k: None },
                k,
                None,
                WireSearchParams::default(),
            )
            .await
    }

    /// Vector-only/BM25-only/hybrid search with full control over mode, filter, and tuning params.
    #[allow(clippy::too_many_arguments)]
    pub async fn search(
        &self,
        query_text: Option<String>,
        query_vector: Option<Vec<f32>>,
        mode: SearchMode,
        k: usize,
        filter: Option<Filter>,
        params: WireSearchParams,
    ) -> Result<SearchOutcome, ClientError> {
        self.client.search(&self.coll, query_text, query_vector, mode, k, filter, params).await
    }

    /// Builds or rebuilds an index for the store's collection from current storage state.
    pub async fn reindex(&self, index_kind: WireIndexKind) -> Result<ReindexSummary, ClientError> {
        self.client.reindex(&self.coll, index_kind).await
    }
}
