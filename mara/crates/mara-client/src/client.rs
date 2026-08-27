//! `mara-client`'s public, ergonomic surface — what `mara-cli` and
//! `mara-sdk` build on. One `MaraClient` owns a pooled set of
//! already-`Hello`'d connections; each call here checks one out, sends one
//! request, returns it to the pool.

use crate::error::ClientError;
use crate::manager::ConnectionManager;
use deadpool::managed::Pool;
use mara_proto::{
    ChunkSpec, DistanceMetric, DocId, ExtraPayload, Filter, PayloadRow, PutItem, Request, Response, Row, ScoredHit, SearchMode, WireFieldType,
    WireIndexKind, WireSearchParams,
};
use std::path::PathBuf;

/// What a successful `put_document` confirms — mirrors
/// `Response::Document` without exposing the wire type directly.
pub struct DocumentSummary {
    /// Identifier assigned to the new document.
    pub doc_id: DocId,
    /// The document's key, as supplied by the caller.
    pub doc_key: String,
    /// Number of chunks the document was split into.
    pub chunk_count: u32,
    /// Document version after this write.
    pub version: u32,
}

/// What a successful `search` returns — mirrors
/// `Response::SearchResults` without exposing the wire type directly.
pub struct SearchOutcome {
    /// The matched hits, best first.
    pub hits: Vec<ScoredHit>,
    /// Whether a filtered-ANN escalation hit its budget before finding a full `k`.
    pub truncated_by_filter: bool,
}

/// What a successful `reindex` confirms — mirrors `Response::Reindexed`
/// without exposing the wire type directly.
pub struct ReindexSummary {
    /// The collection that was reindexed.
    pub coll: String,
    /// Which index kind was built.
    pub index_kind: WireIndexKind,
    /// Number of rows now covered by the index.
    pub row_count: u64,
}

/// Cheap to clone — `deadpool::managed::Pool` is itself an `Arc`-backed
/// handle, so every clone shares the same underlying pool of connections
/// rather than opening a second one.
#[derive(Clone)]
pub struct MaraClient {
    pool: Pool<ConnectionManager>,
}

impl MaraClient {
    /// Builds a pooled client for the Unix socket at `socket_path`, with no auth token.
    pub fn connect_unix(socket_path: impl Into<PathBuf>, client_name: impl Into<String>) -> Result<Self, ClientError> {
        Self::connect_unix_with_token(socket_path, client_name, None)
    }

    /// Builds a pooled client for the Unix socket at `socket_path`, authenticating with `auth_token`.
    pub fn connect_unix_with_token(socket_path: impl Into<PathBuf>, client_name: impl Into<String>, auth_token: Option<String>) -> Result<Self, ClientError> {
        let manager = ConnectionManager::new(socket_path, client_name, auth_token);
        let pool = Pool::builder(manager).max_size(16).build().map_err(|e| ClientError::Pool(e.to_string()))?;
        Ok(MaraClient { pool })
    }

    /// The raw request/response escape hatch beneath the typed methods
    /// below — what `mara-cli` uses so its response-printing logic is
    /// identical whether a command ran over the socket or (with
    /// `--embedded`) in-process against `Engine::handle` directly.
    pub async fn call(&self, req: Request) -> Result<Response, ClientError> {
        let mut conn = self.pool.get().await.map_err(|e| ClientError::Pool(e.to_string()))?;
        conn.call(req).await
    }

    /// Creates a new collection with a fixed dimensionality, metric, and payload schema.
    pub async fn create_collection(&self, name: &str, dim: usize, metric: DistanceMetric, schema: Vec<(String, WireFieldType)>) -> Result<(), ClientError> {
        match self
            .call(Request::CreateCollection {
                name: name.to_string(),
                dim,
                metric,
                schema,
            })
            .await?
        {
            Response::Ok => Ok(()),
            Response::Error { code, message } => Err(ClientError::Server { code, message }),
            other => Err(ClientError::Unexpected(format!("{other:?}"))),
        }
    }

    /// Inserts or replaces a single row.
    pub async fn put(&self, coll: &str, key: &str, vector: Vec<f32>, fields: PayloadRow, extra: Option<ExtraPayload>) -> Result<Row, ClientError> {
        match self
            .call(Request::Put {
                coll: coll.to_string(),
                key: key.to_string(),
                text: None,
                vector: Some(vector),
                fields,
                extra,
            })
            .await?
        {
            Response::Row(row) => Ok(row),
            Response::Error { code, message } => Err(ClientError::Server { code, message }),
            other => Err(ClientError::Unexpected(format!("{other:?}"))),
        }
    }

    /// Inserts or replaces many rows in one transaction.
    pub async fn put_batch(&self, coll: &str, items: Vec<PutItem>) -> Result<Vec<Row>, ClientError> {
        match self.call(Request::PutBatch { coll: coll.to_string(), items }).await? {
            Response::Rows(rows) => Ok(rows),
            Response::Error { code, message } => Err(ClientError::Server { code, message }),
            other => Err(ClientError::Unexpected(format!("{other:?}"))),
        }
    }

    /// `Ok(None)` on a `not_found` server response — every other error
    /// still propagates as `Err`.
    pub async fn get_by_key(&self, coll: &str, key: &str) -> Result<Option<Row>, ClientError> {
        match self
            .call(Request::GetByKey {
                coll: coll.to_string(),
                key: key.to_string(),
            })
            .await?
        {
            Response::Row(row) => Ok(Some(row)),
            Response::Error { code, .. } if code == "not_found" => Ok(None),
            Response::Error { code, message } => Err(ClientError::Server { code, message }),
            other => Err(ClientError::Unexpected(format!("{other:?}"))),
        }
    }

    /// Inserts a brand-new document, chunking and embedding `text` server-side.
    pub async fn put_document(
        &self,
        coll: &str,
        doc_key: &str,
        text: &str,
        chunk_spec: ChunkSpec,
        fields: PayloadRow,
        source: Option<String>,
    ) -> Result<DocumentSummary, ClientError> {
        match self
            .call(Request::PutDocument {
                coll: coll.to_string(),
                doc_key: doc_key.to_string(),
                text: text.to_string(),
                chunk_spec,
                fields,
                source,
            })
            .await?
        {
            Response::Document { doc_id, doc_key, chunk_count, version } => Ok(DocumentSummary { doc_id, doc_key, chunk_count, version }),
            Response::Error { code, message } => Err(ClientError::Server { code, message }),
            other => Err(ClientError::Unexpected(format!("{other:?}"))),
        }
    }

    /// Deletes a single row by its key.
    pub async fn delete(&self, coll: &str, key: &str) -> Result<(), ClientError> {
        match self
            .call(Request::Delete {
                coll: coll.to_string(),
                key: key.to_string(),
            })
            .await?
        {
            Response::Ok => Ok(()),
            Response::Error { code, message } => Err(ClientError::Server { code, message }),
            other => Err(ClientError::Unexpected(format!("{other:?}"))),
        }
    }

    /// `query_vector: None` with `query_text: Some(_)` embeds locally on
    /// the daemon side, the same way `put`'s text-only form does — see
    /// `Request::Search`'s doc comment for exactly what each `mode`
    /// requires.
    #[allow(clippy::too_many_arguments)]
    pub async fn search(
        &self,
        coll: &str,
        query_text: Option<String>,
        query_vector: Option<Vec<f32>>,
        mode: SearchMode,
        k: usize,
        filter: Option<Filter>,
        params: WireSearchParams,
    ) -> Result<SearchOutcome, ClientError> {
        match self
            .call(Request::Search {
                coll: coll.to_string(),
                query_text,
                query_vector,
                mode,
                k,
                filter,
                params,
            })
            .await?
        {
            Response::SearchResults { hits, truncated_by_filter } => Ok(SearchOutcome { hits, truncated_by_filter }),
            Response::Error { code, message } => Err(ClientError::Server { code, message }),
            other => Err(ClientError::Unexpected(format!("{other:?}"))),
        }
    }

    /// Builds or rebuilds an index for `coll` from current storage state.
    pub async fn reindex(&self, coll: &str, index_kind: WireIndexKind) -> Result<ReindexSummary, ClientError> {
        match self.call(Request::Reindex { coll: coll.to_string(), index_kind }).await? {
            Response::Reindexed { coll, index_kind, row_count } => Ok(ReindexSummary { coll, index_kind, row_count }),
            Response::Error { code, message } => Err(ClientError::Server { code, message }),
            other => Err(ClientError::Unexpected(format!("{other:?}"))),
        }
    }
}
