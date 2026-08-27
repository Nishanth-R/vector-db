//! Axum handlers — one per REST endpoint, each: resolve the caller's
//! `RequestCtx` from its bearer token, build the matching `mara_proto`
//! `Request`, hand it to `Arc<dyn Engine>::handle` (the same dispatcher
//! and authorization table the native UDS/TCP listener uses), and render
//! the `Response` back as JSON. Authorization itself lives entirely in
//! `Engine::handle` — these handlers never re-check a `Capability`, except
//! `audit`, which bypasses `Engine` entirely (there is no `Request::Audit`
//! wire variant) and so checks `Capability::AuditRead` itself.

use crate::auth::resolve_ctx;
use crate::dto::*;
use crate::error::{error_response, unexpected_response};
use crate::ApiState;
use axum::extract::{ConnectInfo, Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response as AxumResponse};
use axum::Json;
use mara_proto::{Request, Response};
use std::net::SocketAddr;

macro_rules! ctx_or_return {
    ($state:expr, $headers:expr, $addr:expr) => {
        match resolve_ctx(&$state.shared, &$headers, $addr) {
            Ok(ctx) => ctx,
            Err(e) => return e.into_response(),
        }
    };
}

#[utoipa::path(get, path = "/v1/health", responses((status = 200, description = "the process is up", body = HealthResponse)))]
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

#[utoipa::path(
    post,
    path = "/v1/collections",
    request_body = CreateCollectionRequest,
    responses(
        (status = 201, description = "collection created", body = CreateCollectionResponse),
        (status = 400, description = "bad request", body = ErrorBody),
        (status = 401, description = "missing/invalid bearer token", body = ErrorBody),
        (status = 403, description = "principal not permitted", body = ErrorBody),
        (status = 409, description = "collection already exists", body = ErrorBody),
    )
)]
pub async fn create_collection(
    State(state): State<ApiState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<CreateCollectionRequest>,
) -> AxumResponse {
    let ctx = ctx_or_return!(state, headers, addr);
    let schema = body.schema.into_iter().map(|f| (f.name, f.field_type)).collect();
    let req = Request::CreateCollection {
        name: body.name.clone(),
        dim: body.dim,
        metric: body.metric,
        schema,
    };
    match state.shared.engine.handle(&ctx, req).await {
        Response::Ok => (
            StatusCode::CREATED,
            Json(CreateCollectionResponse { name: body.name, dim: body.dim, metric: body.metric }),
        )
            .into_response(),
        Response::Error { code, message } => error_response(&code, message),
        other => unexpected_response(other),
    }
}

/// `key` comes from the URL — REST's `PUT .../{key}` is the idiomatic
/// shape for `mara_proto::Request::Put`'s upsert-by-key semantics.
#[utoipa::path(
    put,
    path = "/v1/collections/{coll}/rows/{key}",
    params(("coll" = String, Path), ("key" = String, Path)),
    request_body = PutRowRequest,
    responses(
        (status = 200, description = "row inserted or updated", body = mara_proto::Row),
        (status = 400, description = "bad request", body = ErrorBody),
        (status = 401, description = "missing/invalid bearer token", body = ErrorBody),
        (status = 403, description = "principal not permitted", body = ErrorBody),
        (status = 503, description = "text given but no embedding model configured", body = ErrorBody),
    )
)]
pub async fn put_row(
    State(state): State<ApiState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Path((coll, key)): Path<(String, String)>,
    headers: HeaderMap,
    Json(body): Json<PutRowRequest>,
) -> AxumResponse {
    let ctx = ctx_or_return!(state, headers, addr);
    let req = Request::Put {
        coll,
        key,
        text: body.text,
        vector: body.vector,
        fields: body.fields,
        extra: body.extra,
    };
    match state.shared.engine.handle(&ctx, req).await {
        Response::Row(row) => (StatusCode::OK, Json(row)).into_response(),
        Response::Error { code, message } => error_response(&code, message),
        other => unexpected_response(other),
    }
}

#[utoipa::path(
    post,
    path = "/v1/collections/{coll}/rows/batch",
    params(("coll" = String, Path)),
    request_body = PutBatchRequest,
    responses(
        (status = 200, description = "rows inserted or updated, in request order", body = PutBatchResponse),
        (status = 400, description = "bad request", body = ErrorBody),
        (status = 401, description = "missing/invalid bearer token", body = ErrorBody),
        (status = 403, description = "principal not permitted", body = ErrorBody),
        (status = 503, description = "text given but no embedding model configured", body = ErrorBody),
    )
)]
pub async fn put_batch(
    State(state): State<ApiState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Path(coll): Path<String>,
    headers: HeaderMap,
    Json(body): Json<PutBatchRequest>,
) -> AxumResponse {
    let ctx = ctx_or_return!(state, headers, addr);
    let req = Request::PutBatch { coll, items: body.items };
    match state.shared.engine.handle(&ctx, req).await {
        Response::Rows(rows) => (StatusCode::OK, Json(PutBatchResponse { rows })).into_response(),
        Response::Error { code, message } => error_response(&code, message),
        other => unexpected_response(other),
    }
}

#[utoipa::path(
    get,
    path = "/v1/collections/{coll}/rows/{key}",
    params(("coll" = String, Path), ("key" = String, Path)),
    responses(
        (status = 200, description = "the row", body = mara_proto::Row),
        (status = 401, description = "missing/invalid bearer token", body = ErrorBody),
        (status = 403, description = "principal not permitted", body = ErrorBody),
        (status = 404, description = "no row with that key", body = ErrorBody),
    )
)]
pub async fn get_row(State(state): State<ApiState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, Path((coll, key)): Path<(String, String)>, headers: HeaderMap) -> AxumResponse {
    let ctx = ctx_or_return!(state, headers, addr);
    match state.shared.engine.handle(&ctx, Request::GetByKey { coll, key }).await {
        Response::Row(row) => (StatusCode::OK, Json(row)).into_response(),
        Response::Error { code, message } => error_response(&code, message),
        other => unexpected_response(other),
    }
}

#[utoipa::path(
    delete,
    path = "/v1/collections/{coll}/rows/{key}",
    params(("coll" = String, Path), ("key" = String, Path)),
    responses(
        (status = 204, description = "row deleted"),
        (status = 401, description = "missing/invalid bearer token", body = ErrorBody),
        (status = 403, description = "principal not permitted", body = ErrorBody),
        (status = 404, description = "no row with that key", body = ErrorBody),
    )
)]
pub async fn delete_row(State(state): State<ApiState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, Path((coll, key)): Path<(String, String)>, headers: HeaderMap) -> AxumResponse {
    let ctx = ctx_or_return!(state, headers, addr);
    match state.shared.engine.handle(&ctx, Request::Delete { coll, key }).await {
        Response::Ok => StatusCode::NO_CONTENT.into_response(),
        Response::Error { code, message } => error_response(&code, message),
        other => unexpected_response(other),
    }
}

/// Chunk-and-insert: splits `text` per `chunk_spec`, embeds every chunk
/// with the daemon's configured local model, and commits the whole
/// document as one transaction — the HTTP mirror of `mara-cli insert-document`.
#[utoipa::path(
    post,
    path = "/v1/collections/{coll}/documents",
    params(("coll" = String, Path)),
    request_body = PutDocumentRequest,
    responses(
        (status = 201, description = "document chunked, embedded, and committed", body = DocumentResponse),
        (status = 400, description = "bad request / invalid chunk_spec", body = ErrorBody),
        (status = 401, description = "missing/invalid bearer token", body = ErrorBody),
        (status = 403, description = "principal not permitted", body = ErrorBody),
        (status = 409, description = "doc_key already exists", body = ErrorBody),
        (status = 503, description = "no embedding model configured", body = ErrorBody),
    )
)]
pub async fn put_document(
    State(state): State<ApiState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Path(coll): Path<String>,
    headers: HeaderMap,
    Json(body): Json<PutDocumentRequest>,
) -> AxumResponse {
    let ctx = ctx_or_return!(state, headers, addr);
    let req = Request::PutDocument {
        coll,
        doc_key: body.doc_key,
        text: body.text,
        chunk_spec: body.chunk_spec,
        fields: body.fields,
        source: body.source,
    };
    match state.shared.engine.handle(&ctx, req).await {
        Response::Document { doc_id, doc_key, chunk_count, version } => {
            (StatusCode::CREATED, Json(DocumentResponse { doc_id, doc_key, chunk_count, version })).into_response()
        }
        Response::Error { code, message } => error_response(&code, message),
        other => unexpected_response(other),
    }
}

#[utoipa::path(
    post,
    path = "/v1/collections/{coll}/search",
    params(("coll" = String, Path)),
    request_body = SearchRequest,
    responses(
        (status = 200, description = "search hits", body = SearchResponse),
        (status = 400, description = "bad request", body = ErrorBody),
        (status = 401, description = "missing/invalid bearer token", body = ErrorBody),
        (status = 403, description = "principal not permitted", body = ErrorBody),
        (status = 503, description = "query_text given but no embedding model configured", body = ErrorBody),
    )
)]
pub async fn search(
    State(state): State<ApiState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Path(coll): Path<String>,
    headers: HeaderMap,
    Json(body): Json<SearchRequest>,
) -> AxumResponse {
    let ctx = ctx_or_return!(state, headers, addr);
    let req = Request::Search {
        coll,
        query_text: body.query_text,
        query_vector: body.query_vector,
        mode: body.mode,
        k: body.k,
        filter: body.filter,
        params: body.params,
    };
    match state.shared.engine.handle(&ctx, req).await {
        Response::SearchResults { hits, truncated_by_filter } => (StatusCode::OK, Json(SearchResponse { hits, truncated_by_filter })).into_response(),
        Response::Error { code, message } => error_response(&code, message),
        other => unexpected_response(other),
    }
}

#[utoipa::path(
    post,
    path = "/v1/collections/{coll}/reindex",
    params(("coll" = String, Path)),
    request_body = ReindexRequest,
    responses(
        (status = 200, description = "index (re)built", body = ReindexResponse),
        (status = 401, description = "missing/invalid bearer token", body = ErrorBody),
        (status = 403, description = "principal not permitted", body = ErrorBody),
    )
)]
pub async fn reindex(
    State(state): State<ApiState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Path(coll): Path<String>,
    headers: HeaderMap,
    Json(body): Json<ReindexRequest>,
) -> AxumResponse {
    let ctx = ctx_or_return!(state, headers, addr);
    match state.shared.engine.handle(&ctx, Request::Reindex { coll, index_kind: body.index_kind }).await {
        Response::Reindexed { coll, index_kind, row_count } => (StatusCode::OK, Json(ReindexResponse { coll, index_kind, row_count })).into_response(),
        Response::Error { code, message } => error_response(&code, message),
        other => unexpected_response(other),
    }
}

#[derive(serde::Deserialize, utoipa::IntoParams)]
pub struct AuditQuery {
    /// Most-recent-first; defaults to 50, capped at 1000 per request.
    limit: Option<usize>,
}

/// Reads back recent audit entries — Admin-only (`Capability::AuditRead`),
/// checked directly here rather than through `Engine::handle`: there is no
/// `Request::Audit` wire variant, since a read-only query over a JSONL log
/// on disk has no undo/WAL/replication story to share with the rest of
/// `Request`.
#[utoipa::path(
    get,
    path = "/v1/audit",
    params(AuditQuery),
    responses(
        (status = 200, description = "recent audit entries, newest first", body = AuditResponse),
        (status = 401, description = "missing/invalid bearer token", body = ErrorBody),
        (status = 403, description = "principal not permitted (admin only)", body = ErrorBody),
    )
)]
pub async fn audit(State(state): State<ApiState>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Query(q): Query<AuditQuery>) -> AxumResponse {
    let ctx = ctx_or_return!(state, headers, addr);
    if !mara_auth::require(ctx.principal.role, mara_auth::Capability::AuditRead) {
        return error_response("forbidden", "principal's role is not permitted to read the audit log".into());
    }
    let limit = q.limit.unwrap_or(50).min(1000);
    match mara_auth::read_recent(&state.audit_dir, limit) {
        Ok(records) => (StatusCode::OK, Json(AuditResponse { records: records.into_iter().map(AuditRecordDto::from).collect() })).into_response(),
        Err(e) => error_response("storage_error", e.to_string()),
    }
}
