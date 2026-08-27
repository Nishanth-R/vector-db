//! REST-shaped request/response bodies for `mara-api`'s handlers.
//!
//! A thin adapter layer over `mara-proto`'s wire `Request`/`Response`
//! enums, not a second, parallel type system: every field here is either
//! a `mara-proto` type reused directly (`PutItem`, `Filter`, `ChunkSpec`,
//! `Row`, `ScoredHit`, …, all already `ToSchema`) or a small wrapper
//! struct that exists only because a URL-keyed REST resource shape
//! (`PUT /v1/collections/{coll}/rows/{key}`) doesn't map 1:1 onto the wire
//! protocol's single flat `Request` enum (which carries `key` as a body
//! field, since a persistent socket connection has no URL to put it in).

use mara_proto::{ChunkSpec, DistanceMetric, ExtraPayload, Filter, PayloadRow, PutItem, Row, ScoredHit, SearchMode, WireFieldType, WireIndexKind, WireSearchParams};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

fn default_metric() -> DistanceMetric {
    DistanceMetric::Cosine
}

fn default_search_mode() -> SearchMode {
    // Mirrors the master plan's stated intent ("`search` defaults to
    // hybrid") for the one entrypoint where a client omitting `mode`
    // entirely is common — a human hitting the HTTP API by hand.
    SearchMode::Hybrid { method: None, overfetch_k: None }
}

fn default_k() -> usize {
    10
}

#[derive(Clone, Serialize, Deserialize, ToSchema)]
pub struct SchemaFieldDto {
    pub name: String,
    pub field_type: WireFieldType,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dim: usize,
    #[serde(default = "default_metric")]
    pub metric: DistanceMetric,
    #[serde(default)]
    pub schema: Vec<SchemaFieldDto>,
}

#[derive(Serialize, ToSchema)]
pub struct CreateCollectionResponse {
    pub name: String,
    pub dim: usize,
    pub metric: DistanceMetric,
}

/// Body for `PUT /v1/collections/{coll}/rows/{key}` — `key` itself comes
/// from the URL, not this struct.
#[derive(Serialize, Deserialize, ToSchema)]
pub struct PutRowRequest {
    pub text: Option<String>,
    pub vector: Option<Vec<f32>>,
    #[serde(default)]
    pub fields: PayloadRow,
    #[schema(value_type = Object)]
    pub extra: Option<ExtraPayload>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct PutBatchRequest {
    pub items: Vec<PutItem>,
}

#[derive(Serialize, ToSchema)]
pub struct PutBatchResponse {
    pub rows: Vec<Row>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct PutDocumentRequest {
    pub doc_key: String,
    pub text: String,
    #[serde(default)]
    pub chunk_spec: ChunkSpec,
    #[serde(default)]
    pub fields: PayloadRow,
    pub source: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub struct DocumentResponse {
    pub doc_id: mara_proto::DocId,
    pub doc_key: String,
    pub chunk_count: u32,
    pub version: u32,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct SearchRequest {
    pub query_text: Option<String>,
    pub query_vector: Option<Vec<f32>>,
    #[serde(default = "default_search_mode")]
    pub mode: SearchMode,
    #[serde(default = "default_k")]
    pub k: usize,
    pub filter: Option<Filter>,
    #[serde(default)]
    pub params: WireSearchParams,
}

#[derive(Serialize, ToSchema)]
pub struct SearchResponse {
    pub hits: Vec<ScoredHit>,
    pub truncated_by_filter: bool,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ReindexRequest {
    pub index_kind: WireIndexKind,
}

#[derive(Serialize, ToSchema)]
pub struct ReindexResponse {
    pub coll: String,
    pub index_kind: WireIndexKind,
    pub row_count: u64,
}

#[derive(Serialize, ToSchema)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
}

#[derive(Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: &'static str,
}

/// One audit entry as shown over HTTP — `mara-auth::AuditRecord` mirrored
/// field-for-field, but with `ts` as RFC 3339 text rather than
/// `chrono::DateTime` directly, so `mara-auth` never needs a `utoipa`
/// dependency just to satisfy this one read-only endpoint's schema.
#[derive(Serialize, ToSchema)]
pub struct AuditRecordDto {
    pub ts: String,
    pub request_id: String,
    pub session: String,
    pub principal_id: String,
    pub principal_name: String,
    pub role: String,
    pub action: String,
    pub outcome: String,
    pub latency_ms: u64,
}

impl From<mara_auth::AuditRecord> for AuditRecordDto {
    fn from(r: mara_auth::AuditRecord) -> Self {
        let outcome = match &r.result {
            mara_auth::AuditOutcome::Ok => "ok".to_string(),
            mara_auth::AuditOutcome::Error { message } => format!("error: {message}"),
        };
        AuditRecordDto {
            ts: r.ts.to_rfc3339(),
            request_id: r.request_id.to_string(),
            session: r.session.0,
            principal_id: r.principal.id.0,
            principal_name: r.principal.name,
            role: format!("{:?}", r.principal.role),
            action: r.action,
            outcome,
            latency_ms: r.latency_ms,
        }
    }
}

#[derive(Serialize, ToSchema)]
pub struct AuditResponse {
    pub records: Vec<AuditRecordDto>,
}
