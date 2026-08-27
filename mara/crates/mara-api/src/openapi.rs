use utoipa::OpenApi;

#[derive(OpenApi)]
#[openapi(
    info(title = "mara", description = "HTTP API over the mara vector database — a REST mirror of the native UDS/TCP protocol, resolving `Authorization: Bearer <token>` to the same `RequestCtx` and dispatching through the same `Engine`."),
    paths(
        crate::handlers::health,
        crate::handlers::create_collection,
        crate::handlers::put_row,
        crate::handlers::put_batch,
        crate::handlers::get_row,
        crate::handlers::delete_row,
        crate::handlers::put_document,
        crate::handlers::search,
        crate::handlers::reindex,
        crate::handlers::audit,
    ),
    components(schemas(
        crate::dto::SchemaFieldDto,
        crate::dto::CreateCollectionRequest,
        crate::dto::CreateCollectionResponse,
        crate::dto::PutRowRequest,
        crate::dto::PutBatchRequest,
        crate::dto::PutBatchResponse,
        crate::dto::PutDocumentRequest,
        crate::dto::DocumentResponse,
        crate::dto::SearchRequest,
        crate::dto::SearchResponse,
        crate::dto::ReindexRequest,
        crate::dto::ReindexResponse,
        crate::dto::ErrorBody,
        crate::dto::HealthResponse,
        crate::dto::AuditRecordDto,
        crate::dto::AuditResponse,
        mara_proto::Row,
        mara_proto::ScoredHit,
        mara_proto::PutItem,
        mara_proto::Filter,
        mara_proto::Scalar,
        mara_proto::PayloadValue,
        mara_proto::ChunkSpec,
        mara_proto::ChunkStrategy,
        mara_proto::DistanceMetric,
        mara_proto::SearchMode,
        mara_proto::WireFusionMethod,
        mara_proto::WireIndexKind,
        mara_proto::WireSearchParams,
        mara_proto::WireFieldType,
        mara_proto::RowId,
        mara_proto::DocId,
    ))
)]
pub struct ApiDoc;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spec_generates_and_serializes_to_valid_json_with_every_path_present() {
        let spec = ApiDoc::openapi();
        let json = spec.to_pretty_json().expect("ApiDoc must always serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("must be valid JSON");
        let paths = value["paths"].as_object().expect("must have a paths object");
        for expected in [
            "/v1/health",
            "/v1/audit",
            "/v1/collections",
            "/v1/collections/{coll}/rows/batch",
            "/v1/collections/{coll}/rows/{key}",
            "/v1/collections/{coll}/documents",
            "/v1/collections/{coll}/search",
            "/v1/collections/{coll}/reindex",
        ] {
            assert!(paths.contains_key(expected), "missing path {expected:?} in generated OpenAPI spec");
        }
        assert!(!value["components"]["schemas"].as_object().unwrap().is_empty());
    }
}
