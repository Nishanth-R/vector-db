use crate::change::ChangeSubscriber;
use crate::collection::{Collection, PutInput};
use crate::error::{StorageError, StorageResult};
use crate::payload::{FieldType, FilterMask, PayloadSchema};
use mara_proto::{DistanceMetric, ExtraPayload, Filter, PayloadRow, RequestCtx, Row, RowId, TxnEntry, TxnId, TxnSummary};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// A single change to a collection's payload schema (`alter_schema`).
/// Deliberately minimal, per the master plan: adding a field (backfilled
/// absent on existing rows) or dropping one. Changing a field's *type*
/// requires a collection rebuild, not supported here.
#[derive(Clone, Debug)]
pub enum SchemaChange {
    AddField { name: String, field_type: FieldType },
    DropField { name: String },
}

/// The storage engine's public surface. Every mutating method takes
/// `&RequestCtx` — threaded from the start (see the master plan's Layer 1)
/// rather than retrofitted later.
///
/// This is `mara-storage`'s vocabulary as of the document + payload/filter
/// build (steps 0-5). History (`undo`/`revert_to`) and `checkpoint` are
/// added to this trait as the layers that back them land — `StorageApi` is
/// meant to grow with the crate, not be fully declared up front against
/// unbuilt functionality.
pub trait StorageApi: Send + Sync {
    fn create_collection(
        &self,
        ctx: &RequestCtx,
        name: &str,
        dim: usize,
        metric: DistanceMetric,
        schema: PayloadSchema,
    ) -> StorageResult<()>;
    fn list_collections(&self) -> Vec<String>;
    fn alter_schema(&self, ctx: &RequestCtx, coll: &str, change: SchemaChange) -> StorageResult<()>;

    fn put(
        &self,
        ctx: &RequestCtx,
        coll: &str,
        key: &str,
        vector: Vec<f32>,
        fields: PayloadRow,
        extra: Option<ExtraPayload>,
    ) -> StorageResult<RowId>;
    fn put_batch(&self, ctx: &RequestCtx, coll: &str, items: Vec<PutInput>) -> StorageResult<Vec<RowId>>;
    fn delete(&self, ctx: &RequestCtx, coll: &str, key: &str) -> StorageResult<()>;
    fn get_by_key(&self, coll: &str, key: &str) -> StorageResult<Option<Row>>;
    fn scan(&self, coll: &str, after: Option<RowId>, limit: usize) -> StorageResult<Vec<Row>>;
    fn fetch_vectors(&self, coll: &str, ids: &[RowId]) -> StorageResult<Vec<Option<Vec<f32>>>>;
    fn active_count(&self, coll: &str) -> StorageResult<u64>;

    fn compile_filter(&self, coll: &str, filter: &Filter) -> StorageResult<FilterMask>;
    fn payload_of(&self, coll: &str, row_id: RowId) -> StorageResult<Option<PayloadRow>>;
    fn payload_batch(&self, coll: &str, ids: &[RowId]) -> StorageResult<Vec<Option<PayloadRow>>>;

    /// Reverses transaction `target_txn` in `coll` — see
    /// `Collection::undo`. Requires a WAL-backed (durable) collection.
    fn undo(&self, ctx: &RequestCtx, coll: &str, target_txn: TxnId, force: bool) -> StorageResult<TxnSummary>;
    /// Recent transactions in `coll`, most-recent first.
    fn history(&self, coll: &str, limit: usize) -> StorageResult<Vec<TxnEntry>>;

    fn subscribe(&self, coll: &str, subscriber: Arc<dyn ChangeSubscriber>) -> StorageResult<()>;
}

/// The default in-memory `StorageApi` implementation: a registry of named
/// `Collection`s. WAL-backed durability and the history/undo layer attach
/// to this same struct in later steps.
#[derive(Default)]
pub struct Storage {
    collections: RwLock<HashMap<String, Arc<Collection>>>,
}

impl Storage {
    pub fn new() -> Self {
        Storage {
            collections: RwLock::new(HashMap::new()),
        }
    }

    pub fn collection(&self, name: &str) -> StorageResult<Arc<Collection>> {
        self.collections
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| StorageError::CollectionNotFound(name.to_string()))
    }
}

impl StorageApi for Storage {
    fn create_collection(
        &self,
        _ctx: &RequestCtx,
        name: &str,
        dim: usize,
        metric: DistanceMetric,
        schema: PayloadSchema,
    ) -> StorageResult<()> {
        let mut guard = self.collections.write();
        if guard.contains_key(name) {
            return Err(StorageError::CollectionAlreadyExists(name.to_string()));
        }
        // A schema declared explicitly at creation is enforced: an
        // undeclared field on a write is a clear error rather than a
        // silently-unindexed one.
        guard.insert(name.to_string(), Arc::new(Collection::with_schema(name, dim, metric, schema, true)));
        Ok(())
    }

    fn list_collections(&self) -> Vec<String> {
        let mut names: Vec<String> = self.collections.read().keys().cloned().collect();
        names.sort();
        names
    }

    fn alter_schema(&self, ctx: &RequestCtx, coll: &str, change: SchemaChange) -> StorageResult<()> {
        self.collection(coll)?.alter_schema(ctx, change)
    }

    fn put(
        &self,
        ctx: &RequestCtx,
        coll: &str,
        key: &str,
        vector: Vec<f32>,
        fields: PayloadRow,
        extra: Option<ExtraPayload>,
    ) -> StorageResult<RowId> {
        self.collection(coll)?.put(ctx, key, vector, fields, extra)
    }

    fn put_batch(&self, ctx: &RequestCtx, coll: &str, items: Vec<PutInput>) -> StorageResult<Vec<RowId>> {
        self.collection(coll)?.put_batch(ctx, items)
    }

    fn delete(&self, ctx: &RequestCtx, coll: &str, key: &str) -> StorageResult<()> {
        self.collection(coll)?.delete(ctx, key)
    }

    fn get_by_key(&self, coll: &str, key: &str) -> StorageResult<Option<Row>> {
        Ok(self.collection(coll)?.get_by_key(key))
    }

    fn scan(&self, coll: &str, after: Option<RowId>, limit: usize) -> StorageResult<Vec<Row>> {
        Ok(self.collection(coll)?.scan(after, limit))
    }

    fn fetch_vectors(&self, coll: &str, ids: &[RowId]) -> StorageResult<Vec<Option<Vec<f32>>>> {
        Ok(self.collection(coll)?.fetch_vectors(ids))
    }

    fn active_count(&self, coll: &str) -> StorageResult<u64> {
        Ok(self.collection(coll)?.active_count())
    }

    fn compile_filter(&self, coll: &str, filter: &Filter) -> StorageResult<FilterMask> {
        self.collection(coll)?.compile_filter(filter)
    }

    fn payload_of(&self, coll: &str, row_id: RowId) -> StorageResult<Option<PayloadRow>> {
        Ok(self.collection(coll)?.payload_of(row_id))
    }

    fn payload_batch(&self, coll: &str, ids: &[RowId]) -> StorageResult<Vec<Option<PayloadRow>>> {
        Ok(self.collection(coll)?.payload_batch(ids))
    }

    fn undo(&self, ctx: &RequestCtx, coll: &str, target_txn: TxnId, force: bool) -> StorageResult<TxnSummary> {
        self.collection(coll)?.undo(ctx, target_txn, force)
    }

    fn history(&self, coll: &str, limit: usize) -> StorageResult<Vec<TxnEntry>> {
        Ok(self.collection(coll)?.history(limit))
    }

    fn subscribe(&self, coll: &str, subscriber: Arc<dyn ChangeSubscriber>) -> StorageResult<()> {
        self.collection(coll)?.subscribe(subscriber);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mara_proto::{Principal, Role, SessionId, Source};

    fn ctx() -> RequestCtx {
        RequestCtx::new(
            SessionId("s".into()),
            Principal {
                id: mara_proto::PrincipalId("p".into()),
                name: "t".into(),
                role: Role::Admin,
            },
            Source::Embedded,
        )
    }

    #[test]
    fn create_collection_twice_is_an_error() {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty()).unwrap();
        assert_eq!(
            s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty())
                .unwrap_err(),
            StorageError::CollectionAlreadyExists("docs".into())
        );
    }

    #[test]
    fn operating_on_an_unknown_collection_errors_cleanly() {
        let s = Storage::new();
        assert_eq!(
            s.put(&ctx(), "nope", "k", vec![1.0], PayloadRow::new(), None).unwrap_err(),
            StorageError::CollectionNotFound("nope".into())
        );
    }

    #[test]
    fn list_collections_is_sorted() {
        let s = Storage::new();
        s.create_collection(&ctx(), "zeta", 2, DistanceMetric::L2, PayloadSchema::empty()).unwrap();
        s.create_collection(&ctx(), "alpha", 2, DistanceMetric::L2, PayloadSchema::empty()).unwrap();
        assert_eq!(s.list_collections(), vec!["alpha".to_string(), "zeta".to_string()]);
    }

    #[test]
    fn put_and_get_through_the_storage_api() {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty()).unwrap();
        let id = s
            .put(&ctx(), "docs", "a", vec![1.0, 2.0, 3.0], PayloadRow::new(), None)
            .unwrap();
        let row = s.get_by_key("docs", "a").unwrap().unwrap();
        assert_eq!(row.id, id);
        assert_eq!(s.active_count("docs").unwrap(), 1);
    }

    #[test]
    fn schema_declared_at_creation_is_enforced_strictly() {
        let s = Storage::new();
        let schema = PayloadSchema::builder().field("title", FieldType::Keyword).build();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, schema).unwrap();

        let mut fields = PayloadRow::new();
        fields.insert("ghost".into(), mara_proto::PayloadValue::Bool(true));
        let err = s.put(&ctx(), "docs", "a", vec![1.0, 2.0, 3.0], fields, None).unwrap_err();
        assert!(matches!(err, StorageError::Filter(_)));
    }

    #[test]
    fn compile_filter_and_payload_of_round_trip_through_storage_api() {
        let s = Storage::new();
        let schema = PayloadSchema::builder().field("title", FieldType::Keyword).build();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, schema).unwrap();

        let mut fields = PayloadRow::new();
        fields.insert("title".into(), mara_proto::PayloadValue::Keyword("hello".into()));
        let id = s.put(&ctx(), "docs", "a", vec![1.0, 2.0, 3.0], fields.clone(), None).unwrap();

        let mask = s
            .compile_filter(
                "docs",
                &Filter::Eq {
                    field: "title".into(),
                    value: mara_proto::Scalar::Str("hello".into()),
                },
            )
            .unwrap();
        assert!(mask.contains(id));

        assert_eq!(s.payload_of("docs", id).unwrap(), Some(fields));
    }

    #[test]
    fn alter_schema_add_then_drop_field_through_storage_api() {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty()).unwrap();
        s.alter_schema(&ctx(), "docs", SchemaChange::AddField { name: "size".into(), field_type: FieldType::I64 })
            .unwrap();

        let mut fields = PayloadRow::new();
        fields.insert("size".into(), mara_proto::PayloadValue::I64(7));
        s.put(&ctx(), "docs", "a", vec![1.0, 2.0, 3.0], fields, None).unwrap();

        s.alter_schema(&ctx(), "docs", SchemaChange::DropField { name: "size".into() }).unwrap();
        let err = s
            .compile_filter("docs", &Filter::Exists { field: "size".into() })
            .unwrap_err();
        assert!(matches!(err, StorageError::Filter(_)));
    }
}
