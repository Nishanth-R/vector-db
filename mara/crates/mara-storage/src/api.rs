use crate::change::ChangeSubscriber;
use crate::collection::{Collection, PutInput};
use crate::document::{DocEntry, PutDocumentInput};
use crate::error::{StorageError, StorageResult};
use crate::payload::{FieldType, FilterMask, PayloadSchema};
use crate::wal::FsyncPolicy;
use mara_proto::{DistanceMetric, DocId, ExtraPayload, Filter, PayloadRow, RequestCtx, Row, RowId, TxnEntry, TxnId, TxnSummary};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
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

/// A collection's fixed-at-creation vector shape — what a caller needs to
/// build a `VectorIndex` (`mara-index-vector`) against it without reaching
/// past `StorageApi` into the concrete `Storage`/`Collection` types.
#[derive(Clone, Copy, Debug)]
pub struct CollectionInfo {
    pub dim: usize,
    pub metric: DistanceMetric,
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
    fn collection_info(&self, coll: &str) -> StorageResult<CollectionInfo>;

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
    fn put_document(&self, ctx: &RequestCtx, coll: &str, input: PutDocumentInput) -> StorageResult<DocId>;
    fn get_document(&self, coll: &str, doc_key: &str) -> StorageResult<Option<DocEntry>>;
    fn delete(&self, ctx: &RequestCtx, coll: &str, key: &str) -> StorageResult<()>;
    fn get_by_key(&self, coll: &str, key: &str) -> StorageResult<Option<Row>>;
    fn scan(&self, coll: &str, after: Option<RowId>, limit: usize) -> StorageResult<Vec<Row>>;
    fn fetch_vectors(&self, coll: &str, ids: &[RowId]) -> StorageResult<Vec<Option<Vec<f32>>>>;
    fn rows_by_id(&self, coll: &str, ids: &[RowId]) -> StorageResult<Vec<Option<Row>>>;
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

/// One collection's identity, as recorded in the durable registry file —
/// just enough to reopen it via `Collection::open` without re-running
/// `CreateCollection`.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct RegistryEntry {
    name: String,
    dim: usize,
    metric: DistanceMetric,
    schema: Vec<(String, FieldType)>,
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct RegistryFile {
    #[serde(default)]
    collections: Vec<RegistryEntry>,
}

fn registry_path(data_dir: &Path) -> PathBuf {
    data_dir.join("collections.json")
}

fn load_registry_file(data_dir: &Path) -> HashMap<String, RegistryEntry> {
    let Ok(text) = std::fs::read_to_string(registry_path(data_dir)) else {
        return HashMap::new();
    };
    let Ok(file) = serde_json::from_str::<RegistryFile>(&text) else {
        return HashMap::new();
    };
    file.collections.into_iter().map(|e| (e.name.clone(), e)).collect()
}

fn save_registry_file(data_dir: &Path, entries: &HashMap<String, RegistryEntry>) -> std::io::Result<()> {
    let mut collections: Vec<RegistryEntry> = entries.values().cloned().collect();
    collections.sort_by(|a, b| a.name.cmp(&b.name));
    let json = serde_json::to_string_pretty(&RegistryFile { collections }).expect("RegistryFile always serializes");
    std::fs::write(registry_path(data_dir), json)
}

/// Where a `Storage` registry persists the collections it creates.
/// `InMemory` never touches disk (tests, one-shot uses that don't need
/// durability); `Durable` opens each collection via `Collection::open`
/// under `data_dir/collections/<name>/`, giving it a real WAL and
/// checkpointing, and records its identity in `data_dir/collections.json`
/// so a *different* process (a fresh daemon, or each separate `--embedded`
/// invocation) can look it up by name without needing `create_collection`
/// called again first — see `Storage::collection`'s lazy-open path.
enum StorageMode {
    InMemory,
    Durable {
        data_dir: PathBuf,
        segment_size_bytes: u32,
        fsync: FsyncPolicy,
    },
}

/// The default `StorageApi` implementation: a registry of named
/// `Collection`s, either purely in-memory or WAL-backed per `StorageMode`.
pub struct Storage {
    collections: RwLock<HashMap<String, Arc<Collection>>>,
    mode: StorageMode,
    /// Durable mode only: which collections exist, with what
    /// dim/metric/schema, loaded from `data_dir/collections.json` at
    /// `Storage::open` time and kept in sync with it on every
    /// `create_collection`.
    registry: RwLock<HashMap<String, RegistryEntry>>,
}

impl Default for Storage {
    fn default() -> Self {
        Self::new()
    }
}

impl Storage {
    pub fn new() -> Self {
        Storage {
            collections: RwLock::new(HashMap::new()),
            mode: StorageMode::InMemory,
            registry: RwLock::new(HashMap::new()),
        }
    }

    /// A `Storage` whose collections are durable: each `create_collection`
    /// call opens (or resumes) `data_dir/collections/<name>/` via
    /// `Collection::open`, and every collection this — or any prior —
    /// process created against `data_dir` is transparently rediscoverable
    /// by name.
    pub fn open(data_dir: impl Into<PathBuf>, segment_size_bytes: u32, fsync: FsyncPolicy) -> Self {
        let data_dir = data_dir.into();
        let registry = load_registry_file(&data_dir);
        Storage {
            collections: RwLock::new(HashMap::new()),
            mode: StorageMode::Durable {
                data_dir,
                segment_size_bytes,
                fsync,
            },
            registry: RwLock::new(registry),
        }
    }

    pub fn collection(&self, name: &str) -> StorageResult<Arc<Collection>> {
        if let Some(c) = self.collections.read().get(name).cloned() {
            return Ok(c);
        }
        // Lazy-open: the collection isn't open in *this* Storage instance
        // yet, but the durable registry remembers it existing (created by
        // this process earlier, or an entirely different one).
        if let StorageMode::Durable { data_dir, segment_size_bytes, fsync } = &self.mode {
            let entry = self.registry.read().get(name).cloned();
            if let Some(entry) = entry {
                let mut guard = self.collections.write();
                // Double-checked: another thread may have opened it while
                // we didn't hold the write lock.
                if let Some(c) = guard.get(name).cloned() {
                    return Ok(c);
                }
                let mut builder = PayloadSchema::builder();
                for (field_name, ty) in entry.schema {
                    builder = builder.field(field_name, ty);
                }
                let collection = Collection::open(
                    name,
                    entry.dim,
                    entry.metric,
                    builder.build(),
                    true,
                    data_dir.join("collections").join(name),
                    *segment_size_bytes,
                    *fsync,
                )
                .map_err(|e| StorageError::Wal(e.to_string()))?;
                let arc = Arc::new(collection);
                guard.insert(name.to_string(), arc.clone());
                return Ok(arc);
            }
        }
        Err(StorageError::CollectionNotFound(name.to_string()))
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
        if guard.contains_key(name) || self.registry.read().contains_key(name) {
            return Err(StorageError::CollectionAlreadyExists(name.to_string()));
        }
        let schema_for_registry: Vec<(String, FieldType)> = schema.iter().cloned().collect();
        // A schema declared explicitly at creation is enforced: an
        // undeclared field on a write is a clear error rather than a
        // silently-unindexed one.
        let collection = match &self.mode {
            StorageMode::InMemory => Collection::with_schema(name, dim, metric, schema, true),
            StorageMode::Durable {
                data_dir,
                segment_size_bytes,
                fsync,
            } => Collection::open(name, dim, metric, schema, true, data_dir.join("collections").join(name), *segment_size_bytes, *fsync)
                .map_err(|e| StorageError::Wal(e.to_string()))?,
        };
        guard.insert(name.to_string(), Arc::new(collection));
        drop(guard);

        if let StorageMode::Durable { data_dir, .. } = &self.mode {
            let mut reg = self.registry.write();
            reg.insert(
                name.to_string(),
                RegistryEntry {
                    name: name.to_string(),
                    dim,
                    metric,
                    schema: schema_for_registry,
                },
            );
            // Best-effort: a failed write here only affects whether a
            // *future* process can rediscover this collection — this one
            // already has it open and fully working.
            let _ = save_registry_file(data_dir, &reg);
        }
        Ok(())
    }

    fn list_collections(&self) -> Vec<String> {
        let mut names: std::collections::BTreeSet<String> = self.collections.read().keys().cloned().collect();
        names.extend(self.registry.read().keys().cloned());
        names.into_iter().collect()
    }

    fn alter_schema(&self, ctx: &RequestCtx, coll: &str, change: SchemaChange) -> StorageResult<()> {
        self.collection(coll)?.alter_schema(ctx, change)
    }

    fn collection_info(&self, coll: &str) -> StorageResult<CollectionInfo> {
        let c = self.collection(coll)?;
        Ok(CollectionInfo { dim: c.dim, metric: c.metric })
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

    fn put_document(&self, ctx: &RequestCtx, coll: &str, input: PutDocumentInput) -> StorageResult<DocId> {
        self.collection(coll)?.put_document(ctx, input)
    }

    fn get_document(&self, coll: &str, doc_key: &str) -> StorageResult<Option<DocEntry>> {
        Ok(self.collection(coll)?.get_document(doc_key))
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

    fn rows_by_id(&self, coll: &str, ids: &[RowId]) -> StorageResult<Vec<Option<Row>>> {
        Ok(self.collection(coll)?.rows_by_id(ids))
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

    #[test]
    fn a_fresh_storage_instance_rediscovers_a_collection_created_by_a_prior_one() {
        // The scenario that bit `--embedded` mode in practice: each
        // invocation is a brand-new `Storage::open`, so without the
        // durable registry, `create_collection` in one process would be
        // invisible to the next.
        let dir = tempfile::tempdir().unwrap();
        {
            let s1 = Storage::open(dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always);
            let schema = PayloadSchema::builder().field("title", FieldType::Keyword).build();
            s1.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, schema).unwrap();
            let mut fields = PayloadRow::new();
            fields.insert("title".into(), mara_proto::PayloadValue::Keyword("hello".into()));
            s1.put(&ctx(), "docs", "a", vec![1.0, 2.0, 3.0], fields, None).unwrap();
        }

        let s2 = Storage::open(dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always);
        assert_eq!(s2.list_collections(), vec!["docs".to_string()], "list_collections must include registry-only entries");
        let row = s2.get_by_key("docs", "a").unwrap().expect("the row must survive across Storage instances");
        assert_eq!(row.vector, Some(vec![1.0, 2.0, 3.0]));

        // The schema (and its strictness) must also carry over: an
        // undeclared field is still rejected after rediscovery.
        let mut bad_fields = PayloadRow::new();
        bad_fields.insert("ghost".into(), mara_proto::PayloadValue::Bool(true));
        assert!(s2.put(&ctx(), "docs", "b", vec![0.0, 0.0, 0.0], bad_fields, None).is_err());

        // create_collection must also see registry-only entries as taken,
        // not just currently-open ones.
        assert_eq!(
            s2.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty())
                .unwrap_err(),
            StorageError::CollectionAlreadyExists("docs".into())
        );
    }
}
