//! The `Engine` trait and its one implementation (master plan Layer 3,
//! *Composition*): `mara-api`'s axum handlers and the native socket loop
//! both call the same `Arc<dyn Engine>` in-process — one dispatcher, one
//! authorization table, one audit path.

use mara_auth::{require, AuditOutcome, AuditRecord, AuditSink, Capability};
use mara_chunker::chunk_text;
use mara_embed::EmbeddingBackend;
use mara_index_bm25::{Bm25Index, Bm25Params};
use mara_index_vector::{FlatIndex, IvfParams, IvfPqIndex, LiveIndex, LiveIndexParams, PqParams, SearchParams, VectorIndex};
use mara_proto::{DistanceMetric, ModelFingerprint, Request, RequestCtx, Response, Row, RowId, ScoredHit, SearchMode, WireFieldType, WireIndexKind, WireSearchParams};
use mara_storage::document::{ChunkInput, PutDocumentInput};
use mara_storage::{ChangeBatch, ChangeEvent, ChangeSubscriber, FieldType, FilterMask, PayloadSchema, StorageApi, StorageError, StorageResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

#[async_trait::async_trait]
pub trait Engine: Send + Sync {
    async fn handle(&self, ctx: &RequestCtx, req: Request) -> Response;
}

/// Feeds `Collection::subscribe`'s `ChangeBatch`es into a live
/// `Bm25Index` — the "same `ChangeBatch`-driven live-update pattern" the
/// master plan's BM25 section describes. A row with no `text` (a plain,
/// non-document row) is simply not BM25-relevant and skipped; `Update` is
/// a documented no-op — `mara-storage`'s document lifecycle never emits
/// one for a chunk row (`put_document`/`replace_document`/
/// `delete_document` are Insert/Delete only, confirmed by inspection), so
/// there's currently no case where this needs the old text an `Update`
/// event doesn't carry.
struct Bm25Subscriber {
    index: Arc<RwLock<Bm25Index>>,
}

impl ChangeSubscriber for Bm25Subscriber {
    fn on_change(&self, batch: &ChangeBatch) {
        let mut idx = self.index.write();
        for event in &batch.events {
            match event {
                ChangeEvent::Insert { row_id, text: Some(t), .. } => idx.insert(*row_id, t),
                ChangeEvent::Delete { row_id, text: Some(t), .. } => idx.remove(*row_id, t),
                ChangeEvent::Update { .. } => {
                    tracing::warn!("Bm25Subscriber received an Update event, which it doesn't yet handle (no chunk-update code path exists today)");
                }
                _ => {}
            }
        }
    }
}

pub struct EngineImpl {
    storage: Arc<dyn StorageApi>,
    audit: Arc<dyn AuditSink>,
    /// `None` when `[embedding] model` is unset — see the master plan's
    /// *Embedding model selection*: local embedding is opt-in, not a
    /// silent first-request model download.
    embedder: Option<Arc<dyn EmbeddingBackend>>,
    /// `[embedding] batch_size` — how many texts go into one
    /// `EmbeddingBackend::embed_batch` call. Irrelevant (but harmless)
    /// when `embedder` is `None`.
    embed_batch_size: usize,
    /// Per-collection cached vector index, populated by `Reindex`. Absent
    /// means "no index built yet" — `Search{mode: VectorOnly}` falls back
    /// to a fresh `FlatIndex` on the fly rather than erroring, since Flat
    /// needs no build step and is the master plan's stated default at
    /// this project's scale. `Flat`-kind entries are inserted directly
    /// (already always-live, so wrapping them in `LiveIndex` buys
    /// nothing); `IvfPq`-kind entries are always `LiveIndex`-wrapped —
    /// see `live_indexes` below for what that buys.
    vector_indexes: RwLock<HashMap<String, Arc<dyn VectorIndex>>>,
    /// The same `LiveIndex` instances also present in `vector_indexes`
    /// (as `Arc<dyn VectorIndex>`, for searching), kept here too in their
    /// concrete type so `handle` can check `should_rebuild`/call `rebuild`
    /// on them after a write — `Arc<dyn VectorIndex>` alone can't be
    /// downcast back to call `LiveIndex`-specific methods.
    live_indexes: RwLock<HashMap<String, Arc<LiveIndex>>>,
    /// Per-collection live BM25 index, attached automatically by
    /// `CreateCollection` and (re-)attached by `Reindex{index_kind: Bm25}`
    /// (for a collection whose creation this daemon process never saw —
    /// rediscovered from the on-disk registry instead). Unlike
    /// `vector_indexes`, this stays current on its own via `Bm25Subscriber`.
    bm25_indexes: RwLock<HashMap<String, Arc<RwLock<Bm25Index>>>>,
    /// `true` only for an `EngineImpl` running behind a replication
    /// follower's own daemon — see [`EngineImpl::as_follower`]. Gates row
    /// and collection lifecycle mutations (`Put`/`PutBatch`/`PutDocument`/
    /// `Delete`/`CreateCollection`) with a clear "not the leader" error;
    /// `Reindex` is deliberately exempt — it only ever rebuilds *derived*
    /// index state from whatever rows this follower already has, which a
    /// follower is free to do independently, the same way it's never part
    /// of storage's own durable, replicated state on the leader either.
    is_follower: bool,
}

impl EngineImpl {
    pub fn new(storage: Arc<dyn StorageApi>, audit: Arc<dyn AuditSink>) -> Self {
        EngineImpl {
            storage,
            audit,
            embedder: None,
            embed_batch_size: 32,
            vector_indexes: RwLock::new(HashMap::new()),
            live_indexes: RwLock::new(HashMap::new()),
            bm25_indexes: RwLock::new(HashMap::new()),
            is_follower: false,
        }
    }

    pub fn with_embedder(
        storage: Arc<dyn StorageApi>,
        audit: Arc<dyn AuditSink>,
        embedder: Option<Arc<dyn EmbeddingBackend>>,
        embed_batch_size: usize,
    ) -> Self {
        EngineImpl {
            storage,
            audit,
            embedder,
            embed_batch_size,
            vector_indexes: RwLock::new(HashMap::new()),
            live_indexes: RwLock::new(HashMap::new()),
            bm25_indexes: RwLock::new(HashMap::new()),
            is_follower: false,
        }
    }

    /// Opts this engine into follower behavior — see `is_follower`. Called
    /// once at boot, from `[replication] role = "follower"`; never
    /// toggled at runtime (a role change is a restart, not a live
    /// transition, at this build's scope).
    pub fn as_follower(mut self) -> Self {
        self.is_follower = true;
        self
    }

    fn vector_index_for(&self, coll: &str) -> StorageResult<Arc<dyn VectorIndex>> {
        if let Some(idx) = self.vector_indexes.read().get(coll).cloned() {
            return Ok(idx);
        }
        let info = self.storage.collection_info(coll)?;
        Ok(Arc::new(FlatIndex::new(info.dim, info.metric)))
    }

    /// Checks `coll`'s attached `LiveIndex` (if any) and, if it's due,
    /// rebuilds it off the request path via `spawn_blocking` — matching
    /// `ChangeSubscriber::on_change`'s own contract that real work
    /// belongs off the synchronous write path. Called once per write
    /// request from `handle`, rather than a separate periodic timer task:
    /// the delta-size trigger (the primary one) only ever needs checking
    /// after a write anyway, and the 6h safety timer still fires
    /// correctly on the next write to a otherwise-quiet collection — the
    /// one gap is a collection that stops receiving writes entirely
    /// forever, which also has no staleness to fix.
    fn maybe_trigger_rebuild(&self, coll: &str) {
        let Some(live) = self.live_indexes.read().get(coll).cloned() else {
            return;
        };
        if !live.should_rebuild() {
            return;
        }
        let storage = self.storage.clone();
        let coll = coll.to_string();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = live.rebuild(&*storage, &coll) {
                tracing::warn!(coll, error = %e, "background LiveIndex rebuild failed");
            }
        });
    }

    /// (Re-)attaches a live BM25 index for `coll`: subscribe *first*, then
    /// backfill from a full scan — subscribing after the scan would leave
    /// a window where a concurrent write lands in neither the scan
    /// snapshot nor the not-yet-attached feed and is silently missed
    /// entirely. Subscribing first can instead double-insert a row that's
    /// written during the backfill scan (once live, once from the scan
    /// page that already contains it), which only skews `N`/`avgdl`
    /// slightly rather than dropping a row from the index outright — the
    /// safer failure mode of the two, and an accepted, narrow race for an
    /// infrequent admin operation. A proper fix (an atomic scan+subscribe
    /// primitive) is `LiveIndex` territory, a later build step.
    /// Builds `builder` wrapped in a fresh `LiveIndex`, subscribes it to
    /// `coll`'s change feed, and caches it (both as the `Arc<dyn
    /// VectorIndex>` `Search` looks up and, concretely, as the
    /// `Arc<LiveIndex>` rebuild-scheduling needs) — the shared tail of
    /// `Reindex{IvfPq}` and `Reindex{Lsh}`, which differ only in which
    /// index algorithm `builder` actually builds. Returns the resulting
    /// row count, or the `Response::Error` to return as-is on failure.
    fn attach_live_vector_index(&self, coll: &str, dim: usize, metric: DistanceMetric, builder: mara_index_vector::Builder) -> Result<u64, Response> {
        let live = LiveIndex::build(&*self.storage, coll, dim, metric, builder, LiveIndexParams::default()).map_err(index_err)?;
        let live = Arc::new(live);
        self.storage.subscribe(coll, live.clone()).map_err(storage_err)?;
        self.vector_indexes.write().insert(coll.to_string(), live.clone() as Arc<dyn VectorIndex>);
        self.live_indexes.write().insert(coll.to_string(), live);
        Ok(self.storage.active_count(coll).unwrap_or(0))
    }

    fn attach_and_backfill_bm25(&self, coll: &str) -> StorageResult<u64> {
        let index = Arc::new(RwLock::new(Bm25Index::new(Bm25Params::default())));
        self.storage.subscribe(coll, Arc::new(Bm25Subscriber { index: index.clone() }))?;

        let mut count = 0u64;
        let mut cursor = None;
        loop {
            let page = self.storage.scan(coll, cursor, 10_000)?;
            let page_len = page.len();
            cursor = page.last().map(|r| r.id).or(cursor);
            for row in &page {
                if let Some(text) = &row.text {
                    index.write().insert(row.id, text);
                    count += 1;
                }
            }
            if page_len < 10_000 {
                break;
            }
        }

        self.bm25_indexes.write().insert(coll.to_string(), index);
        Ok(count)
    }

    /// Resolves `filter` into a `FilterMask`. Everything except a
    /// top-level `Filter::TextMatch` delegates to `storage.compile_filter`
    /// unchanged; `TextMatch` nested inside a larger `And`/`Or`/`Not` tree
    /// still surfaces storage's existing `TextMatchUnsupported` error —
    /// combining a lexical predicate with the rest of a filter tree in one
    /// mask is real work this step doesn't take on (see `mara-fusion`,
    /// where hybrid search makes that combination worth building).
    fn compile_filter(&self, coll: &str, filter: &mara_proto::Filter) -> Result<FilterMask, Response> {
        if let mara_proto::Filter::TextMatch { terms, .. } = filter {
            let bm25 = self.bm25_indexes.read().get(coll).cloned();
            let Some(bm25) = bm25 else {
                return Err(Response::Error {
                    code: "bm25_not_available".into(),
                    message: format!("no BM25 index attached for {coll:?} — call Reindex{{index_kind: bm25}} first"),
                });
            };
            let allowed = bm25.read().resolve_text_match(terms);
            let estimated_cardinality = allowed.len();
            return Ok(FilterMask { allowed, estimated_cardinality });
        }
        self.storage.compile_filter(coll, filter).map_err(storage_err)
    }
}

fn to_search_params(w: &WireSearchParams) -> SearchParams {
    let d = SearchParams::default();
    SearchParams {
        // `None` (unspecified) must resolve to the default *grouping cap*
        // (`Some(3)`), not to "grouping disabled" — those aren't the same
        // internal value, unlike `rerank_k`/`filter_nprobe_max` below,
        // where "unspecified" and the index's own dynamic-default meaning
        // happen to both be `None`. A client that genuinely wants
        // grouping off has no wire-level way to say so yet (a narrow,
        // documented v0 gap — pass a very large cap as a workaround).
        max_chunks_per_doc: w.max_chunks_per_doc.map_or(d.max_chunks_per_doc, Some),
        nprobe: w.nprobe.unwrap_or(d.nprobe),
        beam_width: w.beam_width.unwrap_or(d.beam_width),
        rerank_k: w.rerank_k,
        filter_exact_threshold: w.filter_exact_threshold.unwrap_or(d.filter_exact_threshold),
        filter_selectivity_high: w.filter_selectivity_high.unwrap_or(d.filter_selectivity_high),
        filter_nprobe_max: w.filter_nprobe_max,
    }
}

/// Batch-fetches full `Row`s for a `(RowId, score, exact)` list — shared
/// by both `SearchMode`s, which otherwise differ only in how they got
/// there (a vector index's `SearchHit`s vs. BM25's `LexicalHit`s already
/// carry `score`/`exact`, never a full `Row`).
fn scored_hits(storage: &dyn StorageApi, coll: &str, scored: Vec<(RowId, f32, bool)>) -> Vec<ScoredHit> {
    let ids: Vec<RowId> = scored.iter().map(|(id, _, _)| *id).collect();
    let rows = storage.rows_by_id(coll, &ids).unwrap_or_default();
    scored
        .into_iter()
        .zip(rows)
        .filter_map(|((_, score, exact), row)| row.map(|row| ScoredHit { row, score, exact }))
        .collect()
}

/// `SearchMode::Hybrid`'s post-fusion grouping — mirrors
/// `mara_index_vector::grouping::apply_doc_grouping`, reimplemented here
/// rather than reused across the crate boundary since it operates on
/// `(f32, Row)` pairs from a *fused* ranking, not a single index's own
/// `SearchHit`s. `ranked` must already be sorted descending by score.
fn group_by_doc(ranked: Vec<(f32, Row)>, k: usize, max_chunks_per_doc: Option<u32>) -> Vec<(f32, Row)> {
    let mut ranked = ranked;
    if let Some(max) = max_chunks_per_doc {
        let mut per_doc: HashMap<mara_proto::DocId, u32> = HashMap::new();
        ranked.retain(|(_, row)| match row.doc_id {
            Some(doc_id) => {
                let count = per_doc.entry(doc_id).or_insert(0);
                let keep = *count < max;
                if keep {
                    *count += 1;
                }
                keep
            }
            None => true,
        });
    }
    ranked.truncate(k);
    ranked
}

fn index_err(e: mara_index_vector::IndexError) -> Response {
    Response::Error {
        code: "index_error".into(),
        message: e.to_string(),
    }
}

/// `PqParams::default()`'s `m=48` (tuned for 384-dim sentence-transformer
/// output, per the master plan's stated defaults) doesn't evenly divide
/// every embedding dimension a collection might actually use — 1024 is a
/// common one it fails on. `Reindex{index_kind: IvfPq}` has no per-request
/// override for `pq_m` yet, so it needs a value that's *always* valid:
/// the largest divisor of `dim` at or below the target, falling back
/// toward smaller (and, worst case, `m=1` — one subspace, no real
/// compression, but never a build failure) rather than erroring on a
/// dimension that just doesn't factor conveniently near 48.
fn safe_pq_m(dim: usize) -> usize {
    const TARGET: usize = 48;
    (1..=dim.clamp(1, TARGET)).rev().find(|m| dim.is_multiple_of(*m)).unwrap_or(1)
}

fn invalid_argument(message: impl Into<String>) -> Response {
    Response::Error {
        code: "invalid_argument".into(),
        message: message.into(),
    }
}

fn embedding_not_configured(detail: &str) -> Response {
    Response::Error {
        code: "embedding_not_configured".into(),
        message: format!("{detail} — set `[embedding] model` in the daemon config to enable local embedding"),
    }
}

fn embed_err(e: mara_embed::EmbedError) -> Response {
    Response::Error {
        code: "embedding_error".into(),
        message: e.to_string(),
    }
}

fn wire_field_type(t: WireFieldType) -> FieldType {
    match t {
        WireFieldType::Keyword => FieldType::Keyword,
        WireFieldType::KeywordList => FieldType::KeywordList,
        WireFieldType::I64 => FieldType::I64,
        WireFieldType::F64 => FieldType::F64,
        WireFieldType::DateTime => FieldType::DateTime,
        WireFieldType::Bool => FieldType::Bool,
        WireFieldType::Text => FieldType::Text,
    }
}

fn build_schema(wire: Vec<(String, WireFieldType)>) -> PayloadSchema {
    let mut builder = PayloadSchema::builder();
    for (name, ty) in wire {
        builder = builder.field(name, wire_field_type(ty));
    }
    builder.build()
}

fn forbidden(action: &str) -> Response {
    Response::Error {
        code: "forbidden".into(),
        message: format!("principal's role is not permitted to {action}"),
    }
}

fn storage_err(e: StorageError) -> Response {
    let code = match &e {
        StorageError::CollectionNotFound(_) | StorageError::KeyNotFound(_) | StorageError::DocumentNotFound(_) => "not_found",
        StorageError::CollectionAlreadyExists(_) => "already_exists",
        StorageError::DimensionMismatch { .. } => "dimension_mismatch",
        _ => "storage_error",
    };
    Response::Error {
        code: code.into(),
        message: e.to_string(),
    }
}

fn row_response(row: Option<Row>) -> Response {
    match row {
        Some(r) => Response::Row(r),
        None => Response::Error {
            code: "not_found".into(),
            message: "no row with that key".into(),
        },
    }
}

/// The collection name touched by a request, if any — folded into the
/// audit entry's `coll` field.
fn coll_of(req: &Request) -> Option<String> {
    match req {
        Request::CreateCollection { name, .. } => Some(name.clone()),
        Request::Put { coll, .. }
        | Request::PutBatch { coll, .. }
        | Request::GetByKey { coll, .. }
        | Request::Delete { coll, .. }
        | Request::PutDocument { coll, .. }
        | Request::Search { coll, .. }
        | Request::Reindex { coll, .. } => Some(coll.clone()),
        Request::Hello { .. } | Request::ReplicaHello { .. } => None,
    }
}

impl EngineImpl {
    async fn dispatch(&self, ctx: &RequestCtx, req: Request) -> Response {
        if self.is_follower
            && matches!(
                req,
                Request::Put { .. } | Request::PutBatch { .. } | Request::PutDocument { .. } | Request::Delete { .. } | Request::CreateCollection { .. }
            )
        {
            return Response::Error {
                code: "not_the_leader".into(),
                message: "this node is a replication follower and only accepts writes on the leader".into(),
            };
        }
        match req {
            Request::Hello { session_id, .. } => Response::HelloAck {
                session_id,
                server_version: env!("CARGO_PKG_VERSION").to_string(),
            },
            Request::CreateCollection { name, dim, metric, schema } => {
                if !require(ctx.principal.role, Capability::CreateCollection) {
                    return forbidden("create_collection");
                }
                match self.storage.create_collection(ctx, &name, dim, metric, build_schema(schema)) {
                    Ok(()) => match self.attach_and_backfill_bm25(&name) {
                        // A fresh collection is empty, so this is really
                        // just "subscribe" — the backfill scan finds
                        // nothing. Still routed through the same helper
                        // Reindex uses so there's exactly one way BM25
                        // ever gets attached to a collection.
                        Ok(_row_count) => Response::Ok,
                        Err(e) => storage_err(e),
                    },
                    Err(e) => storage_err(e),
                }
            }
            Request::Put { coll, key, text, vector, fields, extra } => {
                if !require(ctx.principal.role, Capability::Put) {
                    return forbidden("put");
                }
                let vector = match (vector, &text) {
                    (Some(v), _) => v,
                    (None, Some(t)) => {
                        let Some(embedder) = &self.embedder else {
                            return embedding_not_configured(&format!("text-only put (text={t:?}) requires local embedding"));
                        };
                        match embedder.embed_batch(std::slice::from_ref(t)) {
                            Ok(mut vs) => vs.pop().expect("embed_batch(1 text) returns exactly 1 vector"),
                            Err(e) => return embed_err(e),
                        }
                    }
                    (None, None) => {
                        return Response::Error {
                            code: "invalid_argument".into(),
                            message: "put requires either `vector` or `text`".into(),
                        };
                    }
                };
                match self.storage.put(ctx, &coll, &key, vector, fields, extra) {
                    Ok(_row_id) => row_response(self.storage.get_by_key(&coll, &key).ok().flatten()),
                    Err(e) => storage_err(e),
                }
            }
            Request::PutBatch { coll, items } => {
                if !require(ctx.principal.role, Capability::PutBatch) {
                    return forbidden("put_batch");
                }
                // Items missing a vector are embedded together in one
                // batch call — cheaper than one `embed_batch` round trip
                // per item when a real ONNX-backed model is configured.
                let need_embedding: Vec<usize> = items
                    .iter()
                    .enumerate()
                    .filter_map(|(i, item)| (item.vector.is_none() && item.text.is_some()).then_some(i))
                    .collect();
                let embedded: Vec<Vec<f32>> = if need_embedding.is_empty() {
                    Vec::new()
                } else {
                    let Some(embedder) = &self.embedder else {
                        return embedding_not_configured("text-only batch items require local embedding");
                    };
                    let texts: Vec<String> = need_embedding.iter().map(|&i| items[i].text.clone().unwrap()).collect();
                    match embedder.embed_batched(&texts, self.embed_batch_size) {
                        Ok(vs) => vs,
                        Err(e) => return embed_err(e),
                    }
                };
                let mut embedded_iter = embedded.into_iter();
                let mut inputs = Vec::with_capacity(items.len());
                for item in &items {
                    let vector = match &item.vector {
                        Some(v) => v.clone(),
                        None if item.text.is_some() => embedded_iter.next().expect("one embedded vector per need_embedding entry"),
                        None => {
                            return Response::Error {
                                code: "invalid_argument".into(),
                                message: format!("batch item {:?} requires either `vector` or `text`", item.key),
                            };
                        }
                    };
                    inputs.push(mara_storage::PutInput {
                        key: item.key.clone(),
                        vector,
                        fields: item.fields.clone(),
                        extra: item.extra.clone(),
                    });
                }
                match self.storage.put_batch(ctx, &coll, inputs) {
                    Ok(_ids) => {
                        // Re-fetch by key (order-preserving) rather than
                        // threading RowIds through a dedicated batch-fetch
                        // — `StorageApi` has no "fetch full Row by id"
                        // method, only vector-only/payload-only batch
                        // fetches, so this stays simple until that's worth
                        // adding.
                        let rows: Vec<Row> = items
                            .iter()
                            .filter_map(|item| self.storage.get_by_key(&coll, &item.key).ok().flatten())
                            .collect();
                        Response::Rows(rows)
                    }
                    Err(e) => storage_err(e),
                }
            }
            Request::GetByKey { coll, key } => {
                if !require(ctx.principal.role, Capability::Get) {
                    return forbidden("get_by_key");
                }
                match self.storage.get_by_key(&coll, &key) {
                    Ok(row) => row_response(row),
                    Err(e) => storage_err(e),
                }
            }
            Request::Delete { coll, key } => {
                if !require(ctx.principal.role, Capability::Delete) {
                    return forbidden("delete");
                }
                match self.storage.delete(ctx, &coll, &key) {
                    Ok(()) => Response::Ok,
                    Err(e) => storage_err(e),
                }
            }
            Request::PutDocument { coll, doc_key, text, chunk_spec, fields, source } => {
                if !require(ctx.principal.role, Capability::PutDocument) {
                    return forbidden("put_document");
                }
                let Some(embedder) = &self.embedder else {
                    return embedding_not_configured(&format!("put_document {doc_key:?} requires local embedding"));
                };
                let chunks = match chunk_text(&text, &chunk_spec) {
                    Ok(c) => c,
                    Err(e) => {
                        return Response::Error {
                            code: "invalid_chunk_spec".into(),
                            message: e.to_string(),
                        }
                    }
                };
                if chunks.is_empty() {
                    return Response::Error {
                        code: "invalid_argument".into(),
                        message: format!("document {doc_key:?} produced no chunks from its text under the given chunk_spec"),
                    };
                }
                let vectors = match embedder.embed_batched(&chunks, self.embed_batch_size) {
                    Ok(v) => v,
                    Err(e) => return embed_err(e),
                };
                let chunk_inputs: Vec<ChunkInput> = chunks
                    .into_iter()
                    .zip(vectors)
                    .map(|(text, vector)| ChunkInput { text, vector })
                    .collect();
                let embedding_model: ModelFingerprint = embedder.fingerprint().clone();
                let input = PutDocumentInput {
                    doc_key: doc_key.clone(),
                    chunks: chunk_inputs,
                    doc_payload: fields,
                    chunk_spec,
                    source,
                    embedding_model,
                };
                match self.storage.put_document(ctx, &coll, input) {
                    Ok(_doc_id) => match self.storage.get_document(&coll, &doc_key) {
                        Ok(Some(doc)) => Response::Document {
                            doc_id: doc.doc_id,
                            doc_key: doc.doc_key,
                            chunk_count: doc.chunk_count,
                            version: doc.version,
                        },
                        Ok(None) => Response::Error {
                            code: "storage_error".into(),
                            message: format!("document {doc_key:?} was just written but is not readable back"),
                        },
                        Err(e) => storage_err(e),
                    },
                    Err(e) => storage_err(e),
                }
            }
            Request::Search { coll, query_text, query_vector, mode, k, filter, params } => {
                if !require(ctx.principal.role, Capability::Search) {
                    return forbidden("search");
                }
                let filter_mask = match &filter {
                    None => None,
                    Some(f) => match self.compile_filter(&coll, f) {
                        Ok(mask) => Some(mask),
                        Err(resp) => return resp,
                    },
                };

                match mode {
                    SearchMode::VectorOnly => {
                        let query_vector = match (query_vector, &query_text) {
                            (Some(v), _) => v,
                            (None, Some(t)) => {
                                let Some(embedder) = &self.embedder else {
                                    return embedding_not_configured(&format!("vector search (query_text={t:?}) requires local embedding"));
                                };
                                match embedder.embed_batch(std::slice::from_ref(t)) {
                                    Ok(mut vs) => vs.pop().expect("embed_batch(1 text) returns exactly 1 vector"),
                                    Err(e) => return embed_err(e),
                                }
                            }
                            (None, None) => return invalid_argument("search requires either `query_vector` or `query_text`"),
                        };
                        let index = match self.vector_index_for(&coll) {
                            Ok(idx) => idx,
                            Err(e) => return storage_err(e),
                        };
                        let search_params = to_search_params(&params);
                        match index.search(&*self.storage, &coll, &query_vector, k, filter_mask.as_ref(), &search_params) {
                            Ok(result) => {
                                let scored: Vec<(RowId, f32, bool)> = result.hits.iter().map(|h| (h.id, h.score, h.exact)).collect();
                                Response::SearchResults {
                                    hits: scored_hits(&*self.storage, &coll, scored),
                                    truncated_by_filter: result.truncated_by_filter,
                                }
                            }
                            Err(e) => index_err(e),
                        }
                    }
                    SearchMode::Bm25Only => {
                        let Some(query_text) = query_text else {
                            return invalid_argument("mode: bm25_only requires `query_text`");
                        };
                        let bm25 = self.bm25_indexes.read().get(&coll).cloned();
                        let Some(bm25) = bm25 else {
                            return Response::Error {
                                code: "bm25_not_available".into(),
                                message: format!("no BM25 index attached for {coll:?} — call Reindex{{index_kind: bm25}} first"),
                            };
                        };
                        let hits = bm25.read().search(&query_text, k, filter_mask.as_ref());
                        let scored: Vec<(RowId, f32, bool)> = hits.into_iter().map(|h| (h.id, h.score, true)).collect();
                        Response::SearchResults {
                            hits: scored_hits(&*self.storage, &coll, scored),
                            truncated_by_filter: false,
                        }
                    }
                    SearchMode::Hybrid { method, overfetch_k } => {
                        let Some(query_text) = query_text else {
                            return invalid_argument("mode: hybrid requires `query_text` (BM25's arm always needs it)");
                        };
                        let bm25 = self.bm25_indexes.read().get(&coll).cloned();
                        let Some(bm25) = bm25 else {
                            return Response::Error {
                                code: "bm25_not_available".into(),
                                message: format!("no BM25 index attached for {coll:?} — call Reindex{{index_kind: bm25}} first"),
                            };
                        };
                        let vector = match query_vector {
                            Some(v) => v,
                            None => {
                                let Some(embedder) = &self.embedder else {
                                    return embedding_not_configured(&format!("hybrid search (query_text={query_text:?}) requires local embedding for its vector arm"));
                                };
                                match embedder.embed_batch(std::slice::from_ref(&query_text)) {
                                    Ok(mut vs) => vs.pop().expect("embed_batch(1 text) returns exactly 1 vector"),
                                    Err(e) => return embed_err(e),
                                }
                            }
                        };
                        let index = match self.vector_index_for(&coll) {
                            Ok(idx) => idx,
                            Err(e) => return storage_err(e),
                        };
                        // Master plan default: max(k*4, 50). Ungrouped —
                        // per-arm max_chunks_per_doc would let one arm
                        // discard a candidate the fused, grouped result
                        // might have kept; grouping runs once, after fusion.
                        let overfetch = overfetch_k.unwrap_or_else(|| (k * 4).max(50));
                        let vector_params = SearchParams {
                            max_chunks_per_doc: None,
                            ..to_search_params(&params)
                        };
                        let vector_result = match index.search(&*self.storage, &coll, &vector, overfetch, filter_mask.as_ref(), &vector_params) {
                            Ok(r) => r,
                            Err(e) => return index_err(e),
                        };
                        let bm25_hits = bm25.read().search(&query_text, overfetch, filter_mask.as_ref());

                        let vector_pairs: Vec<(RowId, f32)> = vector_result.hits.iter().map(|h| (h.id, h.score)).collect();
                        let bm25_pairs: Vec<(RowId, f32)> = bm25_hits.iter().map(|h| (h.id, h.score)).collect();
                        let fusion_method = match method {
                            None => mara_fusion::FusionMethod::default(),
                            Some(mara_proto::WireFusionMethod::ReciprocalRankFusion { k }) => mara_fusion::FusionMethod::ReciprocalRankFusion { k: k.unwrap_or(60) },
                            Some(mara_proto::WireFusionMethod::WeightedSum { vector_weight, bm25_weight }) => mara_fusion::FusionMethod::WeightedSum { vector_weight, bm25_weight },
                        };
                        let fused = mara_fusion::fuse(&vector_pairs, &bm25_pairs, fusion_method);

                        let ids: Vec<RowId> = fused.iter().map(|h| h.id).collect();
                        let rows = self.storage.rows_by_id(&coll, &ids).unwrap_or_default();
                        let mut ranked: Vec<(f32, Row)> = fused.into_iter().zip(rows).filter_map(|(h, row)| row.map(|row| (h.score, row))).collect();
                        // Already fused-and-ranked descending; a stable
                        // sort here is a no-op unless a caller-provided
                        // `filter`/id-fetch reordered anything — kept only
                        // as a defensive invariant, not because it's
                        // expected to change the order.
                        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                        let grouped = group_by_doc(ranked, k, to_search_params(&params).max_chunks_per_doc);

                        Response::SearchResults {
                            hits: grouped
                                .into_iter()
                                .map(|(score, row)| ScoredHit { row, score, exact: true })
                                .collect(),
                            truncated_by_filter: vector_result.truncated_by_filter,
                        }
                    }
                }
            }
            Request::Reindex { coll, index_kind } => {
                if !require(ctx.principal.role, Capability::Reindex) {
                    return forbidden("reindex");
                }
                // `row_count` means "how many rows this index kind now
                // actually covers" — for `Flat`/`IvfPq` that's every live
                // row (`active_count`), but for `Bm25` it's specifically
                // how many of those rows had text to index, which can be
                // far fewer in a collection that mixes plain vector rows
                // with document chunks. Reporting `active_count()`
                // uniformly here would silently overstate BM25 coverage.
                let row_count = match index_kind {
                    WireIndexKind::Flat => match self.storage.collection_info(&coll) {
                        Ok(info) => {
                            self.vector_indexes.write().insert(coll.clone(), Arc::new(FlatIndex::new(info.dim, info.metric)) as Arc<dyn VectorIndex>);
                            self.storage.active_count(&coll).unwrap_or(0)
                        }
                        Err(e) => return storage_err(e),
                    },
                    WireIndexKind::IvfPq => {
                        let info = match self.storage.collection_info(&coll) {
                            Ok(info) => info,
                            Err(e) => return storage_err(e),
                        };
                        let pq_params = PqParams {
                            m: safe_pq_m(info.dim),
                            ..PqParams::default()
                        };
                        // IvfPqIndex genuinely caches build-time state (no
                        // per-search re-scan, unlike FlatIndex), so it's
                        // the one that benefits from LiveIndex-wrapping —
                        // otherwise every write after this Reindex would
                        // silently stay invisible until the next one.
                        let ivf_params = IvfParams::default();
                        let builder: mara_index_vector::Builder = Arc::new(move |storage: &dyn StorageApi, coll: &str| -> mara_index_vector::IndexResult<Box<dyn VectorIndex>> {
                            let info = storage.collection_info(coll)?;
                            let idx = IvfPqIndex::build(storage, coll, info.dim, info.metric, &ivf_params, &pq_params)?;
                            Ok(Box::new(idx))
                        });
                        match self.attach_live_vector_index(&coll, info.dim, info.metric, builder) {
                            Ok(row_count) => row_count,
                            Err(resp) => return resp,
                        }
                    }
                    WireIndexKind::Lsh => {
                        let info = match self.storage.collection_info(&coll) {
                            Ok(info) => info,
                            Err(e) => return storage_err(e),
                        };
                        let lsh_params = mara_index_vector::LshParams::default();
                        let builder: mara_index_vector::Builder = Arc::new(move |storage: &dyn StorageApi, coll: &str| -> mara_index_vector::IndexResult<Box<dyn VectorIndex>> {
                            let info = storage.collection_info(coll)?;
                            let idx = mara_index_vector::LshIndex::build(storage, coll, info.dim, info.metric, &lsh_params)?;
                            Ok(Box::new(idx))
                        });
                        match self.attach_live_vector_index(&coll, info.dim, info.metric, builder) {
                            Ok(row_count) => row_count,
                            Err(resp) => return resp,
                        }
                    }
                    WireIndexKind::Bm25 => match self.attach_and_backfill_bm25(&coll) {
                        Ok(indexed) => indexed,
                        Err(e) => return storage_err(e),
                    },
                };
                Response::Reindexed { coll, index_kind, row_count }
            }
            // Never reaches `Engine::handle` in practice — the leader's
            // dedicated replication listener intercepts `ReplicaHello`
            // before it ever gets here (see `mara-daemon::replication`),
            // exactly the way `Hello` itself is handled by the connection
            // handshake, not this dispatcher. A client that sends one over
            // the normal port gets a clear rejection rather than a panic.
            Request::ReplicaHello { .. } => Response::Error {
                code: "invalid_argument".into(),
                message: "replica_hello is only valid on the replication listener".into(),
            },
        }
    }
}

#[async_trait::async_trait]
impl Engine for EngineImpl {
    async fn handle(&self, ctx: &RequestCtx, req: Request) -> Response {
        let start = Instant::now();
        let action = req.action_name();
        let coll = coll_of(&req);
        let is_write = matches!(req, Request::Put { .. } | Request::PutBatch { .. } | Request::PutDocument { .. } | Request::Delete { .. });

        let response = self.dispatch(ctx, req).await;

        if is_write
            && !matches!(response, Response::Error { .. })
            && let Some(coll) = &coll
        {
            self.maybe_trigger_rebuild(coll);
        }

        let latency_ms = start.elapsed().as_millis() as u64;
        let outcome = match &response {
            Response::Error { message, .. } => AuditOutcome::Error { message: message.clone() },
            _ => AuditOutcome::Ok,
        };
        let mut record = AuditRecord::new(ctx, action, outcome, latency_ms);
        if let Some(coll) = coll {
            record = record.with_extra("coll", coll);
        }
        match self.audit.record(record) {
            Ok(()) => response,
            // A wedged strict audit sink degrades the daemon into errors
            // rather than an unbounded silent freeze or a silently
            // unaudited mutation — see the master plan's *Audit log*
            // durability policy.
            Err(e) => Response::Error {
                code: "audit_sink_unavailable".into(),
                message: e.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mara_auth::{AuditConfig, AuditFsync, AuditMode, JsonlAuditSink};
    use mara_embed::DeterministicTestBackend;
    use mara_proto::{ChunkSpec, ChunkStrategy, DistanceMetric, PayloadRow, Principal, PrincipalId, Role, SessionId, Source};
    use mara_storage::Storage;

    fn ctx(role: Role) -> RequestCtx {
        RequestCtx::new(
            SessionId("s".into()),
            Principal {
                id: PrincipalId("p".into()),
                name: "t".into(),
                role,
            },
            Source::Embedded,
        )
    }

    fn audit_sink() -> Arc<dyn AuditSink> {
        let dir = tempfile::tempdir().unwrap();
        Arc::new(
            JsonlAuditSink::open(
                dir.keep(),
                AuditConfig {
                    mode: AuditMode::Lossy,
                    fsync: AuditFsync::Never,
                    max_stall: std::time::Duration::from_millis(1000),
                    channel_capacity: 64,
                    rotate_size_mb: 64,
                    retention_days: 1,
                },
            )
            .unwrap(),
        )
    }

    async fn engine_with_embedder(dim: usize) -> (EngineImpl, Arc<dyn StorageApi>) {
        let storage: Arc<dyn StorageApi> = Arc::new(Storage::new());
        let embedder: Arc<dyn EmbeddingBackend> = Arc::new(DeterministicTestBackend::new(dim));
        let engine = EngineImpl::with_embedder(storage.clone(), audit_sink(), Some(embedder), 8);
        (engine, storage)
    }

    async fn create_collection(engine: &EngineImpl, name: &str, dim: usize) {
        let resp = engine
            .handle(
                &ctx(Role::Admin),
                Request::CreateCollection {
                    name: name.to_string(),
                    dim,
                    metric: DistanceMetric::Cosine,
                    schema: Vec::new(),
                },
            )
            .await;
        assert!(matches!(resp, Response::Ok), "create_collection failed: {resp:?}");
    }

    fn markdown_doc(sections: usize) -> String {
        let mut text = String::new();
        for i in 0..sections {
            text.push_str(&format!("## Section {i}\n\nSome body text for section {i} that takes up a bit of room.\n\n"));
        }
        text
    }

    #[tokio::test]
    async fn put_document_end_to_end_creates_retrievable_chunks() {
        let (engine, storage) = engine_with_embedder(8).await;
        create_collection(&engine, "docs", 8).await;

        let resp = engine
            .handle(
                &ctx(Role::Writer),
                Request::PutDocument {
                    coll: "docs".into(),
                    doc_key: "onboarding.md".into(),
                    text: markdown_doc(20),
                    chunk_spec: ChunkSpec {
                        strategy: ChunkStrategy::Markdown { respect_headings: true },
                        max_tokens: 100,
                        overlap_tokens: 10,
                        trim: true,
                    },
                    fields: PayloadRow::new(),
                    source: Some("onboarding.md".into()),
                },
            )
            .await;

        let Response::Document { doc_key, chunk_count, version, .. } = resp else {
            panic!("expected Response::Document, got {resp:?}");
        };
        assert_eq!(doc_key, "onboarding.md");
        assert_eq!(version, 1);
        assert!(chunk_count > 1, "expected multiple chunks, got {chunk_count}");

        let doc = storage.get_document("docs", "onboarding.md").unwrap().unwrap();
        assert_eq!(doc.chunk_count, chunk_count);
        assert_eq!(doc.embedding_model.model_id, "mara-embed/deterministic-test-backend");

        let chunks = storage.scan("docs", None, 1000).unwrap();
        assert_eq!(chunks.len() as u32, chunk_count);
        for row in &chunks {
            assert_eq!(row.doc_id, Some(doc.doc_id));
            assert!(row.chunk_ord.is_some());
            assert!(row.text.is_some());
            assert_eq!(row.vector.as_ref().unwrap().len(), 8);
        }
    }

    #[tokio::test]
    async fn put_document_without_a_configured_embedder_is_a_clear_error() {
        let storage: Arc<dyn StorageApi> = Arc::new(Storage::new());
        let engine = EngineImpl::new(storage, audit_sink());
        create_collection(&engine, "docs", 8).await;

        let resp = engine
            .handle(
                &ctx(Role::Writer),
                Request::PutDocument {
                    coll: "docs".into(),
                    doc_key: "a.md".into(),
                    text: "hello world".into(),
                    chunk_spec: ChunkSpec::default(),
                    fields: PayloadRow::new(),
                    source: None,
                },
            )
            .await;
        match resp {
            Response::Error { code, .. } => assert_eq!(code, "embedding_not_configured"),
            other => panic!("expected embedding_not_configured error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_document_rejects_a_duplicate_doc_key() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "docs", 8).await;
        let req = || Request::PutDocument {
            coll: "docs".into(),
            doc_key: "dup.md".into(),
            text: "# Title\n\nsome text".into(),
            chunk_spec: ChunkSpec::default(),
            fields: PayloadRow::new(),
            source: None,
        };
        let first = engine.handle(&ctx(Role::Writer), req()).await;
        assert!(matches!(first, Response::Document { .. }), "first insert failed: {first:?}");

        let second = engine.handle(&ctx(Role::Writer), req()).await;
        match second {
            Response::Error { message, .. } => assert!(message.contains("already exists"), "unexpected message: {message}"),
            other => panic!("expected a duplicate-doc_key error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_document_is_forbidden_for_a_reader_role() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "docs", 8).await;

        let resp = engine
            .handle(
                &ctx(Role::Reader),
                Request::PutDocument {
                    coll: "docs".into(),
                    doc_key: "a.md".into(),
                    text: "hello world".into(),
                    chunk_spec: ChunkSpec::default(),
                    fields: PayloadRow::new(),
                    source: None,
                },
            )
            .await;
        match resp {
            Response::Error { code, .. } => assert_eq!(code, "forbidden"),
            other => panic!("expected forbidden, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_with_text_and_no_vector_embeds_via_the_configured_backend() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "rows", 8).await;

        let resp = engine
            .handle(
                &ctx(Role::Writer),
                Request::Put {
                    coll: "rows".into(),
                    key: "k1".into(),
                    text: Some("hello world".into()),
                    vector: None,
                    fields: PayloadRow::new(),
                    extra: None,
                },
            )
            .await;
        let Response::Row(row) = resp else { panic!("expected Response::Row, got {resp:?}") };
        let expected = DeterministicTestBackend::new(8).embed_batch(&["hello world".to_string()]).unwrap();
        assert_eq!(row.vector, Some(expected[0].clone()));
    }

    #[tokio::test]
    async fn put_without_vector_or_text_is_a_clear_invalid_argument_error() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "rows", 8).await;

        let resp = engine
            .handle(
                &ctx(Role::Writer),
                Request::Put {
                    coll: "rows".into(),
                    key: "k1".into(),
                    text: None,
                    vector: None,
                    fields: PayloadRow::new(),
                    extra: None,
                },
            )
            .await;
        match resp {
            Response::Error { code, .. } => assert_eq!(code, "invalid_argument"),
            other => panic!("expected invalid_argument, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_batch_embeds_text_only_items_alongside_vector_items() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "rows", 8).await;

        let resp = engine
            .handle(
                &ctx(Role::Writer),
                Request::PutBatch {
                    coll: "rows".into(),
                    items: vec![
                        mara_proto::PutItem {
                            key: "explicit".into(),
                            text: None,
                            vector: Some(vec![0.1; 8]),
                            fields: PayloadRow::new(),
                            extra: None,
                        },
                        mara_proto::PutItem {
                            key: "from_text".into(),
                            text: Some("batched text".into()),
                            vector: None,
                            fields: PayloadRow::new(),
                            extra: None,
                        },
                    ],
                },
            )
            .await;
        let Response::Rows(rows) = resp else { panic!("expected Response::Rows, got {resp:?}") };
        assert_eq!(rows.len(), 2);
        let expected = DeterministicTestBackend::new(8).embed_batch(&["batched text".to_string()]).unwrap();
        let from_text = rows.iter().find(|r| r.key == "from_text").unwrap();
        assert_eq!(from_text.vector, Some(expected[0].clone()));
        let explicit = rows.iter().find(|r| r.key == "explicit").unwrap();
        assert_eq!(explicit.vector, Some(vec![0.1; 8]));
    }

    fn default_wire_params() -> WireSearchParams {
        WireSearchParams::default()
    }

    #[tokio::test]
    async fn search_vector_only_finds_the_nearest_neighbor_without_any_reindex() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "rows", 8).await;

        engine
            .handle(
                &ctx(Role::Writer),
                Request::Put {
                    coll: "rows".into(),
                    key: "a".into(),
                    text: None,
                    vector: Some(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    fields: PayloadRow::new(),
                    extra: None,
                },
            )
            .await;
        engine
            .handle(
                &ctx(Role::Writer),
                Request::Put {
                    coll: "rows".into(),
                    key: "b".into(),
                    text: None,
                    vector: Some(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    fields: PayloadRow::new(),
                    extra: None,
                },
            )
            .await;

        let resp = engine
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "rows".into(),
                    query_text: None,
                    query_vector: Some(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    mode: SearchMode::VectorOnly,
                    k: 1,
                    filter: None,
                    params: default_wire_params(),
                },
            )
            .await;
        let Response::SearchResults { hits, truncated_by_filter } = resp else { panic!("expected SearchResults, got {resp:?}") };
        assert!(!truncated_by_filter);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].row.key, "a");
        assert!(hits[0].exact);
    }

    #[tokio::test]
    async fn search_vector_only_embeds_query_text_when_no_vector_is_given() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "rows", 8).await;
        let embedded = DeterministicTestBackend::new(8).embed_batch(&["hello world".to_string()]).unwrap();
        engine
            .handle(
                &ctx(Role::Writer),
                Request::Put {
                    coll: "rows".into(),
                    key: "a".into(),
                    text: None,
                    vector: Some(embedded[0].clone()),
                    fields: PayloadRow::new(),
                    extra: None,
                },
            )
            .await;

        let resp = engine
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "rows".into(),
                    query_text: Some("hello world".into()),
                    query_vector: None,
                    mode: SearchMode::VectorOnly,
                    k: 1,
                    filter: None,
                    params: default_wire_params(),
                },
            )
            .await;
        let Response::SearchResults { hits, .. } = resp else { panic!("expected SearchResults, got {resp:?}") };
        assert_eq!(hits[0].row.key, "a");
    }

    #[tokio::test]
    async fn search_bm25_only_finds_a_document_by_keyword() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "docs", 8).await;

        engine
            .handle(
                &ctx(Role::Writer),
                Request::PutDocument {
                    coll: "docs".into(),
                    doc_key: "onboarding.md".into(),
                    text: "## Setup\n\nInstall the aardvark package before running migrations.".into(),
                    chunk_spec: ChunkSpec::default(),
                    fields: PayloadRow::new(),
                    source: None,
                },
            )
            .await;

        let resp = engine
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "docs".into(),
                    query_text: Some("aardvark".into()),
                    query_vector: None,
                    mode: SearchMode::Bm25Only,
                    k: 5,
                    filter: None,
                    params: default_wire_params(),
                },
            )
            .await;
        let Response::SearchResults { hits, truncated_by_filter } = resp else { panic!("expected SearchResults, got {resp:?}") };
        assert!(!truncated_by_filter);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].row.text.as_deref().unwrap().contains("aardvark"));
        assert!(hits[0].exact);
    }

    #[tokio::test]
    async fn search_bm25_only_without_query_text_is_invalid_argument() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "docs", 8).await;

        let resp = engine
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "docs".into(),
                    query_text: None,
                    query_vector: None,
                    mode: SearchMode::Bm25Only,
                    k: 5,
                    filter: None,
                    params: default_wire_params(),
                },
            )
            .await;
        match resp {
            Response::Error { code, .. } => assert_eq!(code, "invalid_argument"),
            other => panic!("expected invalid_argument, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn search_is_forbidden_for_a_replica_role() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "rows", 8).await;
        let resp = engine
            .handle(
                &ctx(Role::Replica),
                Request::Search {
                    coll: "rows".into(),
                    query_text: None,
                    query_vector: Some(vec![0.0; 8]),
                    mode: SearchMode::VectorOnly,
                    k: 1,
                    filter: None,
                    params: default_wire_params(),
                },
            )
            .await;
        match resp {
            Response::Error { code, .. } => assert_eq!(code, "forbidden"),
            other => panic!("expected forbidden, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn reindex_ivf_pq_then_vector_search_still_finds_the_true_neighbor() {
        let (engine, _storage) = engine_with_embedder(8).await;
        // A fresh collection (not `create_collection`'s shared Cosine
        // helper): this test's synthetic points are collinear along one
        // axis, which is a degenerate, all-scores-tied setup under Cosine
        // (every positive multiple of the same direction has similarity
        // 1.0 to every other) — L2, where magnitude genuinely
        // distinguishes points, is what "nearest neighbor" means here.
        engine
            .handle(
                &ctx(Role::Admin),
                Request::CreateCollection {
                    name: "rows".into(),
                    dim: 8,
                    metric: DistanceMetric::L2,
                    schema: Vec::new(),
                },
            )
            .await;

        for i in 0..40u32 {
            let mut v = vec![0.0f32; 8];
            v[0] = i as f32;
            engine
                .handle(
                    &ctx(Role::Writer),
                    Request::Put {
                        coll: "rows".into(),
                        key: format!("k{i}"),
                        text: None,
                        vector: Some(v),
                        fields: PayloadRow::new(),
                        extra: None,
                    },
                )
                .await;
        }

        let reindex_resp = engine
            .handle(
                &ctx(Role::Admin),
                Request::Reindex {
                    coll: "rows".into(),
                    index_kind: mara_proto::WireIndexKind::IvfPq,
                },
            )
            .await;
        match reindex_resp {
            Response::Reindexed { row_count, index_kind, .. } => {
                assert_eq!(row_count, 40);
                assert_eq!(index_kind, mara_proto::WireIndexKind::IvfPq);
            }
            other => panic!("expected Reindexed, got {other:?}"),
        }

        let mut query = vec![0.0f32; 8];
        query[0] = 5.0;
        let resp = engine
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "rows".into(),
                    query_text: None,
                    query_vector: Some(query),
                    mode: SearchMode::VectorOnly,
                    k: 1,
                    filter: None,
                    params: default_wire_params(),
                },
            )
            .await;
        let Response::SearchResults { hits, .. } = resp else { panic!("expected SearchResults, got {resp:?}") };
        assert_eq!(hits[0].row.key, "k5", "IVF-PQ's default rerank_k must still exact-rerank to the true nearest neighbor");
    }

    #[tokio::test]
    async fn a_write_after_reindex_ivf_pq_is_immediately_searchable_via_live_index() {
        let (engine, _storage) = engine_with_embedder(8).await;
        engine
            .handle(
                &ctx(Role::Admin),
                Request::CreateCollection {
                    name: "rows".into(),
                    dim: 8,
                    metric: DistanceMetric::L2,
                    schema: Vec::new(),
                },
            )
            .await;
        for i in 0..40u32 {
            let mut v = vec![0.0f32; 8];
            v[0] = i as f32;
            engine
                .handle(
                    &ctx(Role::Writer),
                    Request::Put {
                        coll: "rows".into(),
                        key: format!("k{i}"),
                        text: None,
                        vector: Some(v),
                        fields: PayloadRow::new(),
                        extra: None,
                    },
                )
                .await;
        }
        engine
            .handle(&ctx(Role::Admin), Request::Reindex { coll: "rows".into(), index_kind: mara_proto::WireIndexKind::IvfPq })
            .await;

        // Written *after* the bake — IvfPqIndex alone would never have
        // scanned this row; LiveIndex's delta buffer is what makes it
        // find-able without a second explicit Reindex.
        engine
            .handle(
                &ctx(Role::Writer),
                Request::Put {
                    coll: "rows".into(),
                    key: "late".into(),
                    text: None,
                    vector: Some(vec![5.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    fields: PayloadRow::new(),
                    extra: None,
                },
            )
            .await;

        let resp = engine
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "rows".into(),
                    query_text: None,
                    query_vector: Some(vec![5.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    mode: SearchMode::VectorOnly,
                    k: 1,
                    filter: None,
                    params: default_wire_params(),
                },
            )
            .await;
        let Response::SearchResults { hits, .. } = resp else { panic!("expected SearchResults, got {resp:?}") };
        assert_eq!(hits[0].row.key, "late", "a post-Reindex write must be immediately searchable via LiveIndex's delta buffer, with no second Reindex needed");
    }

    #[tokio::test]
    async fn reindex_lsh_then_vector_search_finds_the_exact_match() {
        let (engine, _storage) = engine_with_embedder(8).await;
        engine
            .handle(
                &ctx(Role::Admin),
                Request::CreateCollection {
                    name: "rows".into(),
                    dim: 8,
                    metric: DistanceMetric::L2,
                    schema: Vec::new(),
                },
            )
            .await;

        // Well-separated clusters, not a fine-grained line of points: LSH
        // is approximate (SimHash bucket membership, not a true k-NN
        // structure), so a query for a genuinely *near* neighbor a small
        // distance from other candidates isn't a guaranteed find at this
        // scale — an exact-match query always is (it hashes into its own
        // bucket in every table), which is exactly what this checks: the
        // Reindex{Lsh}+Search wiring, not LSH's approximate recall itself
        // (already covered at the crate level in lsh.rs).
        let centers = [10.0f32, -10.0, 30.0];
        for (ci, c) in centers.iter().enumerate() {
            for i in 0..10u32 {
                let mut v = vec![0.0f32; 8];
                v[0] = *c;
                v[1] = i as f32 * 0.01;
                engine
                    .handle(
                        &ctx(Role::Writer),
                        Request::Put {
                            coll: "rows".into(),
                            key: format!("k{ci}_{i}"),
                            text: None,
                            vector: Some(v),
                            fields: PayloadRow::new(),
                            extra: None,
                        },
                    )
                    .await;
            }
        }

        let reindex_resp = engine
            .handle(&ctx(Role::Admin), Request::Reindex { coll: "rows".into(), index_kind: mara_proto::WireIndexKind::Lsh })
            .await;
        match reindex_resp {
            Response::Reindexed { row_count, index_kind, .. } => {
                assert_eq!(row_count, 30);
                assert_eq!(index_kind, mara_proto::WireIndexKind::Lsh);
            }
            other => panic!("expected Reindexed, got {other:?}"),
        }

        let mut query = vec![0.0f32; 8];
        query[0] = 10.0;
        query[1] = 0.05;
        let resp = engine
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "rows".into(),
                    query_text: None,
                    query_vector: Some(query),
                    mode: SearchMode::VectorOnly,
                    k: 1,
                    filter: None,
                    params: default_wire_params(),
                },
            )
            .await;
        let Response::SearchResults { hits, .. } = resp else { panic!("expected SearchResults, got {resp:?}") };
        assert_eq!(hits[0].row.key, "k0_5", "an exact-match query must be found via the LSH-backed LiveIndex");
    }

    #[tokio::test]
    async fn a_write_after_reindex_lsh_is_immediately_searchable_via_live_index() {
        let (engine, _storage) = engine_with_embedder(8).await;
        engine
            .handle(
                &ctx(Role::Admin),
                Request::CreateCollection {
                    name: "rows".into(),
                    dim: 8,
                    metric: DistanceMetric::L2,
                    schema: Vec::new(),
                },
            )
            .await;
        for i in 0..40u32 {
            let mut v = vec![0.0f32; 8];
            v[0] = i as f32;
            engine
                .handle(
                    &ctx(Role::Writer),
                    Request::Put {
                        coll: "rows".into(),
                        key: format!("k{i}"),
                        text: None,
                        vector: Some(v),
                        fields: PayloadRow::new(),
                        extra: None,
                    },
                )
                .await;
        }
        engine
            .handle(&ctx(Role::Admin), Request::Reindex { coll: "rows".into(), index_kind: mara_proto::WireIndexKind::Lsh })
            .await;

        // Written *after* the bake — LshIndex alone would never have
        // scanned this row; LiveIndex's delta buffer is what makes it
        // find-able without a second explicit Reindex.
        engine
            .handle(
                &ctx(Role::Writer),
                Request::Put {
                    coll: "rows".into(),
                    key: "late".into(),
                    text: None,
                    vector: Some(vec![500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    fields: PayloadRow::new(),
                    extra: None,
                },
            )
            .await;

        let resp = engine
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "rows".into(),
                    query_text: None,
                    query_vector: Some(vec![500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    mode: SearchMode::VectorOnly,
                    k: 1,
                    filter: None,
                    params: default_wire_params(),
                },
            )
            .await;
        let Response::SearchResults { hits, .. } = resp else { panic!("expected SearchResults, got {resp:?}") };
        assert_eq!(hits[0].row.key, "late", "a post-Reindex write must be immediately searchable via LiveIndex's delta buffer, with no second Reindex needed");
    }

    #[tokio::test]
    async fn reindex_bm25_row_count_reflects_only_text_bearing_rows() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "mixed", 8).await;

        // Two plain vector rows (no text) plus one document chunk (has
        // text) — active_count() would report 3, but only the chunk is
        // actually BM25-searchable.
        engine
            .handle(
                &ctx(Role::Writer),
                Request::Put {
                    coll: "mixed".into(),
                    key: "v1".into(),
                    text: None,
                    vector: Some(vec![1.0; 8]),
                    fields: PayloadRow::new(),
                    extra: None,
                },
            )
            .await;
        engine
            .handle(
                &ctx(Role::Writer),
                Request::Put {
                    coll: "mixed".into(),
                    key: "v2".into(),
                    text: None,
                    vector: Some(vec![2.0; 8]),
                    fields: PayloadRow::new(),
                    extra: None,
                },
            )
            .await;
        engine
            .handle(
                &ctx(Role::Writer),
                Request::PutDocument {
                    coll: "mixed".into(),
                    doc_key: "doc.md".into(),
                    text: "a single chunk of real text".into(),
                    chunk_spec: ChunkSpec::default(),
                    fields: PayloadRow::new(),
                    source: None,
                },
            )
            .await;

        let resp = engine
            .handle(
                &ctx(Role::Admin),
                Request::Reindex {
                    coll: "mixed".into(),
                    index_kind: mara_proto::WireIndexKind::Bm25,
                },
            )
            .await;
        match resp {
            Response::Reindexed { row_count, .. } => assert_eq!(row_count, 1, "only the one text-bearing chunk should count, not all 3 rows"),
            other => panic!("expected Reindexed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn reindex_is_forbidden_for_a_writer_role() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "rows", 8).await;
        let resp = engine
            .handle(
                &ctx(Role::Writer),
                Request::Reindex {
                    coll: "rows".into(),
                    index_kind: mara_proto::WireIndexKind::Flat,
                },
            )
            .await;
        match resp {
            Response::Error { code, .. } => assert_eq!(code, "forbidden"),
            other => panic!("expected forbidden, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_follower_engine_rejects_row_and_collection_writes_but_still_allows_reindex_and_search() {
        let storage: Arc<dyn StorageApi> = Arc::new(Storage::new());
        let embedder: Arc<dyn EmbeddingBackend> = Arc::new(DeterministicTestBackend::new(8));
        let follower = EngineImpl::with_embedder(storage.clone(), audit_sink(), Some(embedder), 8).as_follower();

        // Seed a collection directly through storage — a follower's own
        // engine must never be the thing that creates one; a real one
        // arrives via replication, applied straight against `storage`.
        storage
            .create_collection(&ctx(Role::Admin), "rows", 8, DistanceMetric::Cosine, PayloadSchema::empty())
            .unwrap();

        for (label, req) in [
            (
                "create_collection",
                Request::CreateCollection { name: "other".into(), dim: 8, metric: DistanceMetric::Cosine, schema: Vec::new() },
            ),
            (
                "put",
                Request::Put { coll: "rows".into(), key: "a".into(), text: None, vector: Some(vec![0.0; 8]), fields: PayloadRow::new(), extra: None },
            ),
            ("put_batch", Request::PutBatch { coll: "rows".into(), items: Vec::new() }),
            ("delete", Request::Delete { coll: "rows".into(), key: "a".into() }),
        ] {
            let resp = follower.handle(&ctx(Role::Admin), req).await;
            match resp {
                Response::Error { code, .. } => assert_eq!(code, "not_the_leader", "{label} must be rejected on a follower"),
                other => panic!("{label}: expected not_the_leader, got {other:?}"),
            }
        }

        // Reindex (rebuilding a *derived* index from rows this follower
        // already has) and Search are both still allowed — neither
        // touches replicated row data.
        let reindex_resp = follower
            .handle(&ctx(Role::Admin), Request::Reindex { coll: "rows".into(), index_kind: mara_proto::WireIndexKind::Flat })
            .await;
        assert!(matches!(reindex_resp, Response::Reindexed { .. }), "expected Reindexed, got {reindex_resp:?}");

        let search_resp = follower
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "rows".into(),
                    query_text: None,
                    query_vector: Some(vec![0.0; 8]),
                    mode: SearchMode::VectorOnly,
                    k: 1,
                    filter: None,
                    params: default_wire_params(),
                },
            )
            .await;
        assert!(matches!(search_resp, Response::SearchResults { .. }), "expected SearchResults, got {search_resp:?}");
    }

    #[tokio::test]
    async fn text_match_filter_resolves_via_bm25_during_vector_search() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "docs", 8).await;

        engine
            .handle(
                &ctx(Role::Writer),
                Request::PutDocument {
                    coll: "docs".into(),
                    doc_key: "keep.md".into(),
                    text: "this chunk mentions the secret codeword zephyr".into(),
                    chunk_spec: ChunkSpec::default(),
                    fields: PayloadRow::new(),
                    source: None,
                },
            )
            .await;
        engine
            .handle(
                &ctx(Role::Writer),
                Request::PutDocument {
                    coll: "docs".into(),
                    doc_key: "drop.md".into(),
                    text: "this chunk has nothing special in it at all".into(),
                    chunk_spec: ChunkSpec::default(),
                    fields: PayloadRow::new(),
                    source: None,
                },
            )
            .await;

        let resp = engine
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "docs".into(),
                    query_text: None,
                    query_vector: Some(vec![0.0; 8]),
                    mode: SearchMode::VectorOnly,
                    k: 10,
                    filter: Some(mara_proto::Filter::TextMatch {
                        field: "text".into(),
                        terms: vec!["zephyr".into()],
                    }),
                    params: default_wire_params(),
                },
            )
            .await;
        let Response::SearchResults { hits, .. } = resp else { panic!("expected SearchResults, got {resp:?}") };
        assert_eq!(hits.len(), 1, "only the chunk containing \"zephyr\" should survive the TextMatch filter");
        assert!(hits[0].row.text.as_deref().unwrap().contains("zephyr"));
    }

    #[tokio::test]
    async fn search_hybrid_without_query_text_is_invalid_argument() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "docs", 8).await;
        let resp = engine
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "docs".into(),
                    query_text: None,
                    query_vector: Some(vec![0.0; 8]),
                    mode: SearchMode::Hybrid { method: None, overfetch_k: None },
                    k: 5,
                    filter: None,
                    params: default_wire_params(),
                },
            )
            .await;
        match resp {
            Response::Error { code, .. } => assert_eq!(code, "invalid_argument"),
            other => panic!("expected invalid_argument, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn search_hybrid_promotes_a_bm25_only_match_vector_search_alone_would_miss() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "docs", 8).await;

        // The deterministic test backend's embeddings are a hash of the
        // text — unrelated to real semantic content — so a chunk sharing
        // no vocabulary with the query has an essentially arbitrary
        // cosine similarity to it. "zzzznomatch" chunks pad the
        // collection with plausible vector-arm noise; only "keyword.md"
        // contains the literal query term.
        for i in 0..10 {
            engine
                .handle(
                    &ctx(Role::Writer),
                    Request::PutDocument {
                        coll: "docs".into(),
                        doc_key: format!("noise{i}.md"),
                        text: format!("zzzznomatch filler content number {i} with no relation to the query"),
                        chunk_spec: ChunkSpec::default(),
                        fields: PayloadRow::new(),
                        source: None,
                    },
                )
                .await;
        }
        engine
            .handle(
                &ctx(Role::Writer),
                Request::PutDocument {
                    coll: "docs".into(),
                    doc_key: "keyword.md".into(),
                    text: "this chunk contains the exact rare keyword tanglewood".into(),
                    chunk_spec: ChunkSpec::default(),
                    fields: PayloadRow::new(),
                    source: None,
                },
            )
            .await;

        let bm25_only = engine
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "docs".into(),
                    query_text: Some("tanglewood".into()),
                    query_vector: None,
                    mode: SearchMode::Bm25Only,
                    k: 1,
                    filter: None,
                    params: default_wire_params(),
                },
            )
            .await;
        let Response::SearchResults { hits: bm25_hits, .. } = bm25_only else { panic!("expected SearchResults") };
        assert_eq!(bm25_hits[0].row.doc_id, Some(mara_proto::DocId(10)), "sanity check: BM25 alone must find the keyword chunk");

        let hybrid = engine
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "docs".into(),
                    query_text: Some("tanglewood".into()),
                    query_vector: None,
                    mode: SearchMode::Hybrid { method: None, overfetch_k: None },
                    k: 3,
                    filter: None,
                    params: default_wire_params(),
                },
            )
            .await;
        let Response::SearchResults { hits, .. } = hybrid else { panic!("expected SearchResults, got {hybrid:?}") };
        assert!(
            hits.iter().any(|h| h.row.text.as_deref().unwrap().contains("tanglewood")),
            "the keyword chunk, found via BM25 alone, must survive fusion into the hybrid top-k"
        );
    }

    #[tokio::test]
    async fn search_hybrid_applies_max_chunks_per_doc_after_fusion() {
        let (engine, _storage) = engine_with_embedder(8).await;
        create_collection(&engine, "docs", 8).await;

        let chunks_text = (0..5).map(|i| format!("## Section {i}\n\nEach section repeats the word gizmo for search testing.\n\n")).collect::<String>();
        engine
            .handle(
                &ctx(Role::Writer),
                Request::PutDocument {
                    coll: "docs".into(),
                    doc_key: "multi.md".into(),
                    text: chunks_text,
                    chunk_spec: ChunkSpec {
                        strategy: ChunkStrategy::Markdown { respect_headings: true },
                        max_tokens: 20,
                        overlap_tokens: 0,
                        trim: true,
                    },
                    fields: PayloadRow::new(),
                    source: None,
                },
            )
            .await;

        let resp = engine
            .handle(
                &ctx(Role::Reader),
                Request::Search {
                    coll: "docs".into(),
                    query_text: Some("gizmo".into()),
                    query_vector: None,
                    mode: SearchMode::Hybrid { method: None, overfetch_k: None },
                    k: 10,
                    filter: None,
                    params: WireSearchParams {
                        max_chunks_per_doc: Some(2),
                        ..WireSearchParams::default()
                    },
                },
            )
            .await;
        let Response::SearchResults { hits, .. } = resp else { panic!("expected SearchResults, got {resp:?}") };
        assert!(hits.len() <= 2, "max_chunks_per_doc=2 must cap hits from the one document even after fusion, got {}", hits.len());
    }
}
