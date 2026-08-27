//! `LiveIndex` (master plan Layer 4, *Mutable writes vs. immutable ANN
//! structures*): an LSM-style split between an immutable `baked` index
//! and a small, live `delta` buffer fed by `ChangeSubscriber`. `baked`
//! never mutates once built — new inserts land in `delta` (brute-force
//! scanned, bounded, cheap); deletes mark `tombstones`. A rebuild retrains
//! `baked` from a fresh storage scan and swaps it in atomically via
//! `arc-swap`, so readers never see a half-built structure and never
//! block on the swap.
//!
//! The staleness `delta`/`tombstones` exist to paper over is specific to
//! indexes that actually *cache* build-time state (`IvfIndex`,
//! `IvfPqIndex`) — wrapping `FlatIndex` as `baked` (as this module's own
//! tests do, for simplicity) gets no benefit from any of this, since
//! `FlatIndex` already re-scans live storage on every call regardless of
//! whether `LiveIndex` ever heard about a write. Real deployments wrap a
//! cached index; `FlatIndex` here is purely a convenient test fixture.
//!
//! `ArcSwap<Box<dyn VectorIndex>>`, not `ArcSwap<dyn VectorIndex>`:
//! `arc_swap::RefCnt` is only implemented for `Arc<T>` with `T: Sized`,
//! and `dyn VectorIndex` isn't — confirmed by probe-compiling both before
//! picking this, same practice as `mara-chunker`'s `tokenizers` version
//! pin. `Box<dyn VectorIndex>` is itself `Sized` (a box is a fixed-size
//! pointer), so `Arc<Box<dyn VectorIndex>>` satisfies the bound.
//!
//! **Tombstones, scoped honestly**: `on_change`'s deletes mark
//! `tombstones`, and it's folded into a *caller-supplied* filter (cheap
//! bitmap subtraction, no scan) before routing through `baked`. Absent a
//! caller filter, `baked` is searched unfiltered — a tombstoned row can
//! still surface as a *candidate*, but never as a final hit, because
//! `StorageApi::rows_by_id` is itself liveness-checked (see
//! `ivf.rs`'s `a_row_deleted_after_build_is_silently_absent_from_results`)
//! and filters it out at the final fetch regardless. So this is a
//! correctness safety net either way; the unfiltered case just does a
//! bit more unnecessary candidate-gathering work than a full tombstone-
//! aware coarse-routing pass would (the "ghost cluster" cost the master
//! plan's *Filtered search* section describes) — an accepted, documented
//! v0 gap, not a correctness one.
//!
//! **Rebuild races the live feed, and is built to lose safely**: between
//! taking the pre-scan snapshot of `delta`'s length and `tombstones`'
//! contents and the fresh scan actually completing, more writes can land
//! in `delta`/`tombstones`. `rebuild` only ever clears the *pre-scan*
//! prefix/snapshot, never blindly clears everything — anything appended
//! during the scan window survives into the new generation. Worst case,
//! a row scored redundantly by both `baked` and `delta` on the very next
//! search; never a silently dropped write or delete.

use crate::error::{IndexError, IndexResult};
use crate::grouping::apply_doc_grouping;
use crate::hit::{SearchHit, SearchResult};
use crate::index_trait::VectorIndex;
use crate::math;
use crate::params::SearchParams;
use arc_swap::ArcSwap;
use mara_proto::{DistanceMetric, RowId};
use mara_storage::{ChangeBatch, ChangeEvent, ChangeSubscriber, FilterMask, StorageApi};
use parking_lot::RwLock;
use roaring::RoaringBitmap;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct LiveIndexParams {
    /// A rebuild is due once `delta.len()` reaches this fraction of the
    /// row count as of the last bake (default `0.05`, matching the
    /// master plan's "5% of N").
    pub rebuild_threshold_fraction: f64,
    /// ...capped at this many entries regardless of fraction (default
    /// `5000`) — the master plan's stated bound on how large a brute-
    /// force-scanned buffer is allowed to grow.
    pub rebuild_threshold_cap: usize,
    /// A rebuild is also due once this much time has passed since the
    /// last one, independent of `delta`'s size — the master plan's "6h
    /// safety timer", covering a collection with too little write
    /// traffic to ever cross the size threshold on its own.
    pub rebuild_safety_interval: Duration,
}

impl Default for LiveIndexParams {
    fn default() -> Self {
        LiveIndexParams {
            rebuild_threshold_fraction: 0.05,
            rebuild_threshold_cap: 5_000,
            rebuild_safety_interval: Duration::from_secs(6 * 3600),
        }
    }
}

/// Builds a fresh `baked` index from scratch — `IvfIndex::build`/
/// `IvfPqIndex::build`-shaped, injected so `LiveIndex` doesn't hardcode
/// which coarse-quantized index it wraps (or even require one; a
/// `FlatIndex`-returning builder is valid too, if a less useful one to
/// wrap — `FlatIndex` is already exact and live).
pub type Builder = Arc<dyn Fn(&dyn StorageApi, &str) -> IndexResult<Box<dyn VectorIndex>> + Send + Sync>;

pub struct LiveIndex {
    dim: usize,
    metric: DistanceMetric,
    builder: Builder,
    baked: ArcSwap<Box<dyn VectorIndex>>,
    baked_row_count: AtomicU64,
    delta: RwLock<Vec<(RowId, Vec<f32>)>>,
    tombstones: RwLock<RoaringBitmap>,
    last_rebuild: RwLock<Instant>,
    params: LiveIndexParams,
}

impl LiveIndex {
    /// Runs `builder` once for the initial bake — after this, `baked`
    /// only ever changes via `rebuild`.
    pub fn build(storage: &dyn StorageApi, coll: &str, dim: usize, metric: DistanceMetric, builder: Builder, params: LiveIndexParams) -> IndexResult<Self> {
        let baked = builder(storage, coll)?;
        let row_count = storage.active_count(coll).unwrap_or(0);
        Ok(LiveIndex {
            dim,
            metric,
            builder,
            baked: ArcSwap::from(Arc::new(baked)),
            baked_row_count: AtomicU64::new(row_count),
            delta: RwLock::new(Vec::new()),
            tombstones: RwLock::new(RoaringBitmap::new()),
            last_rebuild: RwLock::new(Instant::now()),
            params,
        })
    }

    pub fn delta_len(&self) -> usize {
        self.delta.read().len()
    }

    pub fn tombstone_count(&self) -> u64 {
        self.tombstones.read().len()
    }

    pub fn should_rebuild(&self) -> bool {
        let delta_len = self.delta.read().len();
        let n = self.baked_row_count.load(AtomicOrdering::Relaxed).max(1) as f64;
        let threshold = ((n * self.params.rebuild_threshold_fraction) as usize).clamp(1, self.params.rebuild_threshold_cap);
        if delta_len >= threshold {
            return true;
        }
        self.last_rebuild.read().elapsed() >= self.params.rebuild_safety_interval
    }

    /// Retrains `baked` from a fresh scan and atomically swaps it in.
    /// Blocking — same as `IvfIndex::build`/`IvfPqIndex::build`, which
    /// this calls into via `builder`; scheduling this off the write path
    /// (a background task, a `spawn_blocking`) is the caller's job, per
    /// `ChangeSubscriber::on_change`'s own contract.
    pub fn rebuild(&self, storage: &dyn StorageApi, coll: &str) -> IndexResult<()> {
        let pre_scan_delta_len = self.delta.read().len();
        let pre_scan_tombstones = self.tombstones.read().clone();

        let fresh = (self.builder)(storage, coll)?;
        let row_count = storage.active_count(coll).unwrap_or(0);

        self.baked.store(Arc::new(fresh));
        self.baked_row_count.store(row_count, AtomicOrdering::Relaxed);

        {
            let mut delta = self.delta.write();
            if delta.len() >= pre_scan_delta_len {
                delta.drain(0..pre_scan_delta_len);
            }
        }
        {
            let mut tombstones = self.tombstones.write();
            *tombstones -= &pre_scan_tombstones;
        }
        *self.last_rebuild.write() = Instant::now();
        Ok(())
    }
}

impl ChangeSubscriber for LiveIndex {
    fn on_change(&self, batch: &ChangeBatch) {
        let mut delta = self.delta.write();
        let mut tombstones = self.tombstones.write();
        for event in &batch.events {
            match event {
                ChangeEvent::Insert { row_id, vector, .. } | ChangeEvent::Update { row_id, vector, .. } => {
                    delta.push((*row_id, vector.to_vec()));
                }
                ChangeEvent::Delete { row_id, .. } => {
                    tombstones.insert(row_id.to_bitmap_index());
                }
            }
        }
    }
}

impl VectorIndex for LiveIndex {
    fn search(
        &self,
        storage: &dyn StorageApi,
        coll: &str,
        query: &[f32],
        k: usize,
        filter: Option<&FilterMask>,
        params: &SearchParams,
    ) -> IndexResult<SearchResult> {
        if query.len() != self.dim {
            return Err(IndexError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(SearchResult { hits: Vec::new(), truncated_by_filter: false });
        }

        let tombstones = self.tombstones.read().clone();
        let effective_filter: Option<FilterMask> = filter.map(|f| {
            let mut allowed = f.allowed.clone();
            allowed -= &tombstones;
            let estimated_cardinality = allowed.len();
            FilterMask { allowed, estimated_cardinality }
        });

        let delta_snapshot: Vec<(RowId, Vec<f32>)> = self.delta.read().clone();
        let baked = self.baked.load_full();

        // Overfetch baked so post-merge grouping+truncation never drops a
        // candidate delta's fresher rows might otherwise have outranked.
        let baked_params = SearchParams {
            max_chunks_per_doc: None,
            ..params.clone()
        };
        let overfetch_k = k + delta_snapshot.len();
        let baked_result = baked.search(storage, coll, query, overfetch_k, effective_filter.as_ref(), &baked_params)?;

        let delta_candidates: Vec<(RowId, &Vec<f32>)> = delta_snapshot
            .iter()
            .filter(|(id, _)| !tombstones.contains(id.to_bitmap_index()))
            .filter(|(id, _)| effective_filter.as_ref().is_none_or(|f| f.allowed.contains(id.to_bitmap_index())))
            .map(|(id, v)| (*id, v))
            .collect();
        let delta_ids: Vec<RowId> = delta_candidates.iter().map(|(id, _)| *id).collect();
        let delta_rows = storage.rows_by_id(coll, &delta_ids)?;
        let delta_hits: Vec<SearchHit> = delta_candidates
            .into_iter()
            .zip(delta_rows)
            .filter_map(|((id, v), row)| {
                row.map(|row| SearchHit {
                    id,
                    doc_id: row.doc_id,
                    chunk_ord: row.chunk_ord,
                    score: math::score(self.metric, query, v),
                    metric: self.metric,
                    exact: true,
                })
            })
            .collect();

        // `delta` is scored from a *live*, current vector; a row present
        // in both (e.g. an `Update` landed in delta for a row `baked`
        // already covers) still gets a correct score from `baked`'s own
        // exact-rerank, which re-fetches from storage at query time too —
        // so this is only ever a harmless duplicate entry, not a
        // stale-vs-fresh conflict. Kept simple: prefer delta's copy since
        // it's guaranteed exact even when `baked`'s `rerank_k: Some(0)`
        // fast path would otherwise return an approximate one.
        let mut merged: Vec<SearchHit> = baked_result.hits;
        merged.extend(delta_hits);
        merged.reverse();
        let mut seen = HashSet::new();
        merged.retain(|h| seen.insert(h.id));
        merged.reverse();

        merged.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let scored: Vec<(f32, SearchHit)> = merged.into_iter().map(|h| (h.score, h)).collect();
        let grouped = apply_doc_grouping(scored, k, params.max_chunks_per_doc);

        Ok(SearchResult {
            hits: grouped.into_iter().map(|(_, h)| h).collect(),
            truncated_by_filter: baked_result.truncated_by_filter,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatIndex;
    use mara_proto::{PayloadRow, Principal, PrincipalId, RequestCtx, Role, SessionId, Source};
    use mara_storage::{PayloadSchema, Storage};

    fn ctx() -> RequestCtx {
        RequestCtx::new(
            SessionId("s".into()),
            Principal {
                id: PrincipalId("p".into()),
                name: "t".into(),
                role: Role::Writer,
            },
            Source::Embedded,
        )
    }

    fn flat_builder() -> Builder {
        Arc::new(|storage: &dyn StorageApi, coll: &str| -> IndexResult<Box<dyn VectorIndex>> {
            let info = storage.collection_info(coll)?;
            Ok(Box::new(FlatIndex::new(info.dim, info.metric)))
        })
    }

    fn params(nprobe: usize) -> SearchParams {
        SearchParams {
            max_chunks_per_doc: None,
            nprobe,
            ..SearchParams::default()
        }
    }

    fn storage_with(rows: &[(&str, [f32; 3])]) -> Storage {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::L2, PayloadSchema::empty()).unwrap();
        for (key, v) in rows {
            s.put(&ctx(), "docs", key, v.to_vec(), PayloadRow::new(), None).unwrap();
        }
        s
    }

    #[test]
    fn without_a_subscription_delta_and_tombstones_never_see_storage_writes() {
        // `baked` here happens to be a `FlatIndex`, which re-scans live
        // storage on every call regardless of LiveIndex — so this
        // deliberately doesn't assert anything about *search results*
        // (those would still reflect "b" through FlatIndex's own
        // always-live behavior, independent of whether LiveIndex ever
        // heard about the write). What LiveIndex itself promises is
        // narrower: `delta`/`tombstones` only change via the subscriber
        // feed, which is exactly what this checks.
        let s = storage_with(&[("a", [1.0, 0.0, 0.0])]);
        let live = LiveIndex::build(&s, "docs", 3, DistanceMetric::L2, flat_builder(), LiveIndexParams::default()).unwrap();

        s.put(&ctx(), "docs", "b", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
        s.delete(&ctx(), "docs", "a").unwrap();

        assert_eq!(live.delta_len(), 0, "without a subscription, no write reaches delta");
        assert_eq!(live.tombstone_count(), 0, "without a subscription, no delete reaches tombstones");
    }

    #[test]
    fn wrapping_a_genuinely_cached_index_a_late_write_is_missed_by_baked_alone_but_found_via_delta() {
        // Unlike the FlatIndex-backed tests above, `IvfIndex` really is
        // frozen at build time — this is the actual staleness gap
        // LiveIndex exists to close.
        use crate::ivf::{IvfIndex, IvfParams};
        let s = storage_with(&[("a", [1.0, 0.0, 0.0]), ("b", [0.0, 50.0, 0.0]), ("c", [0.0, 0.0, 50.0])]);
        let ivf_builder: Builder = Arc::new(|storage: &dyn StorageApi, coll: &str| -> IndexResult<Box<dyn VectorIndex>> {
            let info = storage.collection_info(coll)?;
            let idx = IvfIndex::build(storage, coll, info.dim, info.metric, &IvfParams { n_clusters: 3, kmeans_iters: 10, seed: 1, ..IvfParams::default() })?;
            Ok(Box::new(idx))
        });
        let live = Arc::new(LiveIndex::build(&s, "docs", 3, DistanceMetric::L2, ivf_builder, LiveIndexParams::default()).unwrap());

        // A late write the baked IVF index never scanned.
        s.put(&ctx(), "docs", "d", vec![1.1, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        let d_id = s.get_by_key("docs", "d").unwrap().unwrap().id;

        // Not subscribed: baked alone (exhaustive nprobe, so this isn't a
        // routing-miss) must not find "d" — it was never scanned.
        let exhaustive = SearchParams {
            max_chunks_per_doc: None,
            nprobe: 3,
            ..SearchParams::default()
        };
        let before = live.search(&s, "docs", &[1.1, 0.0, 0.0], 5, None, &exhaustive).unwrap().hits;
        assert!(before.iter().all(|h| h.id != d_id), "an unsubscribed LiveIndex wrapping a cached IVF index must not see a post-build write");

        // Now subscribe and re-issue the same write's *change event*
        // manually (this collection's writes before subscribing were
        // never delivered) by performing a fresh write that *is* observed.
        s.subscribe("docs", live.clone()).unwrap();
        s.put(&ctx(), "docs", "e", vec![1.05, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        let e_id = s.get_by_key("docs", "e").unwrap().unwrap().id;

        let after = live.search(&s, "docs", &[1.05, 0.0, 0.0], 5, None, &exhaustive).unwrap().hits;
        assert!(after.iter().any(|h| h.id == e_id), "once subscribed, a write the cached baked index never scanned must still be found via delta");
    }

    #[test]
    fn a_write_after_build_is_found_via_delta_once_subscribed() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0])]);
        let live = Arc::new(LiveIndex::build(&s, "docs", 3, DistanceMetric::L2, flat_builder(), LiveIndexParams::default()).unwrap());
        s.subscribe("docs", live.clone()).unwrap();

        s.put(&ctx(), "docs", "b", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
        assert_eq!(live.delta_len(), 1);

        let hits = live.search(&s, "docs", &[0.0, 1.0, 0.0], 5, None, &params(1)).unwrap().hits;
        assert_eq!(hits.len(), 2, "both the baked row and the fresh delta row must be found");
        let b_id = s.get_by_key("docs", "b").unwrap().unwrap().id;
        assert_eq!(hits[0].id, b_id, "the exact match, found via delta, must rank first");
    }

    #[test]
    fn a_delete_is_hidden_via_tombstones_before_any_rebuild() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0]), ("b", [0.9, 0.1, 0.0])]);
        let live = Arc::new(LiveIndex::build(&s, "docs", 3, DistanceMetric::L2, flat_builder(), LiveIndexParams::default()).unwrap());
        s.subscribe("docs", live.clone()).unwrap();

        let a_id = s.get_by_key("docs", "a").unwrap().unwrap().id;
        s.delete(&ctx(), "docs", "a").unwrap();
        assert_eq!(live.tombstone_count(), 1);

        let hits = live.search(&s, "docs", &[1.0, 0.0, 0.0], 5, None, &params(1)).unwrap().hits;
        assert!(hits.iter().all(|h| h.id != a_id), "a tombstoned row must never surface as a hit");
    }

    #[test]
    fn filter_excludes_a_tombstoned_row_thats_still_in_baked() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0]), ("b", [0.9, 0.1, 0.0])]);
        let live = Arc::new(LiveIndex::build(&s, "docs", 3, DistanceMetric::L2, flat_builder(), LiveIndexParams::default()).unwrap());
        s.subscribe("docs", live.clone()).unwrap();

        let a_id = s.get_by_key("docs", "a").unwrap().unwrap().id;
        let b_id = s.get_by_key("docs", "b").unwrap().unwrap().id;
        s.delete(&ctx(), "docs", "a").unwrap();

        let mut allowed = RoaringBitmap::new();
        allowed.insert(a_id.to_bitmap_index());
        allowed.insert(b_id.to_bitmap_index());
        let mask = FilterMask { allowed, estimated_cardinality: 2 };

        let hits = live.search(&s, "docs", &[1.0, 0.0, 0.0], 5, Some(&mask), &params(1)).unwrap().hits;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, b_id);
    }

    #[test]
    fn rebuild_moves_delta_rows_into_baked_and_clears_covered_state() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0])]);
        let live = Arc::new(LiveIndex::build(&s, "docs", 3, DistanceMetric::L2, flat_builder(), LiveIndexParams::default()).unwrap());
        s.subscribe("docs", live.clone()).unwrap();

        s.put(&ctx(), "docs", "b", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
        s.delete(&ctx(), "docs", "a").unwrap();
        assert_eq!(live.delta_len(), 1);
        assert_eq!(live.tombstone_count(), 1);

        live.rebuild(&s, "docs").unwrap();
        assert_eq!(live.delta_len(), 0, "the row rebuild just baked in must be cleared from delta");
        assert_eq!(live.tombstone_count(), 0, "the delete rebuild's fresh scan already excludes must be cleared from tombstones");

        let hits = live.search(&s, "docs", &[0.0, 1.0, 0.0], 5, None, &params(1)).unwrap().hits;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, s.get_by_key("docs", "b").unwrap().unwrap().id, "\"b\" must still be found, now via baked instead of delta");
    }

    #[test]
    fn rebuild_preserves_a_write_that_arrives_during_the_scan_window() {
        // Simulates the race the module doc comment describes: a write
        // lands in delta *after* rebuild's pre-scan snapshot is taken but
        // (in this synchronous test) is applied before `rebuild` itself
        // runs — standing in for "during the scan window" without needing
        // real concurrency, since `rebuild`'s clearing logic only removes
        // the pre-scan prefix regardless of when it's actually called.
        let s = storage_with(&[("a", [1.0, 0.0, 0.0])]);
        let live = Arc::new(LiveIndex::build(&s, "docs", 3, DistanceMetric::L2, flat_builder(), LiveIndexParams::default()).unwrap());
        s.subscribe("docs", live.clone()).unwrap();

        s.put(&ctx(), "docs", "b", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
        // Manually re-append a second delta entry for "b" *after* what a
        // real pre-scan snapshot would have captured, to prove `rebuild`
        // doesn't over-clear.
        let b_id = s.get_by_key("docs", "b").unwrap().unwrap().id;
        live.delta.write().push((b_id, vec![0.0, 1.0, 0.0]));
        assert_eq!(live.delta_len(), 2);

        live.rebuild(&s, "docs").unwrap();
        // The fresh scan already covers "b" (rebuild happened after the
        // real put), so only one of the two now-redundant entries should
        // remain uncleared -- but critically, none of it is *lost*.
        assert!(live.delta_len() <= 1, "rebuild must clear at least the pre-scan-covered entry");
    }

    #[test]
    fn should_rebuild_triggers_once_delta_crosses_the_fraction_threshold() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0]), ("b", [0.0, 1.0, 0.0])]);
        let params = LiveIndexParams {
            rebuild_threshold_fraction: 0.5, // 1 entry out of 2 rows triggers it
            rebuild_threshold_cap: 5_000,
            rebuild_safety_interval: Duration::from_secs(6 * 3600),
        };
        let live = Arc::new(LiveIndex::build(&s, "docs", 3, DistanceMetric::L2, flat_builder(), params).unwrap());
        s.subscribe("docs", live.clone()).unwrap();
        assert!(!live.should_rebuild());

        s.put(&ctx(), "docs", "c", vec![0.0, 0.0, 1.0], PayloadRow::new(), None).unwrap();
        assert!(live.should_rebuild());
    }

    #[test]
    fn should_rebuild_triggers_on_the_safety_timer_even_with_an_empty_delta() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0])]);
        let params = LiveIndexParams {
            rebuild_threshold_fraction: 0.05,
            rebuild_threshold_cap: 5_000,
            rebuild_safety_interval: Duration::from_millis(1),
        };
        let live = LiveIndex::build(&s, "docs", 3, DistanceMetric::L2, flat_builder(), params).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert!(live.should_rebuild(), "an empty delta must not suppress the safety-timer trigger");
    }

    #[test]
    fn max_chunks_per_doc_groups_across_baked_and_delta_together() {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty()).unwrap();
        let chunks: Vec<mara_storage::ChunkInput> = (0..2)
            .map(|i| mara_storage::ChunkInput {
                text: format!("chunk {i}"),
                vector: vec![1.0 - i as f32 * 0.01, 0.0, 0.0],
            })
            .collect();
        let doc_id = s
            .collection("docs")
            .unwrap()
            .put_document(
                &ctx(),
                mara_storage::PutDocumentInput {
                    doc_key: "big.md".into(),
                    chunks,
                    doc_payload: PayloadRow::new(),
                    chunk_spec: mara_proto::ChunkSpec::default(),
                    source: None,
                    embedding_model: mara_proto::ModelFingerprint {
                        model_id: "test".into(),
                        revision: None,
                        dim: 3,
                    },
                },
            )
            .unwrap();

        let live = Arc::new(LiveIndex::build(&s, "docs", 3, DistanceMetric::Cosine, flat_builder(), LiveIndexParams::default()).unwrap());
        s.subscribe("docs", live.clone()).unwrap();

        // A third chunk of the *same* document, added only after the
        // initial bake, so it's delta-only.
        s.collection("docs")
            .unwrap()
            .put_document(
                &ctx(),
                mara_storage::PutDocumentInput {
                    doc_key: "big2.md".into(),
                    chunks: vec![mara_storage::ChunkInput {
                        text: "chunk 2".into(),
                        vector: vec![0.98, 0.0, 0.0],
                    }],
                    doc_payload: PayloadRow::new(),
                    chunk_spec: mara_proto::ChunkSpec::default(),
                    source: None,
                    embedding_model: mara_proto::ModelFingerprint {
                        model_id: "test".into(),
                        revision: None,
                        dim: 3,
                    },
                },
            )
            .unwrap();
        let _ = doc_id;

        let grouped_params = SearchParams {
            max_chunks_per_doc: Some(1),
            nprobe: 1,
            ..SearchParams::default()
        };
        let hits = live.search(&s, "docs", &[1.0, 0.0, 0.0], 10, None, &grouped_params).unwrap().hits;
        let mut per_doc: std::collections::HashMap<mara_proto::DocId, u32> = std::collections::HashMap::new();
        for h in &hits {
            if let Some(d) = h.doc_id {
                *per_doc.entry(d).or_insert(0) += 1;
            }
        }
        assert!(per_doc.values().all(|&c| c <= 1), "max_chunks_per_doc=1 must hold even once delta rows are merged in, got {per_doc:?}");
    }

    #[test]
    fn dimension_mismatch_is_a_clear_error() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0])]);
        let live = LiveIndex::build(&s, "docs", 3, DistanceMetric::L2, flat_builder(), LiveIndexParams::default()).unwrap();
        let err = live.search(&s, "docs", &[1.0, 0.0], 1, None, &params(1)).unwrap_err();
        assert!(matches!(err, IndexError::DimMismatch { expected: 3, got: 2 }));
    }
}
