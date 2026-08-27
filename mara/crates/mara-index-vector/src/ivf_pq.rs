//! `IvfPqIndex`: `IvfIndex`'s coarse routing plus PQ/ADC candidate
//! scoring and an exact-rerank stage (master plan Layer 4, *IVF-PQ +
//! OPQ + beam search*, PQ half — OPQ's rotation is the next build step).
//! Search shape: route to `nprobe` clusters exactly as `IvfIndex` does,
//! rank that candidate set cheaply via ADC (`O(m)`/candidate), then
//! exact-score only the top `rerank_k` against their real stored vectors
//! before returning — so a normal search (`rerank_k > 0`, the default)
//! never reports an approximate distance mislabeled as a real score, per
//! the master plan's cosine-correctness invariant.

use crate::error::{IndexError, IndexResult};
use crate::grouping::apply_doc_grouping;
use crate::hit::{SearchHit, SearchResult};
use crate::index_trait::VectorIndex;
use crate::ivf::{cluster_repr, scan_all, IvfIndex, IvfParams};
use crate::opq::{train_opq, OpqParams, OpqRotation};
use crate::params::SearchParams;
use crate::pq::{PqCodebook, PqParams};
use crate::scoring::finish_exact;
use mara_proto::{DistanceMetric, Row, RowId};
use mara_storage::{FilterMask, StorageApi};
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug)]
pub struct IvfPqIndex {
    dim: usize,
    metric: DistanceMetric,
    /// `Some` only when built via `build_with_opq` — see the master
    /// plan's `opq_enabled` toggle. Applied, when present, after
    /// `cluster_repr` and before everything downstream (coarse
    /// clustering, PQ training/encoding, query routing/ADC) — never
    /// touching the exact-rerank stage, which always scores real,
    /// unrotated vectors.
    rotation: Option<OpqRotation>,
    coarse: IvfIndex,
    codebook: PqCodebook,
    codes: HashMap<RowId, Vec<u8>>,
}

/// A best-effort score for the `rerank_k == 0` "max-speed, lower-accuracy"
/// mode (see the master plan's *Exact rerank*), where no real vector is
/// ever fetched — only ADC's approximate squared-L2-in-`cluster_repr`-
/// space distance is available. `L2` and `Cosine` both have a clean,
/// principled conversion (`Cosine` via the unit-sphere identity
/// `||u-v||^2 = 2 - 2cos(u,v)` that `cluster_repr` sets up); `DotProduct`
/// does not — `cluster_repr` leaves those vectors unnormalized, so there's
/// no closed-form mapping from squared-L2 to inner product, and negated
/// distance is used as a directional proxy only. This entire function is
/// skipped whenever `rerank_k > 0` (the default), which always scores
/// from the real vector instead.
fn adc_score_estimate(metric: DistanceMetric, sq_dist: f32) -> f32 {
    match metric {
        DistanceMetric::L2 | DistanceMetric::DotProduct => -sq_dist,
        DistanceMetric::Cosine => 1.0 - sq_dist / 2.0,
    }
}

impl IvfPqIndex {
    /// Scans `coll` once, building both the coarse IVF structure and PQ
    /// codebooks/codes from the same scanned rows — see
    /// `IvfIndex::build_from_rows`, which this reuses rather than
    /// scanning storage a second time. No OPQ preprocessing; see
    /// `build_with_opq` for that.
    pub fn build(
        storage: &dyn StorageApi,
        coll: &str,
        dim: usize,
        metric: DistanceMetric,
        ivf_params: &IvfParams,
        pq_params: &PqParams,
    ) -> IndexResult<Self> {
        Self::build_opt(storage, coll, dim, metric, ivf_params, pq_params, None)
    }

    /// Same as `build`, but with OPQ's orthogonal rotation trained first
    /// and applied as one global preprocessing step before *both* coarse
    /// clustering and PQ training/encoding — per the master plan's
    /// `opq_enabled` toggle, meant to be benchmarked against plain
    /// `build` (see `benches/recall.rs`), not assumed to win.
    pub fn build_with_opq(
        storage: &dyn StorageApi,
        coll: &str,
        dim: usize,
        metric: DistanceMetric,
        ivf_params: &IvfParams,
        opq_params: &OpqParams,
    ) -> IndexResult<Self> {
        Self::build_opt(storage, coll, dim, metric, ivf_params, &opq_params.pq, Some(opq_params))
    }

    fn build_opt(
        storage: &dyn StorageApi,
        coll: &str,
        dim: usize,
        metric: DistanceMetric,
        ivf_params: &IvfParams,
        pq_params: &PqParams,
        opq_params: Option<&OpqParams>,
    ) -> IndexResult<Self> {
        let rows = scan_all(storage, coll)?;
        let base: Vec<(RowId, Vec<f32>)> = rows
            .iter()
            .filter_map(|r| r.vector.as_deref().map(|v| (r.id, cluster_repr(metric, v))))
            .collect();

        let (rotation, codebook, final_repr) = match opq_params {
            Some(opq_params) => {
                let plain: Vec<Vec<f32>> = base.iter().map(|(_, v)| v.clone()).collect();
                let (rotation, codebook) = train_opq(&plain, dim, opq_params)?;
                let rotated: Vec<(RowId, Vec<f32>)> = base.into_iter().map(|(id, v)| (id, rotation.apply(&v))).collect();
                (Some(rotation), codebook, rotated)
            }
            None => {
                let plain: Vec<Vec<f32>> = base.iter().map(|(_, v)| v.clone()).collect();
                // Trained even on an empty `plain` — `PqCodebook::train`
                // validates `m`/`ksub` against `dim` regardless of
                // whether there's data yet (a config mistake should
                // surface at build time, not stay latent until the first
                // row lands), and degrades to empty per-subspace
                // codebooks that `encode` is simply never called against
                // below.
                let codebook = PqCodebook::train(&plain, dim, pq_params)?;
                (None, codebook, base)
            }
        };

        let coarse = IvfIndex::build_from_repr(&final_repr, dim, metric, ivf_params);
        let codes: HashMap<RowId, Vec<u8>> = final_repr.into_iter().map(|(id, v)| (id, codebook.encode(&v))).collect();

        Ok(IvfPqIndex {
            dim,
            metric,
            rotation,
            coarse,
            codebook,
            codes,
        })
    }

    pub fn n_clusters(&self) -> usize {
        self.coarse.n_clusters()
    }

    pub fn pq_m(&self) -> usize {
        self.codebook.m()
    }

    pub fn opq_enabled(&self) -> bool {
        self.rotation.is_some()
    }
}

impl VectorIndex for IvfPqIndex {
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
        if k == 0 || self.coarse.n_clusters() == 0 {
            return Ok(SearchResult { hits: Vec::new(), truncated_by_filter: false });
        }

        let mut query_repr = cluster_repr(self.metric, query);
        if let Some(rotation) = &self.rotation {
            query_repr = rotation.apply(&query_repr);
        }
        let (candidates, truncated_by_filter) = self.coarse.resolve_candidates(storage, coll, &query_repr, params, filter, k)?;

        let adc_table = self.codebook.build_adc_table(&query_repr);
        // Ascending: ADC produces a *distance* (lower is better), unlike
        // every other ranking in this crate, which sorts descending by a
        // higher-is-better score — kept local to this function so that
        // distinction never leaks into `SearchHit::score`.
        let mut by_adc: Vec<(RowId, f32)> = candidates
            .iter()
            .map(RowId::from_bitmap_index)
            .filter_map(|id| self.codes.get(&id).map(|code| (id, adc_table.distance(code))))
            .collect();
        by_adc.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let target_rerank_k = params.rerank_k.unwrap_or_else(|| (20 * k).max(200));

        let hits = if target_rerank_k == 0 {
            let scored: Vec<(f32, RowId, f32)> = by_adc.into_iter().take(k).map(|(id, dist)| (adc_score_estimate(self.metric, dist), id, dist)).collect();
            let ids: Vec<RowId> = scored.iter().map(|(_, id, _)| *id).collect();
            let rows = storage.rows_by_id(coll, &ids)?;
            let mut scored: Vec<(f32, Row)> = scored
                .into_iter()
                .zip(rows)
                .filter_map(|((score, _, _), row)| row.map(|r| (score, r)))
                .collect();
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            let grouped = apply_doc_grouping(scored, k, params.max_chunks_per_doc);
            grouped
                .into_iter()
                .map(|(score, row)| SearchHit {
                    id: row.id,
                    doc_id: row.doc_id,
                    chunk_ord: row.chunk_ord,
                    score,
                    metric: self.metric,
                    exact: false,
                })
                .collect()
        } else {
            let shortlist: Vec<RowId> = by_adc.into_iter().take(target_rerank_k).map(|(id, _)| id).collect();
            finish_exact(storage, coll, self.metric, query, k, &shortlist, params.max_chunks_per_doc)?
        };

        Ok(SearchResult { hits, truncated_by_filter })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatIndex;
    use mara_proto::{Filter, PayloadRow, Principal, PrincipalId, RequestCtx, Role, Scalar, SessionId, Source};
    use mara_storage::{PayloadSchema, Storage};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use roaring::RoaringBitmap;

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

    fn default_params(nprobe: usize) -> SearchParams {
        SearchParams {
            max_chunks_per_doc: None,
            nprobe,
            ..SearchParams::default()
        }
    }

    fn synthetic_collection(dim: usize, n_clusters: usize, per_cluster: usize, seed: u64) -> Storage {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", dim, DistanceMetric::L2, PayloadSchema::empty()).unwrap();
        let mut i = 0;
        for c in 0..n_clusters {
            let center: Vec<f32> = (0..dim).map(|d| if d == c % dim { 50.0 } else { 0.0 }).collect();
            for _ in 0..per_cluster {
                let v: Vec<f32> = center.iter().map(|x| x + rng.gen_range(-1.0..1.0)).collect();
                s.put(&ctx(), "docs", &format!("k{i}"), v, PayloadRow::new(), None).unwrap();
                i += 1;
            }
        }
        s
    }

    #[test]
    fn exact_rerank_default_recovers_flat_index_top_hit_for_a_centered_query() {
        let dim = 8;
        let s = synthetic_collection(dim, 4, 20, 1);
        let ivf_pq = IvfPqIndex::build(
            &s,
            "docs",
            dim,
            DistanceMetric::L2,
            &IvfParams { n_clusters: 4, kmeans_iters: 15, seed: 2, ..IvfParams::default() },
            &PqParams { m: 4, ksub: 16, kmeans_iters: 10, seed: 3 },
        )
        .unwrap();
        let flat = FlatIndex::new(dim, DistanceMetric::L2);

        let mut query = vec![0.0; dim];
        query[0] = 50.0;
        let k = 3;
        let pq_hits = ivf_pq.search(&s, "docs", &query, k, None, &default_params(4)).unwrap().hits;
        let flat_hits = flat.search(&s, "docs", &query, k, None, &default_params(0)).unwrap().hits;

        assert!(pq_hits.iter().all(|h| h.exact), "default rerank_k must always exact-score");
        assert_eq!(pq_hits[0].id, flat_hits[0].id, "the nearest true neighbor must survive ADC shortlisting + exact rerank");
    }

    #[test]
    fn build_with_opq_still_recovers_the_true_nearest_neighbor() {
        let dim = 8;
        let s = synthetic_collection(dim, 4, 20, 9);
        let ivf_pq = IvfPqIndex::build_with_opq(
            &s,
            "docs",
            dim,
            DistanceMetric::L2,
            &IvfParams { n_clusters: 4, kmeans_iters: 15, seed: 10, ..IvfParams::default() },
            &OpqParams {
                pq: PqParams { m: 4, ksub: 16, kmeans_iters: 10, seed: 11 },
                iterations: 5,
            },
        )
        .unwrap();
        assert!(ivf_pq.opq_enabled());
        let flat = FlatIndex::new(dim, DistanceMetric::L2);

        let mut query = vec![0.0; dim];
        query[0] = 50.0;
        let k = 3;
        let pq_hits = ivf_pq.search(&s, "docs", &query, k, None, &default_params(4)).unwrap().hits;
        let flat_hits = flat.search(&s, "docs", &query, k, None, &default_params(0)).unwrap().hits;

        assert!(pq_hits.iter().all(|h| h.exact));
        assert_eq!(pq_hits[0].id, flat_hits[0].id, "OPQ's rotation must not break exact-rerank correctness — same invariant as the non-OPQ path");

        // The exact-rerank score is computed from the real, unrotated
        // vector regardless of OPQ — this is the master plan's
        // cosine-correctness invariant applied to L2: the reported score
        // must match FlatIndex's exactly, not a rotated-space distance.
        assert!((pq_hits[0].score - flat_hits[0].score).abs() < 1e-3);
    }

    #[test]
    fn rerank_k_zero_returns_fast_inexact_results() {
        let dim = 8;
        let s = synthetic_collection(dim, 2, 15, 5);
        let ivf_pq = IvfPqIndex::build(
            &s,
            "docs",
            dim,
            DistanceMetric::L2,
            &IvfParams { n_clusters: 2, kmeans_iters: 15, seed: 6, ..IvfParams::default() },
            &PqParams { m: 4, ksub: 16, kmeans_iters: 10, seed: 7 },
        )
        .unwrap();

        let mut query = vec![0.0; dim];
        query[0] = 50.0;
        let hits = ivf_pq
            .search(
                &s,
                "docs",
                &query,
                5,
                None,
                &SearchParams {
                    max_chunks_per_doc: None,
                    nprobe: 2,
                    ..default_params(2)
                },
            )
            .unwrap().hits;
        // rerank_k defaults to None (dynamic); explicitly request 0 via a
        // second call using the params struct's rerank_k field directly.
        let hits_no_rerank = ivf_pq
            .search(
                &s,
                "docs",
                &query,
                5,
                None,
                &SearchParams {
                    max_chunks_per_doc: None,
                    nprobe: 2,
                    rerank_k: Some(0),
                    ..SearchParams::default()
                },
            )
            .unwrap().hits;
        assert!(!hits_no_rerank.is_empty());
        assert!(hits_no_rerank.iter().all(|h| !h.exact), "rerank_k=0 must mark hits inexact");
        assert!(!hits.is_empty());
    }

    #[test]
    fn m_not_dividing_dim_surfaces_as_a_build_error() {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 10, DistanceMetric::Cosine, PayloadSchema::empty()).unwrap();
        s.put(&ctx(), "docs", "a", vec![1.0; 10], PayloadRow::new(), None).unwrap();
        let err = IvfPqIndex::build(
            &s,
            "docs",
            10,
            DistanceMetric::Cosine,
            &IvfParams::default(),
            &PqParams { m: 3, ksub: 4, kmeans_iters: 5, seed: 1 },
        )
        .unwrap_err();
        assert!(matches!(err, IndexError::Pq(_)));
    }

    #[test]
    fn dimension_mismatch_is_a_clear_error() {
        let s = synthetic_collection(4, 1, 5, 1);
        let ivf_pq = IvfPqIndex::build(
            &s,
            "docs",
            4,
            DistanceMetric::L2,
            &IvfParams { n_clusters: 1, kmeans_iters: 5, seed: 1, ..IvfParams::default() },
            &PqParams { m: 2, ksub: 4, kmeans_iters: 5, seed: 1 },
        )
        .unwrap();
        let err = ivf_pq.search(&s, "docs", &[1.0, 0.0], 1, None, &default_params(1)).unwrap_err();
        assert!(matches!(err, IndexError::DimMismatch { expected: 4, got: 2 }));
    }

    #[test]
    fn empty_collection_builds_and_searches_without_panicking() {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 8, DistanceMetric::Cosine, PayloadSchema::empty()).unwrap();
        let ivf_pq = IvfPqIndex::build(
            &s,
            "docs",
            8,
            DistanceMetric::Cosine,
            &IvfParams::default(),
            &PqParams { m: 4, ksub: 16, kmeans_iters: 5, seed: 1 },
        )
        .unwrap();
        assert_eq!(ivf_pq.n_clusters(), 0);
        let hits = ivf_pq.search(&s, "docs", &[0.0; 8], 5, None, &default_params(4)).unwrap().hits;
        assert!(hits.is_empty());
    }

    #[test]
    fn filter_mask_excludes_non_matching_rows() {
        let s = synthetic_collection(8, 1, 10, 2);
        let id_first = s.get_by_key("docs", "k0").unwrap().unwrap().id;
        let mut allowed = RoaringBitmap::new();
        allowed.insert(id_first.to_bitmap_index());
        let mask = FilterMask {
            allowed: allowed.clone(),
            estimated_cardinality: allowed.len(),
        };

        let ivf_pq = IvfPqIndex::build(
            &s,
            "docs",
            8,
            DistanceMetric::L2,
            &IvfParams { n_clusters: 1, kmeans_iters: 5, seed: 1, ..IvfParams::default() },
            &PqParams { m: 4, ksub: 16, kmeans_iters: 5, seed: 1 },
        )
        .unwrap();
        let hits = ivf_pq.search(&s, "docs", &[50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5, Some(&mask), &default_params(1)).unwrap().hits;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id_first);
    }

    #[test]
    fn compiled_filter_mask_from_storage_round_trips_through_search() {
        let s = Storage::new();
        let schema = PayloadSchema::builder().field("tag", mara_storage::FieldType::Keyword).build();
        s.create_collection(&ctx(), "docs", 4, DistanceMetric::Cosine, schema).unwrap();
        let mut fields_a = PayloadRow::new();
        fields_a.insert("tag".into(), mara_proto::PayloadValue::Keyword("keep".into()));
        s.put(&ctx(), "docs", "a", vec![1.0, 0.0, 0.0, 0.0], fields_a, None).unwrap();
        let mut fields_b = PayloadRow::new();
        fields_b.insert("tag".into(), mara_proto::PayloadValue::Keyword("drop".into()));
        s.put(&ctx(), "docs", "b", vec![0.9, 0.1, 0.0, 0.0], fields_b, None).unwrap();

        let mask = s
            .compile_filter("docs", &Filter::Eq { field: "tag".into(), value: Scalar::Str("keep".into()) })
            .unwrap();
        let ivf_pq = IvfPqIndex::build(
            &s,
            "docs",
            4,
            DistanceMetric::Cosine,
            &IvfParams { n_clusters: 1, kmeans_iters: 5, seed: 1, ..IvfParams::default() },
            &PqParams { m: 2, ksub: 4, kmeans_iters: 5, seed: 1 },
        )
        .unwrap();
        let hits = ivf_pq.search(&s, "docs", &[1.0, 0.0, 0.0, 0.0], 5, Some(&mask), &default_params(1)).unwrap().hits;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, s.get_by_key("docs", "a").unwrap().unwrap().id);
    }

    #[test]
    fn max_chunks_per_doc_caps_hits_from_one_document() {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 4, DistanceMetric::Cosine, PayloadSchema::empty()).unwrap();
        let chunks: Vec<mara_storage::ChunkInput> = (0..5)
            .map(|i| mara_storage::ChunkInput {
                text: format!("chunk {i}"),
                vector: vec![1.0 - i as f32 * 0.01, 0.0, 0.0, 0.0],
            })
            .collect();
        s.collection("docs")
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
                        dim: 4,
                    },
                },
            )
            .unwrap();

        let ivf_pq = IvfPqIndex::build(
            &s,
            "docs",
            4,
            DistanceMetric::Cosine,
            &IvfParams { n_clusters: 1, kmeans_iters: 5, seed: 1, ..IvfParams::default() },
            &PqParams { m: 2, ksub: 4, kmeans_iters: 5, seed: 1 },
        )
        .unwrap();
        let hits = ivf_pq
            .search(
                &s,
                "docs",
                &[1.0, 0.0, 0.0, 0.0],
                10,
                None,
                &SearchParams {
                    max_chunks_per_doc: Some(2),
                    nprobe: 1,
                    ..default_params(1)
                },
            )
            .unwrap().hits;
        assert_eq!(hits.len(), 2, "only 2 of the document's 5 chunks may appear");
    }
}
