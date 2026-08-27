//! `LshIndex` — a selectable alternate vector index (master plan Layer 4,
//! *LSH*), deliberately **not** fused into IVF-PQ: the two solve
//! different problems with different lifecycles. LSH needs no training
//! ever (hyperplanes drawn once at build time from a seeded RNG); IVF-PQ
//! needs periodic k-means/OPQ retraining as data drifts. LSH's real role
//! is a zero-training index usable immediately after collection creation
//! (before any IVF-PQ build has run) and a recall/correctness cross-check
//! against it.
//!
//! SimHash: `num_tables` independent hash tables, each with its own
//! `num_hyperplanes` random hyperplanes drawn once at build time. A
//! vector's bucket key in a table is the sign bit of its dot product
//! against each of that table's hyperplanes, packed into one `u32` (so
//! `num_hyperplanes` is capped at 32) — the classic SimHash construction,
//! whose bucket-collision probability approximates cosine similarity
//! (the angle between two vectors), which is why the master plan calls it
//! "cosine-appropriate". Candidates are the *union* of every table's
//! matching bucket, deduplicated by `RowId` via `RoaringBitmap`, then
//! routed through the same exact-rerank shortlist logic (`finish_exact`)
//! `IvfPqIndex` uses — a bucket hit is only ever a candidate, never a
//! reported score.
//!
//! No multi-probe (checking neighboring buckets one bit-flip away):
//! independent multi-table diversity is the standard way LSH recovers
//! recall lost to bucket-boundary near-misses, and is what the master
//! plan's own description asks for ("candidates unioned/deduped across
//! tables") — multi-probing each table individually is a real, separate
//! recall/latency trade this crate doesn't take on.

use crate::error::{IndexError, IndexResult};
use crate::hit::SearchResult;
use crate::index_trait::VectorIndex;
use crate::ivf::{cluster_repr, scan_all};
use crate::math;
use crate::scoring::finish_exact;
use mara_proto::{DistanceMetric, RowId};
use mara_storage::{FilterMask, StorageApi};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use roaring::RoaringBitmap;
use std::collections::HashMap;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LshError {
    #[error("lsh_num_hyperplanes ({0}) must be in 1..=32 so one bucket signature fits a u32")]
    NumHyperplanesOutOfRange(usize),
}

#[derive(Clone, Debug)]
pub struct LshParams {
    pub num_tables: usize,
    pub num_hyperplanes: usize,
    pub seed: u64,
}

impl Default for LshParams {
    fn default() -> Self {
        LshParams {
            num_tables: 8,
            num_hyperplanes: 20,
            seed: 0,
        }
    }
}

#[derive(Debug)]
struct HashTable {
    hyperplanes: Vec<Vec<f32>>,
    buckets: HashMap<u32, RoaringBitmap>,
}

impl HashTable {
    fn bucket_key(&self, v: &[f32]) -> u32 {
        let mut key = 0u32;
        for (i, hp) in self.hyperplanes.iter().enumerate() {
            if math::dot(v, hp) >= 0.0 {
                key |= 1 << i;
            }
        }
        key
    }
}

#[derive(Debug)]
pub struct LshIndex {
    dim: usize,
    metric: DistanceMetric,
    tables: Vec<HashTable>,
}

impl LshIndex {
    pub fn build(storage: &dyn StorageApi, coll: &str, dim: usize, metric: DistanceMetric, params: &LshParams) -> IndexResult<Self> {
        if params.num_hyperplanes == 0 || params.num_hyperplanes > 32 {
            return Err(IndexError::from(LshError::NumHyperplanesOutOfRange(params.num_hyperplanes)));
        }

        let rows = scan_all(storage, coll)?;
        let vectored: Vec<(RowId, Vec<f32>)> = rows.iter().filter_map(|r| r.vector.as_deref().map(|v| (r.id, cluster_repr(metric, v)))).collect();

        let mut rng = ChaCha8Rng::seed_from_u64(params.seed);
        let mut tables = Vec::with_capacity(params.num_tables);
        for _ in 0..params.num_tables.max(1) {
            let hyperplanes: Vec<Vec<f32>> = (0..params.num_hyperplanes).map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect();
            let mut table = HashTable { hyperplanes, buckets: HashMap::new() };
            for (id, v) in &vectored {
                let key = table.bucket_key(v);
                table.buckets.entry(key).or_default().insert(id.to_bitmap_index());
            }
            tables.push(table);
        }

        Ok(LshIndex { dim, metric, tables })
    }

    pub fn num_tables(&self) -> usize {
        self.tables.len()
    }
}

impl VectorIndex for LshIndex {
    fn search(
        &self,
        storage: &dyn StorageApi,
        coll: &str,
        query: &[f32],
        k: usize,
        filter: Option<&FilterMask>,
        params: &crate::params::SearchParams,
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

        let query_repr = cluster_repr(self.metric, query);
        let mut candidates = RoaringBitmap::new();
        for table in &self.tables {
            let key = table.bucket_key(&query_repr);
            if let Some(bucket) = table.buckets.get(&key) {
                candidates |= bucket;
            }
        }
        if let Some(f) = filter {
            candidates &= &f.allowed;
        }

        let ids: Vec<RowId> = candidates.iter().map(RowId::from_bitmap_index).collect();
        let hits = finish_exact(storage, coll, self.metric, query, k, &ids, params.max_chunks_per_doc)?;
        Ok(SearchResult { hits, truncated_by_filter: false })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SearchParams;
    use mara_proto::{Filter, PayloadRow, Principal, PrincipalId, RequestCtx, Role, Scalar, SessionId, Source};
    use mara_storage::{PayloadSchema, Storage};
    use rand_chacha::ChaCha8Rng as TestRng;

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

    fn storage_with(rows: &[(&str, [f32; 3])], metric: DistanceMetric) -> Storage {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, metric, PayloadSchema::empty()).unwrap();
        for (key, v) in rows {
            s.put(&ctx(), "docs", key, v.to_vec(), PayloadRow::new(), None).unwrap();
        }
        s
    }

    fn params() -> SearchParams {
        SearchParams { max_chunks_per_doc: None, ..SearchParams::default() }
    }

    #[test]
    fn an_exact_match_query_is_always_found() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0]), ("b", [0.0, 1.0, 0.0]), ("c", [0.0, 0.0, 1.0])], DistanceMetric::Cosine);
        let lsh = LshIndex::build(&s, "docs", 3, DistanceMetric::Cosine, &LshParams::default()).unwrap();
        let hits = lsh.search(&s, "docs", &[1.0, 0.0, 0.0], 3, None, &params()).unwrap().hits;
        assert_eq!(hits[0].id, s.get_by_key("docs", "a").unwrap().unwrap().id, "a query identical to a stored vector must land in that vector's own bucket in every table");
    }

    #[test]
    fn well_separated_clusters_are_found_with_full_recall() {
        let mut rng = TestRng::seed_from_u64(1);
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty()).unwrap();
        let centers = [[10.0f32, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        for (ci, c) in centers.iter().enumerate() {
            for i in 0..10 {
                let v = [c[0] + rng.gen_range(-0.1..0.1), c[1] + rng.gen_range(-0.1..0.1), c[2] + rng.gen_range(-0.1..0.1)];
                s.put(&ctx(), "docs", &format!("k{ci}_{i}"), v.to_vec(), PayloadRow::new(), None).unwrap();
            }
        }
        let lsh = LshIndex::build(&s, "docs", 3, DistanceMetric::Cosine, &LshParams::default()).unwrap();
        let hits = lsh.search(&s, "docs", &[10.0, 0.0, 0.0], 10, None, &params()).unwrap().hits;
        assert_eq!(hits.len(), 10, "all 10 points from a tightly-clustered, well-separated group must be found");
        assert!(hits.iter().all(|h| h.exact));
    }

    #[test]
    fn more_tables_never_reduces_the_candidate_pool() {
        let mut rng = TestRng::seed_from_u64(2);
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty()).unwrap();
        for i in 0..60 {
            let v = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
            s.put(&ctx(), "docs", &format!("k{i}"), v.to_vec(), PayloadRow::new(), None).unwrap();
        }
        let few = LshIndex::build(&s, "docs", 3, DistanceMetric::Cosine, &LshParams { num_tables: 1, num_hyperplanes: 20, seed: 3 }).unwrap();
        let many = LshIndex::build(&s, "docs", 3, DistanceMetric::Cosine, &LshParams { num_tables: 16, num_hyperplanes: 20, seed: 3 }).unwrap();

        let query = [0.5, 0.5, 0.5];
        let few_hits = few.search(&s, "docs", &query, 60, None, &params()).unwrap().hits.len();
        let many_hits = many.search(&s, "docs", &query, 60, None, &params()).unwrap().hits.len();
        assert!(many_hits >= few_hits, "unioning more independent tables must never find fewer candidates, got {few_hits} (1 table) vs {many_hits} (16 tables)");
    }

    #[test]
    fn dimension_mismatch_is_a_clear_error() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0])], DistanceMetric::Cosine);
        let lsh = LshIndex::build(&s, "docs", 3, DistanceMetric::Cosine, &LshParams::default()).unwrap();
        let err = lsh.search(&s, "docs", &[1.0, 0.0], 1, None, &params()).unwrap_err();
        assert!(matches!(err, IndexError::DimMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn empty_collection_builds_and_searches_without_panicking() {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty()).unwrap();
        let lsh = LshIndex::build(&s, "docs", 3, DistanceMetric::Cosine, &LshParams::default()).unwrap();
        let hits = lsh.search(&s, "docs", &[1.0, 0.0, 0.0], 5, None, &params()).unwrap().hits;
        assert!(hits.is_empty());
    }

    #[test]
    fn num_hyperplanes_out_of_range_is_a_clear_error() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0])], DistanceMetric::Cosine);
        let err = LshIndex::build(&s, "docs", 3, DistanceMetric::Cosine, &LshParams { num_tables: 4, num_hyperplanes: 33, seed: 1 }).unwrap_err();
        assert!(matches!(err, IndexError::Lsh(LshError::NumHyperplanesOutOfRange(33))));

        let err = LshIndex::build(&s, "docs", 3, DistanceMetric::Cosine, &LshParams { num_tables: 4, num_hyperplanes: 0, seed: 1 }).unwrap_err();
        assert!(matches!(err, IndexError::Lsh(LshError::NumHyperplanesOutOfRange(0))));
    }

    #[test]
    fn filter_mask_excludes_non_matching_rows() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0]), ("b", [0.9, 0.1, 0.0])], DistanceMetric::Cosine);
        let id_a = s.get_by_key("docs", "a").unwrap().unwrap().id;
        let mut allowed = RoaringBitmap::new();
        allowed.insert(id_a.to_bitmap_index());
        let mask = FilterMask { allowed, estimated_cardinality: 1 };

        let lsh = LshIndex::build(&s, "docs", 3, DistanceMetric::Cosine, &LshParams::default()).unwrap();
        let hits = lsh.search(&s, "docs", &[1.0, 0.0, 0.0], 5, Some(&mask), &params()).unwrap().hits;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id_a);
    }

    #[test]
    fn compiled_filter_mask_from_storage_round_trips_through_search() {
        let s = Storage::new();
        let schema = PayloadSchema::builder().field("tag", mara_storage::FieldType::Keyword).build();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, schema).unwrap();
        let mut fields_a = PayloadRow::new();
        fields_a.insert("tag".into(), mara_proto::PayloadValue::Keyword("keep".into()));
        s.put(&ctx(), "docs", "a", vec![1.0, 0.0, 0.0], fields_a, None).unwrap();
        let mut fields_b = PayloadRow::new();
        fields_b.insert("tag".into(), mara_proto::PayloadValue::Keyword("drop".into()));
        s.put(&ctx(), "docs", "b", vec![0.9, 0.1, 0.0], fields_b, None).unwrap();

        let mask = s
            .compile_filter("docs", &Filter::Eq { field: "tag".into(), value: Scalar::Str("keep".into()) })
            .unwrap();
        let lsh = LshIndex::build(&s, "docs", 3, DistanceMetric::Cosine, &LshParams::default()).unwrap();
        let hits = lsh.search(&s, "docs", &[1.0, 0.0, 0.0], 5, Some(&mask), &params()).unwrap().hits;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, s.get_by_key("docs", "a").unwrap().unwrap().id);
    }

    #[test]
    fn max_chunks_per_doc_caps_hits_from_one_document() {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty()).unwrap();
        let chunks: Vec<mara_storage::ChunkInput> = (0..5)
            .map(|i| mara_storage::ChunkInput {
                text: format!("chunk {i}"),
                vector: vec![1.0 - i as f32 * 0.01, 0.0, 0.0],
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
                        dim: 3,
                    },
                },
            )
            .unwrap();

        let lsh = LshIndex::build(&s, "docs", 3, DistanceMetric::Cosine, &LshParams::default()).unwrap();
        let hits = lsh
            .search(&s, "docs", &[1.0, 0.0, 0.0], 10, None, &SearchParams { max_chunks_per_doc: Some(2), ..SearchParams::default() })
            .unwrap()
            .hits;
        assert_eq!(hits.len(), 2, "only 2 of the document's 5 chunks may appear");
    }

    #[test]
    fn the_same_seed_always_produces_the_same_buckets() {
        let s = storage_with(&[("a", [1.0, 0.2, -0.3]), ("b", [0.5, -0.7, 0.1])], DistanceMetric::Cosine);
        let a = LshIndex::build(&s, "docs", 3, DistanceMetric::Cosine, &LshParams { num_tables: 4, num_hyperplanes: 10, seed: 42 }).unwrap();
        let b = LshIndex::build(&s, "docs", 3, DistanceMetric::Cosine, &LshParams { num_tables: 4, num_hyperplanes: 10, seed: 42 }).unwrap();
        let query = [0.3, 0.1, -0.2];
        let hits_a = a.search(&s, "docs", &query, 5, None, &params()).unwrap();
        let hits_b = b.search(&s, "docs", &query, 5, None, &params()).unwrap();
        assert_eq!(hits_a.hits, hits_b.hits, "the same seed must reproduce identical hyperplanes and therefore identical results");
    }
}
