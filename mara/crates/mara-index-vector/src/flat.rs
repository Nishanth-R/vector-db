//! The brute-force exact index. Not a last resort — the *default* at this
//! project's scale (see the master plan's *Flat/exact fallback*): a full
//! SIMD+`rayon` scan over a few hundred thousand rows is single-digit
//! milliseconds, which can match or beat IVF-PQ's routing+rerank overhead.
//! This is also the recall oracle every later approximate index gets
//! validated against, filtered and unfiltered — so it deliberately keeps
//! no baked state of its own: every `search` call re-scans storage fresh,
//! trading a little redundant work for zero staleness and zero rebuild
//! logic to get wrong.

use crate::error::{IndexError, IndexResult};
use crate::hit::SearchHit;
use crate::index_trait::VectorIndex;
use crate::math;
use crate::params::SearchParams;
use mara_proto::{DistanceMetric, DocId, Row};
use mara_storage::{FilterMask, StorageApi, StorageResult};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;

const SCAN_PAGE_SIZE: usize = 10_000;

pub struct FlatIndex {
    pub dim: usize,
    pub metric: DistanceMetric,
}

impl FlatIndex {
    pub fn new(dim: usize, metric: DistanceMetric) -> Self {
        FlatIndex { dim, metric }
    }

    /// Always higher-is-better, in `self.metric`'s convention — for `L2`
    /// that means the caller sees `-squared_distance`, never a raw
    /// distance a naive caller might mistake for "lower is better".
    fn score(&self, query: &[f32], vector: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => math::cosine(query, vector),
            DistanceMetric::L2 => -math::l2_sq(query, vector),
            DistanceMetric::DotProduct => math::dot(query, vector),
        }
    }
}

fn scan_all(storage: &dyn StorageApi, coll: &str) -> StorageResult<Vec<Row>> {
    let mut out = Vec::new();
    let mut cursor = None;
    loop {
        let page = storage.scan(coll, cursor, SCAN_PAGE_SIZE)?;
        let page_len = page.len();
        cursor = page.last().map(|r| r.id).or(cursor);
        out.extend(page);
        if page_len < SCAN_PAGE_SIZE {
            break;
        }
    }
    Ok(out)
}

/// Caps how many hits from the same document survive, in score order —
/// applied once here so every `VectorIndex` implementation can share it
/// rather than reimplementing the same grouping logic.
fn apply_doc_grouping(mut ranked: Vec<(f32, &Row)>, k: usize, max_chunks_per_doc: Option<u32>) -> Vec<(f32, &Row)> {
    if let Some(max) = max_chunks_per_doc {
        let mut per_doc: HashMap<DocId, u32> = HashMap::new();
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

impl VectorIndex for FlatIndex {
    fn search(
        &self,
        storage: &dyn StorageApi,
        coll: &str,
        query: &[f32],
        k: usize,
        filter: Option<&FilterMask>,
        params: &SearchParams,
    ) -> IndexResult<Vec<SearchHit>> {
        if query.len() != self.dim {
            return Err(IndexError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let rows = scan_all(storage, coll)?;

        let mut scored: Vec<(f32, &Row)> = rows
            .par_iter()
            .filter(|row| filter.is_none_or(|f| f.contains(row.id)))
            .filter_map(|row| row.vector.as_deref().map(|v| (self.score(query, v), row)))
            .collect();

        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let grouped = apply_doc_grouping(scored, k, params.max_chunks_per_doc);
        Ok(grouped
            .into_iter()
            .map(|(score, row)| SearchHit {
                id: row.id,
                doc_id: row.doc_id,
                chunk_ord: row.chunk_ord,
                score,
                metric: self.metric,
                exact: true,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mara_proto::{Filter, PayloadRow, Principal, PrincipalId, RequestCtx, Role, Scalar, SessionId, Source};
    use mara_storage::{ChunkInput, PayloadSchema, PutDocumentInput, Storage};

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

    fn storage_with(rows: &[(&str, [f32; 3])]) -> Storage {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty())
            .unwrap();
        for (key, v) in rows {
            s.put(&ctx(), "docs", key, v.to_vec(), PayloadRow::new(), None).unwrap();
        }
        s
    }

    #[test]
    fn nearest_neighbor_ranks_first_under_cosine() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0]), ("b", [0.0, 1.0, 0.0]), ("c", [0.9, 0.1, 0.0])]);
        let idx = FlatIndex::new(3, DistanceMetric::Cosine);
        let hits = idx
            .search(&s, "docs", &[1.0, 0.0, 0.0], 3, None, &SearchParams { max_chunks_per_doc: None })
            .unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(s.get_by_key("docs", "a").unwrap().unwrap().id, hits[0].id, "exact match must rank first");
        assert!(hits[0].exact);
        assert!(hits.windows(2).all(|w| w[0].score >= w[1].score), "must be sorted descending by score");
    }

    #[test]
    fn l2_metric_ranks_closest_point_first() {
        let s = storage_with(&[("far", [10.0, 10.0, 10.0]), ("near", [1.1, 0.0, 0.0]), ("mid", [3.0, 0.0, 0.0])]);
        let idx = FlatIndex::new(3, DistanceMetric::L2);
        let hits = idx
            .search(&s, "docs", &[1.0, 0.0, 0.0], 1, None, &SearchParams { max_chunks_per_doc: None })
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(s.get_by_key("docs", "near").unwrap().unwrap().id, hits[0].id);
    }

    #[test]
    fn dot_product_favors_larger_magnitude_in_the_same_direction() {
        let s = storage_with(&[("small", [1.0, 0.0, 0.0]), ("large", [5.0, 0.0, 0.0])]);
        let idx = FlatIndex::new(3, DistanceMetric::DotProduct);
        let hits = idx
            .search(&s, "docs", &[1.0, 0.0, 0.0], 2, None, &SearchParams { max_chunks_per_doc: None })
            .unwrap();
        assert_eq!(s.get_by_key("docs", "large").unwrap().unwrap().id, hits[0].id);
    }

    #[test]
    fn dimension_mismatch_is_a_clear_error() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0])]);
        let idx = FlatIndex::new(3, DistanceMetric::Cosine);
        let err = idx
            .search(&s, "docs", &[1.0, 0.0], 1, None, &SearchParams::default())
            .unwrap_err();
        assert!(matches!(err, IndexError::DimMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn k_larger_than_the_collection_returns_everything_without_panicking() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0]), ("b", [0.0, 1.0, 0.0])]);
        let idx = FlatIndex::new(3, DistanceMetric::Cosine);
        let hits = idx
            .search(&s, "docs", &[1.0, 0.0, 0.0], 100, None, &SearchParams { max_chunks_per_doc: None })
            .unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn empty_collection_returns_empty_without_panicking() {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty())
            .unwrap();
        let idx = FlatIndex::new(3, DistanceMetric::Cosine);
        let hits = idx.search(&s, "docs", &[1.0, 0.0, 0.0], 5, None, &SearchParams::default()).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn filter_mask_excludes_non_matching_rows() {
        let s = storage_with(&[("a", [1.0, 0.0, 0.0]), ("b", [0.9, 0.1, 0.0])]);
        let id_a = s.get_by_key("docs", "a").unwrap().unwrap().id;
        let mut allowed = roaring::RoaringBitmap::new();
        allowed.insert(id_a.to_bitmap_index());
        let mask = FilterMask {
            allowed: allowed.clone(),
            estimated_cardinality: allowed.len(),
        };

        let idx = FlatIndex::new(3, DistanceMetric::Cosine);
        let hits = idx
            .search(&s, "docs", &[1.0, 0.0, 0.0], 5, Some(&mask), &SearchParams { max_chunks_per_doc: None })
            .unwrap();
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
        let idx = FlatIndex::new(3, DistanceMetric::Cosine);
        let hits = idx
            .search(&s, "docs", &[1.0, 0.0, 0.0], 5, Some(&mask), &SearchParams { max_chunks_per_doc: None })
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, s.get_by_key("docs", "a").unwrap().unwrap().id);
    }

    #[test]
    fn max_chunks_per_doc_caps_hits_from_one_document() {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty())
            .unwrap();
        let chunks: Vec<ChunkInput> = (0..5)
            .map(|i| ChunkInput {
                text: format!("chunk {i}"),
                vector: vec![1.0 - i as f32 * 0.01, 0.0, 0.0],
            })
            .collect();
        s.collection("docs")
            .unwrap()
            .put_document(
                &ctx(),
                PutDocumentInput {
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

        let idx = FlatIndex::new(3, DistanceMetric::Cosine);
        let hits = idx
            .search(&s, "docs", &[1.0, 0.0, 0.0], 10, None, &SearchParams { max_chunks_per_doc: Some(2) })
            .unwrap();
        assert_eq!(hits.len(), 2, "only 2 of the document's 5 chunks may appear");
    }

    #[test]
    fn no_grouping_when_max_chunks_per_doc_is_none() {
        let s = Storage::new();
        s.create_collection(&ctx(), "docs", 3, DistanceMetric::Cosine, PayloadSchema::empty())
            .unwrap();
        let chunks: Vec<ChunkInput> = (0..5)
            .map(|i| ChunkInput {
                text: format!("chunk {i}"),
                vector: vec![1.0 - i as f32 * 0.01, 0.0, 0.0],
            })
            .collect();
        s.collection("docs")
            .unwrap()
            .put_document(
                &ctx(),
                PutDocumentInput {
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

        let idx = FlatIndex::new(3, DistanceMetric::Cosine);
        let hits = idx
            .search(&s, "docs", &[1.0, 0.0, 0.0], 10, None, &SearchParams { max_chunks_per_doc: None })
            .unwrap();
        assert_eq!(hits.len(), 5);
    }
}
