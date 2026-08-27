//! The fetch-score-sort-group tail every exact scoring path shares:
//! `FlatIndex`'s full/masked scan, `IvfIndex`'s post-routing candidate
//! set, and `IvfPqIndex`'s post-rerank shortlist all end here. Also
//! `FlatIndex`'s and `IvfIndex`/`IvfPqIndex`'s regime-1 fast path (master
//! plan *Filtered search*): a filter mask small enough that scanning
//! *exactly* the allowed rows beats touching an ANN index at all.

use crate::error::IndexResult;
use crate::grouping::apply_doc_grouping;
use crate::hit::SearchHit;
use crate::math;
use mara_proto::{DistanceMetric, Row, RowId};
use mara_storage::StorageApi;
use rayon::prelude::*;
use std::cmp::Ordering;

/// Fetches `ids` by `RowId`, exact-scores each against `query`, ranks
/// descending, applies doc grouping, and truncates to `k` — every hit
/// here is `exact: true` by construction (scored from the real stored
/// vector, never an ADC/LSH approximation).
pub(crate) fn finish_exact(
    storage: &dyn StorageApi,
    coll: &str,
    metric: DistanceMetric,
    query: &[f32],
    k: usize,
    ids: &[RowId],
    max_chunks_per_doc: Option<u32>,
) -> IndexResult<Vec<SearchHit>> {
    let rows = storage.rows_by_id(coll, ids)?;
    let mut scored: Vec<(f32, Row)> = rows
        .into_par_iter()
        .flatten()
        .filter_map(|row| {
            let score = row.vector.as_deref().map(|v| math::score(metric, query, v))?;
            Some((score, row))
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let grouped = apply_doc_grouping(scored, k, max_chunks_per_doc);
    Ok(grouped
        .into_iter()
        .map(|(score, row)| SearchHit {
            id: row.id,
            doc_id: row.doc_id,
            chunk_ord: row.chunk_ord,
            score,
            metric,
            exact: true,
        })
        .collect())
}
