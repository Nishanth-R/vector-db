//! Hybrid retrieval fusion (master plan Layer 5, *Fusion*): combines a
//! vector index's and BM25's independently-ranked hit lists into one
//! ranked list, since their raw scores live on incompatible scales (a
//! cosine similarity and a Lucene-style BM25 score aren't comparable by
//! summing) that make naive score addition fragile.
//!
//! Deliberately decoupled from `mara-index-vector`/`mara-index-bm25`:
//! `fuse` takes plain `(RowId, score)` pairs, not `SearchHit`/`LexicalHit`,
//! so it has no opinion on where a candidate came from and no dependency
//! on either index crate. It's also decoupled from storage — `doc_id`-
//! based `max_chunks_per_doc` grouping and final-`k` truncation are the
//! caller's job (`mara-daemon`'s `Engine`, which already fetches full
//! `Row`s to build the response and has `doc_id` right there), applied
//! *after* fusion, per the master plan: fusing pre-grouped per-arm
//! results could drop a candidate that would have survived grouping the
//! fused list instead.
//!
//! **Input contract**: both hit lists must already be rank-ordered,
//! best-first — exactly what `VectorIndex::search`/`Bm25Index::search`
//! return. `fuse` uses each list's own order for ranking rather than
//! re-sorting by score, so passing an unsorted list silently produces a
//! wrong fusion, not a panic.

use mara_proto::RowId;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FusionMethod {
    /// `Σ 1/(k + rank)`, `rank` 1-indexed. The master plan's default:
    /// robust to incomparable score scales because it fuses on rank
    /// position, never touching either arm's raw score.
    ReciprocalRankFusion { k: u32 },
    /// Each arm's scores are min-max normalized to `[0, 1]` independently
    /// before the weighted sum — a candidate absent from an arm
    /// contributes `0` for that arm's term, the same convention a
    /// min-max scale already treats as "worst".
    WeightedSum { vector_weight: f32, bm25_weight: f32 },
}

impl Default for FusionMethod {
    fn default() -> Self {
        FusionMethod::ReciprocalRankFusion { k: 60 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FusedHit {
    pub id: RowId,
    pub score: f32,
}

/// Fuses `vector_hits` and `bm25_hits` into one ranked, deduplicated list
/// — every `RowId` appearing in either input appears exactly once in the
/// output, ranked by fused score descending, ties broken by `RowId`
/// ascending for fully deterministic output (score-only comparison would
/// leave tie order dependent on `HashMap` iteration, which isn't stable
/// run to run). Not truncated to any `k` — the caller decides how much of
/// the fused, ranked list it needs after grouping.
pub fn fuse(vector_hits: &[(RowId, f32)], bm25_hits: &[(RowId, f32)], method: FusionMethod) -> Vec<FusedHit> {
    let mut hits = match method {
        FusionMethod::ReciprocalRankFusion { k } => reciprocal_rank_fusion(vector_hits, bm25_hits, k),
        FusionMethod::WeightedSum { vector_weight, bm25_weight } => weighted_sum(vector_hits, bm25_hits, vector_weight, bm25_weight),
    };
    hits.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal).then_with(|| a.id.cmp(&b.id)));
    hits
}

fn reciprocal_rank_fusion(vector_hits: &[(RowId, f32)], bm25_hits: &[(RowId, f32)], k: u32) -> Vec<FusedHit> {
    let mut scores: HashMap<RowId, f32> = HashMap::new();
    for (rank, (id, _)) in vector_hits.iter().enumerate() {
        *scores.entry(*id).or_insert(0.0) += 1.0 / (k as f32 + (rank + 1) as f32);
    }
    for (rank, (id, _)) in bm25_hits.iter().enumerate() {
        *scores.entry(*id).or_insert(0.0) += 1.0 / (k as f32 + (rank + 1) as f32);
    }
    scores.into_iter().map(|(id, score)| FusedHit { id, score }).collect()
}

/// `None` for an empty input (nothing to normalize); every score maps to
/// `1.0` when every hit is tied (a zero-width range would otherwise
/// divide by zero) — treating a tie as "all equally best" rather than
/// picking an arbitrary winner.
fn min_max_normalize(hits: &[(RowId, f32)]) -> HashMap<RowId, f32> {
    if hits.is_empty() {
        return HashMap::new();
    }
    let min = hits.iter().map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
    let max = hits.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    hits.iter()
        .map(|(id, s)| {
            let norm = if range > 0.0 { (s - min) / range } else { 1.0 };
            (*id, norm)
        })
        .collect()
}

fn weighted_sum(vector_hits: &[(RowId, f32)], bm25_hits: &[(RowId, f32)], vector_weight: f32, bm25_weight: f32) -> Vec<FusedHit> {
    let v_norm = min_max_normalize(vector_hits);
    let b_norm = min_max_normalize(bm25_hits);
    let ids: HashSet<RowId> = v_norm.keys().chain(b_norm.keys()).copied().collect();
    ids.into_iter()
        .map(|id| {
            let vs = v_norm.get(&id).copied().unwrap_or(0.0);
            let bs = b_norm.get(&id).copied().unwrap_or(0.0);
            FusedHit {
                id,
                score: vector_weight * vs + bm25_weight * bs,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u64) -> RowId {
        RowId(n)
    }

    #[test]
    fn default_method_is_rrf_with_k_60() {
        assert_eq!(FusionMethod::default(), FusionMethod::ReciprocalRankFusion { k: 60 });
    }

    #[test]
    fn rrf_ranks_a_candidate_present_in_both_arms_above_one_present_in_only_one() {
        // "both" is rank 1 in both arms; "vector_only" is rank 1 in the
        // vector arm alone.
        let vector_hits = [(id(1), 0.9), (id(2), 0.5)];
        let bm25_hits = [(id(1), 10.0)];
        let fused = fuse(&vector_hits, &bm25_hits, FusionMethod::default());
        assert_eq!(fused[0].id, id(1), "present-in-both must outrank present-in-one-arm-only");
    }

    #[test]
    fn a_candidate_ranked_high_on_one_arm_and_mediocre_on_the_other_still_lands_near_the_top() {
        // id(99) is rank 80 on vector and rank 2 on BM25 — mirrors the
        // master plan's own stated example of why both arms get
        // overfetched (well past the final k) before fusing: a candidate
        // this deep on one axis alone wouldn't even be visible without it.
        let mut vector_hits: Vec<(RowId, f32)> = (0..100).map(|i| (id(i), 1.0 - i as f32 * 0.001)).collect();
        let pos = vector_hits.iter().position(|(rid, _)| *rid == id(99)).unwrap();
        vector_hits.remove(pos);
        vector_hits.insert(79, (id(99), 0.5));

        let bm25_hits: Vec<(RowId, f32)> = vec![(id(1), 20.0), (id(99), 18.0), (id(2), 15.0)];

        let fused = fuse(&vector_hits, &bm25_hits, FusionMethod::default());
        let top10: Vec<RowId> = fused.iter().take(10).map(|h| h.id).collect();
        assert!(top10.contains(&id(99)), "rank-80-on-vector + rank-2-on-bm25 should land in the fused top 10, got {top10:?}");
    }

    #[test]
    fn every_candidate_from_either_arm_appears_exactly_once() {
        let vector_hits = [(id(1), 0.9), (id(2), 0.5)];
        let bm25_hits = [(id(2), 10.0), (id(3), 5.0)];
        let fused = fuse(&vector_hits, &bm25_hits, FusionMethod::default());
        let mut ids: Vec<RowId> = fused.iter().map(|h| h.id).collect();
        ids.sort();
        assert_eq!(ids, vec![id(1), id(2), id(3)]);
    }

    #[test]
    fn empty_inputs_produce_no_hits() {
        let fused = fuse(&[], &[], FusionMethod::default());
        assert!(fused.is_empty());
    }

    #[test]
    fn one_empty_arm_still_fuses_the_other() {
        let vector_hits = [(id(1), 0.9), (id(2), 0.5)];
        let fused = fuse(&vector_hits, &[], FusionMethod::default());
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].id, id(1));
    }

    #[test]
    fn ties_are_broken_deterministically_by_row_id() {
        // Both candidates rank 1 in exactly one, disjoint arm each ->
        // identical RRF contributions -> a genuine score tie.
        let vector_hits = [(id(5), 1.0)];
        let bm25_hits = [(id(2), 1.0)];
        let a = fuse(&vector_hits, &bm25_hits, FusionMethod::default());
        let b = fuse(&vector_hits, &bm25_hits, FusionMethod::default());
        assert_eq!(a, b, "fusion must be deterministic across repeated calls with identical input");
        assert_eq!(a[0].id, id(2), "tied scores must break by ascending RowId");
    }

    #[test]
    fn weighted_sum_respects_configured_weights() {
        // id(1) is best-on-vector-only; id(2) is best-on-bm25-only.
        let vector_hits = [(id(1), 1.0), (id(2), 0.0)];
        let bm25_hits = [(id(2), 1.0), (id(1), 0.0)];

        let vector_favored = fuse(&vector_hits, &bm25_hits, FusionMethod::WeightedSum { vector_weight: 0.9, bm25_weight: 0.1 });
        assert_eq!(vector_favored[0].id, id(1));

        let bm25_favored = fuse(&vector_hits, &bm25_hits, FusionMethod::WeightedSum { vector_weight: 0.1, bm25_weight: 0.9 });
        assert_eq!(bm25_favored[0].id, id(2));
    }

    #[test]
    fn weighted_sum_normalizes_each_arm_independently_of_scale() {
        // BM25 scores routinely run 0-20+; vector scores here run -1..1.
        // Equal weights should still let the top-ranked-on-each-arm
        // candidates tie for first, unaffected by the raw scale gap.
        let vector_hits = [(id(1), 0.95), (id(2), -0.3)];
        let bm25_hits = [(id(2), 18.0), (id(1), 2.0)];
        let fused = fuse(&vector_hits, &bm25_hits, FusionMethod::WeightedSum { vector_weight: 0.5, bm25_weight: 0.5 });
        assert!((fused[0].score - fused[1].score).abs() < 1e-6, "each candidate is top-of-one-arm and bottom-of-the-other, so equal weights must tie them exactly");
    }

    #[test]
    fn weighted_sum_all_tied_scores_normalize_to_one_not_nan() {
        let vector_hits = [(id(1), 0.5), (id(2), 0.5)];
        let fused = fuse(&vector_hits, &[], FusionMethod::WeightedSum { vector_weight: 1.0, bm25_weight: 0.0 });
        for h in &fused {
            assert!(!h.score.is_nan());
            assert!((h.score - 1.0).abs() < 1e-6);
        }
    }
}
