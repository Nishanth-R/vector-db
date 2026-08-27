//! The coarse quantizer `IvfIndex`/`IvfPqIndex` route queries through
//! (master plan Layer 4, *IVF-PQ + OPQ + beam search*, coarse-quantizer
//! half): a flat centroid list at or below `hierarchical_threshold`
//! clusters, 2-level hierarchical (k-means-of-k-means) above it. Beam
//! search only means something over a tree of partial routing decisions
//! — over a short flat list it degenerates to plain top-`nprobe`, so
//! hierarchy is what makes it real rather than decorative, and staying
//! flat below the threshold is both simpler and exactly as capable.
//!
//! `route_candidates` returns a single merged `RoaringBitmap` regardless
//! of which shape backs it, so `IvfIndex`/`IvfPqIndex`'s `search` never
//! needs to know which one it's dealing with.

use crate::kmeans::kmeans;
use crate::math::l2_sq;
use crate::params::SearchParams;
use mara_proto::RowId;
use mara_storage::{FilterMask, StorageApi, StorageResult};
use roaring::RoaringBitmap;
use std::cmp::Ordering;

pub(crate) struct Leaf {
    centroid: Vec<f32>,
    posting_list: RoaringBitmap,
}

pub(crate) enum CoarseQuantizer {
    Flat {
        centroids: Vec<Vec<f32>>,
        posting_lists: Vec<RoaringBitmap>,
    },
    Hierarchical {
        parents: Vec<Vec<f32>>,
        /// `leaves_by_parent[p]` holds parent `p`'s own children — a
        /// parent with zero assigned training points simply contributes
        /// an empty `Vec` here, not a crash.
        leaves_by_parent: Vec<Vec<Leaf>>,
    },
}

impl std::fmt::Debug for CoarseQuantizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoarseQuantizer::Flat { centroids, .. } => write!(f, "CoarseQuantizer::Flat({} clusters)", centroids.len()),
            CoarseQuantizer::Hierarchical { parents, .. } => write!(f, "CoarseQuantizer::Hierarchical({} parents)", parents.len()),
        }
    }
}

impl CoarseQuantizer {
    pub(crate) fn empty() -> Self {
        CoarseQuantizer::Flat {
            centroids: Vec::new(),
            posting_lists: Vec::new(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            CoarseQuantizer::Flat { centroids, .. } => centroids.is_empty(),
            CoarseQuantizer::Hierarchical { parents, .. } => parents.is_empty(),
        }
    }

    /// Total leaf (posting-list-bearing) cluster count, flat or
    /// hierarchical — what `IvfIndex::n_clusters` reports either way.
    pub(crate) fn n_leaves(&self) -> usize {
        match self {
            CoarseQuantizer::Flat { centroids, .. } => centroids.len(),
            CoarseQuantizer::Hierarchical { leaves_by_parent, .. } => leaves_by_parent.iter().map(|l| l.len()).sum(),
        }
    }

    /// `n_clusters` at or below `hierarchical_threshold` builds a flat
    /// list (unchanged from before hierarchy existed); above it, points
    /// are clustered into `ceil(sqrt(n_clusters))` parents, each of which
    /// is independently sub-clustered into its own children, so the
    /// total leaf count still lands near `n_clusters` (k-means's own
    /// clamping keeps a sparse parent's leaf count from exceeding its
    /// member count).
    pub(crate) fn build(vectored: &[(RowId, Vec<f32>)], n_clusters: usize, hierarchical_threshold: usize, kmeans_iters: usize, seed: u64) -> Self {
        if vectored.is_empty() {
            return CoarseQuantizer::empty();
        }
        if n_clusters <= hierarchical_threshold {
            return Self::build_flat(vectored, n_clusters, kmeans_iters, seed);
        }
        Self::build_hierarchical(vectored, n_clusters, kmeans_iters, seed)
    }

    fn build_flat(vectored: &[(RowId, Vec<f32>)], n_clusters: usize, kmeans_iters: usize, seed: u64) -> Self {
        let repr: Vec<Vec<f32>> = vectored.iter().map(|(_, v)| v.clone()).collect();
        let result = kmeans(&repr, n_clusters, kmeans_iters, seed);
        let mut posting_lists = vec![RoaringBitmap::new(); result.centroids.len()];
        for ((row_id, _), &cluster) in vectored.iter().zip(&result.assignments) {
            posting_lists[cluster].insert(row_id.to_bitmap_index());
        }
        CoarseQuantizer::Flat {
            centroids: result.centroids,
            posting_lists,
        }
    }

    fn build_hierarchical(vectored: &[(RowId, Vec<f32>)], n_clusters: usize, kmeans_iters: usize, seed: u64) -> Self {
        let n_parents = (n_clusters as f64).sqrt().ceil() as usize;
        let children_per_parent = n_clusters.div_ceil(n_parents.max(1));

        let repr: Vec<Vec<f32>> = vectored.iter().map(|(_, v)| v.clone()).collect();
        let level1 = kmeans(&repr, n_parents, kmeans_iters, seed);

        let mut members_by_parent: Vec<Vec<(RowId, Vec<f32>)>> = vec![Vec::new(); level1.centroids.len()];
        for ((row_id, v), &parent) in vectored.iter().zip(&level1.assignments) {
            members_by_parent[parent].push((*row_id, v.clone()));
        }

        let leaves_by_parent: Vec<Vec<Leaf>> = members_by_parent
            .into_iter()
            .enumerate()
            .map(|(p, members)| {
                if members.is_empty() {
                    return Vec::new();
                }
                let sub_repr: Vec<Vec<f32>> = members.iter().map(|(_, v)| v.clone()).collect();
                // A distinct seed per parent so every parent's k-means++
                // init doesn't draw from the same RNG stream in lockstep.
                let level2 = kmeans(&sub_repr, children_per_parent, kmeans_iters, seed.wrapping_add(1).wrapping_add(p as u64));
                let mut posting_lists = vec![RoaringBitmap::new(); level2.centroids.len()];
                for ((row_id, _), &child) in members.iter().zip(&level2.assignments) {
                    posting_lists[child].insert(row_id.to_bitmap_index());
                }
                level2
                    .centroids
                    .into_iter()
                    .zip(posting_lists)
                    .map(|(centroid, posting_list)| Leaf { centroid, posting_list })
                    .collect()
            })
            .collect();

        CoarseQuantizer::Hierarchical {
            parents: level1.centroids,
            leaves_by_parent,
        }
    }

    /// Routes `query_repr` to candidate rows: for `Flat`, the union of
    /// the `nprobe` nearest centroids' posting lists (`beam_width`
    /// unused). For `Hierarchical`, beam search — keep the `beam_width`
    /// nearest *parents* alive (not just the single best, which is what
    /// guards against the classic k-means-tree boundary problem where the
    /// true nearest leaf's parent isn't top-ranked), pool every leaf
    /// under those parents, then take the globally `nprobe`-nearest
    /// leaves from that pool.
    pub(crate) fn route_candidates(&self, query_repr: &[f32], nprobe: usize, beam_width: usize) -> RoaringBitmap {
        match self {
            CoarseQuantizer::Flat { centroids, posting_lists } => {
                let mut by_distance: Vec<(usize, f32)> = centroids.iter().enumerate().map(|(i, c)| (i, l2_sq(query_repr, c))).collect();
                by_distance.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                let mut candidates = RoaringBitmap::new();
                for (i, _) in by_distance.into_iter().take(nprobe.clamp(1, centroids.len())) {
                    candidates |= &posting_lists[i];
                }
                candidates
            }
            CoarseQuantizer::Hierarchical { parents, leaves_by_parent } => {
                let mut parent_dist: Vec<(usize, f32)> = parents.iter().enumerate().map(|(i, c)| (i, l2_sq(query_repr, c))).collect();
                parent_dist.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                let beam: Vec<usize> = parent_dist.into_iter().take(beam_width.clamp(1, parents.len())).map(|(i, _)| i).collect();

                let mut leaf_dist: Vec<(usize, usize, f32)> = Vec::new();
                for &p in &beam {
                    for (li, leaf) in leaves_by_parent[p].iter().enumerate() {
                        leaf_dist.push((p, li, l2_sq(query_repr, &leaf.centroid)));
                    }
                }
                leaf_dist.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

                let mut candidates = RoaringBitmap::new();
                for (p, li, _) in leaf_dist.into_iter().take(nprobe.max(1)) {
                    candidates |= &leaves_by_parent[p][li].posting_list;
                }
                candidates
            }
        }
    }

    /// Master plan *Filtered search*, regimes 2 and 3's shared escalation
    /// tail: re-route with a doubling `nprobe`, intersecting `allowed`
    /// each round, until the post-filter candidate count reaches `target`
    /// or `nprobe` hits `requested_cap` — whichever comes first. Hitting
    /// the cap short of `target` is reported via `truncated_by_filter`
    /// rather than a silent short return — but only when `requested_cap`
    /// itself cut the search short of the coarse quantizer's full leaf
    /// count. A collection that genuinely doesn't have `target` matching
    /// rows at all is a complete, correct answer, not a truncated one:
    /// probing every single leaf and still coming up short means there's
    /// nothing left to find, not that the budget ran out first.
    ///
    /// A known v0 simplification: each round re-derives the *whole*
    /// candidate set from scratch at the new `nprobe` rather than
    /// incrementally adding just the newly-reached leaves — simpler to
    /// reason about, at the cost of some redundant re-scoring of
    /// already-seen leaves on later rounds. Correct, not maximally
    /// efficient.
    pub(crate) fn route_with_escalation(&self, query_repr: &[f32], start_nprobe: usize, beam_width: usize, allowed: &RoaringBitmap, target: usize, requested_cap: usize) -> (RoaringBitmap, bool) {
        let n_leaves = self.n_leaves().max(1);
        let cap = requested_cap.clamp(1, n_leaves);
        let cap_is_exhaustive = requested_cap >= n_leaves;
        let mut nprobe = start_nprobe.clamp(1, cap);
        loop {
            let mut candidates = self.route_candidates(query_repr, nprobe, beam_width);
            candidates &= allowed;
            let reached_cap = nprobe >= cap;
            if candidates.len() as usize >= target || reached_cap {
                let truncated = !cap_is_exhaustive && (candidates.len() as usize) < target && reached_cap;
                return (candidates, truncated);
            }
            nprobe = (nprobe * 2).clamp(nprobe + 1, cap);
        }
    }

    /// The full master-plan *Filtered search* dispatch in one place: no
    /// filter routes plainly; a filter selective enough for regime 1
    /// (exact scan) hands back `allowed` untouched, letting the caller
    /// skip ANN entirely; otherwise regime 2 (filtered ANN, escalate from
    /// the start) or regime 3 (plain ANN, mask, escalate only if short of
    /// `k`) applies depending on `|allowed| / active_count` against
    /// `filter_selectivity_high`. `IvfIndex` and `IvfPqIndex` both call
    /// this and only differ in what they do with the resulting candidate
    /// set (exact-score directly vs. ADC-shortlist-then-rerank).
    pub(crate) fn resolve_candidates(
        &self,
        storage: &dyn StorageApi,
        coll: &str,
        query_repr: &[f32],
        params: &SearchParams,
        filter: Option<&FilterMask>,
        k: usize,
    ) -> StorageResult<(RoaringBitmap, bool)> {
        let Some(f) = filter else {
            return Ok((self.route_candidates(query_repr, params.nprobe, params.beam_width), false));
        };

        if (f.estimated_cardinality as usize) <= params.filter_exact_threshold {
            return Ok((f.allowed.clone(), false));
        }

        let n = storage.active_count(coll)?;
        let selectivity = f.estimated_cardinality as f64 / (n.max(1) as f64);
        let target = params.rerank_k.unwrap_or_else(|| (20 * k).max(200));
        let nprobe_max = params.filter_nprobe_max.unwrap_or_else(|| params.nprobe.saturating_mul(4));

        if selectivity < params.filter_selectivity_high {
            return Ok(self.route_with_escalation(query_repr, params.nprobe, params.beam_width, &f.allowed, target, nprobe_max));
        }

        let mut plain = self.route_candidates(query_repr, params.nprobe, params.beam_width);
        plain &= &f.allowed;
        if plain.len() as usize >= k {
            Ok((plain, false))
        } else {
            Ok(self.route_with_escalation(query_repr, params.nprobe, params.beam_width, &f.allowed, target, nprobe_max))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mara_proto::RowId;

    fn row(id: u64) -> RowId {
        RowId(id)
    }

    fn bitmap(ids: &[u64]) -> RoaringBitmap {
        ids.iter().map(|&id| row(id).to_bitmap_index()).collect()
    }

    /// A hand-built (not k-means-trained) two-parent tree, engineered so
    /// row 200's leaf sits under the parent that is *not* nearest to the
    /// query — the classic k-means-tree boundary problem: parent
    /// centroids are averages, so a query near the boundary between two
    /// parents can be closer to parent A's centroid while a real point
    /// just inside parent B's territory is actually nearer.
    fn boundary_case() -> CoarseQuantizer {
        CoarseQuantizer::Hierarchical {
            parents: vec![vec![0.0, 0.0], vec![10.0, 0.0]],
            leaves_by_parent: vec![
                vec![Leaf {
                    centroid: vec![0.0, 0.0],
                    posting_list: bitmap(&[100]),
                }],
                vec![Leaf {
                    centroid: vec![10.0, 0.0],
                    posting_list: bitmap(&[200]),
                }],
            ],
        }
    }

    #[test]
    fn beam_width_one_cannot_reach_the_non_nearest_parents_leaf() {
        let coarse = boundary_case();
        // Query sits just closer to parent 0 (dist 24.01) than parent 1
        // (dist 26.01) — but row 200, under parent 1, is the query's true
        // nearest point overall (dist 26.01 to its leaf centroid here,
        // since the leaf *is* the point in this hand-built case).
        let query = vec![4.9, 0.0];
        let candidates = coarse.route_candidates(&query, 2, 1);
        assert_eq!(candidates, bitmap(&[100]), "beam_width=1 structurally cannot see a leaf under any parent but the single nearest one");
    }

    #[test]
    fn beam_width_two_recovers_the_boundary_case() {
        let coarse = boundary_case();
        let query = vec![4.9, 0.0];
        let candidates = coarse.route_candidates(&query, 2, 2);
        assert_eq!(candidates, bitmap(&[100, 200]), "beam_width=2 keeps both parents alive, so both leaves enter the nprobe pool");
    }

    #[test]
    fn nprobe_still_limits_the_pool_even_with_a_wide_beam() {
        let coarse = boundary_case();
        let query = vec![4.9, 0.0];
        // Both parents are in the beam, but nprobe=1 only keeps the
        // single closest leaf across the whole pooled set.
        let candidates = coarse.route_candidates(&query, 1, 2);
        assert_eq!(candidates, bitmap(&[100]));
    }

    #[test]
    fn build_hierarchical_produces_leaves_close_to_the_requested_n_clusters() {
        let mut rng_vectors: Vec<(RowId, Vec<f32>)> = Vec::new();
        let mut id = 0u64;
        for cx in 0..4 {
            for i in 0..50 {
                let x = cx as f32 * 100.0 + (i as f32 * 0.1);
                rng_vectors.push((row(id), vec![x, 0.0]));
                id += 1;
            }
        }
        let coarse = CoarseQuantizer::build(&rng_vectors, 16, 0, 15, 1);
        assert!(matches!(coarse, CoarseQuantizer::Hierarchical { .. }), "hierarchical_threshold=0 must force hierarchical mode");
        assert!(coarse.n_leaves() > 0 && coarse.n_leaves() <= 16, "leaf count should land at or under the requested n_clusters, got {}", coarse.n_leaves());
    }

    #[test]
    fn below_threshold_stays_flat() {
        let vectored: Vec<(RowId, Vec<f32>)> = vec![(row(1), vec![0.0, 0.0]), (row(2), vec![1.0, 1.0])];
        let coarse = CoarseQuantizer::build(&vectored, 2, 512, 5, 1);
        assert!(matches!(coarse, CoarseQuantizer::Flat { .. }));
    }

    #[test]
    fn empty_input_is_empty_regardless_of_threshold() {
        let coarse = CoarseQuantizer::build(&[], 10, 0, 5, 1);
        assert!(coarse.is_empty());
        assert_eq!(coarse.n_leaves(), 0);
    }
}
