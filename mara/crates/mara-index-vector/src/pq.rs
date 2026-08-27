//! Product Quantization + asymmetric distance computation (master plan
//! Layer 4, *IVF-PQ + OPQ + beam search*, PQ half only — OPQ's orthogonal
//! rotation is the next build step, not this one). A vector is split into
//! `m` contiguous subvectors of `dsub = dim / m` dimensions each; each
//! subspace gets its own k-means codebook of `ksub` centroids (`ksub` is
//! validated `<= 256` so a subcode fits one `u8`), and a vector is encoded
//! as `m` centroid indices — `pq_m=48` at `dim=384` is `48` bytes vs. the
//! `1536`-byte raw `f32` vector the master plan's defaults cite.
//!
//! ADC scores a candidate *without ever decoding it back to floats*: build
//! one `m * ksub` lookup table per query (squared distance from each of
//! the query's own `m` subvectors to that subspace's `ksub` centroids),
//! then a candidate's approximate distance is `m` table lookups summed —
//! `O(m)` instead of `O(dim)`. This approximation is only ever used to
//! shortlist candidates; `ivf_pq.rs` always exact-reranks the shortlist
//! against real stored vectors before returning a score to a caller (the
//! master plan's cosine-correctness invariant: no internally-computed
//! approximate distance is ever mislabeled as a real score).

use rayon::prelude::*;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PqError {
    #[error("pq_m ({m}) must evenly divide dim ({dim}); try one of: {suggested:?}")]
    MDoesNotDivideDim { m: usize, dim: usize, suggested: Vec<usize> },
    #[error("ksub ({0}) must be in 1..=256 so a subcode fits one byte")]
    KsubOutOfRange(usize),
}

#[derive(Clone, Debug)]
pub struct PqParams {
    pub m: usize,
    pub ksub: usize,
    pub kmeans_iters: usize,
    pub seed: u64,
}

impl Default for PqParams {
    fn default() -> Self {
        PqParams {
            m: 48,
            ksub: 256,
            kmeans_iters: 20,
            seed: 0,
        }
    }
}

/// Every divisor of `dim` in `2..=dim` — what `MDoesNotDivideDim` suggests
/// instead of leaving a caller to guess, per the master plan ("validated
/// at config load with a suggested divisor, never silently guessed").
fn divisors(dim: usize) -> Vec<usize> {
    (1..=dim).filter(|d| dim.is_multiple_of(*d)).collect()
}

#[derive(Debug)]
pub struct PqCodebook {
    m: usize,
    dsub: usize,
    ksub: usize,
    /// `codebooks[i]` holds subspace `i`'s `ksub` centroids, each `dsub`
    /// floats.
    codebooks: Vec<Vec<Vec<f32>>>,
}

/// One query's per-subspace distance table — `table[i][j]` is the squared
/// L2 distance from the query's subvector `i` to subspace `i`'s centroid
/// `j`. Reranking a whole shortlist against one query builds this once and
/// reuses it for every candidate.
///
/// Deliberately `Vec<Vec<f32>>`, not one flat `Vec<f32>` at a fixed
/// `ksub`-wide stride: `kmeans` clamps a subspace's actual centroid count
/// below the requested `ksub` whenever there are fewer training points
/// than `ksub` (the same graceful-degradation `IvfIndex` relies on for
/// `n_clusters`), so a subspace's codebook can legitimately be shorter
/// than `ksub` — a fixed stride would either misalign every subspace
/// after the first short one or index straight past the flat array's end.
pub struct AdcTable {
    m: usize,
    table: Vec<Vec<f32>>,
}

impl AdcTable {
    /// Sums `m` table lookups — the whole point of ADC: no `dsub`-wide
    /// float math per candidate, just byte-indexed adds.
    pub fn distance(&self, code: &[u8]) -> f32 {
        debug_assert_eq!(code.len(), self.m);
        code.iter().enumerate().map(|(i, &c)| self.table[i][c as usize]).sum()
    }
}

impl PqCodebook {
    pub fn m(&self) -> usize {
        self.m
    }

    /// The *configured* `ksub` — a subspace's actual trained centroid
    /// count (`codebooks[i].len()`) can be smaller when there were fewer
    /// training points than `ksub` (see `AdcTable`'s doc comment).
    pub fn ksub(&self) -> usize {
        self.ksub
    }

    /// Trains one k-means codebook per subspace, in parallel across
    /// subspaces (each subspace's own k-means is independent of every
    /// other's, so this nests cleanly inside `kmeans`'s own `rayon`
    /// parallelism over points). `vectors` must already be in whatever
    /// representation the caller wants distances computed in (raw, or
    /// `ivf::cluster_repr`'s L2-normalized form for `Cosine` — this
    /// module has no opinion on that, matching how `kmeans` itself is
    /// representation-agnostic).
    pub fn train(vectors: &[Vec<f32>], dim: usize, params: &PqParams) -> Result<Self, PqError> {
        if !dim.is_multiple_of(params.m) {
            return Err(PqError::MDoesNotDivideDim {
                m: params.m,
                dim,
                suggested: divisors(dim),
            });
        }
        if params.ksub == 0 || params.ksub > 256 {
            return Err(PqError::KsubOutOfRange(params.ksub));
        }
        let dsub = dim / params.m;

        let codebooks: Vec<Vec<Vec<f32>>> = (0..params.m)
            .into_par_iter()
            .map(|i| {
                let start = i * dsub;
                let sub_vectors: Vec<Vec<f32>> = vectors.iter().map(|v| v[start..start + dsub].to_vec()).collect();
                if sub_vectors.is_empty() {
                    return Vec::new();
                }
                crate::kmeans::kmeans(&sub_vectors, params.ksub, params.kmeans_iters, params.seed.wrapping_add(i as u64)).centroids
            })
            .collect();

        Ok(PqCodebook {
            m: params.m,
            dsub,
            ksub: params.ksub,
            codebooks,
        })
    }

    /// Encodes `v` as `m` centroid indices, one per subspace.
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        (0..self.m)
            .map(|i| {
                let start = i * self.dsub;
                let sub = &v[start..start + self.dsub];
                nearest(sub, &self.codebooks[i]) as u8
            })
            .collect()
    }

    /// Reconstructs a `dim`-length approximation of the original vector
    /// by concatenating each subspace's chosen centroid — the "decode"
    /// half of what makes `encode` lossy compression rather than a
    /// discard. Used by OPQ's alternating optimization to get the
    /// quantization targets it solves the next rotation against; not
    /// needed on the plain PQ/ADC search path, which never decodes a
    /// candidate back to floats.
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        debug_assert_eq!(code.len(), self.m);
        let mut out = Vec::with_capacity(self.m * self.dsub);
        for (i, &c) in code.iter().enumerate() {
            out.extend_from_slice(&self.codebooks[i][c as usize]);
        }
        out
    }

    pub fn build_adc_table(&self, query: &[f32]) -> AdcTable {
        let table: Vec<Vec<f32>> = (0..self.m)
            .map(|i| {
                let start = i * self.dsub;
                let sub = &query[start..start + self.dsub];
                self.codebooks[i].iter().map(|centroid| crate::math::l2_sq(sub, centroid)).collect()
            })
            .collect();
        AdcTable { m: self.m, table }
    }
}

fn nearest(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, crate::math::l2_sq(point, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .expect("codebooks is never empty for a nonempty training set")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn synthetic_vectors(dim: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n).map(|_| (0..dim).map(|_| rng.gen_range(-10.0..10.0)).collect()).collect()
    }

    #[test]
    fn m_not_dividing_dim_is_a_clear_error_with_suggestions() {
        let vectors = synthetic_vectors(10, 5, 1);
        let err = PqCodebook::train(&vectors, 10, &PqParams { m: 3, ksub: 4, kmeans_iters: 5, seed: 1 }).unwrap_err();
        match err {
            PqError::MDoesNotDivideDim { m, dim, suggested } => {
                assert_eq!(m, 3);
                assert_eq!(dim, 10);
                assert!(suggested.contains(&2));
                assert!(suggested.contains(&5));
                assert!(!suggested.contains(&3));
            }
            other => panic!("expected MDoesNotDivideDim, got {other:?}"),
        }
    }

    #[test]
    fn ksub_out_of_range_is_a_clear_error() {
        let vectors = synthetic_vectors(8, 5, 1);
        let err = PqCodebook::train(&vectors, 8, &PqParams { m: 2, ksub: 300, kmeans_iters: 5, seed: 1 }).unwrap_err();
        assert_eq!(err, PqError::KsubOutOfRange(300));

        let err = PqCodebook::train(&vectors, 8, &PqParams { m: 2, ksub: 0, kmeans_iters: 5, seed: 1 }).unwrap_err();
        assert_eq!(err, PqError::KsubOutOfRange(0));
    }

    #[test]
    fn encode_then_adc_self_distance_is_near_zero() {
        let vectors = synthetic_vectors(16, 200, 7);
        let codebook = PqCodebook::train(&vectors, 16, &PqParams { m: 4, ksub: 16, kmeans_iters: 15, seed: 3 }).unwrap();

        for v in vectors.iter().take(10) {
            let code = codebook.encode(v);
            assert_eq!(code.len(), 4);
            let table = codebook.build_adc_table(v);
            // A vector's ADC distance to its own code is the sum of its
            // per-subspace quantization error — not exactly zero (that's
            // the whole cost of compression), but must be far smaller
            // than its distance to an arbitrary other point.
            let self_dist = table.distance(&code);
            let other_code = codebook.encode(&vectors[(vectors.iter().position(|x| x == v).unwrap() + 100) % vectors.len()]);
            let other_dist = table.distance(&other_code);
            assert!(self_dist <= other_dist, "a vector's own code must not score worse than an arbitrary other code");
        }
    }

    #[test]
    fn adc_distance_correlates_with_true_l2_distance() {
        let vectors = synthetic_vectors(32, 300, 11);
        let codebook = PqCodebook::train(&vectors, 32, &PqParams { m: 8, ksub: 32, kmeans_iters: 15, seed: 5 }).unwrap();
        let codes: Vec<Vec<u8>> = vectors.iter().map(|v| codebook.encode(v)).collect();

        let query = &vectors[0];
        let table = codebook.build_adc_table(query);

        // Rank every point by true L2 and by ADC; the true nearest
        // neighbor (itself) must also be the ADC-nearest, and overall
        // rank correlation must be reasonably strong — PQ is lossy, not
        // useless.
        let mut by_true: Vec<(usize, f32)> = vectors.iter().enumerate().map(|(i, v)| (i, crate::math::l2_sq(query, v))).collect();
        by_true.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut by_adc: Vec<(usize, f32)> = codes.iter().enumerate().map(|(i, c)| (i, table.distance(c))).collect();
        by_adc.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        assert_eq!(by_true[0].0, 0, "query is its own true nearest neighbor");
        assert_eq!(by_adc[0].0, 0, "query's own code must also be its ADC-nearest");

        let true_top10: std::collections::HashSet<_> = by_true.iter().take(10).map(|(i, _)| *i).collect();
        let adc_top10: std::collections::HashSet<_> = by_adc.iter().take(10).map(|(i, _)| *i).collect();
        let overlap = true_top10.intersection(&adc_top10).count();
        assert!(overlap >= 5, "ADC top-10 should substantially overlap true top-10, got {overlap}/10");
    }

    #[test]
    fn compression_ratio_matches_m_bytes_per_code() {
        let vectors = synthetic_vectors(16, 50, 1);
        let codebook = PqCodebook::train(&vectors, 16, &PqParams { m: 4, ksub: 16, kmeans_iters: 5, seed: 1 }).unwrap();
        let code = codebook.encode(&vectors[0]);
        assert_eq!(code.len(), 4, "4 subspaces -> 4-byte code, vs. 16 f32s (64 bytes) raw");
    }
}
