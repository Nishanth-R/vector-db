//! SIMD (portable, via `wide` — stable Rust, no nightly `std::simd`)
//! distance kernels. These operate on raw, un-normalized, un-rotated
//! vectors — `FlatIndex` has no coarse/PQ/rotation stage to leak an
//! internal representation from, so "the exact rerank" and "the only
//! computation" are the same thing here.

use mara_proto::DistanceMetric;
use wide::f32x8;

const LANES: usize = 8;

/// Every exact-scoring `VectorIndex` (`Flat`, `Ivf`, `IvfPq`'s rerank
/// stage) converts a raw distance/similarity into the same higher-is-
/// better convention this one way — shared so "what score means" can
/// never quietly drift between index implementations.
pub fn score(metric: DistanceMetric, query: &[f32], vector: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine(query, vector),
        DistanceMetric::L2 => -l2_sq(query, vector),
        DistanceMetric::DotProduct => dot(query, vector),
    }
}

/// Dot product, SIMD over 8-lane chunks with a scalar tail.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = f32x8::ZERO;
    let chunks = a.len() / LANES;
    for i in 0..chunks {
        let start = i * LANES;
        let va = f32x8::new(a[start..start + LANES].try_into().unwrap());
        let vb = f32x8::new(b[start..start + LANES].try_into().unwrap());
        sum += va * vb;
    }
    let mut total = sum.reduce_add();
    for i in (chunks * LANES)..a.len() {
        total += a[i] * b[i];
    }
    total
}

/// Squared L2 distance (no `sqrt` — ranking by squared distance gives the
/// same order as ranking by distance, since `sqrt` is monotonic on
/// non-negative inputs, and it's what `score()` below relies on).
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = f32x8::ZERO;
    let chunks = a.len() / LANES;
    for i in 0..chunks {
        let start = i * LANES;
        let va = f32x8::new(a[start..start + LANES].try_into().unwrap());
        let vb = f32x8::new(b[start..start + LANES].try_into().unwrap());
        let d = va - vb;
        sum += d * d;
    }
    let mut total = sum.reduce_add();
    for i in (chunks * LANES)..a.len() {
        let d = a[i] - b[i];
        total += d * d;
    }
    total
}

pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// True cosine similarity between raw vectors. `0.0` if either vector is
/// zero (undefined direction) rather than `NaN`, so a degenerate stored
/// vector can't poison a ranked result set with a NaN comparison.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar_reference_across_lane_boundaries() {
        for len in [0, 1, 7, 8, 9, 15, 16, 17, 100] {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
            let b: Vec<f32> = (0..len).map(|i| (len - i) as f32 * 0.25).collect();
            let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot(&a, &b) - expected).abs() < 1e-3, "len={len}");
        }
    }

    #[test]
    fn l2_sq_matches_scalar_reference() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = [9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum();
        assert!((l2_sq(&a, &b) - expected).abs() < 1e-3);
    }

    #[test]
    fn cosine_of_identical_vectors_is_one() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_of_opposite_vectors_is_negative_one() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [-1.0f32, -2.0, -3.0];
        assert!((cosine(&a, &b) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_a_zero_vector_is_zero_not_nan() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 3.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }
}
