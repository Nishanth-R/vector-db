//! OPQ — Optimized Product Quantization (master plan Layer 4, *IVF-PQ +
//! OPQ + beam search*, OPQ half): an orthogonal rotation learned jointly
//! with the PQ codebooks via alternating optimization (Ge et al. 2013),
//! applied as one global preprocessing step before *both* coarse and PQ
//! training. Orthogonal rotations preserve L2 distance and, on unit
//! vectors, cosine similarity exactly — so this never distorts coarse
//! routing, only reshapes the space PQ quantizes to better align its
//! per-subspace axes with the data's actual variance.
//!
//! `opq_enabled` in the master plan is a toggle rather than an assumed
//! win: OPQ was designed against feature spaces with markedly
//! non-isotropic per-dimension variance (classic image descriptors like
//! SIFT/GIST); modern sentence-transformer embeddings are comparatively
//! isotropic, so whether OPQ's rotation earns back its training cost here
//! is a real empirical question, not a given — see the `benches/`
//! criterion harness that checks this against the `Flat` oracle.

use crate::pq::{PqCodebook, PqError, PqParams};
use nalgebra::{DMatrix, SVD};

#[derive(Clone, Debug)]
pub struct OpqParams {
    pub pq: PqParams,
    pub iterations: usize,
}

impl Default for OpqParams {
    fn default() -> Self {
        OpqParams {
            pq: PqParams::default(),
            iterations: 20,
        }
    }
}

/// A learned `dim x dim` orthogonal rotation. Stored transposed
/// (`matrix_t[j*dim+i] = R[i][j]`) so `apply` — the hot path, run on
/// every query — is a contiguous SIMD dot product per output dimension
/// via `math::dot`, not a strided column read.
#[derive(Clone, Debug)]
pub struct OpqRotation {
    dim: usize,
    matrix_t: Vec<f32>,
}

impl OpqRotation {
    fn identity(dim: usize) -> Self {
        let mut matrix_t = vec![0.0f32; dim * dim];
        for i in 0..dim {
            matrix_t[i * dim + i] = 1.0;
        }
        OpqRotation { dim, matrix_t }
    }

    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        debug_assert_eq!(v.len(), self.dim);
        (0..self.dim).map(|j| crate::math::dot(v, &self.matrix_t[j * self.dim..(j + 1) * self.dim])).collect()
    }
}

/// Solves the orthogonal Procrustes problem: the orthogonal `R`
/// minimizing `||X @ R - Y||_F`, via `R = U @ V^T` where `X^T @ Y = U S
/// V^T` (SVD) — the standard closed-form solution. `x` and `y` are
/// row-major point sets of equal length; `M = X^T @ Y` is computed via
/// `nalgebra`'s dense matmul (this is the one place a `matrixmultiply`-
/// backed GEMM earns its keep over `wide`'s per-vector SIMD kernels — a
/// full `N x dim` by `N x dim` contraction, not a per-pair distance).
fn solve_procrustes(x: &[Vec<f32>], y: &[Vec<f32>], dim: usize) -> OpqRotation {
    let n = x.len();
    let x_flat: Vec<f32> = x.iter().flatten().copied().collect();
    let y_flat: Vec<f32> = y.iter().flatten().copied().collect();
    let xm = DMatrix::from_row_slice(n, dim, &x_flat);
    let ym = DMatrix::from_row_slice(n, dim, &y_flat);
    let m = xm.transpose() * ym;

    let svd = SVD::new(m, true, true);
    let u = svd.u.expect("SVD of a square matrix with compute_u=true always yields U");
    let v_t = svd.v_t.expect("SVD of a square matrix with compute_v=true always yields V^T");
    let r = u * v_t;

    let mut matrix_t = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            matrix_t[j * dim + i] = r[(i, j)];
        }
    }
    OpqRotation { dim, matrix_t }
}

/// Alternating optimization: refit PQ codebooks under the current
/// rotation, then solve for the rotation that best aligns the original
/// (unrotated) points with those codebooks' reconstructions, repeat.
/// Returns the final rotation paired with the codebook trained *under*
/// that exact rotation — the last iteration always ends with a codebook
/// refit, never a rotation update, so the pair returned is always mutually
/// consistent (never "codebook trained under the second-to-last R").
pub fn train_opq(vectors: &[Vec<f32>], dim: usize, params: &OpqParams) -> Result<(OpqRotation, PqCodebook), PqError> {
    if vectors.is_empty() {
        // No data to align a rotation to — identity, and let
        // `PqCodebook::train` do its usual m/ksub validation regardless.
        let codebook = PqCodebook::train(vectors, dim, &params.pq)?;
        return Ok((OpqRotation::identity(dim), codebook));
    }

    let iterations = params.iterations.max(1);
    let mut rotation = OpqRotation::identity(dim);
    let mut codebook: Option<PqCodebook> = None;

    for iter in 0..iterations {
        let rotated: Vec<Vec<f32>> = vectors.iter().map(|v| rotation.apply(v)).collect();
        let mut pq_params = params.pq.clone();
        pq_params.seed = pq_params.seed.wrapping_add(iter as u64);
        let cb = PqCodebook::train(&rotated, dim, &pq_params)?;

        if iter + 1 == iterations {
            codebook = Some(cb);
            break;
        }

        let reconstructed: Vec<Vec<f32>> = rotated.iter().map(|r| cb.decode(&cb.encode(r))).collect();
        rotation = solve_procrustes(vectors, &reconstructed, dim);
        codebook = Some(cb);
    }

    Ok((rotation, codebook.expect("the loop always runs at least once (iterations.max(1))")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn identity_rotation_leaves_vectors_unchanged() {
        let id = OpqRotation::identity(4);
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let out = id.apply(&v);
        for (a, b) in v.iter().zip(&out) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn a_learned_rotation_is_orthogonal_norm_preserving() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let dim = 8;
        let vectors: Vec<Vec<f32>> = (0..200).map(|_| (0..dim).map(|_| rng.gen_range(-5.0..5.0)).collect()).collect();

        let (rotation, _codebook) = train_opq(&vectors, dim, &OpqParams { pq: PqParams { m: 4, ksub: 16, kmeans_iters: 10, seed: 2 }, iterations: 5 }).unwrap();

        for v in vectors.iter().take(10) {
            let rotated = rotation.apply(v);
            let orig_norm = crate::math::norm(v);
            let rotated_norm = crate::math::norm(&rotated);
            assert!((orig_norm - rotated_norm).abs() < 1e-2, "orthogonal rotation must preserve vector norm: {orig_norm} vs {rotated_norm}");
        }
    }

    #[test]
    fn rotation_and_codebook_are_mutually_consistent() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let dim = 8;
        let vectors: Vec<Vec<f32>> = (0..200).map(|_| (0..dim).map(|_| rng.gen_range(-5.0..5.0)).collect()).collect();

        let (rotation, codebook) = train_opq(&vectors, dim, &OpqParams { pq: PqParams { m: 4, ksub: 16, kmeans_iters: 10, seed: 4 }, iterations: 4 }).unwrap();

        // The codebook was trained on `vectors` rotated by the *final*
        // `rotation` — so encoding a freshly-rotated training point and
        // decoding it back must land close to that same rotated point,
        // the same self-distance sanity check `pq.rs` applies.
        for v in vectors.iter().take(5) {
            let rotated = rotation.apply(v);
            let code = codebook.encode(&rotated);
            let decoded = codebook.decode(&code);
            let err = crate::math::l2_sq(&rotated, &decoded);
            let scale = crate::math::l2_sq(&rotated, &vec![0.0; dim]);
            assert!(err < scale, "quantization error ({err}) should be well below the vector's own scale ({scale})");
        }
    }

    #[test]
    fn empty_input_returns_identity_and_a_trained_but_empty_codebook() {
        let (rotation, codebook) = train_opq(&[], 8, &OpqParams { pq: PqParams { m: 4, ksub: 16, kmeans_iters: 5, seed: 1 }, iterations: 5 }).unwrap();
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = rotation.apply(&v);
        for (a, b) in v.iter().zip(&out) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(codebook.m(), 4);
    }

    #[test]
    fn m_not_dividing_dim_surfaces_as_a_build_error() {
        let vectors: Vec<Vec<f32>> = vec![vec![1.0; 10]; 5];
        let err = train_opq(&vectors, 10, &OpqParams { pq: PqParams { m: 3, ksub: 4, kmeans_iters: 5, seed: 1 }, iterations: 3 }).unwrap_err();
        assert!(matches!(err, PqError::MDoesNotDivideDim { .. }));
    }
}
