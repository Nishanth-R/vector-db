//! Lloyd's k-means with k-means++ seeding — the coarse quantizer `IvfIndex`
//! clusters vectors around (master plan Layer 4, *IVF*). Pure and
//! storage-agnostic: works on whatever vector representation the caller
//! hands it (raw for `L2`/`DotProduct`, L2-normalized for `Cosine` — see
//! `ivf.rs`), always measuring in squared L2, the quantity k-means
//! minimizes and the same one `math::l2_sq` already computes with SIMD.

use crate::math::l2_sq;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

pub struct KmeansResult {
    pub centroids: Vec<Vec<f32>>,
    /// Cluster index per input point, same order as `points`.
    pub assignments: Vec<usize>,
}

/// `k` is clamped to `[1, points.len()]` — asking for more clusters than
/// points is a config mistake, not a crash. `points.is_empty()` short
/// circuits to an empty result before `k` is even looked at.
pub fn kmeans(points: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> KmeansResult {
    if points.is_empty() {
        return KmeansResult {
            centroids: Vec::new(),
            assignments: Vec::new(),
        };
    }
    let k = k.clamp(1, points.len());
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut centroids = kmeans_plus_plus_init(points, k, &mut rng);
    // A sentinel no real cluster index (`0..k`) can ever equal — plain
    // `vec![0; n]` would spuriously match the very first real assignment
    // whenever every point's nearest centroid happens to be cluster 0
    // (guaranteed when `k == 1`), short-circuiting the "converged" check
    // before `update_centroids` ever moves the centroid off its
    // arbitrary k-means++ starting pick.
    let mut assignments = vec![usize::MAX; points.len()];

    for _ in 0..iters.max(1) {
        let new_assignments: Vec<usize> = points
            .par_iter()
            .map(|p| nearest_centroid(p, &centroids))
            .collect();
        let converged = new_assignments == assignments;
        assignments = new_assignments;
        if converged {
            break;
        }
        update_centroids(points, &assignments, &mut centroids);
    }

    KmeansResult { centroids, assignments }
}

fn nearest_centroid(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_sq(point, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .expect("centroids is never empty inside kmeans")
}

/// Recomputes each centroid as the mean of its currently-assigned points.
/// A cluster that lost every point keeps its previous centroid rather than
/// becoming a `NaN`-filled vector from a divide-by-zero — a known v0
/// simplification (no re-seeding a starved cluster from the point farthest
/// from its centroid, as some production k-means implementations do); an
/// empty cluster just contributes an empty posting list at search time.
fn update_centroids(points: &[Vec<f32>], assignments: &[usize], centroids: &mut [Vec<f32>]) {
    let dim = points[0].len();
    let mut sums = vec![vec![0.0f32; dim]; centroids.len()];
    let mut counts = vec![0u32; centroids.len()];
    for (point, &cluster) in points.iter().zip(assignments) {
        counts[cluster] += 1;
        for (s, v) in sums[cluster].iter_mut().zip(point) {
            *s += v;
        }
    }
    for (cluster, count) in counts.into_iter().enumerate() {
        if count == 0 {
            continue;
        }
        for (c, s) in centroids[cluster].iter_mut().zip(&sums[cluster]) {
            *c = s / count as f32;
        }
    }
}

/// k-means++: the first centroid is uniform-random; every subsequent one
/// is sampled with probability proportional to its squared distance from
/// the nearest centroid already chosen — points far from existing
/// centroids are disproportionately likely to seed a new one, which is
/// what keeps k-means++'s expected clustering quality provably better
/// than picking `k` centroids uniformly at random.
fn kmeans_plus_plus_init(points: &[Vec<f32>], k: usize, rng: &mut ChaCha8Rng) -> Vec<Vec<f32>> {
    let mut centroids = Vec::with_capacity(k);
    centroids.push(points[rng.gen_range(0..points.len())].clone());

    while centroids.len() < k {
        let weights: Vec<f32> = points
            .iter()
            .map(|p| centroids.iter().map(|c| l2_sq(p, c)).fold(f32::INFINITY, f32::min))
            .collect();
        // Every remaining weight is exactly 0.0 (every point coincides
        // with an already-chosen centroid, e.g. many duplicate rows) —
        // `WeightedIndex` rejects an all-zero distribution, so fall back
        // to uniform choice rather than propagating that as a panic.
        let next = if weights.iter().any(|&w| w > 0.0) {
            let dist = WeightedIndex::new(&weights).expect("at least one positive weight was just checked");
            dist.sample(rng)
        } else {
            rng.gen_range(0..points.len())
        };
        centroids.push(points[next].clone());
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster_around(center: [f32; 2], n: usize, spread: f32, rng: &mut ChaCha8Rng) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| vec![center[0] + rng.gen_range(-spread..spread), center[1] + rng.gen_range(-spread..spread)])
            .collect()
    }

    #[test]
    fn well_separated_clusters_are_recovered_with_correct_assignments() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut points = Vec::new();
        points.extend(cluster_around([0.0, 0.0], 20, 0.5, &mut rng));
        points.extend(cluster_around([100.0, 0.0], 20, 0.5, &mut rng));
        points.extend(cluster_around([0.0, 100.0], 20, 0.5, &mut rng));

        let result = kmeans(&points, 3, 20, 7);
        assert_eq!(result.centroids.len(), 3);

        // Every point in the same input group must land in the same
        // output cluster as every other point in that group (which
        // cluster index each group gets is unspecified — k-means doesn't
        // promise a label ordering, only a consistent partition).
        let group_of = |i: usize| i / 20;
        for i in 0..points.len() {
            for j in 0..points.len() {
                if group_of(i) == group_of(j) {
                    assert_eq!(
                        result.assignments[i], result.assignments[j],
                        "points {i} and {j} are in the same input group but landed in different clusters"
                    );
                }
            }
        }
    }

    #[test]
    fn k_greater_than_point_count_is_clamped_not_a_panic() {
        let points = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = kmeans(&points, 10, 5, 1);
        assert_eq!(result.centroids.len(), 2);
    }

    #[test]
    fn empty_input_returns_empty_result() {
        let result = kmeans(&[], 5, 5, 1);
        assert!(result.centroids.is_empty());
        assert!(result.assignments.is_empty());
    }

    #[test]
    fn duplicate_points_do_not_panic_kmeans_plus_plus_init() {
        let points = vec![vec![1.0, 1.0]; 10];
        let result = kmeans(&points, 3, 5, 1);
        assert_eq!(result.centroids.len(), 3);
        assert_eq!(result.assignments.len(), 10);
    }

    #[test]
    fn same_seed_gives_deterministic_clustering() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let mut points = Vec::new();
        points.extend(cluster_around([0.0, 0.0], 15, 1.0, &mut rng));
        points.extend(cluster_around([50.0, 50.0], 15, 1.0, &mut rng));

        let a = kmeans(&points, 2, 10, 123);
        let b = kmeans(&points, 2, 10, 123);
        assert_eq!(a.centroids, b.centroids);
        assert_eq!(a.assignments, b.assignments);
    }

    #[test]
    fn single_cluster_centroid_is_the_mean_of_all_points() {
        let points = vec![vec![0.0, 0.0], vec![2.0, 0.0], vec![0.0, 2.0], vec![2.0, 2.0]];
        let result = kmeans(&points, 1, 10, 1);
        assert_eq!(result.centroids.len(), 1);
        assert!((result.centroids[0][0] - 1.0).abs() < 1e-5);
        assert!((result.centroids[0][1] - 1.0).abs() < 1e-5);
    }
}
