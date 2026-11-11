use kiddo::{ImmutableKdTree, SquaredEuclidean};
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use std::collections::HashSet;

/// Squared Euclidean distance between two equal-length slices.
/// Using squared distance avoids sqrt, which is sufficient for comparisons.
#[inline(always)]
fn l2_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Core Ball Mapper algorithm optimized for large datasets.
///
/// Returns `(nodes, edges)` where:
/// - `nodes[k]` is the list of point indices in ball `k`
/// - `edges` contains pairs `(a, b)` with `a < b` for balls sharing points
///
/// Complexity: O(n * k) for centre selection + O(n * k) for membership
/// where k = number of balls (typically k << n for reasonable epsilon).
///
/// Callers must pass finite points and a finite `eps > 0`; this function does
/// not re-check them. A non-finite row is not within any distance of itself,
/// so it would become a centre with an empty ball, and `eps < 0` behaves as
/// `|eps|`. The projections that feed Ball Mapper (`jl_grid`, `pca_grid`)
/// reject non-finite input, and the pipeline rejects non-positive epsilons.
pub fn fit_inner(points: ArrayView2<f64>, eps: f64) -> (Vec<Vec<usize>>, Vec<(usize, usize)>) {
    let n = points.nrows();
    let d = points.ncols();
    let eps_sq = eps * eps;
    assert!(d > 0, "BallMapper input must have at least one column");
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Step 1: greedy centre selection - O(n * k) where k = num centres
    let mut center_indices: Vec<usize> = Vec::new();
    'outer: for i in 0..n {
        let pi = points.row(i).to_slice().unwrap();
        for &c in &center_indices {
            let pc = points.row(c).to_slice().unwrap();
            if l2_sq(pi, pc) <= eps_sq {
                continue 'outer;
            }
        }
        center_indices.push(i);
    }

    // Step 2: build membership. Low-dimensional projected data uses a KD-tree
    // radius query; wider embeddings keep the allocation-free linear path.
    let nodes = match d {
        1 => fit_inner_kd::<1>(points, &center_indices, eps_sq),
        2 => fit_inner_kd::<2>(points, &center_indices, eps_sq),
        3 => fit_inner_kd::<3>(points, &center_indices, eps_sq),
        4 => fit_inner_kd::<4>(points, &center_indices, eps_sq),
        5 => fit_inner_kd::<5>(points, &center_indices, eps_sq),
        6 => fit_inner_kd::<6>(points, &center_indices, eps_sq),
        7 => fit_inner_kd::<7>(points, &center_indices, eps_sq),
        8 => fit_inner_kd::<8>(points, &center_indices, eps_sq),
        9 => fit_inner_kd::<9>(points, &center_indices, eps_sq),
        10 => fit_inner_kd::<10>(points, &center_indices, eps_sq),
        11 => fit_inner_kd::<11>(points, &center_indices, eps_sq),
        12 => fit_inner_kd::<12>(points, &center_indices, eps_sq),
        13 => fit_inner_kd::<13>(points, &center_indices, eps_sq),
        14 => fit_inner_kd::<14>(points, &center_indices, eps_sq),
        15 => fit_inner_kd::<15>(points, &center_indices, eps_sq),
        16 => fit_inner_kd::<16>(points, &center_indices, eps_sq),
        _ => fit_inner_linear(points, &center_indices, eps_sq),
    };

    // Step 3: build edges - O(k²) but k is small
    let n_balls = nodes.len();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    #[allow(clippy::needless_range_loop)]
    for a in 0..n_balls {
        let set_a: HashSet<usize> = nodes[a].iter().copied().collect();
        for b in (a + 1)..n_balls {
            if nodes[b].iter().any(|x| set_a.contains(x)) {
                edges.push((a, b));
            }
        }
    }

    (nodes, edges)
}

fn fit_inner_linear(
    points: ArrayView2<f64>,
    center_indices: &[usize],
    eps_sq: f64,
) -> Vec<Vec<usize>> {
    let n = points.nrows();
    center_indices
        .iter()
        .map(|&c| {
            let pc = points.row(c).to_slice().unwrap();
            (0..n)
                .filter(|&i| {
                    let pi = points.row(i).to_slice().unwrap();
                    l2_sq(pi, pc) <= eps_sq
                })
                .collect()
        })
        .collect()
}

fn fit_inner_kd<const K: usize>(
    points: ArrayView2<f64>,
    center_indices: &[usize],
    eps_sq: f64,
) -> Vec<Vec<usize>> {
    // Built in one pass by median splits, the immutable tree accepts any number
    // of repeated points. The mutable `KdTree::add` panics once more points than
    // fit in a leaf bucket are identical (nothing left to split them on), and
    // repeated rows are routine in tabular data.
    let coords: Vec<[f64; K]> = (0..points.nrows())
        .map(|i| point_array::<K>(points, i))
        .collect();
    let tree: ImmutableKdTree<f64, K> = ImmutableKdTree::new_from_slice(&coords);

    center_indices
        .iter()
        .map(|&c| {
            let mut members: Vec<usize> = tree
                .within_unsorted::<SquaredEuclidean>(&coords[c], eps_sq)
                .into_iter()
                .map(|entry| entry.item as usize)
                .collect();
            members.sort_unstable();
            members
        })
        .collect()
}

fn point_array<const K: usize>(points: ArrayView2<f64>, row: usize) -> [f64; K] {
    let mut out = [0.0; K];
    for j in 0..K {
        out[j] = points[[row, j]];
    }
    out
}

/// A fitted Ball Mapper complex.
///
/// Ball Mapper decomposes a point cloud into overlapping balls and
/// represents connectivity as a graph. Designed for large-scale EHR data.
pub struct BallMapper {
    pub eps: f64,
    pub nodes: Vec<Vec<usize>>,
    pub edges: Vec<(usize, usize)>,
}

impl BallMapper {
    /// Create a new Ball Mapper with given radius.
    pub fn new(eps: f64) -> Self {
        BallMapper {
            eps,
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Fit the Ball Mapper to a point cloud.
    pub fn fit(&mut self, points: ArrayView2<f64>) {
        let (nodes, edges) = fit_inner(points, self.eps);
        self.nodes = nodes;
        self.edges = edges;
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
}

/// Run Ball Mapper for every (embedding, epsilon) pair in parallel.
///
/// This is the main entry point for grid search. Parallelised across all
/// combinations using rayon for maximum throughput on large datasets.
///
/// Complexity per fit: O(n * k) where k = number of balls.
/// No O(n²) memory allocation - scales to large EHR datasets.
pub fn ball_mapper_grid(embeddings: &[Array2<f64>], epsilons: &[f64]) -> Vec<BallMapper> {
    embeddings
        .par_iter()
        .flat_map(|emb| {
            let eps_clone = epsilons.to_vec();
            eps_clone
                .into_par_iter()
                .map(move |eps| {
                    let (nodes, edges) = fit_inner(emb.view(), eps);
                    BallMapper { eps, nodes, edges }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    /// Greedy centres exactly as `fit_inner` selects them.
    fn greedy_centers(points: ArrayView2<f64>, eps_sq: f64) -> Vec<usize> {
        let mut centers: Vec<usize> = Vec::new();
        for i in 0..points.nrows() {
            let pi = points.row(i).to_slice().unwrap();
            if !centers
                .iter()
                .any(|&c| l2_sq(pi, points.row(c).to_slice().unwrap()) <= eps_sq)
            {
                centers.push(i);
            }
        }
        centers
    }

    /// Ball membership from the KD-tree path must equal the linear scan for
    /// every centre. `centers` also probes non-greedy query points.
    fn assert_kd_matches_linear<const K: usize>(points: &Array2<f64>, eps: f64) {
        assert_eq!(points.ncols(), K);
        let eps_sq = eps * eps;
        let mut centers = greedy_centers(points.view(), eps_sq);
        centers.extend([0, points.nrows() / 2, points.nrows() - 1]);
        let kd = fit_inner_kd::<K>(points.view(), &centers, eps_sq);
        let linear = fit_inner_linear(points.view(), &centers, eps_sq);
        assert_eq!(kd, linear, "eps {eps}");
    }

    #[test]
    fn kd_membership_with_33_identical_points() {
        // One more than kiddo's leaf bucket: the mutable tree panicked here.
        let points = Array2::from_elem((33, 2), 1.5);
        for eps in [0.0, 0.5] {
            assert_kd_matches_linear::<2>(&points, eps);
        }
        let (nodes, edges) = fit_inner(points.view(), 0.5);
        assert_eq!(nodes, vec![(0..33).collect::<Vec<_>>()]);
        assert!(edges.is_empty());
    }

    #[test]
    fn kd_membership_with_5000_identical_points() {
        let points = Array2::<f64>::zeros((5_000, 3));
        for eps in [0.0, 0.5] {
            assert_kd_matches_linear::<3>(&points, eps);
        }
        let (nodes, edges) = fit_inner(points.view(), 0.5);
        assert_eq!(nodes, vec![(0..5_000).collect::<Vec<_>>()]);
        assert!(edges.is_empty());
    }

    #[test]
    fn kd_membership_with_shared_axis_value() {
        // Distinct rows whose first coordinate is constant.
        let points = Array2::from_shape_fn(
            (5_000, 2),
            |(i, j)| {
                if j == 0 {
                    0.0
                } else {
                    i as f64 * 1e-3
                }
            },
        );
        for eps in [0.05, 1.0] {
            assert_kd_matches_linear::<2>(&points, eps);
        }
    }

    #[test]
    fn kd_membership_with_random_points_and_duplicate_blobs() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut rows: Vec<[f64; 2]> = (0..2_000)
            .map(|_| [rng.gen_range(-5.0..5.0), rng.gen_range(-5.0..5.0)])
            .collect();
        rows.extend(std::iter::repeat_n([1.0, 1.0], 1_000));
        rows.extend(std::iter::repeat_n([1.0, -2.0], 1_000));
        let points = Array2::from_shape_fn((rows.len(), 2), |(i, j)| rows[i][j]);
        // The blobs are 3.0 apart, so eps 3.0 puts each exactly on the other's
        // ball boundary.
        for eps in [0.5, 1.5, 3.0] {
            assert_kd_matches_linear::<2>(&points, eps);
        }
    }

    #[test]
    fn fit_inner_kd_and_linear_dimensions_agree_on_heavy_repeats() {
        // 20 distinct rows (one constant column) each repeated 1,000 times.
        // Zero-padding to 17 columns leaves distances unchanged but routes
        // membership through the linear path, so both paths must agree.
        let mut rng = StdRng::seed_from_u64(11);
        let distinct: Vec<[f64; 3]> = (0..20)
            .map(|_| [rng.gen_range(0.0..3.0), 0.0, rng.gen_range(0.0..3.0)])
            .collect();
        let narrow = Array2::from_shape_fn((20_000, 3), |(i, j)| distinct[i % 20][j]);
        let mut wide = Array2::<f64>::zeros((20_000, 17));
        wide.slice_mut(ndarray::s![.., ..3]).assign(&narrow);
        for eps in [0.0, 0.3, 0.8, 2.0] {
            assert_eq!(
                fit_inner(narrow.view(), eps),
                fit_inner(wide.view(), eps),
                "eps {eps}"
            );
        }
    }

    #[test]
    fn fit_inner_empty_input_returns_no_balls() {
        // Both membership paths: KD-tree (d = 2) and linear scan (d = 17).
        for d in [2, 17] {
            let points = Array2::<f64>::zeros((0, d));
            let (nodes, edges) = fit_inner(points.view(), 0.5);
            assert!(nodes.is_empty(), "d {d}");
            assert!(edges.is_empty(), "d {d}");
        }
    }

    #[test]
    fn fit_inner_single_point_is_one_ball() {
        for d in [2, 17] {
            let points = Array2::from_elem((1, d), 0.25);
            for eps in [0.0, 0.5] {
                let (nodes, edges) = fit_inner(points.view(), eps);
                assert_eq!(nodes, vec![vec![0]], "d {d} eps {eps}");
                assert!(edges.is_empty(), "d {d} eps {eps}");
            }
        }
    }
}
