//! Representative ball-map selection: k-means++ over cheap structural
//! features `(n_nodes, n_edges, eps)`, one representative per cluster.
//!
//! Replaces `sklearn.cluster.KMeans(n_init=10, random_state=42)`. The
//! clustering is seeded and deterministic, but it is not bit-compatible with
//! scikit-learn: fixtures for this stage are informational only.

use ndarray::Array2;
use pulsar_core::BallMapper;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Indices of up to `n_reps` ball maps, one per k-means cluster (the member
/// closest to its centroid), in cluster order. Returns every index when
/// `n_reps >= maps.len()`. A cluster that ends up empty (fewer distinct
/// feature rows than `n_reps`) contributes nothing.
pub fn select_representatives(maps: &[BallMapper], n_reps: usize, seed: u64) -> Vec<usize> {
    let n_maps = maps.len();
    if n_reps == 0 || n_maps == 0 {
        return Vec::new();
    }
    if n_reps >= n_maps {
        return (0..n_maps).collect();
    }
    let mut features = Array2::<f64>::zeros((n_maps, 3));
    for (i, m) in maps.iter().enumerate() {
        features[[i, 0]] = m.n_nodes() as f64;
        features[[i, 1]] = m.n_edges() as f64;
        features[[i, 2]] = m.eps;
    }
    // Standardize: (x - mean) / (population std + 1e-10).
    for j in 0..3 {
        let col = features.column(j).to_owned();
        let mean = col.sum() / n_maps as f64;
        let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_maps as f64;
        let std = var.sqrt() + 1e-10;
        features.column_mut(j).mapv_inplace(|x| (x - mean) / std);
    }
    let labels = kmeans(&features, n_reps, seed, 10);
    let mut reps = Vec::with_capacity(n_reps);
    for k in 0..n_reps {
        let members: Vec<usize> = (0..n_maps).filter(|&i| labels[i] == k).collect();
        if members.is_empty() {
            continue;
        }
        let mut centroid = [0.0f64; 3];
        for &i in &members {
            for j in 0..3 {
                centroid[j] += features[[i, j]];
            }
        }
        for c in &mut centroid {
            *c /= members.len() as f64;
        }
        let best = members
            .iter()
            .copied()
            .min_by(|&a, &b| {
                dist2(&features, a, &centroid).total_cmp(&dist2(&features, b, &centroid))
            })
            .expect("members is non-empty");
        reps.push(best);
    }
    reps
}

fn dist2(x: &Array2<f64>, i: usize, c: &[f64; 3]) -> f64 {
    (0..3).map(|j| (x[[i, j]] - c[j]).powi(2)).sum()
}

/// Lloyd's algorithm with k-means++ seeding, best of `n_init` runs by
/// inertia. Rows of `x` are points.
fn kmeans(x: &Array2<f64>, k: usize, seed: u64, n_init: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut best: Option<(f64, Vec<usize>)> = None;
    for _ in 0..n_init {
        let run_seed: u64 = rng.gen();
        let (inertia, labels) = kmeans_once(x, k, run_seed);
        if best.as_ref().is_none_or(|(b, _)| inertia < *b) {
            best = Some((inertia, labels));
        }
    }
    best.map(|(_, l)| l).unwrap_or_default()
}

fn kmeans_once(x: &Array2<f64>, k: usize, seed: u64) -> (f64, Vec<usize>) {
    let n = x.nrows();
    let d = x.ncols();
    let mut rng = StdRng::seed_from_u64(seed);
    let row = |i: usize| x.row(i).to_vec();
    let sq = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum() };

    // k-means++ seeding.
    let mut centers: Vec<Vec<f64>> = vec![row(rng.gen_range(0..n))];
    let mut d2: Vec<f64> = (0..n).map(|i| sq(&row(i), &centers[0])).collect();
    while centers.len() < k {
        let total: f64 = d2.iter().sum();
        let next = if total > 0.0 {
            WeightedIndex::new(&d2)
                .map(|w| w.sample(&mut rng))
                .unwrap_or_else(|_| rng.gen_range(0..n))
        } else {
            rng.gen_range(0..n)
        };
        centers.push(row(next));
        let newest = &centers[centers.len() - 1];
        for (i, slot) in d2.iter_mut().enumerate() {
            let dd = sq(&row(i), newest);
            if dd < *slot {
                *slot = dd;
            }
        }
    }

    let mut labels = vec![0usize; n];
    let mut inertia = 0.0;
    for _ in 0..300 {
        inertia = 0.0;
        for (i, label) in labels.iter_mut().enumerate() {
            let r = row(i);
            let (best, dist) = centers
                .iter()
                .enumerate()
                .map(|(c, ctr)| (c, sq(&r, ctr)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("k >= 1");
            *label = best;
            inertia += dist;
        }
        let mut sums = vec![vec![0.0f64; d]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            counts[labels[i]] += 1;
            for j in 0..d {
                sums[labels[i]][j] += x[[i, j]];
            }
        }
        let mut shift = 0.0;
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            let new: Vec<f64> = sums[c].iter().map(|s| s / counts[c] as f64).collect();
            shift += sq(&new, &centers[c]);
            centers[c] = new;
        }
        if shift <= 1e-8 {
            break;
        }
    }
    (inertia, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(nodes: usize, edges: usize, eps: f64) -> BallMapper {
        BallMapper {
            eps,
            nodes: (0..nodes).map(|i| vec![i]).collect(),
            edges: (0..edges).map(|i| (i, i + 1)).collect(),
        }
    }

    #[test]
    fn picks_one_map_per_well_separated_group() {
        // Three obvious groups of ball maps; each group has an exact centre.
        let mut maps = Vec::new();
        for (nodes, edges, eps) in [(10, 5, 0.1), (100, 80, 0.5), (1000, 2000, 1.0)] {
            maps.push(map(nodes - 1, edges, eps));
            maps.push(map(nodes, edges, eps));
            maps.push(map(nodes + 1, edges, eps));
        }
        let reps = select_representatives(&maps, 3, 42);
        let mut groups: Vec<usize> = reps.iter().map(|&i| i / 3).collect();
        groups.sort_unstable();
        assert_eq!(groups, [0, 1, 2], "one representative per group: {reps:?}");
        // The centre of each group is the member closest to the centroid.
        assert!(reps.iter().all(|&i| i % 3 == 1), "{reps:?}");
        assert_eq!(select_representatives(&maps, 20, 42).len(), 9);
        assert!(select_representatives(&maps, 0, 42).is_empty());
    }
}
