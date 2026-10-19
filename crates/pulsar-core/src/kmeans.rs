//! Lloyd's k-means with k-means++ seeding, best of `n_init` runs by inertia.
//!
//! Replaces `sklearn.cluster.KMeans(n_init=10)`: seeded and deterministic,
//! but not bit-compatible with scikit-learn (whose seeding and Elkan/Lloyd
//! variants differ), so parity fixtures that depend on it are partition
//! comparisons (ARI), not label comparisons.

use ndarray::Array2;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::PulsarError;

#[derive(Debug, Clone, PartialEq)]
pub struct KMeans {
    /// Cluster index per row of the input.
    pub labels: Vec<usize>,
    /// `k × d` centroids.
    pub centers: Array2<f64>,
    /// Sum of squared distances of rows to their centroid.
    pub inertia: f64,
}

/// Cluster the rows of `x` into `k` groups. `n_init` independent seeded
/// runs; the one with the lowest inertia wins.
pub fn kmeans(x: &Array2<f64>, k: usize, seed: u64, n_init: usize) -> Result<KMeans, PulsarError> {
    let n = x.nrows();
    if k == 0 || k > n {
        return Err(PulsarError::InvalidParameter {
            msg: format!("k-means needs 1 <= k <= n_rows, got k = {k} for {n} rows"),
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(PulsarError::InvalidParameter {
            msg: "k-means input contains NaN or infinite values".into(),
        });
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut best: Option<KMeans> = None;
    for _ in 0..n_init.max(1) {
        let run_seed: u64 = rng.gen();
        let run = kmeans_once(x, k, run_seed);
        if best.as_ref().is_none_or(|b| run.inertia < b.inertia) {
            best = Some(run);
        }
    }
    Ok(best.expect("n_init >= 1"))
}

fn kmeans_once(x: &Array2<f64>, k: usize, seed: u64) -> KMeans {
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
    let mut center_arr = Array2::<f64>::zeros((k, d));
    for (c, ctr) in centers.iter().enumerate() {
        for (j, v) in ctr.iter().enumerate() {
            center_arr[[c, j]] = *v;
        }
    }
    KMeans {
        labels,
        centers: center_arr,
        inertia,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_well_separated_groups() {
        let mut rows = Vec::new();
        for (cx, cy) in [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)] {
            for k in 0..20 {
                let t = k as f64 * 0.01;
                rows.push([cx + t, cy - t]);
            }
        }
        let x = Array2::from_shape_vec((60, 2), rows.concat()).unwrap();
        let result = kmeans(&x, 3, 7, 10).unwrap();
        for g in 0..3 {
            let first = result.labels[g * 20];
            assert!(result.labels[g * 20..(g + 1) * 20]
                .iter()
                .all(|&l| l == first));
        }
        let mut distinct = result.labels.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 3);
        assert!(result.inertia < 1.0);
        assert_eq!(result.centers.dim(), (3, 2));
        assert_eq!(
            kmeans(&x, 3, 7, 10).unwrap(),
            result,
            "seeded runs are deterministic"
        );
    }

    #[test]
    fn rejects_bad_k_and_non_finite_input() {
        let x = Array2::<f64>::zeros((4, 2));
        assert!(matches!(
            kmeans(&x, 0, 1, 1),
            Err(PulsarError::InvalidParameter { .. })
        ));
        assert!(matches!(
            kmeans(&x, 5, 1, 1),
            Err(PulsarError::InvalidParameter { .. })
        ));
        let mut bad = x.clone();
        bad[[1, 1]] = f64::NAN;
        assert!(matches!(
            kmeans(&bad, 2, 1, 1),
            Err(PulsarError::InvalidParameter { .. })
        ));
    }
}
