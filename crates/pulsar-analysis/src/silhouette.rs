//! Silhouette coefficients (`sklearn.metrics.silhouette_samples` with a
//! precomputed distance), generic over how a pairwise distance is obtained.

use crate::error::AnalysisError;

/// Per-sample silhouette `(b - a) / max(a, b)`, where `a` is the mean
/// distance to the sample's own cluster and `b` the smallest mean distance
/// to another cluster. Singleton clusters score 0, as in scikit-learn.
/// `distance(i, j)` must be symmetric with `distance(i, i) == 0`.
pub fn silhouette_samples(
    labels: &[usize],
    distance: impl Fn(usize, usize) -> f64,
) -> Result<Vec<f64>, AnalysisError> {
    let n = labels.len();
    let k = labels.iter().copied().max().map_or(0, |m| m + 1);
    let mut sizes = vec![0usize; k];
    for &l in labels {
        sizes[l] += 1;
    }
    let n_labels = sizes.iter().filter(|&&s| s > 0).count();
    if n < 2 || n_labels < 2 || n_labels > n - 1 {
        return Err(AnalysisError::InvalidArgument(format!(
            "silhouette needs 2 <= n_labels <= n_samples - 1, got {n_labels} labels for {n} samples"
        )));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let own = labels[i];
        if sizes[own] <= 1 {
            out.push(0.0);
            continue;
        }
        let mut sums = vec![0.0_f64; k];
        for j in 0..n {
            if j != i {
                sums[labels[j]] += distance(i, j);
            }
        }
        let a = sums[own] / (sizes[own] - 1) as f64;
        let b = (0..k)
            .filter(|&c| c != own && sizes[c] > 0)
            .map(|c| sums[c] / sizes[c] as f64)
            .fold(f64::INFINITY, f64::min);
        let denom = a.max(b);
        out.push(if denom > 0.0 { (b - a) / denom } else { 0.0 });
    }
    Ok(out)
}

/// Mean silhouette over all samples.
pub fn silhouette_score(
    labels: &[usize],
    distance: impl Fn(usize, usize) -> f64,
) -> Result<f64, AnalysisError> {
    let s = silhouette_samples(labels, distance)?;
    Ok(s.iter().sum::<f64>() / s.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answer_matches_sklearn() {
        // Two tight pairs far apart on a line. sklearn silhouette_samples:
        // outer points (10.5 - 1) / 10.5, inner points (9.5 - 1) / 9.5.
        let x: [f64; 4] = [0.0, 1.0, 10.0, 11.0];
        let labels = [0, 0, 1, 1];
        let d = |i: usize, j: usize| (x[i] - x[j]).abs();
        let s = silhouette_samples(&labels, d).unwrap();
        let want = [9.5 / 10.5, 8.5 / 9.5, 8.5 / 9.5, 9.5 / 10.5];
        for (got, want) in s.iter().zip(want) {
            assert!((got - want).abs() < 1e-12, "{s:?}");
        }
        let mean = want.iter().sum::<f64>() / 4.0;
        assert!((silhouette_score(&labels, d).unwrap() - mean).abs() < 1e-12);

        // Singleton clusters score 0; sklearn gives [0.9, 0.8889, 0, 0].
        let labels = [0, 0, 1, 2];
        let s = silhouette_samples(&labels, d).unwrap();
        assert_eq!(s[2], 0.0);
        assert_eq!(s[3], 0.0);
        assert!((s[0] - 0.9).abs() < 1e-12, "{s:?}");
        assert!((s[1] - 8.0 / 9.0).abs() < 1e-12, "{s:?}");

        assert_eq!(
            silhouette_score(&[0, 0, 0, 0], d).unwrap_err().code(),
            "INVALID_ARGUMENT"
        );
        assert_eq!(
            silhouette_score(&[0, 1, 2, 3], d).unwrap_err().code(),
            "INVALID_ARGUMENT"
        );
    }
}
