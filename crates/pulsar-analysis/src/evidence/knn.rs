//! k-NN neighbourhood overlap between two representations of the same rows
//! (`neighborhood_overlap`).

use ndarray::{Array2, ArrayView2};

use super::EvidenceError;

/// Z-score each column (ddof = 0); zero-variance columns become 0.
fn zscore_columns(x: ArrayView2<f64>) -> Array2<f64> {
    let (n, p) = x.dim();
    let mut z = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        let col = x.column(j);
        let mean = col.sum() / n as f64;
        let std = (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        if std > 0.0 {
            for i in 0..n {
                z[[i, j]] = (x[[i, j]] - mean) / std;
            }
        }
    }
    z
}

/// `(n, k)` Euclidean k-NN indices excluding the row itself.
///
/// Brute force with `O(n)` scratch per row (no `n x n` matrix). Ties break
/// by index; like the sklearn reference, a row whose own index is not among
/// its `k + 1` nearest (many exact duplicates) drops its farthest instead.
fn knn_indices(z: &Array2<f64>, k: usize) -> Vec<Vec<usize>> {
    use rayon::prelude::*;
    let n = z.nrows();
    // Rows are independent, so the result does not depend on thread count.
    (0..n)
        .into_par_iter()
        .map_init(
            || Vec::<(f64, usize)>::with_capacity(n),
            |dist, i| {
                dist.clear();
                let ri = z.row(i);
                dist.extend((0..n).map(|j| {
                    let d: f64 = ri
                        .iter()
                        .zip(z.row(j))
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (d, j)
                }));
                let cmp =
                    |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
                dist.select_nth_unstable_by(k, cmp);
                let mut near: Vec<(f64, usize)> = dist[..=k].to_vec();
                near.sort_by(cmp);
                match near.iter().position(|&(_, j)| j == i) {
                    Some(p) => {
                        near.remove(p);
                    }
                    None => {
                        near.pop();
                    }
                }
                near.into_iter().map(|(_, j)| j).collect()
            },
        )
        .collect()
}

/// k-NN lists of `x` after column z-scoring, for callers that reuse one side
/// across many comparisons (zero-variance and all-zero padding columns do not
/// change them). Caller guarantees `1 <= k < n_rows` and finite values.
pub(crate) fn neighbor_lists(x: ArrayView2<f64>, k: usize) -> Vec<Vec<usize>> {
    knn_indices(&zscore_columns(x), k)
}

/// Mean per-row Jaccard overlap of two k-NN lists.
pub(crate) fn mean_jaccard(a: &[Vec<usize>], b: &[Vec<usize>], k: usize) -> f64 {
    let total: f64 = a
        .iter()
        .zip(b)
        .map(|(a, b)| {
            let inter = a.iter().filter(|i| b.contains(i)).count() as f64;
            inter / (2.0 * k as f64 - inter)
        })
        .sum();
    total / a.len() as f64
}

/// Mean per-row Jaccard overlap of the Euclidean k-NN sets of `x_a` and
/// `x_b` after column z-scoring.
pub fn neighborhood_overlap(
    x_a: ArrayView2<f64>,
    x_b: ArrayView2<f64>,
    k: usize,
) -> Result<f64, EvidenceError> {
    let invalid = |m: String| Err(EvidenceError::InvalidArgument(m));
    if x_a.dim() != x_b.dim() {
        return invalid(format!(
            "X_a and X_b must have the same shape (2D), got {:?} and {:?}.",
            x_a.dim(),
            x_b.dim()
        ));
    }
    let (n_rows, n_cols) = x_a.dim();
    if n_rows < 2 {
        return invalid(format!(
            "X_a and X_b must have at least 2 rows, got {n_rows}."
        ));
    }
    if n_cols < 1 {
        return invalid("X_a and X_b must have at least 1 column.".into());
    }
    if k < 1 || k >= n_rows {
        return invalid(format!(
            "k must be an integer with 1 <= k < n_rows ({n_rows}), got {k}."
        ));
    }
    if !(x_a.iter().all(|v| v.is_finite()) && x_b.iter().all(|v| v.is_finite())) {
        return invalid("X_a and X_b must contain only finite values.".into());
    }

    Ok(mean_jaccard(
        &neighbor_lists(x_a, k),
        &neighbor_lists(x_b, k),
        k,
    ))
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn identical_representations_overlap_fully() {
        let x = array![[0.0, 0.0], [1.0, 0.1], [2.0, 0.3], [3.0, 0.2], [4.0, 0.9]];
        assert_eq!(neighborhood_overlap(x.view(), x.view(), 2).unwrap(), 1.0);
    }

    #[test]
    fn scale_does_not_change_overlap_but_reordering_does() {
        let x = array![[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]];
        let scaled = &x * 100.0;
        assert_eq!(
            neighborhood_overlap(x.view(), scaled.view(), 2).unwrap(),
            1.0
        );
        // Interleaving the clusters changes every neighbourhood.
        let shuffled = array![[0.0], [10.0], [1.0], [11.0], [2.0], [12.0]];
        assert!(neighborhood_overlap(x.view(), shuffled.view(), 2).unwrap() < 1.0);
    }

    #[test]
    fn known_answer_line_of_points() {
        // Distinct gaps, so no distance ties. k = 1 neighbours of a:
        // 0 -> 1, 1 -> 0, 2 -> 1, 3 -> 2. Swapping the values of rows 0 and
        // 1 in b gives 0 -> 1, 1 -> 0, 2 -> 0, 3 -> 2. Three rows agree
        // (Jaccard 1), row 2 disagrees (0): mean 0.75.
        let a = array![[0.0], [1.0], [3.0], [7.0]];
        let b = array![[1.0], [0.0], [3.0], [7.0]];
        assert_eq!(neighborhood_overlap(a.view(), b.view(), 1).unwrap(), 0.75);
    }

    #[test]
    fn rejects_bad_arguments_with_specific_messages() {
        let x = array![[0.0], [1.0], [2.0]];
        let y = array![[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]];
        let msg = |r: Result<f64, EvidenceError>| r.unwrap_err().to_string();
        assert!(msg(neighborhood_overlap(x.view(), y.view(), 1)).contains("same shape"));
        assert!(msg(neighborhood_overlap(x.view(), x.view(), 3)).contains("1 <= k < n_rows (3)"));
        assert!(msg(neighborhood_overlap(x.view(), x.view(), 0)).contains("got 0"));
        let one = array![[1.0]];
        assert!(msg(neighborhood_overlap(one.view(), one.view(), 1)).contains("at least 2 rows"));
        let nan = array![[0.0], [f64::NAN], [1.0]];
        assert!(msg(neighborhood_overlap(nan.view(), nan.view(), 1)).contains("only finite"));
    }
}
