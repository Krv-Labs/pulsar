use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::ballmapper::BallMapper;

/// Compute pseudo-Laplacian matrix from Ball Mapper node membership.
///
/// Given n data points and a list of balls, the pseudo-Laplacian L is n×n where:
/// - L[i,i] = number of balls containing point i
/// - L[i,j] for i≠j = negative count of balls containing both i and j
///
/// This reflects topological proximity: frequently co-occurring points have
/// strong negative off-diagonal entries.
pub fn pseudo_laplacian_inner(nodes: &[Vec<usize>], n: usize) -> Array2<i64> {
    let mut l = Array2::<i64>::zeros((n, n));
    for members in nodes {
        for &i in members {
            for &j in members {
                if i == j {
                    l[[i, j]] += 1;
                } else {
                    l[[i, j]] -= 1;
                }
            }
        }
    }
    l
}

/// Accumulate pseudo-Laplacians from all ball maps in parallel.
///
/// This is the optimized entry point that replaces sequential Python loops.
/// Uses rayon parallel map-reduce for maximum throughput.
///
/// ```python
/// # Single call replaces 4000+ Python/Rust crossings
/// galactic_L = accumulate_pseudo_laplacians(ball_maps, n)
/// ```
#[pyfunction]
pub fn accumulate_pseudo_laplacians<'py>(
    py: Python<'py>,
    ball_maps: Vec<PyRef<'py, BallMapper>>,
    n: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let all_nodes: Vec<&Vec<Vec<usize>>> = ball_maps.iter().map(|bm| &bm.nodes).collect();

    let galactic_l: Array2<i64> = all_nodes
        .par_iter()
        .map(|nodes| pseudo_laplacian_inner(nodes, n))
        .reduce(
            || Array2::<i64>::zeros((n, n)),
            |mut acc, l| {
                acc += &l;
                acc
            },
        );

    Ok(galactic_l.into_pyarray_bound(py))
}

// ---------------------------------------------------------------------------
// Sparse accumulation
//
// The pseudo-Laplacian of Ball-Mapper co-membership is intrinsically sparse: an
// off-diagonal entry is nonzero only when two points share a ball. Materializing
// the full n×n matrix wastes O(n²) memory on structural zeros. The sparse path
// tracks the O(n) diagonal and the upper-triangle (i<j) off-diagonal as a COO
// edge list, never allocating n×n. This is an *exact* representation change — it
// reconstructs the dense matrix bit-for-bit (see `accumulate_pseudo_laplacians_sparse`
// parity tests), not an approximation.
// ---------------------------------------------------------------------------

/// Per-ball-map sparse contribution: O(n) diagonal counts plus upper-triangle
/// off-diagonal co-occurrence counts (`i < j`, magnitude stored as a positive
/// integer; the negative Laplacian sign is applied at weight time).
struct SparseLaplacianContribution {
    diag: Vec<i64>,
    offdiag: Vec<(usize, usize, i64)>,
}

/// Sparse pseudo-Laplacian: diagonal counts + deduped, `(i,j)`-sorted upper-triangle
/// off-diagonal co-occurrence counts. Feeds `CosmicGraph.from_pseudo_laplacian_sparse`
/// directly without densifying.
#[pyclass]
pub struct SparsePseudoLaplacian {
    pub n: usize,
    pub diag: Vec<i64>,
    pub offdiag: Vec<(usize, usize, i64)>,
}

/// Sort a COO buffer by `(i, j)` and merge duplicate entries by summing counts,
/// dropping any pair whose total is zero. Produces a deterministic edge list.
fn sort_and_merge(mut coo: Vec<(usize, usize, i64)>) -> Vec<(usize, usize, i64)> {
    coo.sort_unstable_by_key(|&(i, j, _)| (i, j));
    let mut out: Vec<(usize, usize, i64)> = Vec::with_capacity(coo.len());
    for (i, j, c) in coo {
        match out.last_mut() {
            Some(last) if last.0 == i && last.1 == j => last.2 += c,
            _ => out.push((i, j, c)),
        }
    }
    out.retain(|&(_, _, c)| c != 0);
    out
}

/// Sparse counterpart of [`pseudo_laplacian_inner`]. Emits one `(i, j, 1)` per
/// co-occurring pair (`i < j`) per ball and a `+1` diagonal bump per membership.
/// Duplicate pairs (a pair sharing several balls within one map) are summed by
/// the caller's merge.
fn pseudo_laplacian_inner_sparse(nodes: &[Vec<usize>], n: usize) -> SparseLaplacianContribution {
    let mut diag = vec![0i64; n];
    let mut offdiag = Vec::new();
    for members in nodes {
        for &i in members {
            diag[i] += 1;
        }
        for a in 0..members.len() {
            for b in (a + 1)..members.len() {
                let (mut i, mut j) = (members[a], members[b]);
                if i == j {
                    continue; // defensive: members are expected to be distinct
                }
                if i > j {
                    std::mem::swap(&mut i, &mut j);
                }
                offdiag.push((i, j, 1));
            }
        }
    }
    SparseLaplacianContribution { diag, offdiag }
}

#[pymethods]
impl SparsePseudoLaplacian {
    /// Number of points (matrix dimension n).
    #[getter]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Diagonal counts `diag[i]` = number of (ball-map, ball) pairs containing i.
    #[getter]
    pub fn diag<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.diag.clone().into_pyarray_bound(py)
    }

    /// Upper-triangle off-diagonal co-occurrence counts `(i, j, count)` with `i < j`,
    /// sorted by `(i, j)`.
    #[getter]
    pub fn offdiag(&self) -> Vec<(usize, usize, i64)> {
        self.offdiag.clone()
    }

    /// Number of stored off-diagonal entries (nonzeros in the upper triangle).
    #[getter]
    pub fn nnz(&self) -> usize {
        self.offdiag.len()
    }

    /// Fold another sparse Laplacian (same n) into this one: sum diagonals and
    /// merge off-diagonal counts. Used to accumulate across datasets in `fit_multi`
    /// without ever building an n×n matrix.
    pub fn merge_in_place(&mut self, other: PyRef<SparsePseudoLaplacian>) -> PyResult<()> {
        if other.n != self.n {
            return Err(crate::error::PulsarError::ShapeMismatch {
                expected: format!("n = {}", self.n),
                got: format!("n = {}", other.n),
            }
            .into());
        }
        for (acc, &add) in self.diag.iter_mut().zip(other.diag.iter()) {
            *acc += add;
        }
        let mut combined =
            Vec::with_capacity(self.offdiag.len() + other.offdiag.len());
        combined.append(&mut self.offdiag);
        combined.extend_from_slice(&other.offdiag);
        self.offdiag = sort_and_merge(combined);
        Ok(())
    }
}

/// Sparse counterpart of [`accumulate_pseudo_laplacians`]. Accumulates the
/// co-membership pseudo-Laplacian across all ball maps as a COO edge list plus an
/// O(n) diagonal, never allocating an n×n matrix.
///
/// Reduce strategy: each ball map maps to a thread-local `(diag, offdiag)`
/// contribution; the rayon reduce concatenates COO buffers (memcpy-cheap) and adds
/// diagonals; a single final sort+merge dedups the off-diagonal deterministically.
#[pyfunction]
pub fn accumulate_pseudo_laplacians_sparse<'py>(
    _py: Python<'py>,
    ball_maps: Vec<PyRef<'py, BallMapper>>,
    n: usize,
) -> PyResult<SparsePseudoLaplacian> {
    let all_nodes: Vec<&Vec<Vec<usize>>> = ball_maps.iter().map(|bm| &bm.nodes).collect();

    let (diag, raw_offdiag): (Vec<i64>, Vec<(usize, usize, i64)>) = all_nodes
        .par_iter()
        .map(|nodes| {
            let c = pseudo_laplacian_inner_sparse(nodes, n);
            (c.diag, c.offdiag)
        })
        .reduce(
            || (vec![0i64; n], Vec::new()),
            |mut acc, mut item| {
                // Sum diagonals, concat off-diagonals (deduped once at the end).
                for (a, b) in acc.0.iter_mut().zip(item.0.iter()) {
                    *a += *b;
                }
                acc.1.append(&mut item.1);
                acc
            },
        );

    let offdiag = sort_and_merge(raw_offdiag);
    Ok(SparsePseudoLaplacian { n, diag, offdiag })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reconstruct a dense matrix from a sparse contribution for comparison.
    fn dense_from_sparse(diag: &[i64], offdiag: &[(usize, usize, i64)], n: usize) -> Array2<i64> {
        let mut l = Array2::<i64>::zeros((n, n));
        for (i, &d) in diag.iter().enumerate() {
            l[[i, i]] = d;
        }
        for &(i, j, c) in offdiag {
            l[[i, j]] -= c;
            l[[j, i]] -= c;
        }
        l
    }

    #[test]
    fn sparse_inner_reconstructs_dense() {
        let nodes = vec![vec![0, 1, 2], vec![1, 2, 3], vec![3, 4]];
        let n = 5;
        let dense = pseudo_laplacian_inner(&nodes, n);
        let c = pseudo_laplacian_inner_sparse(&nodes, n);
        // The per-ball contribution stores raw (possibly duplicate) pairs; merge first.
        let merged = sort_and_merge(c.offdiag);
        assert_eq!(dense_from_sparse(&c.diag, &merged, n), dense);
    }

    #[test]
    fn sort_and_merge_sums_duplicates_and_drops_zeros() {
        let coo = vec![(0, 1, 1), (0, 1, 2), (2, 3, 1), (0, 1, -3)];
        let merged = sort_and_merge(coo);
        // (0,1) sums to 0 and is dropped; (2,3) remains.
        assert_eq!(merged, vec![(2, 3, 1)]);
    }

    #[test]
    fn sparse_inner_emits_upper_triangle_only() {
        let nodes = vec![vec![3, 1, 2]]; // unsorted members
        let c = pseudo_laplacian_inner_sparse(&nodes, 4);
        assert!(c.offdiag.iter().all(|&(i, j, _)| i < j));
    }
}
