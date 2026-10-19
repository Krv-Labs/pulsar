//! Locally optimal block preconditioned conjugate gradient (Knyazev 2001),
//! unpreconditioned, for the `k` smallest eigenpairs of a symmetric
//! positive semi-definite operator given only as a matrix-free `apply`.
//!
//! A block method handles repeated eigenvalues (one zero per connected
//! component of a graph Laplacian), which single-vector Lanczos does not.
//! The basis `[X, W, P]` is re-orthonormalized every iteration, so the
//! Rayleigh–Ritz step is a plain symmetric eigenproblem.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::AnalysisError;

#[derive(Debug, Clone, PartialEq)]
pub struct Eigenpairs {
    /// Ascending.
    pub values: Vec<f64>,
    /// `n × k`, columns matching `values`.
    pub vectors: DMatrix<f64>,
    pub iterations: usize,
    pub max_residual: f64,
}

/// Columns of `m` orthonormalized by Gram–Schmidt (twice, for stability);
/// columns that are numerically dependent are dropped.
fn orthonormal_columns(m: &DMatrix<f64>, drop_tol: f64) -> DMatrix<f64> {
    let n = m.nrows();
    let mut cols: Vec<DVector<f64>> = Vec::with_capacity(m.ncols());
    for c in 0..m.ncols() {
        let mut v = m.column(c).into_owned();
        let original = v.norm();
        for _ in 0..2 {
            for q in &cols {
                let proj = q.dot(&v);
                v -= q * proj;
            }
        }
        let norm = v.norm();
        if norm > drop_tol * original.max(1.0) && norm > 0.0 {
            cols.push(v / norm);
        }
    }
    let mut out = DMatrix::<f64>::zeros(n, cols.len());
    for (c, v) in cols.iter().enumerate() {
        out.set_column(c, v);
    }
    out
}

/// Smallest `k` Ritz pairs of the symmetric matrix `g`.
fn smallest_ritz(g: &DMatrix<f64>, k: usize) -> (Vec<f64>, DMatrix<f64>) {
    let sym = (g + g.transpose()) * 0.5;
    let eig = SymmetricEigen::new(sym);
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let take = k.min(order.len());
    let values: Vec<f64> = order
        .iter()
        .take(take)
        .map(|&i| eig.eigenvalues[i])
        .collect();
    let mut vectors = DMatrix::<f64>::zeros(g.nrows(), take);
    for (c, &i) in order.iter().take(take).enumerate() {
        vectors.set_column(c, &eig.eigenvectors.column(i));
    }
    (values, vectors)
}

/// Find the `k` smallest eigenpairs of the `n × n` operator `apply`.
/// Converges when every residual `‖A x − λ x‖ ≤ tol`; returns
/// `SOLVER_ERROR` if residuals are still above `100 · tol` after
/// `max_iter` iterations.
pub fn lobpcg_smallest(
    apply: &dyn Fn(&DMatrix<f64>) -> DMatrix<f64>,
    n: usize,
    k: usize,
    seed: u64,
    tol: f64,
    max_iter: usize,
) -> Result<Eigenpairs, AnalysisError> {
    if k == 0 || 3 * k > n {
        return Err(AnalysisError::InvalidArgument(format!(
            "lobpcg needs 1 <= k and 3k <= n, got k = {k}, n = {n}"
        )));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let random = DMatrix::<f64>::from_fn(n, k, |_, _| StandardNormal.sample(&mut rng));
    let mut x = orthonormal_columns(&random, 1e-12);
    if x.ncols() < k {
        return Err(AnalysisError::Convergence(
            "random start was rank deficient".into(),
        ));
    }
    let mut p: Option<DMatrix<f64>> = None;
    let mut values = vec![0.0; k];
    let mut max_residual = f64::INFINITY;
    let mut iterations = 0;

    for it in 0..max_iter {
        iterations = it + 1;
        // Rayleigh–Ritz on X so the residuals are of Ritz vectors.
        let ax = apply(&x);
        let (lam, c) = smallest_ritz(&(x.transpose() * &ax), k);
        x = &x * &c;
        let ax = ax * &c;
        values = lam;
        let mut r = ax.clone();
        for (col, &l) in values.iter().enumerate() {
            let xc = x.column(col) * l;
            let rc = r.column(col) - xc;
            r.set_column(col, &rc);
        }
        max_residual = (0..k).map(|c| r.column(c).norm()).fold(0.0, f64::max);
        if max_residual <= tol {
            break;
        }

        // Basis [X, W, P], orthonormalized.
        let mut blocks: Vec<&DMatrix<f64>> = vec![&x, &r];
        if let Some(p) = &p {
            blocks.push(p);
        }
        let total_cols: usize = blocks.iter().map(|b| b.ncols()).sum();
        let mut stacked = DMatrix::<f64>::zeros(n, total_cols);
        let mut offset = 0;
        for b in blocks {
            stacked.view_mut((0, offset), (n, b.ncols())).copy_from(b);
            offset += b.ncols();
        }
        let s = orthonormal_columns(&stacked, 1e-10);
        if s.ncols() <= k {
            // No new directions: the residual is numerically zero.
            break;
        }
        let a_s = apply(&s);
        let (lam, c) = smallest_ritz(&(s.transpose() * &a_s), k);
        let x_new = &s * &c;
        // The part of the step that lies outside the previous X.
        let rest = s.columns(k, s.ncols() - k) * c.rows(k, c.nrows() - k);
        values = lam;
        x = x_new;
        p = Some(rest);
    }

    Ok(Eigenpairs {
        values,
        vectors: x,
        iterations,
        max_residual,
    })
    .and_then(|e| {
        if e.max_residual <= tol * 100.0 {
            Ok(e)
        } else {
            Err(AnalysisError::Convergence(format!(
                "max residual {:.3e} after {} iterations",
                e.max_residual, e.iterations
            )))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_smallest_eigenpairs_of_a_diagonal_operator() {
        let n = 40;
        let diag: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.5).collect();
        let apply = |x: &DMatrix<f64>| {
            let mut y = x.clone();
            for (i, d) in diag.iter().enumerate() {
                for c in 0..x.ncols() {
                    y[(i, c)] *= d;
                }
            }
            y
        };
        let e = lobpcg_smallest(&apply, n, 3, 7, 1e-8, 200).unwrap();
        for (got, want) in e.values.iter().zip([1.0, 1.5, 2.0]) {
            assert!((got - want).abs() < 1e-6, "{:?}", e.values);
        }
        for c in 0..3 {
            let v = e.vectors.column(c);
            assert!(v[c].abs() > 0.999, "eigenvector {c} should be e_{c}: {v}");
        }
    }

    #[test]
    fn handles_a_repeated_zero_eigenvalue() {
        // Laplacian of two disjoint paths of 5 nodes: zero has multiplicity 2.
        let n = 10;
        let edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
        ];
        let apply = |x: &DMatrix<f64>| {
            let mut y = DMatrix::<f64>::zeros(n, x.ncols());
            for &(i, j) in &edges {
                for c in 0..x.ncols() {
                    let d = x[(i, c)] - x[(j, c)];
                    y[(i, c)] += d;
                    y[(j, c)] -= d;
                }
            }
            y
        };
        let e = lobpcg_smallest(&apply, n, 3, 1, 1e-9, 300).unwrap();
        assert!(
            e.values[0].abs() < 1e-7 && e.values[1].abs() < 1e-7,
            "{:?}",
            e.values
        );
        // Third: smallest nonzero of a path P5 = 2 - 2cos(pi/5) ≈ 0.381966.
        assert!(
            (e.values[2] - 0.3819660112501051).abs() < 1e-6,
            "{:?}",
            e.values
        );
        assert_eq!(
            lobpcg_smallest(&apply, n, 4, 1, 1e-9, 10)
                .unwrap_err()
                .code(),
            "INVALID_ARGUMENT"
        );
    }
}
