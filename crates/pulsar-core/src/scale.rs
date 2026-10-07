use ndarray::Array2;

use crate::error::PulsarError;

/// Standard scaler that stores fitted parameters.
///
/// Standard scaling transforms each feature column `x` to `(x - μ) / σ`,
/// where `μ` is the column mean and `σ` is the **population** standard
/// deviation (ddof=0).  This matches scikit-learn's `StandardScaler` default.
///
/// A std of zero would cause a division-by-zero.  Constant columns (σ < 1e-10)
/// have their std clamped to `1.0` so the scaled value is `0.0` everywhere.
pub struct StandardScalerInner {
    /// Per-column means, in column order.
    pub means: Vec<f64>,
    /// Per-column standard deviations (population, clamped ≥ 1.0 if near-zero).
    pub stds: Vec<f64>,
}

impl StandardScalerInner {
    /// Compute column statistics and return the scaled matrix together with the
    /// fitted scaler.
    ///
    /// Uses **population std** (ddof=0):
    /// `σ = sqrt( Σ(xᵢ − μ)² / n )`
    pub fn fit_transform(data: &Array2<f64>) -> (Array2<f64>, StandardScalerInner) {
        let (nrows, ncols) = (data.nrows(), data.ncols());
        let mut means = vec![0.0f64; ncols];
        let mut stds = vec![1.0f64; ncols];

        for j in 0..ncols {
            let col = data.column(j);
            let mean = col.sum() / nrows as f64;
            let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / nrows as f64;
            means[j] = mean;
            stds[j] = if variance.sqrt() < 1e-10 { 1.0 } else { variance.sqrt() };
        }

        let scaler = StandardScalerInner { means: means.clone(), stds: stds.clone() };
        let mut out = data.clone();
        for j in 0..ncols {
            out.column_mut(j).mapv_inplace(|x| (x - means[j]) / stds[j]);
        }
        (out, scaler)
    }

    /// Apply previously fitted statistics to a new matrix.
    ///
    /// # Errors
    /// [`PulsarError::ShapeMismatch`] if `data` has a different number of
    /// columns than the matrix used during `fit_transform`.
    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PulsarError> {
        let ncols = data.ncols();
        if ncols != self.means.len() {
            return Err(PulsarError::ShapeMismatch {
                expected: format!("{} columns", self.means.len()),
                got: format!("{} columns", ncols),
            });
        }
        let mut out = data.clone();
        for j in 0..ncols {
            out.column_mut(j).mapv_inplace(|x| (x - self.means[j]) / self.stds[j]);
        }
        Ok(out)
    }

    /// Reverse a previous `transform`: `x_orig = x_scaled * σ + μ`.
    ///
    /// # Errors
    /// [`PulsarError::ShapeMismatch`] if `data` has a different number of
    /// columns than the matrix used during `fit_transform`.
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PulsarError> {
        let ncols = data.ncols();
        if ncols != self.means.len() {
            return Err(PulsarError::ShapeMismatch {
                expected: format!("{} columns", self.means.len()),
                got: format!("{} columns", ncols),
            });
        }
        let mut out = data.clone();
        for j in 0..ncols {
            out.column_mut(j).mapv_inplace(|x| x * self.stds[j] + self.means[j]);
        }
        Ok(out)
    }
}
