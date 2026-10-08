use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use pulsar_core::{
    find_stable_thresholds, find_stable_thresholds_from_edges, Plateau, StabilityResult,
    DEFAULT_NUM_BINS,
};

/// A plateau in the components-vs-threshold curve (Python-facing).
#[pyclass(name = "Plateau")]
#[derive(Clone)]
pub struct PyPlateau {
    inner: Plateau,
}

#[pymethods]
impl PyPlateau {
    /// Threshold at the start (high end) of the plateau.
    #[getter]
    fn start_threshold(&self) -> f64 {
        self.inner.start_threshold
    }

    /// Threshold at the end (low end) of the plateau.
    #[getter]
    fn end_threshold(&self) -> f64 {
        self.inner.end_threshold
    }

    /// Number of connected components during this plateau.
    #[getter]
    fn component_count(&self) -> usize {
        self.inner.component_count
    }

    /// Length of the plateau (threshold range).
    #[getter]
    fn length(&self) -> f64 {
        self.inner.length()
    }

    /// Midpoint threshold of the plateau.
    #[getter]
    fn midpoint(&self) -> f64 {
        self.inner.midpoint()
    }

    fn __repr__(&self) -> String {
        format!(
            "Plateau(start={:.4}, end={:.4}, components={}, length={:.4})",
            self.inner.start_threshold,
            self.inner.end_threshold,
            self.inner.component_count,
            self.inner.length()
        )
    }
}

/// Result of threshold stability analysis (Python-facing).
#[pyclass(name = "StabilityResult")]
pub struct PyStabilityResult {
    inner: StabilityResult,
}

#[pymethods]
impl PyStabilityResult {
    /// Optimal threshold (midpoint of longest plateau).
    #[getter]
    fn optimal_threshold(&self) -> f64 {
        self.inner.optimal_threshold
    }

    /// All detected plateaus, sorted by length (longest first).
    #[getter]
    fn plateaus(&self) -> Vec<PyPlateau> {
        self.inner
            .plateaus
            .iter()
            .cloned()
            .map(|p| PyPlateau { inner: p })
            .collect()
    }

    /// Threshold values at which component count was measured.
    #[getter]
    fn thresholds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.thresholds.clone().into_pyarray_bound(py)
    }

    /// Component count at each threshold.
    #[getter]
    fn component_counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        self.inner.component_counts.clone().into_pyarray_bound(py)
    }

    /// Get the top k plateaus.
    #[pyo3(signature = (k=3))]
    fn top_k_plateaus(&self, k: usize) -> Vec<PyPlateau> {
        self.inner
            .plateaus
            .iter()
            .take(k)
            .cloned()
            .map(|p| PyPlateau { inner: p })
            .collect()
    }

    /// Get midpoints of the top k plateaus.
    #[pyo3(signature = (k=3))]
    fn top_k_thresholds<'py>(&self, py: Python<'py>, k: usize) -> Bound<'py, PyArray1<f64>> {
        let midpoints: Vec<f64> = self
            .inner
            .plateaus
            .iter()
            .take(k)
            .map(|p| p.midpoint())
            .collect();
        midpoints.into_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "StabilityResult(optimal_threshold={:.4}, num_plateaus={})",
            self.inner.optimal_threshold,
            self.inner.plateaus.len()
        )
    }
}

/// Find threshold values that produce stable connected component structure.
///
/// Sweeps τ from 1 → 0, tracking how connected components evolve in a weighted
/// adjacency matrix. Returns plateaus (stable regions) where small changes to
/// the threshold don't affect the component count.
///
/// This is equivalent to 0-dimensional persistent homology (H₀) computed
/// efficiently using weight quantization and incremental union-find.
///
/// # Parameters
/// - `weighted_adj` (`np.ndarray[float64, 2D]`, shape `(n, n)`) — weighted
///   adjacency matrix with values in `[0, 1]`.
/// - `num_bins` (`int`, optional) — number of quantization bins for the
///   threshold sweep. Higher values give finer resolution at the cost of
///   more computation. Default: 256 (threshold accuracy ±0.004).
///
/// # Returns
/// A `StabilityResult` containing:
/// - `optimal_threshold`: midpoint of the longest plateau
/// - `plateaus`: all detected plateaus, sorted by length (use `plateaus[0].component_count` for the optimal component count)
/// - `thresholds`: descending threshold change-points, plus 0.0
/// - `component_counts`: component count at each threshold change-point
///
/// # Scalability
/// Uses O(n²) time and O(m + n) memory where m = number of edges. For sparse
/// graphs this is efficient; for dense graphs m → n²/2 but sorting is avoided.
///
/// # Example
/// ```python
/// from pulsar._pulsar import CosmicGraph, find_stable_thresholds
///
/// cg = CosmicGraph.from_pseudo_laplacian(galactic_L, threshold=0.0)
/// result = find_stable_thresholds(cg.weighted_adj)
///
/// print(f"Optimal threshold: {result.optimal_threshold:.3f}")
/// print(f"This produces {result.plateaus[0].component_count} stable clusters")
///
/// # Apply the optimal threshold
/// optimal_cg = CosmicGraph.from_pseudo_laplacian(galactic_L, result.optimal_threshold)
///
/// # For higher precision, increase num_bins:
/// result_hires = find_stable_thresholds(cg.weighted_adj, num_bins=1024)
/// ```
#[pyfunction]
#[pyo3(name = "find_stable_thresholds", signature = (weighted_adj, num_bins=None))]
pub fn py_find_stable_thresholds<'py>(
    _py: Python<'py>,
    weighted_adj: PyReadonlyArray2<'py, f64>,
    num_bins: Option<usize>,
) -> PyResult<PyStabilityResult> {
    let arr = weighted_adj.as_array().to_owned();
    let bins = num_bins.unwrap_or(DEFAULT_NUM_BINS);
    let result = find_stable_thresholds(&arr, bins)?;
    Ok(PyStabilityResult { inner: result })
}

/// Find stable thresholds from a weighted edge list (sparse path), with no dense
/// n×n scan. Shares all logic with [`find_stable_thresholds`] and produces
/// identical results on the same graph.
///
/// # Parameters
/// - `n` (`int`) — number of nodes.
/// - `edges` (`list[tuple[int, int, float]]`) — `(i, j, weight)` with weights in
///   `[0, 1]`. Orientation of `(i, j)` is irrelevant.
/// - `num_bins` (`int`, optional) — quantization bins. Default: 256.
///
/// # Returns
/// A `StabilityResult`, identical in shape to `find_stable_thresholds`.
#[pyfunction]
#[pyo3(name = "find_stable_thresholds_sparse", signature = (n, edges, num_bins=None))]
pub fn py_find_stable_thresholds_sparse(
    _py: Python<'_>,
    n: usize,
    edges: Vec<(usize, usize, f64)>,
    num_bins: Option<usize>,
) -> PyResult<PyStabilityResult> {
    let bins = num_bins.unwrap_or(DEFAULT_NUM_BINS);
    let result = find_stable_thresholds_from_edges(n, &edges, bins)?;
    Ok(PyStabilityResult { inner: result })
}
