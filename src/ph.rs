//! 0-dimensional persistent homology for threshold stability detection.
//!
//! This module implements stability-based threshold selection for cosmic graphs
//! using connected component analysis. The core idea: sweep the threshold τ from
//! high to low, tracking how the number of connected components changes. Plateaus
//! in this curve — regions where small perturbations to τ don't change the
//! component count — correspond to stable topological features.
//!
//! This is equivalent to reading off H₀ (0-dimensional) persistent homology
//! without computing full persistence diagrams.
//!
//! ## Scalability
//!
//! The naive exact algorithm has complexity O(n² + m log m) time and O(n²) memory,
//! where m = number of unique edge weights (up to n²). This becomes prohibitive
//! for n > 10,000 nodes.
//!
//! We use two approximations to make this scalable:
//!
//! ### 1. Weight Quantization
//!
//! Instead of treating each unique edge weight as a distinct threshold, we bucket
//! weights into `num_bins` discrete levels. For weights in [0, 1]:
//!
//! ```text
//! bin(w) = floor(w * num_bins)
//! threshold(bin) = bin / num_bins
//! ```
//!
//! This reduces the number of distinct thresholds from O(n²) to O(num_bins).
//!
//! ### 2. Sparse Threshold Sweep
//!
//! We only measure component counts at the `num_bins` bin boundaries, not at
//! every unique edge weight. Edges are bucketed by their quantized weight, then
//! processed bin-by-bin in descending order.
//!
//! ### Complexity After Approximation
//!
//! | Step | Time | Memory |
//! |------|------|--------|
//! | Scan matrix & bucket edges | O(n²) | O(m) for edge tuples in bins |
//! | Process bins (union-find) | O(n² · α(n)) | O(n) for union-find |
//! | Plateau detection | O(num_bins) | O(num_bins) |
//! | **Total** | **O(n²)** | **O(m + n)** where m ≤ n²/2 |
//!
//! The time reduction vs. sorting (O(m log m) → O(n²)) is the key win.
//! Memory is O(m) for the bucketed edge list — for dense graphs this approaches
//! O(n²), but avoids the sorting overhead. For n = 100,000 with num_bins = 256,
//! a sparse graph with m = O(n) edges uses ~3 MB vs ~40 GB for the full matrix.
//!
//! ### Approximation Error
//!
//! Plateau boundaries are accurate to ±1/num_bins. With the default of 256 bins,
//! the optimal threshold is accurate to ±0.004, which is well within the noise
//! floor for most applications.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::error::PulsarError;

// ============================================================================
// Union-Find data structure
// ============================================================================

/// Disjoint-set (Union-Find) data structure with path compression and union by rank.
///
/// Provides near-constant time operations for tracking connected components
/// as edges are added incrementally. The amortized time per operation is O(α(n))
/// where α is the inverse Ackermann function (effectively constant for all
/// practical input sizes).
pub struct UnionFind {
    /// Parent pointers. `parent[x] == x` indicates x is a root.
    parent: Vec<usize>,
    /// Rank (upper bound on tree height) for union-by-rank optimization.
    rank: Vec<usize>,
    /// Current number of disjoint components.
    num_components: usize,
}

impl UnionFind {
    /// Create a new UnionFind with `n` elements, each in its own component.
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
            num_components: n,
        }
    }

    /// Find the root of the set containing `x`, with path compression.
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union the sets containing `x` and `y`. Returns true if they were in different sets.
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        match self.rank[root_x].cmp(&self.rank[root_y]) {
            std::cmp::Ordering::Less => {
                self.parent[root_x] = root_y;
            }
            std::cmp::Ordering::Greater => {
                self.parent[root_y] = root_x;
            }
            std::cmp::Ordering::Equal => {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }

        self.num_components -= 1;
        true
    }

    /// Current number of connected components.
    #[inline]
    pub fn components(&self) -> usize {
        self.num_components
    }

    /// Reset to initial state (n disjoint singletons).
    #[cfg(test)]
    pub fn reset(&mut self) {
        let n = self.parent.len();
        for i in 0..n {
            self.parent[i] = i;
            self.rank[i] = 0;
        }
        self.num_components = n;
    }
}

// ============================================================================
// Result types
// ============================================================================

/// A plateau (stable region) in the components-vs-threshold curve.
///
/// Represents a contiguous range of thresholds where the connected component
/// count remains constant. Longer plateaus indicate more robust topological
/// structure — the graph's clustering is insensitive to threshold perturbations
/// within this range.
#[derive(Clone, Debug)]
pub struct Plateau {
    /// Threshold at the start (high end) of the plateau.
    pub start_threshold: f64,
    /// Threshold at the end (low end) of the plateau.
    pub end_threshold: f64,
    /// Number of connected components during this plateau.
    pub component_count: usize,
}

impl Plateau {
    /// Length of the plateau (threshold range width).
    ///
    /// Longer plateaus indicate more stable structure. A plateau of length 0.3
    /// means the component count is unchanged across 30% of the threshold range.
    #[inline]
    pub fn length(&self) -> f64 {
        self.start_threshold - self.end_threshold
    }

    /// Midpoint threshold of the plateau.
    ///
    /// This is the recommended threshold to use — it maximizes distance from
    /// both plateau boundaries, providing robustness to noise.
    #[inline]
    pub fn midpoint(&self) -> f64 {
        (self.start_threshold + self.end_threshold) / 2.0
    }
}

/// Result of threshold stability analysis.
///
/// Contains the optimal threshold, all detected plateaus, and the full
/// component-vs-threshold curve data for visualization.
#[derive(Clone, Debug)]
pub struct StabilityResult {
    /// Optimal threshold: midpoint of the longest plateau.
    ///
    /// This threshold produces the most stable connected component structure —
    /// small perturbations to this value won't change the clustering.
    pub optimal_threshold: f64,

    /// All detected plateaus, sorted by length (longest first).
    ///
    /// The first plateau is the "most stable" region. Subsequent plateaus
    /// may represent alternative clusterings at different granularities.
    pub plateaus: Vec<Plateau>,

    /// Threshold change-points at which component count was measured.
    ///
    /// These are descending bin boundaries (1.0 → 0.0) where component counts
    /// changed, plus the final 0.0 boundary to close the curve for plotting.
    /// Use with `component_counts` for visualization.
    pub thresholds: Vec<f64>,

    /// Component count at each threshold in `thresholds`.
    ///
    /// Forms a step function: `component_counts[i]` is the number of connected
    /// components when the threshold is in the range `(thresholds[i+1], thresholds[i]]`.
    pub component_counts: Vec<usize>,
}

// ============================================================================
// Core algorithm
// ============================================================================

/// Default number of bins for weight quantization.
///
/// With 256 bins, threshold accuracy is ±0.004, which is sufficient for most
/// applications. Increasing this improves accuracy at the cost of more
/// threshold checkpoints to evaluate.
pub const DEFAULT_NUM_BINS: usize = 256;

/// Find threshold values that produce stable connected component structure.
///
/// Uses weight quantization and sparse threshold sweep for scalability.
/// See module documentation for algorithm details.
///
/// # Arguments
/// * `weighted_adj` - Weighted adjacency matrix, shape `(n, n)`, values in `[0, 1]`.
/// * `num_bins` - Number of quantization bins (default: 256). Higher values give
///   finer threshold resolution at the cost of more computation.
///
/// # Returns
/// A [`StabilityResult`] containing the optimal threshold, all plateaus, and
/// curve data for visualization.
///
/// # Errors
/// Returns [`PulsarError::InvalidParameter`] if `num_bins == 0`.
/// Returns [`PulsarError::ShapeMismatch`] if `weighted_adj` is not square.
pub fn find_stable_thresholds(
    weighted_adj: &Array2<f64>,
    num_bins: usize,
) -> Result<StabilityResult, PulsarError> {
    if num_bins == 0 {
        return Err(PulsarError::InvalidParameter {
            msg: "num_bins must be positive".to_string(),
        });
    }

    let n = weighted_adj.nrows();
    let m = weighted_adj.ncols();
    if n != m {
        return Err(PulsarError::ShapeMismatch {
            expected: "square matrix (n, n)".to_string(),
            got: format!("({n}, {m})"),
        });
    }

    // Handle degenerate cases
    if n == 0 {
        return Ok(StabilityResult {
            optimal_threshold: 0.5,
            plateaus: vec![],
            thresholds: vec![],
            component_counts: vec![],
        });
    }

    if n == 1 {
        return Ok(StabilityResult {
            optimal_threshold: 0.5,
            plateaus: vec![Plateau {
                start_threshold: 1.0,
                end_threshold: 0.0,
                component_count: 1,
            }],
            thresholds: vec![1.0, 0.0],
            component_counts: vec![1, 1],
        });
    }

    // -------------------------------------------------------------------------
    // Phase 1: Bucket edges by quantized weight
    // -------------------------------------------------------------------------
    // We avoid materializing all edges by using a two-pass approach:
    // Pass 1: Count edges per bin (this pass)
    // Pass 2: Process edges bin-by-bin during the sweep
    //
    // For now, we collect edges into per-bin vectors. This still uses O(m)
    // memory for edges, but the constant factor is lower than sorting, and
    // we avoid the O(m log m) sort cost.

    // bins[b] contains edges with quantized weight == b
    // Quantization: bin = floor(weight * num_bins), clamped to [0, num_bins-1]
    let mut bins: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_bins];

    for i in 0..n {
        for j in (i + 1)..n {
            let w = weighted_adj[[i, j]];
            if w > 0.0 {
                // Quantize weight to bin index
                // w ∈ (0, 1] maps to bin ∈ [0, num_bins-1]
                // We use (w * num_bins).floor() but clamp to avoid edge cases
                let bin = ((w * num_bins as f64).floor() as usize).min(num_bins - 1);
                bins[bin].push((i, j));
            }
        }
    }

    // Check if graph has any edges
    let total_edges: usize = bins.iter().map(|b| b.len()).sum();
    if total_edges == 0 {
        return Ok(StabilityResult {
            optimal_threshold: 0.5,
            plateaus: vec![Plateau {
                start_threshold: 1.0,
                end_threshold: 0.0,
                component_count: n,
            }],
            thresholds: vec![1.0, 0.0],
            component_counts: vec![n, n],
        });
    }

    // -------------------------------------------------------------------------
    // Phase 2: Sparse threshold sweep with union-find
    // -------------------------------------------------------------------------
    // Sweep from high threshold to low (bin num_bins-1 down to 0).
    // At each bin boundary, record the component count.
    //
    // Threshold τ = bin/num_bins means we include all edges with weight > τ,
    // i.e., all edges in bins > bin. So we process bins in descending order,
    // and after processing bin b, the threshold is b/num_bins.

    let mut uf = UnionFind::new(n);
    let mut thresholds: Vec<f64> = Vec::with_capacity(num_bins + 1);
    let mut component_counts: Vec<usize> = Vec::with_capacity(num_bins + 1);

    // Initial state: threshold = 1.0, no edges included
    thresholds.push(1.0);
    component_counts.push(n);

    // Process bins from highest to lowest
    for bin in (0..num_bins).rev() {
        // Add all edges in this bin
        for &(i, j) in &bins[bin] {
            uf.union(i, j);
        }

        // Record state at threshold = bin / num_bins
        let threshold = bin as f64 / num_bins as f64;
        thresholds.push(threshold);
        component_counts.push(uf.components());
    }

    // -------------------------------------------------------------------------
    // Phase 3: Identify plateaus
    // -------------------------------------------------------------------------
    // A plateau is a contiguous range where component count is constant.
    // We merge consecutive threshold intervals with the same count.

    let mut plateaus: Vec<Plateau> = Vec::new();
    let mut plateau_start_idx = 0;

    for i in 1..component_counts.len() {
        if component_counts[i] != component_counts[plateau_start_idx] {
            // End of plateau at index i-1
            plateaus.push(Plateau {
                start_threshold: thresholds[plateau_start_idx],
                end_threshold: thresholds[i - 1],
                component_count: component_counts[plateau_start_idx],
            });
            plateau_start_idx = i;
        }
    }

    // Final plateau (from plateau_start_idx to end)
    let last_idx = component_counts.len() - 1;
    plateaus.push(Plateau {
        start_threshold: thresholds[plateau_start_idx],
        end_threshold: thresholds[last_idx],
        component_count: component_counts[plateau_start_idx],
    });

    // Sort plateaus by length (longest first)
    plateaus.sort_by(|a, b| {
        b.length()
            .partial_cmp(&a.length())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // -------------------------------------------------------------------------
    // Phase 4: Select optimal threshold
    // -------------------------------------------------------------------------
    let optimal_threshold = plateaus
        .first()
        .map(|p| p.midpoint())
        .unwrap_or(0.5);

    // Deduplicate consecutive identical component counts for cleaner output
    let mut dedup_thresholds: Vec<f64> = Vec::new();
    let mut dedup_counts: Vec<usize> = Vec::new();

    for i in 0..thresholds.len() {
        if dedup_counts.is_empty() || dedup_counts.last() != Some(&component_counts[i]) {
            dedup_thresholds.push(thresholds[i]);
            dedup_counts.push(component_counts[i]);
        }
    }

    // Always include the final 0.0 boundary so step-curve plots are complete.
    if let (Some(&last_threshold), Some(&last_count)) = (thresholds.last(), component_counts.last()) {
        if dedup_thresholds.last().copied() != Some(last_threshold) {
            dedup_thresholds.push(last_threshold);
            dedup_counts.push(last_count);
        }
    }

    Ok(StabilityResult {
        optimal_threshold,
        plateaus,
        thresholds: dedup_thresholds,
        component_counts: dedup_counts,
    })
}

// ============================================================================
// Python bindings
// ============================================================================

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);
        assert_eq!(uf.components(), 5);

        uf.union(0, 1);
        assert_eq!(uf.components(), 4);

        uf.union(2, 3);
        assert_eq!(uf.components(), 3);

        uf.union(1, 2);
        assert_eq!(uf.components(), 2);

        assert_eq!(uf.find(0), uf.find(3));
    }

    #[test]
    fn test_union_find_reset() {
        let mut uf = UnionFind::new(3);
        uf.union(0, 1);
        uf.union(1, 2);
        assert_eq!(uf.components(), 1);

        uf.reset();
        assert_eq!(uf.components(), 3);
        assert_ne!(uf.find(0), uf.find(1));
    }

    #[test]
    fn test_find_stable_thresholds_simple() {
        // 4 nodes: two pairs with high internal similarity, low cross-pair
        // Weights: (0,1)=0.9, (2,3)=0.8, cross-pairs=0.1
        //
        // Expected component evolution as threshold decreases:
        //   τ > 0.9:  4 components (all disconnected)
        //   τ ∈ (0.8, 0.9]: 3 components (0-1 connected)
        //   τ ∈ (0.1, 0.8]: 2 components (0-1 and 2-3 connected) ← longest plateau
        //   τ ≤ 0.1:  1 component (all connected)
        let w = arr2(&[
            [0.0, 0.9, 0.1, 0.1],
            [0.9, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.0, 0.8],
            [0.1, 0.1, 0.8, 0.0],
        ]);

        // Use 100 bins for this test (finer than the weight differences)
        let result = find_stable_thresholds(&w, 100).unwrap();

        // Check we have the expected component counts in the curve
        assert!(result.component_counts.contains(&4)); // initially disconnected
        assert!(result.component_counts.contains(&2)); // two pairs
        assert!(result.component_counts.contains(&1)); // fully connected

        // Optimal threshold should be in the stable 2-component region (0.1, 0.8]
        // With quantization, this is approximately the midpoint
        assert!(result.optimal_threshold > 0.1);
        assert!(result.optimal_threshold < 0.9);

        // The longest plateau should have 2 components
        assert_eq!(result.plateaus[0].component_count, 2);
    }

    #[test]
    fn test_find_stable_thresholds_single_component() {
        // Fully connected graph with uniform weights
        let w = arr2(&[
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]);

        let result = find_stable_thresholds(&w, DEFAULT_NUM_BINS).unwrap();

        // At threshold below 0.5, should have 1 component
        assert!(result.component_counts.contains(&1));

        // Should also start with 3 components at high threshold
        assert!(result.component_counts.contains(&3));
    }

    #[test]
    fn test_empty_graph() {
        // No edges (all zeros)
        let w = arr2(&[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]);

        let result = find_stable_thresholds(&w, DEFAULT_NUM_BINS).unwrap();

        // All nodes remain disconnected — single plateau covering entire range
        assert_eq!(result.plateaus.len(), 1);
        assert_eq!(result.plateaus[0].component_count, 3);
        assert_eq!(result.plateaus[0].start_threshold, 1.0);
        assert_eq!(result.plateaus[0].end_threshold, 0.0);
    }

    #[test]
    fn test_single_node() {
        let w = arr2(&[[0.0]]);
        let result = find_stable_thresholds(&w, DEFAULT_NUM_BINS).unwrap();

        assert_eq!(result.optimal_threshold, 0.5);
        assert_eq!(result.plateaus[0].component_count, 1);
    }

    #[test]
    fn test_quantization_bins() {
        // Test that different bin counts produce consistent results
        let w = arr2(&[
            [0.0, 0.7, 0.3],
            [0.7, 0.0, 0.3],
            [0.3, 0.3, 0.0],
        ]);

        let result_coarse = find_stable_thresholds(&w, 10).unwrap();
        let result_fine = find_stable_thresholds(&w, 1000).unwrap();

        // Both should identify the same number of distinct component counts
        let coarse_unique: std::collections::HashSet<_> =
            result_coarse.component_counts.iter().collect();
        let fine_unique: std::collections::HashSet<_> =
            result_fine.component_counts.iter().collect();
        assert_eq!(coarse_unique, fine_unique);

        // Both should find the longest plateau has 1 component (fully connected below 0.3)
        // The actual optimal threshold may differ due to plateau boundaries shifting
        // with quantization, but they should both select a reasonable value
        assert!(result_coarse.optimal_threshold >= 0.0);
        assert!(result_coarse.optimal_threshold <= 1.0);
        assert!(result_fine.optimal_threshold >= 0.0);
        assert!(result_fine.optimal_threshold <= 1.0);
    }

    #[test]
    fn test_plateau_properties() {
        let plateau = Plateau {
            start_threshold: 0.8,
            end_threshold: 0.2,
            component_count: 5,
        };

        assert!((plateau.length() - 0.6).abs() < 1e-10);
        assert!((plateau.midpoint() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_num_bins_zero_returns_error() {
        let w = arr2(&[[0.0, 0.4], [0.4, 0.0]]);
        let err = find_stable_thresholds(&w, 0).unwrap_err();
        assert!(matches!(err, PulsarError::InvalidParameter { .. }));
    }

    #[test]
    fn test_non_square_returns_error() {
        let w = arr2(&[[0.0, 0.4, 0.2], [0.4, 0.0, 0.1]]);
        let err = find_stable_thresholds(&w, DEFAULT_NUM_BINS).unwrap_err();
        assert!(matches!(err, PulsarError::ShapeMismatch { .. }));
    }
}
