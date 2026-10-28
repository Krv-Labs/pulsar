//! Temporal Cosmic Graph support for longitudinal time-series data.
//!
//! This module extends Pulsar to handle data where the same set of nodes (e.g., patients)
//! are observed across multiple time steps. Instead of a single 2D pseudo-Laplacian,
//! we accumulate a 3D tensor `L[i, j, t]` representing connectivity at each time step.
//!
//! ## Core Data Structure
//!
//! The temporal pseudo-Laplacian tensor has shape `(n, n, T)` where:
//! - `n` is the number of nodes (fixed across time)
//! - `T` is the number of time steps
//! - `L[i, j, t]` follows the same semantics as the standard pseudo-Laplacian at time `t`
//!
//! ## Parallelization Strategy
//!
//! Time steps are processed in parallel using rayon. Each time step's ball maps
//! are accumulated independently, then results are combined into the 3D tensor.
//! Aggregations run in parallel over the contiguous time lanes `W[i, j, :]`.
//!
//! ## TemporalCosmicGraph
//!
//! [`TemporalCosmicGraph`] ports `pulsar/representations/temporal.py`: the
//! normalized tensor `W[i, j, t]` and its aggregations into summary 2D graphs
//! (persistence, mean, recency, volatility, trend, change point). The tensor
//! is dense, `O(n^2 T)`: callers bound its size before building it (the
//! longitudinal tools' memory ceiling).
//!
//! Known divergences from Python:
//! - `to_networkx` is not ported (no consumer; threshold an aggregation instead).
//! - Floating-point sums follow a sequential order, not numpy's pairwise one;
//!   results agree to within 1e-12 relative.
//! - An empty time range (`slice` past the end) aggregates to zeros; numpy
//!   returned NaN with a RuntimeWarning.

use ndarray::{s, Array2, Array3, ArrayD, Axis, Ix3};
use rayon::prelude::*;

use crate::ballmapper::ball_mapper_grid;
use crate::error::{ensure_members_in_range, PulsarError};
use crate::pca::pca_grid;
use crate::pseudolaplacian::dense_contribution;
use crate::scale::StandardScalerInner;

/// `pca_grid` oversampling and power iterations used by `from_snapshots`
/// (the Python binding's defaults).
const PCA_OVERSAMPLES: usize = 10;
const PCA_POWER_ITER: usize = 2;

/// Accumulate pseudo-Laplacians for each time step, returning a 3D tensor.
///
/// This is the temporal extension of `accumulate_pseudo_laplacians`. Each time step
/// has its own set of BallMapper results from running the standard Pulsar pipeline
/// on that time step's data.
///
/// # Arguments
/// * `ball_maps_per_time` - Vector of length T, where each element is a vector of
///   BallMapper references for that time step
/// * `n` - Number of nodes (must be consistent across all time steps)
///
/// # Returns
/// A 3D array of shape `(n, n, T)` containing the accumulated pseudo-Laplacian
/// for each time step.
///
/// # Errors
/// [`PulsarError::InvalidParameter`] if any member index is `>= n`.
///
/// # Example
/// ```python
/// # Run Pulsar pipeline at each time step
/// ball_maps_per_time = []
/// for t in range(T):
///     embeddings_t = pca_grid(X_scaled[t], dims, seeds)
///     ball_maps_t = ball_mapper_grid(embeddings_t, epsilons)
///     ball_maps_per_time.append(ball_maps_t)
///
/// # Accumulate into 3D tensor
/// L_tensor = accumulate_temporal_pseudo_laplacians(ball_maps_per_time, n)
/// # L_tensor has shape (n, n, T)
/// ```
pub fn accumulate_temporal_pseudo_laplacians_inner(
    ball_maps_per_time: &[Vec<&Vec<Vec<usize>>>],
    n: usize,
) -> Result<Array3<i64>, PulsarError> {
    ensure_members_in_range(
        ball_maps_per_time
            .iter()
            .flat_map(|maps| maps.iter().flat_map(|nodes| nodes.iter())),
        n,
    )?;
    let t_steps = ball_maps_per_time.len();

    // Process each time step in parallel
    let laplacians: Vec<Array2<i64>> = ball_maps_per_time
        .par_iter()
        .map(|ball_maps_at_t| {
            // Accumulate all ball maps at this time step
            ball_maps_at_t
                .iter()
                .map(|nodes| dense_contribution(nodes, n))
                .fold(Array2::<i64>::zeros((n, n)), |mut acc, l| {
                    acc += &l;
                    acc
                })
        })
        .collect();

    // Stack into 3D tensor (n, n, T)
    let mut tensor = Array3::<i64>::zeros((n, n, t_steps));
    for (t, laplacian) in laplacians.into_iter().enumerate() {
        tensor.index_axis_mut(Axis(2), t).assign(&laplacian);
    }

    Ok(tensor)
}

/// Normalize a 3D pseudo-Laplacian tensor into weighted adjacency matrices.
///
/// Applies the same normalization formula as `CosmicGraph::from_pseudo_laplacian`
/// independently at each time step.
///
/// # Weight formula (per time step)
///
/// For each off-diagonal pair `(i, j)` at time `t`:
///
/// ```text
/// denom = L[i,i,t] + L[j,j,t] + L[i,j,t]
///
///                 -L[i,j,t]
/// W[i,j,t] =  ─────────────────   if denom > 0
///                  denom
///
///           = 0                    otherwise
/// ```
pub fn normalize_temporal_laplacian(l: &Array3<i64>) -> Array3<f64> {
    let (n, _, t_steps) = l.dim();
    let mut w = Array3::<f64>::zeros((n, n, t_steps));

    for t in 0..t_steps {
        for i in 0..n {
            // Exploit symmetry: compute only for j > i and mirror to (j, i)
            for j in (i + 1)..n {
                let denom = l[[i, i, t]] + l[[j, j, t]] + l[[i, j, t]];
                if denom > 0 {
                    let weight = -(l[[i, j, t]] as f64) / (denom as f64);
                    w[[i, j, t]] = weight;
                    w[[j, i, t]] = weight;
                }
            }
        }
    }

    w
}
/// Python's `str(tuple)` of a shape: `(2, 2)`, `(3,)`.
fn py_shape(shape: &[usize]) -> String {
    match shape {
        [one] => format!("({one},)"),
        dims => format!(
            "({})",
            dims.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Python's `repr(float)` for the values a parameter check reports.
fn py_float(x: f64) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.into()
    } else if x.fract() == 0.0 && x.abs() < 1e16 {
        format!("{x:.1}")
    } else {
        format!("{x}")
    }
}

fn invalid(msg: String) -> PulsarError {
    PulsarError::InvalidParameter { msg }
}

/// Cosmic Graph for longitudinal data: a tensor `W[i, j, t]` of edge weights
/// in `[0, 1]` over a fixed node set, aggregated into summary 2D graphs.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalCosmicGraph {
    /// Shape `(n, n, T)`, standard (row-major) layout.
    tensor: Array3<f64>,
    threshold: f64,
}

impl TemporalCosmicGraph {
    /// Wrap a precomputed weighted adjacency tensor of shape `(n, n, T)`.
    pub fn new(tensor: ArrayD<f64>, threshold: f64) -> Result<Self, PulsarError> {
        let shape = tensor.shape().to_vec();
        if shape.len() != 3 {
            return Err(invalid(format!(
                "Expected 3D tensor, got shape {}",
                py_shape(&shape)
            )));
        }
        if shape[0] != shape[1] {
            return Err(invalid(format!(
                "Expected square slices, got shape {}",
                py_shape(&shape)
            )));
        }
        let tensor = tensor
            .into_dimensionality::<Ix3>()
            .map_err(|e| invalid(e.to_string()))?
            .as_standard_layout()
            .into_owned();
        Ok(Self { tensor, threshold })
    }

    /// Run scale -> PCA grid -> Ball Mapper grid -> pseudo-Laplacian
    /// independently on each time step's `(n, features_t)` snapshot and stack
    /// the normalized results. Every snapshot must have the same `n` rows, in
    /// the same entity order.
    pub fn from_snapshots(
        snapshots: &[Array2<f64>],
        dimensions: &[usize],
        seeds: &[u64],
        epsilons: &[f64],
        threshold: f64,
    ) -> Result<Self, PulsarError> {
        let Some(first) = snapshots.first() else {
            return Err(invalid("snapshots list cannot be empty".into()));
        };
        let n = first.nrows();
        for (t, snap) in snapshots.iter().enumerate() {
            if snap.nrows() != n {
                return Err(invalid(format!(
                    "Inconsistent node count: snapshot 0 has {n} nodes, snapshot {t} has {} nodes",
                    snap.nrows()
                )));
            }
        }
        if let Some(&eps) = epsilons.iter().find(|e| !e.is_finite() || **e <= 0.0) {
            return Err(invalid(format!(
                "epsilon must be a finite positive number, got {}",
                py_float(eps)
            )));
        }
        let ball_maps_per_time = snapshots
            .iter()
            .map(|snap| {
                let (scaled, _) = StandardScalerInner::fit_transform(snap);
                let embeddings =
                    pca_grid(&scaled, dimensions, seeds, PCA_OVERSAMPLES, PCA_POWER_ITER)?;
                Ok(ball_mapper_grid(&embeddings, epsilons))
            })
            .collect::<Result<Vec<_>, PulsarError>>()?;
        let node_refs: Vec<Vec<&Vec<Vec<usize>>>> = ball_maps_per_time
            .iter()
            .map(|maps| maps.iter().map(|bm| &bm.nodes).collect())
            .collect();
        let laplacian = accumulate_temporal_pseudo_laplacians_inner(&node_refs, n)?;
        Ok(Self {
            tensor: normalize_temporal_laplacian(&laplacian),
            threshold,
        })
    }

    /// The tensor, shape `(n, n, T)`.
    pub fn tensor(&self) -> &Array3<f64> {
        &self.tensor
    }

    /// Number of nodes.
    pub fn n(&self) -> usize {
        self.tensor.dim().0
    }

    /// Number of time steps.
    #[allow(non_snake_case)]
    pub fn T(&self) -> usize {
        self.tensor.dim().2
    }

    /// `(n, n, T)`.
    pub fn shape(&self) -> (usize, usize, usize) {
        self.tensor.dim()
    }

    /// Default threshold for binary operations.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Bytes held by the tensor (numpy's `nbytes`).
    pub fn nbytes(&self) -> usize {
        self.tensor.len() * std::mem::size_of::<f64>()
    }

    /// Apply `f` to every time lane `W[i, j, :]`.
    fn aggregate(&self, f: impl Fn(&[f64]) -> f64 + Sync + Send) -> Array2<f64> {
        let (n, _, t_steps) = self.tensor.dim();
        if t_steps == 0 {
            return Array2::zeros((n, n));
        }
        let data = self
            .tensor
            .as_slice()
            .expect("the tensor is kept in standard layout");
        let out: Vec<f64> = data.par_chunks(t_steps).map(f).collect();
        Array2::from_shape_vec((n, n), out).expect("one value per (i, j) lane")
    }

    /// Fraction of time steps where the edge exceeds `threshold` (default: the
    /// instance threshold): `mean_t(W > tau)`.
    pub fn persistence_graph(&self, threshold: Option<f64>) -> Array2<f64> {
        let tau = threshold.unwrap_or(self.threshold);
        self.aggregate(|lane| lane.iter().filter(|&&w| w > tau).count() as f64 / lane.len() as f64)
    }

    /// Average edge weight over time: `mean_t(W)`.
    pub fn mean_graph(&self) -> Array2<f64> {
        self.aggregate(lane_mean)
    }

    /// Exponentially decayed average favouring recent steps:
    /// `sum_t decay^(T-1-t) W / sum_t decay^(T-1-t)`, `decay` in (0, 1).
    pub fn recency_graph(&self, decay: f64) -> Result<Array2<f64>, PulsarError> {
        if !(decay > 0.0 && decay < 1.0) {
            return Err(invalid(format!(
                "decay must be in (0, 1), got {}",
                py_float(decay)
            )));
        }
        let t_steps = self.T();
        let raw: Vec<f64> = (0..t_steps)
            .map(|t| decay.powf((t_steps - 1 - t) as f64))
            .collect();
        let total: f64 = raw.iter().sum();
        let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
        Ok(self.aggregate(|lane| lane.iter().zip(&weights).map(|(w, k)| w * k).sum()))
    }

    /// Temporal (population) variance of each edge: `var_t(W)`.
    pub fn volatility_graph(&self) -> Array2<f64> {
        self.aggregate(|lane| {
            let mean = lane_mean(lane);
            lane.iter().map(|w| (w - mean) * (w - mean)).sum::<f64>() / lane.len() as f64
        })
    }

    /// Least-squares slope of each edge over time: positive converges,
    /// negative diverges. Zero with one time step.
    pub fn trend_graph(&self) -> Array2<f64> {
        let t_steps = self.T();
        let t_mean = (0..t_steps).map(|t| t as f64).sum::<f64>() / t_steps.max(1) as f64;
        let centered: Vec<f64> = (0..t_steps).map(|t| t as f64 - t_mean).collect();
        let t_var: f64 = centered.iter().map(|c| c * c).sum();
        if t_var == 0.0 {
            return Array2::zeros((self.n(), self.n()));
        }
        self.aggregate(|lane| {
            let mean = lane_mean(lane);
            lane.iter()
                .zip(&centered)
                .map(|(w, c)| (w - mean) * c)
                .sum::<f64>()
                / t_var
        })
    }

    /// Largest absolute change between consecutive steps:
    /// `max_t |W[t+1] - W[t]|`. Zero with fewer than two steps.
    pub fn change_point_graph(&self) -> Array2<f64> {
        if self.T() < 2 {
            return Array2::zeros((self.n(), self.n()));
        }
        self.aggregate(|lane| {
            lane.windows(2)
                .map(|p| (p[1] - p[0]).abs())
                .fold(0.0, f64::max)
        })
    }

    /// The time steps `start..end` (`end` defaults to `T`; both clamp like a
    /// Python slice of non-negative indices).
    pub fn slice(&self, start: usize, end: Option<usize>) -> Self {
        let t_steps = self.T();
        let end = end.unwrap_or(t_steps).min(t_steps);
        let start = start.min(end);
        Self {
            tensor: self.tensor.slice(s![.., .., start..end]).to_owned(),
            threshold: self.threshold,
        }
    }
}

fn lane_mean(lane: &[f64]) -> f64 {
    lane.iter().sum::<f64>() / lane.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_accumulation_single_timestep() {
        // Single time step should match non-temporal behavior
        let nodes1 = vec![vec![0, 1], vec![1, 2]];
        let ball_maps_per_time = vec![vec![&nodes1]];

        let tensor = accumulate_temporal_pseudo_laplacians_inner(&ball_maps_per_time, 3).unwrap();

        assert_eq!(tensor.dim(), (3, 3, 1));
        // Check diagonal (membership counts)
        assert_eq!(tensor[[0, 0, 0]], 1); // node 0 in 1 ball
        assert_eq!(tensor[[1, 1, 0]], 2); // node 1 in 2 balls
        assert_eq!(tensor[[2, 2, 0]], 1); // node 2 in 1 ball
    }

    #[test]
    fn test_temporal_accumulation_multiple_timesteps() {
        let nodes1 = vec![vec![0, 1]];
        let nodes2 = vec![vec![1, 2]];
        let ball_maps_per_time = vec![vec![&nodes1], vec![&nodes2]];

        let tensor = accumulate_temporal_pseudo_laplacians_inner(&ball_maps_per_time, 3).unwrap();

        assert_eq!(tensor.dim(), (3, 3, 2));

        // Time step 0: nodes 0,1 connected
        assert_eq!(tensor[[0, 1, 0]], -1);
        assert_eq!(tensor[[1, 2, 0]], 0);

        // Time step 1: nodes 1,2 connected
        assert_eq!(tensor[[0, 1, 1]], 0);
        assert_eq!(tensor[[1, 2, 1]], -1);
    }

    #[test]
    fn test_normalize_temporal() {
        // Create a simple 3D Laplacian tensor
        let mut l = Array3::<i64>::zeros((2, 2, 1));
        l[[0, 0, 0]] = 1;
        l[[1, 1, 0]] = 1;
        l[[0, 1, 0]] = -1;
        l[[1, 0, 0]] = -1;

        let w = normalize_temporal_laplacian(&l);

        // Both nodes in same ball: weight should be 1.0
        // denom = 1 + 1 + (-1) = 1
        // weight = -(-1) / 1 = 1.0
        assert!((w[[0, 1, 0]] - 1.0).abs() < 1e-10);
        assert!((w[[1, 0, 0]] - 1.0).abs() < 1e-10);
    }

    /// Two nodes whose edge weight rises 0 -> 0.5 -> 1 over three steps.
    fn ramp() -> TemporalCosmicGraph {
        let mut w = Array3::<f64>::zeros((2, 2, 3));
        for (t, v) in [0.0, 0.5, 1.0].into_iter().enumerate() {
            w[[0, 1, t]] = v;
            w[[1, 0, t]] = v;
        }
        TemporalCosmicGraph::new(w.into_dyn(), 0.25).unwrap()
    }

    #[test]
    fn aggregations_have_known_answers() {
        let g = ramp();
        let close = |got: f64, want: f64| assert!((got - want).abs() < 1e-12, "{got} != {want}");
        close(g.persistence_graph(None)[[0, 1]], 2.0 / 3.0);
        close(g.persistence_graph(Some(0.75))[[0, 1]], 1.0 / 3.0);
        close(g.mean_graph()[[0, 1]], 0.5);
        // Weights 1/7, 2/7, 4/7 for decay 0.5.
        close(g.recency_graph(0.5).unwrap()[[0, 1]], 5.0 / 7.0);
        close(g.volatility_graph()[[0, 1]], 1.0 / 6.0);
        close(g.trend_graph()[[0, 1]], 0.5);
        close(g.trend_graph()[[1, 0]], 0.5);
        close(g.change_point_graph()[[0, 1]], 0.5);
        close(g.mean_graph()[[0, 0]], 0.0);
        assert_eq!(g.shape(), (2, 2, 3));
        assert_eq!(g.nbytes(), 2 * 2 * 3 * 8);
    }

    #[test]
    fn one_time_step_has_no_trend_or_change() {
        let g = ramp().slice(2, None);
        assert_eq!(g.T(), 1);
        assert_eq!(g.trend_graph()[[0, 1]], 0.0);
        assert_eq!(g.change_point_graph()[[0, 1]], 0.0);
        assert_eq!(g.mean_graph()[[0, 1]], 1.0);
        // Out-of-range bounds clamp like a Python slice.
        assert_eq!(ramp().slice(5, Some(9)).T(), 0);
        assert_eq!(ramp().slice(0, Some(9)).T(), 3);
        assert_eq!(ramp().slice(5, None).mean_graph().dim(), (2, 2));
    }

    #[test]
    fn input_checks_use_the_python_messages() {
        let err = |e: PulsarError| e.to_string();
        assert_eq!(
            err(TemporalCosmicGraph::new(ArrayD::zeros(vec![3]), 0.0).unwrap_err()),
            "Invalid parameter: Expected 3D tensor, got shape (3,)"
        );
        assert_eq!(
            err(ramp().recency_graph(f64::NAN).unwrap_err()),
            "Invalid parameter: decay must be in (0, 1), got nan"
        );
        assert_eq!(
            err(ramp().recency_graph(1.5).unwrap_err()),
            "Invalid parameter: decay must be in (0, 1), got 1.5"
        );
        assert!(TemporalCosmicGraph::from_snapshots(
            &[Array2::zeros((4, 2))],
            &[3],
            &[42],
            &[0.5],
            0.0
        )
        .is_err());
    }

    #[test]
    fn temporal_accumulate_rejects_member_index_beyond_n() {
        // Ten points, but the second time step's ball map covers 40 rows.
        let ok = vec![vec![0, 1]];
        let forty_rows: Vec<Vec<usize>> = vec![(0..5).collect(), (30..40).collect()];
        let err = accumulate_temporal_pseudo_laplacians_inner(&[vec![&ok], vec![&forty_rows]], 10)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "Invalid parameter: ball member index 30 is out of range for n = 10 points"
        );
    }

    #[test]
    fn from_snapshots_rejects_non_positive_or_non_finite_epsilon() {
        let snaps = [Array2::from_shape_fn((6, 2), |(i, j)| (i + 2 * j) as f64)];
        for eps in [0.0, -0.5, f64::NAN, f64::INFINITY] {
            let err = TemporalCosmicGraph::from_snapshots(&snaps, &[2], &[42], &[0.5, eps], 0.0)
                .unwrap_err();
            assert!(
                err.to_string()
                    .contains("epsilon must be a finite positive number"),
                "{err}"
            );
        }
    }
}
