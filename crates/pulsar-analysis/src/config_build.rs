//! The baseline sweep config `create_config` emits
//! (`_build_initial_config_yaml`, `_variance_elbow_dimension` and
//! `_initial_pca_grid` in `pulsar/mcp/config_tools.py`).
//!
//! Epsilon and projection dimensions come from the geometry of the processed
//! feature space (or the raw numeric columns when calibration is unavailable).
//! The text is a fixed template, not a serialization, so it matches the
//! Python output byte for byte.
//!
//! Known divergences from Python:
//! - Python clamped the dimension grid at 16 (the KD-tree envelope). The Rust
//!   grid is bounded only by the variance elbow, the feature count and the
//!   profiled row count. `initial_pca_grid` and `build_initial_config_yaml`
//!   take an optional `cap`: production passes `None`, and the golden parity
//!   test passes `Some(16)` to reproduce the frozen Python output.
//! - Python's variance curve was probed only up to 20 dimensions, so its
//!   elbow never exceeded 20; see the divergence in `characterization`. The
//!   functions here take whatever curve they are given.
//! - Python did not bound the grid by the row count. With a measured curve
//!   that changes nothing (every probe is within the profiled rows); it only
//!   bites in the feature-count heuristic used when no curve exists, which
//!   could otherwise ask PCA for more components than there are rows.

use crate::characterization::{DatasetProfile, NumericProfile};
use crate::plan::{plan_to_preprocessing_yaml, PreprocessingPlan};

const VARIANCE_FRONTIER_TARGETS: [f64; 3] = [0.50, 0.70, 0.85];
const PCA_FLOOR: usize = 3;

/// The dimension where cumulative variance first reaches 90%.
pub fn variance_elbow_dimension(cumulative: &[(usize, f64)], n_features: usize) -> usize {
    if cumulative.is_empty() {
        return if n_features <= 4 {
            n_features.max(2)
        } else if n_features <= 12 {
            5
        } else {
            let log = ((n_features.max(1)) as f64).log2() * 3.0;
            (log as usize).max(10).min(n_features)
        };
    }
    let mut points = cumulative.to_vec();
    points.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    points
        .iter()
        .find(|(_, var)| *var >= 0.90)
        .map_or(points[points.len() - 1].0, |(dim, _)| *dim)
}

/// Where the top of the dimension grid comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CeilingSource {
    /// A probed dimension below the largest probe reached 90% cumulative
    /// variance (or, with a probe set that stops short of both bounds, the
    /// largest probe did).
    VarianceElbow,
    /// The largest probed dimension, below both the feature and the row
    /// count, stayed under 90%: the probe set ran out. Reachable only with a
    /// probe set that stops short, such as Python's.
    ProbeLimit,
    /// The feature count: 90% was not reached before it, or the elbow lies
    /// above it.
    NFeatures,
    /// The profiled row count, below the feature count: PCA cannot keep more
    /// components than rows, so the curve could not be measured further.
    RowLimit,
    /// No variance curve: the feature-count heuristic of
    /// [`variance_elbow_dimension`].
    Fallback,
}

impl CeilingSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::VarianceElbow => "variance_elbow",
            Self::ProbeLimit => "probe_limit",
            Self::NFeatures => "n_features",
            Self::RowLimit => "row_limit",
            Self::Fallback => "fallback",
        }
    }
}

/// The top dimension of the uncapped grid and its provenance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridCeiling {
    pub dim: usize,
    pub source: CeilingSource,
    /// Cumulative variance at `dim`, when the curve probed that dimension.
    pub variance: Option<f64>,
}

/// The top of the grid: the elbow, bounded by the feature count, the
/// profiled row count and the optional `cap`. Shared by [`grid_ceiling`] and
/// [`initial_pca_grid`] so the reported ceiling is the grid's maximum.
fn grid_top(knee: usize, n_features: usize, n_rows: usize, cap: Option<usize>) -> usize {
    knee.min(n_features)
        .min(n_rows)
        .min(cap.unwrap_or(usize::MAX))
}

/// The ceiling [`initial_pca_grid`] uses with `cap: None`. `n_rows` is the
/// number of rows the curve was profiled on.
pub fn grid_ceiling(cumulative: &[(usize, f64)], n_features: usize, n_rows: usize) -> GridCeiling {
    let knee = variance_elbow_dimension(cumulative, n_features);
    let dim = grid_top(knee, n_features, n_rows, None);
    let variance = cumulative
        .iter()
        .filter(|(d, _)| *d == dim)
        .map(|(_, v)| *v)
        .reduce(f64::max);
    let largest_probe = cumulative.iter().map(|(d, _)| *d).max();
    let reached = cumulative.iter().any(|(_, v)| *v >= 0.90);
    // A probe at the largest measurable dimension holds all the variance, so
    // "reached 90%" alone cannot tell an elbow from a bound: attribute by
    // where the elbow landed.
    let source = match largest_probe {
        None => CeilingSource::Fallback,
        Some(_) if dim < knee && dim == n_features => CeilingSource::NFeatures,
        Some(_) if dim < knee => CeilingSource::RowLimit,
        Some(last) if knee < last => CeilingSource::VarianceElbow,
        Some(last) if last >= n_features => CeilingSource::NFeatures,
        Some(last) if last >= n_rows => CeilingSource::RowLimit,
        Some(_) if reached => CeilingSource::VarianceElbow,
        Some(_) => CeilingSource::ProbeLimit,
    };
    GridCeiling {
        dim,
        source,
        variance,
    }
}

/// Multi-scale grid at the 50/70/85% variance frontiers plus the elbow,
/// bounded by the feature count and by `n_rows` (PCA keeps at most as many
/// components as rows), optionally clamped at `cap`.
pub fn initial_pca_grid(
    n_features: usize,
    n_rows: usize,
    knee: usize,
    cumulative: &[(usize, f64)],
    cap: Option<usize>,
) -> Vec<usize> {
    let mut points = cumulative.to_vec();
    points.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let ceiling = grid_top(knee, n_features, n_rows, cap);
    if ceiling == 0 {
        return vec![];
    }
    let floor = PCA_FLOOR.min(ceiling);
    let dim_at = |target: f64| points.iter().find(|(_, v)| *v >= target).map(|(d, _)| *d);

    let mut dims: Vec<usize> = VARIANCE_FRONTIER_TARGETS
        .iter()
        .filter_map(|&t| dim_at(t))
        .collect();
    dims.push(ceiling);

    let mut grid: Vec<usize> = dims
        .into_iter()
        .filter(|&d| d >= 2)
        .map(|d| d.min(ceiling).max(floor))
        .collect();
    grid.sort_unstable();
    grid.dedup();
    let target_count = 3.min((ceiling + 1).saturating_sub(floor).max(1));
    if grid.len() >= target_count {
        return grid;
    }
    if ceiling <= 4 {
        grid.extend(floor..=ceiling);
    } else {
        // Python's round() ties to even.
        grid.extend([
            floor.max(ceiling / 2),
            floor.max((ceiling as f64 * 0.75).round_ties_even() as usize),
            ceiling,
        ]);
    }
    grid.sort_unstable();
    grid.dedup();
    grid
}

fn int_list(values: &[usize]) -> String {
    let items: Vec<String> = values.iter().map(usize::to_string).collect();
    format!("[{}]", items.join(", "))
}

/// The epsilon range of the sweep, `(min, max)`.
pub fn epsilon_range(knn_mean: f64, p25: f64, p75: f64) -> (f64, f64) {
    if p25 > 0.0 && p75 > 0.0 {
        (p25 * 0.75, p75 * 1.35)
    } else {
        (knn_mean * 0.8, knn_mean * 1.5)
    }
}

/// The variance curve, feature count and profiled row count the grid is
/// built from: the processed calibration when it succeeded, else the raw
/// characterization.
pub fn calibration_curve<'a>(
    geo: &'a DatasetProfile,
    processed: Option<&'a NumericProfile>,
) -> (&'a [(usize, f64)], usize, usize) {
    match processed {
        Some(p) => (
            p.pca_cumulative_variance.as_slice(),
            p.n_features,
            p.n_samples_profiled,
        ),
        None => (
            geo.pca_cumulative_variance.as_slice(),
            geo.n_features,
            geo.n_samples_profiled,
        ),
    }
}

/// The baseline config: `geo` is the raw characterization, `processed` the
/// calibration under `plan` when it succeeded, `cap` an optional clamp on the
/// dimension grid.
pub fn build_initial_config_yaml(
    geo: &DatasetProfile,
    data_path: &str,
    run_name: &str,
    processed: Option<&NumericProfile>,
    plan: &PreprocessingPlan,
    cap: Option<usize>,
) -> String {
    let (knn_mean, cumulative, p25, p75, n_features, n_rows) = match processed {
        Some(p) => (
            if p.knn_k5_mean != 0.0 {
                p.knn_k5_mean
            } else {
                0.5
            },
            p.pca_cumulative_variance.as_slice(),
            p.knn_p25,
            p.knn_p75,
            p.n_features,
            p.n_samples_profiled,
        ),
        None => (
            if geo.knn_k5_mean != 0.0 {
                geo.knn_k5_mean
            } else {
                0.5
            },
            geo.pca_cumulative_variance.as_slice(),
            0.0,
            0.0,
            geo.n_features,
            geo.n_samples_profiled,
        ),
    };
    let knee = variance_elbow_dimension(cumulative, n_features);
    let dims = int_list(&initial_pca_grid(n_features, n_rows, knee, cumulative, cap));
    let (eps_min, eps_max) = epsilon_range(knn_mean, p25, p75);
    let preprocessing = plan_to_preprocessing_yaml(plan, true);

    format!(
        "run:\n  name: {run_name}\n  data: {data_path}\n{preprocessing}\nsweep:\n  projection:\n    method: jl\n    dimensions:\n      values: {dims}\n    seed:\n      values: [42, 7, 13]\n    center: true\n  # Legacy mirror for older configs. sweep.projection is authoritative.\n  pca:\n    dimensions:\n      values: {dims}\n    seed:\n      values: [42, 7, 13]\n  ball_mapper:\n    epsilon:\n      range:\n        min: {eps_min:.4}\n        max: {eps_max:.4}\n        steps: 24\ncosmic_graph:\n  construction_threshold: auto\noutput:\n  n_reps: 4\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elbow_falls_back_on_feature_count_without_a_curve() {
        assert_eq!(variance_elbow_dimension(&[], 3), 3);
        assert_eq!(variance_elbow_dimension(&[], 1), 2);
        assert_eq!(variance_elbow_dimension(&[], 9), 5);
        // log2(100) * 3 = 19.9 -> 19
        assert_eq!(variance_elbow_dimension(&[], 100), 19);
        assert_eq!(variance_elbow_dimension(&[], 13), 11);
    }

    #[test]
    fn elbow_is_the_first_dimension_at_ninety_percent() {
        let curve = [(1, 0.5), (2, 0.8), (3, 0.91), (4, 1.0)];
        assert_eq!(variance_elbow_dimension(&curve, 4), 3);
        assert_eq!(variance_elbow_dimension(&[(1, 0.2), (2, 0.4)], 4), 2);
    }

    #[test]
    fn grid_targets_variance_frontiers_and_honours_an_optional_cap() {
        let curve: Vec<(usize, f64)> = (1..=20).map(|d| (d, d as f64 / 20.0)).collect();
        // Frontiers 10, 14, 17 and a ceiling of 16.
        assert_eq!(
            initial_pca_grid(20, 1000, 16, &curve, None),
            vec![10, 14, 16]
        );
        // Frontiers 10, 14, 17 and the elbow 18 (0.90), clamped at 16 as Python did.
        assert_eq!(
            initial_pca_grid(20, 1000, 18, &curve, Some(16)),
            vec![10, 14, 16]
        );
        // Uncapped, an elbow of 20 tops the grid.
        assert_eq!(
            initial_pca_grid(20, 1000, 20, &curve, None),
            vec![10, 14, 17, 20]
        );
        assert_eq!(
            initial_pca_grid(20, 1000, 20, &curve, Some(16)),
            vec![10, 14, 16]
        );
        assert_eq!(
            initial_pca_grid(0, 1000, 5, &curve, None),
            Vec::<usize>::new()
        );
        // Two features: floor == ceiling == 2.
        assert_eq!(
            initial_pca_grid(2, 1000, 2, &[(1, 0.9), (2, 1.0)], None),
            vec![2]
        );
    }

    #[test]
    fn uncapped_grid_reaches_the_probe_limit() {
        // Never reaches 90%: the elbow is the last probed dimension, 20.
        let curve: Vec<(usize, f64)> = (1..=20).map(|d| (d, d as f64 / 23.0)).collect();
        let knee = variance_elbow_dimension(&curve, 40);
        assert_eq!(knee, 20);
        assert_eq!(
            initial_pca_grid(40, 1000, knee, &curve, None),
            vec![12, 17, 20]
        );
        assert_eq!(
            initial_pca_grid(40, 1000, knee, &curve, Some(16)),
            vec![8, 12, 16]
        );
        let ceiling = grid_ceiling(&curve, 40, 1000);
        assert_eq!(ceiling.dim, 20);
        assert_eq!(ceiling.source, CeilingSource::ProbeLimit);
        assert_eq!(ceiling.variance, Some(20.0 / 23.0));
    }

    #[test]
    fn grid_ceiling_names_its_source() {
        let elbow = grid_ceiling(&[(2, 0.5), (3, 0.91), (5, 1.0)], 8, 1000);
        assert_eq!((elbow.dim, elbow.source), (3, CeilingSource::VarianceElbow));
        assert_eq!(elbow.variance, Some(0.91));
        let fallback = grid_ceiling(&[], 9, 1000);
        assert_eq!(
            (fallback.dim, fallback.source),
            (5, CeilingSource::Fallback)
        );
        assert_eq!(fallback.variance, None);
        // Elbow at 10 but only 6 features.
        let narrow = grid_ceiling(&[(2, 0.3), (10, 0.95)], 6, 1000);
        assert_eq!((narrow.dim, narrow.source), (6, CeilingSource::NFeatures));
        assert_eq!(narrow.variance, None);
        // A curve probed up to the feature count: 90% only at the last probe.
        let full = grid_ceiling(&[(20, 0.67), (30, 1.0)], 30, 200);
        assert_eq!((full.dim, full.source), (30, CeilingSource::NFeatures));
        assert_eq!(full.variance, Some(1.0));
        // Only 25 rows for 40 features: the curve stops at the row count.
        let short = grid_ceiling(&[(20, 0.8), (25, 1.0)], 40, 25);
        assert_eq!((short.dim, short.source), (25, CeilingSource::RowLimit));
        // A full curve reaching 90% before its last probe.
        let early = grid_ceiling(&[(10, 0.85), (15, 0.92), (30, 1.0)], 30, 200);
        assert_eq!(
            (early.dim, early.source),
            (15, CeilingSource::VarianceElbow)
        );
    }

    #[test]
    fn the_fallback_grid_is_bounded_by_the_row_count() {
        // No curve: the heuristic gives 19 for 100 features, but 5 rows
        // cannot support more than 5 components.
        assert_eq!(initial_pca_grid(100, 1000, 19, &[], None), vec![9, 14, 19]);
        assert_eq!(initial_pca_grid(100, 5, 19, &[], None), vec![3, 4, 5]);
        let ceiling = grid_ceiling(&[], 100, 5);
        assert_eq!((ceiling.dim, ceiling.source), (5, CeilingSource::Fallback));
    }

    #[test]
    fn epsilon_uses_percentiles_when_calibrated() {
        assert_eq!(epsilon_range(1.0, 2.0, 4.0), (1.5, 5.4));
        assert_eq!(epsilon_range(1.0, 0.0, 4.0), (0.8, 1.5));
    }
}
