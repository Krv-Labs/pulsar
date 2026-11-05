//! The baseline sweep config `create_config` emits
//! (`_build_initial_config_yaml`, `_variance_elbow_dimension` and
//! `_initial_pca_grid` in `pulsar/mcp/config_tools.py`).
//!
//! Epsilon and projection dimensions come from the geometry of the processed
//! feature space (or the raw numeric columns when calibration is unavailable).
//! The text is a fixed template, not a serialization, so it matches the
//! Python output byte for byte.

use crate::characterization::{DatasetProfile, NumericProfile};
use crate::plan::{plan_to_preprocessing_yaml, PreprocessingPlan};

const VARIANCE_FRONTIER_TARGETS: [f64; 3] = [0.50, 0.70, 0.85];
const PCA_FLOOR: usize = 3;
/// Dimensions above this leave the KD-tree envelope.
const MAX_DIMENSION: usize = 16;

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

/// Multi-scale grid at the 50/70/85% variance frontiers plus the elbow.
pub fn initial_pca_grid(n_features: usize, knee: usize, cumulative: &[(usize, f64)]) -> Vec<usize> {
    let mut points = cumulative.to_vec();
    points.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let ceiling = knee.min(n_features).min(MAX_DIMENSION);
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

/// The baseline config: `geo` is the raw characterization, `processed` the
/// calibration under `plan` when it succeeded.
pub fn build_initial_config_yaml(
    geo: &DatasetProfile,
    data_path: &str,
    run_name: &str,
    processed: Option<&NumericProfile>,
    plan: &PreprocessingPlan,
) -> String {
    let (knn_mean, cumulative, p25, p75, n_features) = match processed {
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
        ),
    };
    let knee = variance_elbow_dimension(cumulative, n_features);
    let dims = int_list(&initial_pca_grid(n_features, knee, cumulative));
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
    fn grid_targets_variance_frontiers_and_is_capped() {
        let curve: Vec<(usize, f64)> = (1..=20).map(|d| (d, d as f64 / 20.0)).collect();
        // Frontiers 10, 14, 17 and a ceiling of 16.
        assert_eq!(initial_pca_grid(20, 16, &curve), vec![10, 14, 16]);
        assert_eq!(initial_pca_grid(0, 5, &curve), Vec::<usize>::new());
        // Two features: floor == ceiling == 2.
        assert_eq!(initial_pca_grid(2, 2, &[(1, 0.9), (2, 1.0)]), vec![2]);
    }

    #[test]
    fn epsilon_uses_percentiles_when_calibrated() {
        assert_eq!(epsilon_range(1.0, 2.0, 4.0), (1.5, 5.4));
        assert_eq!(epsilon_range(1.0, 0.0, 4.0), (0.8, 1.5));
    }
}
