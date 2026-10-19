//! Sweep history synthesis over persisted run records
//! (`pulsar/mcp/history.py`). Observations and rationale only: no
//! recommended config, so the agent's next decision stays explicit.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::record::{RunMetrics, RunRecord};

const HAIRBALL: &str = "hairball";
const FRAGMENTED_SINGLETON_FRACTION: f64 = 0.25;
const ICE_CHIP_NONTRIVIAL_DELTA: f64 = 0.02;
const MEANINGFUL_NONTRIVIAL_DELTA: f64 = 0.03;
const SINGLETON_SPIKE_DELTA: f64 = 0.05;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ComponentProfile {
    pub n_nodes: usize,
    pub component_count: usize,
    pub giant_fraction: f64,
    pub non_giant_mass: f64,
    pub nontrivial_component_count: usize,
    pub nontrivial_component_mass: f64,
    pub largest_non_giant_component_pct: f64,
    pub singleton_fraction: f64,
    pub small_component_mass: f64,
}

impl ComponentProfile {
    /// Cross-run trend surface with its own absolute non-giant floor
    /// (`max(10, ceil(1% of n))`), separate from the gate-side significance
    /// rule.
    pub fn from_metrics(m: &RunMetrics) -> Self {
        let sizes: Vec<usize> = m
            .component_sizes
            .iter()
            .copied()
            .filter(|&s| s > 0)
            .collect();
        let n_total = if m.n_nodes > 0 {
            m.n_nodes
        } else {
            sizes.iter().sum()
        };
        if n_total == 0 || sizes.is_empty() {
            return Self {
                n_nodes: n_total,
                component_count: m.component_count,
                giant_fraction: m.giant_fraction,
                non_giant_mass: 0.0,
                nontrivial_component_count: 0,
                nontrivial_component_mass: 0.0,
                largest_non_giant_component_pct: 0.0,
                singleton_fraction: m.singleton_fraction,
                small_component_mass: 0.0,
            };
        }
        let mut sizes = sizes;
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let giant = sizes[0];
        let tail = &sizes[1..];
        let nontrivial_min = 10.max((n_total as f64 * 0.01).ceil() as usize);
        let nontrivial: usize = tail.iter().filter(|&&s| s >= nontrivial_min).sum();
        let nontrivial_count = tail.iter().filter(|&&s| s >= nontrivial_min).count();
        let small: usize = tail.iter().filter(|&&s| s < nontrivial_min).sum();
        let n = n_total as f64;
        Self {
            n_nodes: n_total,
            component_count: if m.component_count > 0 {
                m.component_count
            } else {
                sizes.len()
            },
            giant_fraction: giant as f64 / n,
            non_giant_mass: (n_total - giant) as f64 / n,
            nontrivial_component_count: nontrivial_count,
            nontrivial_component_mass: nontrivial as f64 / n,
            largest_non_giant_component_pct: tail.first().copied().unwrap_or(0) as f64 / n,
            singleton_fraction: m.singleton_fraction,
            small_component_mass: small as f64 / n,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FragmentationTrend {
    pub status: &'static str,
    pub next_decision: &'static str,
    pub agent_action: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous: Option<ComponentProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current: Option<ComponentProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nontrivial_component_mass_delta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub singleton_fraction_delta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub largest_non_giant_component_pct: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component_count_delta: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub significant_component_threshold_pct: Option<f64>,
}

impl FragmentationTrend {
    fn insufficient() -> Self {
        Self {
            status: "insufficient_history",
            next_decision: "run_one_more_sweep",
            agent_action: "Run at least two sweeps before judging fragmentation trend.",
            previous: None,
            current: None,
            nontrivial_component_mass_delta: None,
            singleton_fraction_delta: None,
            largest_non_giant_component_pct: None,
            component_count_delta: None,
            significant_component_threshold_pct: None,
        }
    }

    fn from_history(history: &[RunRecord]) -> Self {
        let [.., prev_rec, cur_rec] = history else {
            return Self::insufficient();
        };
        let previous = ComponentProfile::from_metrics(&prev_rec.metrics);
        let current = ComponentProfile::from_metrics(&cur_rec.metrics);
        let n_total = if current.n_nodes > 0 {
            current.n_nodes
        } else {
            previous.n_nodes
        };
        let n = n_total.max(1) as f64;
        let significant_pct = 25.0_f64.max((n * 0.03).ceil()) / n;

        let nontrivial_delta =
            current.nontrivial_component_mass - previous.nontrivial_component_mass;
        let singleton_delta = current.singleton_fraction - previous.singleton_fraction;
        let largest_non_giant = current.largest_non_giant_component_pct;
        let component_count_delta =
            current.component_count as i64 - previous.component_count as i64;

        let (status, next_decision, agent_action) = if singleton_delta > SINGLETON_SPIKE_DELTA {
            (
                "over_fragmentation",
                "back_off_threshold",
                "Back off the latest refinement: singleton mass increased faster than coherent \
                 non-giant structure.",
            )
        } else if largest_non_giant >= significant_pct
            || nontrivial_delta >= MEANINGFUL_NONTRIVIAL_DELTA
        {
            (
                "meaningful_resolution",
                "inspect_features",
                "Inspect the emerging non-giant component(s) with generate_cluster_dossier or \
                 get_cluster_profile before further tuning.",
            )
        } else if component_count_delta > 0 && nontrivial_delta < ICE_CHIP_NONTRIVIAL_DELTA {
            (
                "ice_chipping",
                "run_one_more_sweep_or_back_off",
                "Do not treat higher component count as progress. The refinement mostly chipped \
                 off tiny components; shift the grid only if a non-trivial component starts \
                 gaining mass.",
            )
        } else {
            (
                "stable",
                "compare_feature_evidence",
                "No strong fragmentation trend detected. Compare feature evidence before making \
                 another refinement.",
            )
        };
        Self {
            status,
            next_decision,
            agent_action,
            previous: Some(previous),
            current: Some(current),
            nontrivial_component_mass_delta: Some(round4(nontrivial_delta)),
            singleton_fraction_delta: Some(round4(singleton_delta)),
            largest_non_giant_component_pct: Some(round4(largest_non_giant)),
            component_count_delta: Some(component_count_delta),
            significant_component_threshold_pct: Some(round4(significant_pct)),
        }
    }
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HistorySummary {
    pub n_runs: usize,
    pub observations: Vec<String>,
    pub rationale: String,
    pub fragmentation_trend: FragmentationTrend,
}

/// Synthesize observations from runs in creation order (oldest first).
pub fn summarize_history(history: &[RunRecord]) -> HistorySummary {
    if history.is_empty() {
        return HistorySummary {
            n_runs: 0,
            observations: vec!["No sweeps recorded in this session.".to_owned()],
            rationale: String::new(),
            fragmentation_trend: FragmentationTrend::insufficient(),
        };
    }

    let n = history.len();
    let mut health_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut dim_health: BTreeMap<usize, Vec<&'static str>> = BTreeMap::new();
    let mut thresholds_with_singletons: Vec<String> = Vec::new();

    for record in history {
        let health = record.metrics.health_label();
        *health_counts.entry(health).or_default() += 1;
        for &dim in &record.config.projection_dimensions {
            dim_health.entry(dim).or_default().push(health);
        }
        if record.metrics.singleton_fraction > FRAGMENTED_SINGLETON_FRACTION {
            thresholds_with_singletons.push(record.config.threshold_label());
        }
    }

    let majority = |healths: &[&str], pred: &dyn Fn(&str) -> bool| {
        healths.iter().filter(|h| pred(h)).count() >= 1.max(healths.len() / 2)
    };
    let hairball_dims: Vec<usize> = dim_health
        .iter()
        .filter(|(_, h)| majority(h, &|h| h == HAIRBALL))
        .map(|(d, _)| *d)
        .collect();
    let fragmented_dims: Vec<usize> = dim_health
        .iter()
        .filter(|(_, h)| majority(h, &|h| h == "fragmented" || h == "empty"))
        .map(|(d, _)| *d)
        .collect();

    let mut observations = Vec::new();
    if !hairball_dims.is_empty() {
        observations.push(format!(
            "Projection dimensions {hairball_dims:?} appeared in hairball regimes in the majority \
             of runs containing them."
        ));
    }
    if !fragmented_dims.is_empty() {
        observations.push(format!(
            "Projection dimensions {fragmented_dims:?} appeared in fragmented or empty graphs in \
             the majority of runs containing them."
        ));
    }
    if !thresholds_with_singletons.is_empty() {
        let mut labels = thresholds_with_singletons.clone();
        labels.sort();
        labels.dedup();
        observations.push(format!(
            "Construction thresholds {labels:?} coincided with elevated singleton fractions."
        ));
    }
    let breakdown: Vec<String> = health_counts
        .iter()
        .map(|(label, count)| format!("{label}\u{d7}{count}"))
        .collect();
    observations.push(format!(
        "Graph health across {n} runs: {}.",
        breakdown.join(", ")
    ));

    let fragmentation_trend = FragmentationTrend::from_history(history);
    observations.push(format!(
        "Fragmentation trend: {}.",
        fragmentation_trend.status
    ));

    let mut rationale = Vec::new();
    if !hairball_dims.is_empty() {
        rationale
            .push("Low or extreme projection dimensions correlate with over-connected hairballs.");
    }
    if !fragmented_dims.is_empty() {
        rationale.push(
            "Some projection dimensions consistently shatter the graph; consider shifting away from them.",
        );
    }
    if !thresholds_with_singletons.is_empty() {
        rationale.push("High construction thresholds collapsed the graph into singletons.");
    }

    HistorySummary {
        n_runs: n,
        observations,
        rationale: rationale.join(" "),
        fragmentation_trend,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::run_record;

    fn metrics(n: usize, sizes: &[usize], density: f64) -> RunMetrics {
        let singleton_count = sizes.iter().filter(|&&s| s == 1).count();
        let n_edges = if density > 0.0 { 1 } else { 0 };
        RunMetrics {
            n_nodes: n,
            n_edges,
            density,
            avg_degree: 0.0,
            component_count: sizes.len(),
            component_sizes: sizes.to_vec(),
            giant_fraction: sizes[0] as f64 / n as f64,
            singleton_count,
            singleton_fraction: singleton_count as f64 / n as f64,
        }
    }

    #[test]
    fn empty_history_is_explicit() {
        let s = summarize_history(&[]);
        assert_eq!(s.n_runs, 0);
        assert_eq!(s.fragmentation_trend.status, "insufficient_history");
        assert_eq!(s.observations, vec!["No sweeps recorded in this session."]);
    }

    #[test]
    fn observations_follow_dimension_health_and_singleton_thresholds() {
        // 100 nodes: run 1 hairball at dims [2]; run 2 fragmented at dims [2, 5] with a
        // fixed threshold; run 3 connected at [5].
        let mut r1 = run_record(metrics(100, &[100], 0.9));
        r1.config.projection_dimensions = vec![2];
        let mut sizes2 = vec![60];
        sizes2.extend(std::iter::repeat_n(1, 40));
        let mut r2 = run_record(metrics(100, &sizes2, 0.2));
        r2.config.projection_dimensions = vec![2, 5];
        r2.config.threshold_policy = crate::record::ThresholdPolicy::Fixed;
        r2.config.requested_threshold = Some(0.5);
        let mut r3 = run_record(metrics(100, &[80, 20], 0.2));
        r3.config.projection_dimensions = vec![5];

        let s = summarize_history(&[r1, r2, r3]);
        assert_eq!(s.n_runs, 3);
        assert!(
            s.observations[0].contains("[2] appeared in hairball"),
            "{:?}",
            s.observations
        );
        // Python's majority rule is `count >= max(1, len // 2)`: with two runs
        // per dimension, one fragmented run is enough.
        assert!(s.observations[1].contains("[2, 5] appeared in fragmented"));
        assert!(s.observations[2].contains("[\"0.5\"] coincided"));
        assert_eq!(
            s.observations[3],
            "Graph health across 3 runs: connected\u{d7}1, fragmented\u{d7}1, hairball\u{d7}1."
        );
        // run 2 → run 3: singletons dropped, a 20-node component appeared (>= 3%).
        let trend = &s.fragmentation_trend;
        assert_eq!(trend.status, "meaningful_resolution");
        assert_eq!(trend.component_count_delta, Some(-39));
        assert_eq!(trend.significant_component_threshold_pct, Some(0.25));
        assert!(s.rationale.contains("hairballs") && s.rationale.contains("singletons"));
    }

    #[test]
    fn trend_classifies_spikes_and_ice_chipping() {
        let base = run_record(metrics(1000, &[1000], 0.2));
        let mut spiky = vec![900];
        spiky.extend(std::iter::repeat_n(1, 100));
        let spike = run_record(metrics(1000, &spiky, 0.2));
        assert_eq!(
            summarize_history(&[base.clone(), spike])
                .fragmentation_trend
                .status,
            "over_fragmentation"
        );
        let chips = run_record(metrics(1000, &[995, 3, 2], 0.2));
        assert_eq!(
            summarize_history(&[base.clone(), chips])
                .fragmentation_trend
                .status,
            "ice_chipping"
        );
        let same = run_record(metrics(1000, &[1000], 0.2));
        assert_eq!(
            summarize_history(&[base, same]).fragmentation_trend.status,
            "stable"
        );
    }
}
