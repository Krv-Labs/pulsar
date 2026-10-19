//! Threshold mass dynamics (`pulsar/mcp/thresholds.py`): component mass
//! profiles, plateau and transition candidate ranking under interpretation
//! policies, deduplication, and structural breakpoints.
//!
//! Every number that the Python code rounded is rounded here with
//! [`round_to`] (Python semantics) *and* the rounded value is what later
//! steps read, exactly as the dict-based Python did. The golden fixtures
//! `threshold_options` and `breakpoints` check this port at `rtol 1e-9`.

use std::collections::HashMap;

use pulsar_core::Plateau;
use serde::{Deserialize, Serialize};

use crate::components::{ComponentState, ThresholdGraph};
use crate::error::AnalysisError;
use crate::pyround::{round_int, round_to};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    Balanced,
    ReportReady,
    DetailSeeking,
    OutlierMining,
}

impl Policy {
    pub const ALL: [&'static str; 4] = [
        "balanced",
        "report_ready",
        "detail_seeking",
        "outlier_mining",
    ];

    pub fn parse(name: &str) -> Result<Self, AnalysisError> {
        match name {
            "balanced" => Ok(Self::Balanced),
            "report_ready" => Ok(Self::ReportReady),
            "detail_seeking" => Ok(Self::DetailSeeking),
            "outlier_mining" => Ok(Self::OutlierMining),
            other => Err(AnalysisError::UnknownPolicy(other.to_owned())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::ReportReady => "report_ready",
            Self::DetailSeeking => "detail_seeking",
            Self::OutlierMining => "outlier_mining",
        }
    }
}

/// Tiers that read as cohorts.
const COHORT_READY_TIERS: [&str; 2] = ["report_ready", "balanced"];
/// A plateau must span this fraction of the active threshold range to count
/// as fully persistent.
const PERSISTENCE_ACTIVE_RANGE_FRACTION: f64 = 0.25;
const SINGLETON_FRONTIER_FRACTION: f64 = 0.95;
/// A `>=` this-many-fold drop between neighbouring component sizes marks
/// where the dust tail begins.
const SIGNIFICANCE_CLIFF_RATIO: f64 = 3.0;

// ------------------------------------------------------------- mass profile

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentMassProfile {
    pub n_nodes: usize,
    pub component_count: usize,
    pub top_component_sizes: Vec<usize>,
    pub top_component_sizes_omitted: usize,
    pub largest_component_fraction: f64,
    pub singleton_count: usize,
    pub singleton_fraction: f64,
    pub small_component_count: usize,
    pub small_component_mass_fraction: f64,
}

/// Component mass at one threshold: the top `top_k` sizes, singletons and the
/// "small" tail (components of at most `small_component_max_fraction · n`).
pub fn component_mass_profile(
    graph: &ThresholdGraph,
    threshold: f64,
    top_k: usize,
    small_component_max_fraction: f64,
) -> ComponentMassProfile {
    profile_from_state(
        graph.n_nodes(),
        &graph.state_at(threshold),
        top_k,
        small_component_max_fraction,
    )
}

fn profile_from_state(
    n_nodes: usize,
    state: &ComponentState,
    top_k: usize,
    small_component_max_fraction: f64,
) -> ComponentMassProfile {
    let sizes = &state.sizes;
    let top: Vec<usize> = sizes.iter().copied().take(top_k).collect();
    let singleton_count = sizes.iter().filter(|&&s| s == 1).count();
    let small_size_limit = 1.max((n_nodes as f64 * small_component_max_fraction) as usize);
    let small: Vec<usize> = sizes
        .iter()
        .copied()
        .filter(|&s| s <= small_size_limit)
        .collect();
    let frac = |num: usize| {
        if n_nodes > 0 {
            num as f64 / n_nodes as f64
        } else {
            0.0
        }
    };
    ComponentMassProfile {
        n_nodes,
        component_count: sizes.len(),
        top_component_sizes_omitted: sizes.len().saturating_sub(top.len()),
        largest_component_fraction: round_to(top.first().map_or(0.0, |&t| frac(t)), 4),
        singleton_count,
        singleton_fraction: round_to(frac(singleton_count), 4),
        small_component_count: small.len(),
        small_component_mass_fraction: round_to(frac(small.iter().sum()), 4),
        top_component_sizes: top,
    }
}

pub fn mass_profile_hint(profile: &ComponentMassProfile) -> &'static str {
    let largest = profile.largest_component_fraction;
    let singletons = profile.singleton_fraction;
    let small_mass = profile.small_component_mass_fraction;
    if profile.component_count <= 1 {
        "mostly connected graph"
    } else if largest >= 0.95 {
        "mostly connected graph with a small tail"
    } else if singletons >= 0.2 {
        "singleton-heavy fragmentation"
    } else if largest >= 0.8 && small_mass >= 0.05 {
        "giant component with small-tail fragmentation"
    } else if largest <= 0.8 && singletons <= 0.05 {
        "stable nontrivial multi-component structure"
    } else {
        "mixed component-size structure"
    }
}

/// `max(3, min(round(sqrt n), round(0.005 n)))`; descriptive context only.
pub fn useful_component_size_floor(n_nodes: usize) -> usize {
    if n_nodes == 0 {
        return 1;
    }
    let n = n_nodes as f64;
    let a = round_int(n.sqrt());
    let b = round_int(0.005 * n);
    3.max(a.min(b).max(0) as usize)
}

/// Components that are real modes vs. dust, judged relative to the size
/// distribution: the dust tail ends at the lowest `>= 3×` cliff between
/// neighbouring descending sizes; everything above it (except a lone
/// singleton) is a mode. Returns `(modes_desc, reason)`.
pub fn significant_component_sizes(sizes: &[usize]) -> (Vec<usize>, &'static str) {
    let mut ordered = sizes.to_vec();
    ordered.sort_unstable_by(|a, b| b.cmp(a));
    if ordered.is_empty() {
        return (Vec::new(), "no_components");
    }
    let mut cut = ordered.len();
    for i in 0..ordered.len() - 1 {
        if ordered[i] as f64 / ordered[i + 1].max(1) as f64 >= SIGNIFICANCE_CLIFF_RATIO {
            cut = i + 1;
        }
    }
    let modes: Vec<usize> = ordered[..cut].iter().copied().filter(|&s| s > 1).collect();
    if modes.is_empty() {
        return (Vec::new(), "no_modes");
    }
    let reason = if cut < ordered.len() {
        "modes_above_dust_cliff"
    } else {
        "all_modes"
    };
    (modes, reason)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MassShape {
    pub useful_component_size_floor: usize,
    pub nontrivial_component_count: usize,
    pub nontrivial_mass_fraction: f64,
    pub multi_component_coverage: f64,
    pub component_size_entropy: f64,
    pub top_two_balance: f64,
}

pub fn mass_shape_metrics(profile: &ComponentMassProfile) -> MassShape {
    let n_nodes = profile.n_nodes;
    let top = &profile.top_component_sizes;
    let floor = useful_component_size_floor(n_nodes);
    let (nontrivial, _) = significant_component_sizes(top);
    let nontrivial_mass: usize = nontrivial.iter().sum();
    let multi_mass: usize = if nontrivial.len() > 1 {
        nontrivial[1..].iter().sum()
    } else {
        0
    };

    let normalized_entropy = if top.is_empty() {
        0.0
    } else {
        let total = top.iter().sum::<usize>().max(1) as f64;
        let entropy: f64 = -top
            .iter()
            .map(|&s| {
                let p = s as f64 / total;
                p * (p + 1e-12).ln()
            })
            .sum::<f64>();
        if top.len() > 1 {
            entropy / (top.len() as f64).ln()
        } else {
            0.0
        }
    };
    let balance = if nontrivial.len() >= 2 {
        let (a, b) = (nontrivial[0] as f64, nontrivial[1] as f64);
        1.0 - (a - b).abs() / (a + b).max(1.0)
    } else {
        0.0
    };
    let frac = |num: usize| {
        if n_nodes > 0 {
            num as f64 / n_nodes as f64
        } else {
            0.0
        }
    };
    MassShape {
        useful_component_size_floor: floor,
        nontrivial_component_count: nontrivial.len(),
        nontrivial_mass_fraction: round_to(frac(nontrivial_mass), 4),
        multi_component_coverage: round_to(frac(multi_mass), 4),
        component_size_entropy: round_to(normalized_entropy, 4),
        top_two_balance: round_to(balance, 4),
    }
}

/// One descriptive row of the threshold morphology sample.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MorphologyRow {
    pub threshold: f64,
    pub component_count: usize,
    pub top_component_sizes: Vec<usize>,
    pub top_component_sizes_omitted: usize,
    pub giant_fraction: f64,
    pub second_largest_ratio: f64,
    pub singleton_count: usize,
    pub singleton_fraction: f64,
    pub small_component_mass_fraction: f64,
    pub nontrivial_component_count: usize,
    pub nontrivial_mass_fraction: f64,
    pub multi_component_coverage: f64,
    pub component_size_entropy: f64,
    pub interpretation_hint: &'static str,
}

pub fn threshold_morphology_profile(
    graph: &ThresholdGraph,
    threshold: f64,
    top_k: usize,
) -> MorphologyRow {
    let profile = component_mass_profile(graph, threshold, top_k, 0.01);
    let metrics = mass_shape_metrics(&profile);
    let giant = profile.top_component_sizes.first().copied().unwrap_or(0);
    let second = profile.top_component_sizes.get(1).copied().unwrap_or(0);
    let ratio = if giant > 0 {
        second as f64 / giant as f64
    } else {
        0.0
    };
    MorphologyRow {
        threshold,
        component_count: profile.component_count,
        top_component_sizes: profile.top_component_sizes.clone(),
        top_component_sizes_omitted: profile.top_component_sizes_omitted,
        giant_fraction: profile.largest_component_fraction,
        second_largest_ratio: round_to(ratio, 4),
        singleton_count: profile.singleton_count,
        singleton_fraction: profile.singleton_fraction,
        small_component_mass_fraction: profile.small_component_mass_fraction,
        nontrivial_component_count: metrics.nontrivial_component_count,
        nontrivial_mass_fraction: metrics.nontrivial_mass_fraction,
        multi_component_coverage: metrics.multi_component_coverage,
        component_size_entropy: metrics.component_size_entropy,
        interpretation_hint: mass_profile_hint(&profile),
    }
}

// ------------------------------------------------------- plateau candidates

fn plateau_tier(profile: &ComponentMassProfile, metrics: &MassShape) -> &'static str {
    let largest = profile.largest_component_fraction;
    let singletons = profile.singleton_fraction;
    let count = metrics.nontrivial_component_count;
    let mass = metrics.nontrivial_mass_fraction;
    if singletons >= 0.35 {
        "outlier_frontier"
    } else if largest >= 0.95 {
        "giant_component_with_dust"
    } else if count >= 2 && singletons <= 0.05 && mass >= 0.8 {
        "report_ready"
    } else if count >= 2 && singletons <= 0.2 && mass >= 0.55 {
        "balanced"
    } else if count >= 2 {
        "exploratory"
    } else {
        "weak_candidate"
    }
}

/// Width of the threshold window where `1 < component_count < max count`.
pub fn active_threshold_range(thresholds: &[f64], component_counts: &[usize]) -> f64 {
    let Some(&n_max) = component_counts.iter().max() else {
        return 0.0;
    };
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (&t, &c) in thresholds.iter().zip(component_counts) {
        if 1 < c && c < n_max {
            lo = lo.min(t);
            hi = hi.max(t);
        }
    }
    if lo.is_finite() {
        (hi - lo).abs()
    } else {
        0.0
    }
}

/// The active range approximated from plateau bounds alone (used when the
/// raw curve is not available).
pub fn active_range_from_plateaus(plateaus: &[Plateau], n_nodes: usize) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for p in plateaus {
        if 1 < p.component_count && p.component_count < n_nodes.max(2) {
            for t in [p.start_threshold, p.end_threshold] {
                lo = lo.min(t);
                hi = hi.max(t);
            }
        }
    }
    if lo.is_finite() {
        (hi - lo).abs()
    } else {
        0.0
    }
}

fn rank_plateau_candidate(
    plateau_width: f64,
    profile: &ComponentMassProfile,
    metrics: &MassShape,
    policy: Policy,
    active_range: Option<f64>,
) -> f64 {
    let eps = 1e-9;
    let denom = match active_range {
        Some(r) if r > eps => (r * PERSISTENCE_ACTIVE_RANGE_FRACTION).max(eps),
        _ => 0.05,
    };
    let persistence = (plateau_width / denom).clamp(0.0, 1.0);
    let balance = metrics.top_two_balance;
    let coverage = metrics.nontrivial_mass_fraction;
    let entropy = metrics.component_size_entropy;
    let singletons = profile.singleton_fraction;
    let small_mass = profile.small_component_mass_fraction;
    let (score, penalty) = match policy {
        Policy::ReportReady => (
            0.35 * persistence + 0.25 * balance + 0.3 * coverage + 0.1 * entropy,
            0.55 * singletons + 0.25 * small_mass,
        ),
        Policy::DetailSeeking => (
            0.25 * persistence + 0.3 * balance + 0.25 * coverage + 0.2 * entropy,
            0.2 * singletons + 0.1 * small_mass,
        ),
        Policy::OutlierMining => (
            0.15 * persistence + 0.1 * balance + 0.2 * coverage + 0.35 * entropy + 0.2 * singletons,
            0.05 * small_mass,
        ),
        Policy::Balanced => (
            0.3 * persistence + 0.25 * balance + 0.3 * coverage + 0.15 * entropy,
            0.35 * singletons + 0.15 * small_mass,
        ),
    };
    round_to((score - penalty).max(0.0), 4)
}

fn candidate_use_guidance(
    tier: &str,
    singleton_fraction: f64,
) -> (Vec<&'static str>, Vec<&'static str>, &'static str) {
    match tier {
        "report_ready" => (
            vec!["report_ready", "balanced"],
            vec!["outlier_mining"],
            "Candidate has broad nontrivial coverage with low singleton residual.",
        ),
        "balanced" => (
            vec!["balanced", "detail_seeking"],
            vec![],
            "Candidate exposes nontrivial components with manageable residual mass.",
        ),
        "exploratory" => (
            vec!["detail_seeking"],
            vec!["report_ready"],
            "Candidate may expose detailed archetypes, but needs caution before final naming.",
        ),
        _ if tier == "outlier_frontier" || singleton_fraction >= 0.35 => (
            vec!["outlier_mining"],
            vec!["report_ready"],
            "Candidate is singleton-rich; use for anomaly or frontier discovery, not final cohort reporting.",
        ),
        _ => (
            vec!["detail_seeking"],
            vec!["report_ready"],
            "Candidate has weak cohort structure; treat as a diagnostic lens only.",
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlateauCandidate {
    pub candidate_kind: &'static str,
    pub threshold: f64,
    pub plateau_start: f64,
    pub plateau_end: f64,
    pub persistence_width: f64,
    pub component_count: usize,
    pub interpretability_tier: &'static str,
    pub rank_within_policy: f64,
    pub component_mass_profile: ComponentMassProfile,
    pub mass_shape: MassShape,
    pub best_for: Vec<&'static str>,
    pub avoid_for: Vec<&'static str>,
    pub why: &'static str,
}

/// Rank plateau midpoints under `policy`; the top `max_candidates`.
pub fn plateau_threshold_candidates(
    graph: &ThresholdGraph,
    plateaus: &[Plateau],
    policy: Policy,
    max_candidates: usize,
    active_range: Option<f64>,
) -> Vec<PlateauCandidate> {
    let active_range =
        active_range.unwrap_or_else(|| active_range_from_plateaus(plateaus, graph.n_nodes()));
    let mut candidates: Vec<PlateauCandidate> = plateaus
        .iter()
        .map(|plateau| {
            let midpoint = plateau.midpoint();
            let width = plateau.length();
            let profile = component_mass_profile(graph, midpoint, 5, 0.01);
            let metrics = mass_shape_metrics(&profile);
            let tier = plateau_tier(&profile, &metrics);
            let rank =
                rank_plateau_candidate(width, &profile, &metrics, policy, Some(active_range));
            let (best_for, avoid_for, why) =
                candidate_use_guidance(tier, profile.singleton_fraction);
            PlateauCandidate {
                candidate_kind: "stable_plateau",
                threshold: midpoint,
                plateau_start: plateau.start_threshold,
                plateau_end: plateau.end_threshold,
                persistence_width: round_to(width, 4),
                component_count: plateau.component_count,
                interpretability_tier: tier,
                rank_within_policy: rank,
                component_mass_profile: profile,
                mass_shape: metrics,
                best_for,
                avoid_for,
                why,
            }
        })
        .collect();
    candidates.sort_by(|a, b| b.rank_within_policy.total_cmp(&a.rank_within_policy));
    candidates.truncate(max_candidates);
    candidates
}

// ---------------------------------------------------- transition candidates

fn transition_candidate_tier(profile: &ComponentMassProfile, metrics: &MassShape) -> &'static str {
    let singletons = profile.singleton_fraction;
    let multi = metrics.multi_component_coverage;
    let balance = metrics.top_two_balance;
    let count = metrics.nontrivial_component_count;
    if singletons >= 0.35 {
        "outlier_frontier"
    } else if count >= 2 && multi >= 0.2 && balance >= 0.35 {
        "exploratory"
    } else if count >= 2 && multi >= 0.05 {
        "detail_lens"
    } else {
        "weak_candidate"
    }
}

fn rank_transition_candidate(
    profile: &ComponentMassProfile,
    metrics: &MassShape,
    event: &Breakpoint,
    policy: Policy,
) -> f64 {
    let multi = metrics.multi_component_coverage;
    let balance = metrics.top_two_balance;
    let entropy = metrics.component_size_entropy;
    let singletons = profile.singleton_fraction;
    let absorbed = event.absorbed_mass_fraction;
    let (score, penalty) = match policy {
        Policy::OutlierMining => (
            0.3 * entropy + 0.25 * absorbed + 0.25 * multi + 0.2 * singletons,
            0.05 * profile.small_component_mass_fraction,
        ),
        Policy::DetailSeeking => (
            0.3 * multi + 0.25 * balance + 0.25 * absorbed + 0.2 * entropy,
            0.12 * singletons,
        ),
        _ => (
            0.35 * multi + 0.25 * balance + 0.25 * absorbed + 0.15 * entropy,
            0.35 * singletons,
        ),
    };
    round_to((score - penalty).max(0.0), 4)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TransitionCandidate {
    pub candidate_kind: &'static str,
    pub threshold: f64,
    pub side: &'static str,
    pub event: &'static str,
    pub component_count_before: usize,
    pub component_count_after: usize,
    pub top_component_sizes: Vec<usize>,
    pub singleton_fraction: f64,
    pub absorbed_mass_fraction: f64,
    pub resulting_component_fraction: f64,
    pub interpretability_tier: &'static str,
    pub sort_key: f64,
    pub mass_shape: MassShape,
    pub best_for: Vec<&'static str>,
    pub avoid_for: Vec<&'static str>,
    pub why: &'static str,
}

fn with_policy(mut best_for: Vec<&'static str>, policy: &'static str) -> Vec<&'static str> {
    best_for.push(policy);
    best_for.sort_unstable();
    best_for.dedup();
    best_for
}

/// Candidates just above a structural transition, ranked under `policy`.
pub fn transition_adjacent_candidates(
    graph: &ThresholdGraph,
    thresholds: &[f64],
    component_counts: &[usize],
    policy: Policy,
    max_candidates: usize,
) -> Vec<TransitionCandidate> {
    let events = structural_breakpoints(
        graph,
        thresholds,
        component_counts,
        (max_candidates * 3).max(6),
        32,
    );
    let mut candidates = Vec::new();
    for event in &events {
        if event.event == "small_component_absorption" && policy != Policy::OutlierMining {
            continue;
        }
        let profile = component_mass_profile(graph, event.threshold, 5, 0.01);
        let metrics = mass_shape_metrics(&profile);
        let tier = transition_candidate_tier(&profile, &metrics);
        let rank = rank_transition_candidate(&profile, &metrics, event, policy);
        let (mut best_for, avoid_for, why) =
            candidate_use_guidance(tier, profile.singleton_fraction);
        if policy == Policy::DetailSeeking && (tier == "detail_lens" || tier == "exploratory") {
            best_for = with_policy(best_for, "detail_seeking");
        }
        if policy == Policy::OutlierMining {
            best_for = with_policy(best_for, "outlier_mining");
        }
        candidates.push(TransitionCandidate {
            candidate_kind: "transition_adjacent",
            threshold: event.threshold,
            side: "stricter_before_merge",
            event: event.event,
            component_count_before: event.component_count_before,
            component_count_after: event.component_count_after,
            top_component_sizes: profile.top_component_sizes.clone(),
            singleton_fraction: profile.singleton_fraction,
            absorbed_mass_fraction: event.absorbed_mass_fraction,
            resulting_component_fraction: event.resulting_component_fraction,
            interpretability_tier: tier,
            sort_key: rank,
            mass_shape: metrics,
            best_for,
            avoid_for,
            why,
        });
    }
    candidates.sort_by(|a, b| b.sort_key.total_cmp(&a.sort_key));
    candidates.truncate(max_candidates);
    candidates
}

// ------------------------------------------------------------ merged options

/// Either candidate kind, serialized as its own shape.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Candidate {
    Plateau(PlateauCandidate),
    Transition(TransitionCandidate),
}

impl Candidate {
    pub fn threshold(&self) -> f64 {
        match self {
            Self::Plateau(c) => c.threshold,
            Self::Transition(c) => c.threshold,
        }
    }

    pub fn interpretability_tier(&self) -> &'static str {
        match self {
            Self::Plateau(c) => c.interpretability_tier,
            Self::Transition(c) => c.interpretability_tier,
        }
    }

    pub fn singleton_fraction(&self) -> f64 {
        match self {
            Self::Plateau(c) => c.component_mass_profile.singleton_fraction,
            Self::Transition(c) => c.singleton_fraction,
        }
    }

    fn largest_fraction(&self, n_nodes: usize) -> f64 {
        match self {
            Self::Plateau(c) => c.component_mass_profile.largest_component_fraction,
            Self::Transition(c) => c
                .top_component_sizes
                .first()
                .map_or(0.0, |&s| s as f64 / n_nodes.max(1) as f64),
        }
    }

    fn component_count(&self) -> usize {
        match self {
            Self::Plateau(c) => c.component_count,
            Self::Transition(c) => c.component_count_after,
        }
    }

    fn eligible_for(&self, policy: Policy) -> bool {
        policy == Policy::OutlierMining || self.singleton_fraction() < SINGLETON_FRONTIER_FRACTION
    }
}

/// The first candidate whose tier reads as a cohort (`report_ready` or
/// `balanced`).
pub fn first_report_ready_candidate(candidates: &[Candidate]) -> Option<&Candidate> {
    candidates
        .iter()
        .find(|c| COHORT_READY_TIERS.contains(&c.interpretability_tier()))
}

fn policy_caps(policy: Policy, max_candidates: usize) -> (usize, usize) {
    match policy {
        Policy::ReportReady => (max_candidates.min(3), max_candidates.min(1)),
        Policy::DetailSeeking | Policy::OutlierMining => (1, max_candidates),
        Policy::Balanced => (max_candidates.min(2), max_candidates.min(4)),
    }
}

/// Drop candidates that are threshold-wise or topologically redundant with
/// an earlier (higher-ranked) one.
pub fn deduplicate_candidates(candidates: Vec<Candidate>, n_nodes: usize) -> Vec<Candidate> {
    let mut unique: Vec<Candidate> = Vec::new();
    'outer: for cand in candidates {
        let (t, f, cc, tier) = (
            cand.threshold(),
            cand.largest_fraction(n_nodes),
            cand.component_count(),
            cand.interpretability_tier(),
        );
        for existing in &unique {
            let (te, fe, cce, tiere) = (
                existing.threshold(),
                existing.largest_fraction(n_nodes),
                existing.component_count(),
                existing.interpretability_tier(),
            );
            // Rule 1: practically the same threshold.
            if (t - te).abs() < 0.03 {
                continue 'outer;
            }
            // Rule 2: both giant-dominated.
            if f >= 0.90 && fe >= 0.90 {
                continue 'outer;
            }
            // Rule 3: same tier, similar topology, similar component count.
            if tier == tiere && (f - fe).abs() < 0.05 {
                let diff = cc.abs_diff(cce);
                if diff <= 2 || diff as f64 / (cc.max(cce).max(1) as f64) < 0.15 {
                    continue 'outer;
                }
            }
        }
        unique.push(cand);
    }
    unique
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThresholdOptions {
    pub policy: Policy,
    pub selection_strategy: &'static str,
    pub auto_readout: String,
    pub recommended_action: &'static str,
    pub stable_plateau_candidates: Vec<PlateauCandidate>,
    pub transition_adjacent_candidates: Vec<TransitionCandidate>,
    pub stable_plateau_candidates_omitted: usize,
    pub transition_adjacent_candidates_omitted: usize,
    pub candidates: Vec<Candidate>,
}

/// Interpretation-threshold candidates under `policy`: ranked plateau
/// midpoints and transition-adjacent cuts, deduplicated and merged by the
/// policy's selection strategy. `plateaus` are the (longest-first) plateaus
/// to consider; `thresholds` / `component_counts` the H0 curve.
pub fn agent_threshold_options(
    graph: &ThresholdGraph,
    plateaus: &[Plateau],
    thresholds: &[f64],
    component_counts: &[usize],
    policy: Policy,
    max_candidates: usize,
) -> ThresholdOptions {
    let n_nodes = graph.n_nodes();
    let (stable_cap, transition_cap) = policy_caps(policy, max_candidates);
    let active_range = active_threshold_range(thresholds, component_counts);
    let stable =
        plateau_threshold_candidates(graph, plateaus, policy, stable_cap, Some(active_range));
    let transition =
        transition_adjacent_candidates(graph, thresholds, component_counts, policy, transition_cap);

    let stable = deduplicate_candidates(
        stable.into_iter().map(Candidate::Plateau).collect(),
        n_nodes,
    );
    let transition = deduplicate_candidates(
        transition.into_iter().map(Candidate::Transition).collect(),
        n_nodes,
    );
    let selectable_stable: Vec<Candidate> = stable
        .iter()
        .filter(|c| c.eligible_for(policy))
        .cloned()
        .collect();
    let selectable_transition: Vec<Candidate> = transition
        .iter()
        .filter(|c| c.eligible_for(policy))
        .cloned()
        .collect();
    let not_dusty = |c: &&Candidate| {
        !matches!(
            c.interpretability_tier(),
            "giant_component_with_dust" | "weak_candidate"
        )
    };

    let (candidates, strategy): (Vec<Candidate>, &'static str) = match policy {
        Policy::DetailSeeking => (
            selectable_transition
                .iter()
                .cloned()
                .chain(selectable_stable.iter().filter(not_dusty).cloned())
                .collect(),
            "transition_adjacent",
        ),
        Policy::OutlierMining => (
            transition
                .iter()
                .cloned()
                .chain(stable.iter().take(1).cloned())
                .collect(),
            "transition_adjacent",
        ),
        Policy::ReportReady => (
            selectable_stable
                .iter()
                .filter(|c| COHORT_READY_TIERS.contains(&c.interpretability_tier()))
                .cloned()
                .collect(),
            "stable_plateau",
        ),
        Policy::Balanced => {
            let useful: Vec<Candidate> = selectable_stable
                .iter()
                .filter(not_dusty)
                .cloned()
                .collect();
            if !useful.is_empty() {
                (useful, "stable_plateau")
            } else if !selectable_transition.is_empty() {
                (
                    selectable_transition.iter().take(2).cloned().collect(),
                    "transition_adjacent",
                )
            } else {
                (
                    selectable_stable.iter().take(1).cloned().collect(),
                    "transition_adjacent",
                )
            }
        }
    };
    let mut candidates = deduplicate_candidates(candidates, n_nodes);
    candidates.truncate(max_candidates);

    let (auto_readout, recommended_action) = match stable.first() {
        None => (
            "No stable threshold candidates were available.".to_owned(),
            "Run a sweep before interpreting threshold options.",
        ),
        Some(auto) if auto.interpretability_tier() == "giant_component_with_dust" => (
            format!(
                "Auto/stable threshold is giant-dominated: {:.2}% of rows sit in the largest component.",
                auto.largest_fraction(n_nodes) * 100.0
            ),
            "Use alternatives as research lenses; do not name final cohorts from a dusty giant component alone.",
        ),
        Some(auto) if COHORT_READY_TIERS.contains(&auto.interpretability_tier()) => (
            "Stable threshold candidates include nontrivial multi-component structure.".to_owned(),
            "Use the report-ready or balanced candidate for cohort naming, then validate features.",
        ),
        Some(_) => (
            "Stable threshold candidates are exploratory rather than report-ready.".to_owned(),
            "Use candidates for detailed or outlier analysis, not final cohort naming.",
        ),
    };

    let unwrap_plateau = |c: Candidate| match c {
        Candidate::Plateau(p) => p,
        Candidate::Transition(_) => unreachable!("stable list holds plateau candidates"),
    };
    let unwrap_transition = |c: Candidate| match c {
        Candidate::Transition(t) => t,
        Candidate::Plateau(_) => unreachable!("transition list holds transition candidates"),
    };
    let stable_plateau_candidates: Vec<PlateauCandidate> =
        stable.into_iter().map(unwrap_plateau).collect();
    let transition_adjacent_candidates: Vec<TransitionCandidate> =
        transition.into_iter().map(unwrap_transition).collect();
    ThresholdOptions {
        policy,
        selection_strategy: strategy,
        auto_readout,
        recommended_action,
        stable_plateau_candidates_omitted: plateaus
            .len()
            .saturating_sub(stable_plateau_candidates.len()),
        transition_adjacent_candidates_omitted: 0,
        stable_plateau_candidates,
        transition_adjacent_candidates,
        candidates,
    }
}

// ------------------------------------------------------------- breakpoints

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Breakpoint {
    pub threshold: f64,
    pub event: &'static str,
    pub component_count_before: usize,
    pub component_count_after: usize,
    pub before_top_component_sizes: Vec<usize>,
    pub after_top_component_sizes: Vec<usize>,
    pub absorbed_mass_fraction: f64,
    pub resulting_component_fraction: f64,
    pub affected_mass_fraction: f64,
    pub interpretation_hint: &'static str,
}

struct Transition {
    index: usize,
    count_delta: usize,
    threshold_gap: f64,
}

/// Threshold cuts where components merge, ranked by the mass they move.
/// `thresholds` descend with the H0 curve; a breakpoint is reported at the
/// lower (after) threshold of each selected transition.
pub fn structural_breakpoints(
    graph: &ThresholdGraph,
    thresholds: &[f64],
    component_counts: &[usize],
    max_breakpoints: usize,
    max_candidates: usize,
) -> Vec<Breakpoint> {
    let n_nodes = graph.n_nodes();
    if n_nodes == 0 {
        return Vec::new();
    }
    let len = thresholds.len().min(component_counts.len());
    let transitions: Vec<Transition> = (1..len)
        .filter(|&i| component_counts[i - 1] != component_counts[i])
        .map(|i| Transition {
            index: i,
            count_delta: component_counts[i - 1].abs_diff(component_counts[i]),
            threshold_gap: (thresholds[i - 1] - thresholds[i]).abs(),
        })
        .collect();

    let mut selected: Vec<usize> = if transitions.len() > max_candidates {
        let half = max_candidates / 2;
        let mut by_count: Vec<&Transition> = transitions.iter().collect();
        by_count.sort_by(|a, b| {
            (b.count_delta, b.threshold_gap)
                .partial_cmp(&(a.count_delta, a.threshold_gap))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut by_gap: Vec<&Transition> = transitions.iter().collect();
        by_gap.sort_by(|a, b| {
            (b.threshold_gap, b.count_delta)
                .partial_cmp(&(a.threshold_gap, a.count_delta))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut picked: Vec<usize> = by_count
            .iter()
            .take(half)
            .chain(by_gap.iter().take(half))
            .map(|t| t.index)
            .collect();
        picked.sort_unstable();
        picked.dedup();
        picked
    } else {
        transitions.iter().map(|t| t.index).collect()
    };
    selected.sort_unstable();

    let mut states: HashMap<u64, ComponentState> = HashMap::new();
    let mut state = |t: f64| -> ComponentState {
        states
            .entry(t.to_bits())
            .or_insert_with(|| graph.state_at(t))
            .clone()
    };

    let mut candidates = Vec::new();
    for i in selected {
        let before_threshold = thresholds[i - 1];
        let after_threshold = thresholds[i];
        let before_count = component_counts[i - 1];
        let after_count = component_counts[i];
        let before = state(before_threshold);
        let after = state(after_threshold);
        let before_size_by_label = before.size_by_label();

        // Prior (before) labels grouped by after label, in ascending label order.
        let n_after = after.sizes.len();
        let mut priors: Vec<Vec<usize>> = vec![Vec::new(); n_after];
        for node in 0..n_nodes {
            priors[after.labels[node]].push(before.labels[node]);
        }
        let mut transition_mass = 0usize;
        let mut absorbed_mass = 0usize;
        let mut merged_prior_sizes: Vec<usize> = Vec::new();
        for prior in priors.iter_mut() {
            prior.sort_unstable();
            prior.dedup();
            if prior.len() <= 1 {
                continue;
            }
            let mut sizes: Vec<usize> = prior.iter().map(|&l| before_size_by_label[l]).collect();
            sizes.sort_unstable_by(|a, b| b.cmp(a));
            let mass: usize = sizes.iter().sum();
            let smaller: usize = sizes[1..].iter().sum();
            if smaller > absorbed_mass {
                transition_mass = mass;
                absorbed_mass = smaller;
                merged_prior_sizes = sizes;
            }
        }
        if absorbed_mass == 0 {
            continue;
        }
        let absorbed_fraction = absorbed_mass as f64 / n_nodes as f64;
        let resulting_fraction = transition_mass as f64 / n_nodes as f64;
        let second_prior_fraction = merged_prior_sizes
            .get(1)
            .map_or(0.0, |&s| s as f64 / n_nodes as f64);
        let (event, hint) = if absorbed_fraction >= 0.05 && second_prior_fraction >= 0.05 {
            (
                "large_component_transition",
                "Major structural transition; lowering the threshold merges large components, while raising it splits them.",
            )
        } else if after_count < before_count && second_prior_fraction < 0.01 {
            (
                "small_component_absorption",
                "Mostly small components or singletons joining larger structure.",
            )
        } else {
            (
                "mixed_component_transition",
                "Component count changes with moderate mass redistribution.",
            )
        };
        candidates.push(Breakpoint {
            threshold: after_threshold,
            event,
            component_count_before: before_count,
            component_count_after: after_count,
            before_top_component_sizes: before.sizes.iter().copied().take(5).collect(),
            after_top_component_sizes: after.sizes.iter().copied().take(5).collect(),
            absorbed_mass_fraction: round_to(absorbed_fraction, 4),
            resulting_component_fraction: round_to(resulting_fraction, 4),
            affected_mass_fraction: round_to(absorbed_fraction, 4),
            interpretation_hint: hint,
        });
    }

    let priority = |event: &str| match event {
        "large_component_transition" => 0,
        "mixed_component_transition" => 1,
        "small_component_absorption" => 2,
        _ => 9,
    };
    candidates.sort_by(|a, b| {
        priority(a.event).cmp(&priority(b.event)).then_with(|| {
            b.absorbed_mass_fraction
                .total_cmp(&a.absorbed_mass_fraction)
        })
    });
    let non_absorption: Vec<Breakpoint> = candidates
        .iter()
        .filter(|c| c.event != "small_component_absorption")
        .cloned()
        .collect();
    let mut out = if non_absorption.is_empty() {
        candidates
    } else {
        non_absorption
    };
    out.truncate(max_breakpoints);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn three_blobs() -> ThresholdGraph {
        // Three cliques of 4 (weights 0.9), bridged by 0.4 and 0.2 edges, plus
        // two singletons.
        let mut edges = Vec::new();
        for blob in 0..3 {
            let base = blob * 4;
            for a in 0..4 {
                for b in a + 1..4 {
                    edges.push((base + a, base + b, 0.9));
                }
            }
        }
        edges.push((3, 4, 0.4));
        edges.push((7, 8, 0.2));
        ThresholdGraph::from_edges(14, &edges)
    }

    #[test]
    fn mass_profile_and_shape_are_rounded_like_python() {
        let g = three_blobs();
        let p = component_mass_profile(&g, 0.5, 8, 0.01);
        assert_eq!(p.component_count, 5);
        assert_eq!(p.top_component_sizes, vec![4, 4, 4, 1, 1]);
        assert_eq!(p.largest_component_fraction, round_to(4.0 / 14.0, 4));
        assert_eq!(p.singleton_count, 2);
        assert_eq!(p.small_component_count, 2);
        assert_eq!(p.small_component_mass_fraction, 0.1429);
        let m = mass_shape_metrics(&p);
        assert_eq!(m.useful_component_size_floor, 3);
        assert_eq!(m.nontrivial_component_count, 3);
        assert_eq!(m.nontrivial_mass_fraction, 0.8571);
        assert_eq!(m.multi_component_coverage, 0.5714);
        assert_eq!(m.top_two_balance, 1.0);
        assert_eq!(mass_profile_hint(&p), "mixed component-size structure");
        assert_eq!(m.component_size_entropy, 0.9014);
        assert_eq!(
            significant_component_sizes(&[121, 111, 106, 1, 1]),
            (vec![121, 111, 106], "modes_above_dust_cliff")
        );
        assert_eq!(
            significant_component_sizes(&[50, 50, 50]),
            (vec![50, 50, 50], "all_modes")
        );
        assert_eq!(significant_component_sizes(&[1, 1]), (vec![], "no_modes"));
        assert_eq!(significant_component_sizes(&[]), (vec![], "no_components"));
        assert_eq!(useful_component_size_floor(344), 3);
        assert_eq!(useful_component_size_floor(100_000), 316);
    }

    #[test]
    fn breakpoints_find_the_bridges() {
        let g = three_blobs();
        let thresholds = [0.95, 0.5, 0.3, 0.1];
        let counts: Vec<usize> = thresholds
            .iter()
            .map(|&t| g.component_count_at(t))
            .collect();
        assert_eq!(counts, vec![14, 5, 4, 3]);
        let bps = structural_breakpoints(&g, &thresholds, &counts, 5, 16);
        // Checked against the Python structural_breakpoints on the same graph:
        // the two bridge merges (4 nodes absorbed) rank above the clique
        // formation (3 nodes absorbed), ties keeping curve order.
        assert_eq!(bps.len(), 3);
        let rows: Vec<(f64, &str, f64)> = bps
            .iter()
            .map(|b| (b.threshold, b.event, b.absorbed_mass_fraction))
            .collect();
        assert_eq!(
            rows,
            vec![
                (0.3, "large_component_transition", 0.2857),
                (0.1, "large_component_transition", 0.2857),
                (0.5, "large_component_transition", 0.2143),
            ]
        );
        assert_eq!(bps[0].before_top_component_sizes, vec![4, 4, 4, 1, 1]);
        assert_eq!(bps[0].after_top_component_sizes, vec![8, 4, 1, 1]);
        assert_eq!(bps[2].before_top_component_sizes, vec![1, 1, 1, 1, 1]);
        assert_eq!(bps[2].resulting_component_fraction, round_to(4.0 / 14.0, 4));
    }

    #[test]
    fn options_select_by_policy_and_validate_the_policy_name() {
        let g = three_blobs();
        let thresholds = [0.95, 0.5, 0.3, 0.1];
        let counts: Vec<usize> = thresholds
            .iter()
            .map(|&t| g.component_count_at(t))
            .collect();
        let plateaus = vec![
            Plateau {
                start_threshold: 0.9,
                end_threshold: 0.4,
                component_count: 5,
            },
            Plateau {
                start_threshold: 0.4,
                end_threshold: 0.2,
                component_count: 4,
            },
        ];
        for policy in Policy::ALL {
            let policy = Policy::parse(policy).unwrap();
            let opts = agent_threshold_options(&g, &plateaus, &thresholds, &counts, policy, 5);
            assert_eq!(opts.policy, policy);
            assert!(opts.candidates.len() <= 5);
            let json = serde_json::to_value(&opts).unwrap();
            assert!(json["candidates"].is_array());
        }
        let opts =
            agent_threshold_options(&g, &plateaus, &thresholds, &counts, Policy::Balanced, 5);
        assert_eq!(opts.selection_strategy, "stable_plateau");
        assert_eq!(opts.stable_plateau_candidates_omitted, 0);
        assert!(
            first_report_ready_candidate(&opts.candidates).is_none() || !opts.candidates.is_empty()
        );
        assert_eq!(Policy::parse("nope").unwrap_err().code(), "UNKNOWN_POLICY");
    }
}
