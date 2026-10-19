//! `diagnose_cosmic_graph`, `get_threshold_stability_curve`,
//! `get_topological_skeleton` (`pulsar/mcp/tools/diagnostics.py`).

use pulsar_analysis::pyround::round_to;
use pulsar_analysis::thresholds::{
    component_mass_profile, mass_profile_hint, significant_component_sizes,
    threshold_morphology_profile, useful_component_size_floor, Breakpoint, Candidate,
};
use pulsar_analysis::{
    agent_threshold_options, skeleton_payload, structural_breakpoints, GraphAnalysis, GraphMetrics,
    Policy, ThresholdGraph,
};
use pulsar_core::{find_stable_thresholds_from_edges, Plateau, StabilityResult, DEFAULT_NUM_BINS};
use rmcp::model::CallToolResult;
use serde_json::{json, Map, Value};

use super::params::{Detail, Format};
use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::markdown;
use crate::schema::{self as s, input_schema};
use crate::state::{RunView, ServerState};

pub const DIAGNOSE: &str = "diagnose_cosmic_graph";
pub const CURVE: &str = "get_threshold_stability_curve";
pub const SKELETON: &str = "get_topological_skeleton";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Surface {
    #[default]
    Constructed,
    Artifact,
}

fn artifact_unavailable(tool: &'static str) -> ToolError {
    ToolError::new(
        tool,
        "ARTIFACT_SURFACE_UNAVAILABLE",
        "surface='artifact' needs create_graph_artifact, which this server does not provide yet.",
        "Use surface='constructed' (the run's persisted cosmic graph).",
    )
}

// ---------------------------------------------------------------- diagnose

#[derive(serde::Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct DiagnoseParams {
    /// 'constructed' (default): the fitted cosmic graph at its construction threshold. 'artifact' is reserved for graph artifacts (not available yet).
    pub surface: Surface,
    /// Artifact handle; only used when surface='artifact'.
    pub artifact_id: String,
    /// 'summary' (default) returns the top 10 component sizes. 'full' additionally lists every component size (O(components) entries).
    pub detail: Detail,
    /// Text view: 'json' (default) or 'markdown'. structured_content is identical.
    pub response_format: Format,
    /// Run to read, from run_topological_sweep. Omit for the latest run. Any completed run works, on any transport.
    pub run_id: String,
}

impl Default for DiagnoseParams {
    fn default() -> Self {
        Self {
            surface: Surface::Constructed,
            artifact_id: String::new(),
            detail: Detail::Summary,
            response_format: Format::Json,
            run_id: String::new(),
        }
    }
}

pub fn diagnose_spec() -> ToolSpec {
    ToolSpec {
        name: DIAGNOSE,
        title: "Diagnose Cosmic Graph",
        description: "Graph-state observables for a fitted cosmic graph. Returns scale, component morphology, weight distribution, sweep support, observed patterns and risk factors (plus MinHash weight accuracy when used). Measurement only, no recommended action. Use after run_topological_sweep; use get_threshold_stability_curve to compare construction thresholds.",
        input_schema: input_schema::<DiagnoseParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::enumeration(&["ok"])),
                ("detail", s::enumeration(&["summary", "full"])),
                ("source", s::enumeration(&["live", "persisted", "artifact"])),
                ("run_id", s::nullable(s::string())),
                (
                    "graph_surface",
                    s::object(
                        &[
                            ("kind", s::string()),
                            ("artifact_id", s::string()),
                            ("construction_threshold", s::number()),
                            ("threshold_role", s::string()),
                            ("preserves_current_fit", s::boolean()),
                        ],
                        &["kind", "construction_threshold", "threshold_role"],
                        true,
                    ),
                ),
                (
                    "scale",
                    s::object(
                        &[
                            ("n_nodes", s::integer()),
                            ("n_edges", s::integer()),
                            ("density", s::number()),
                            ("avg_degree", s::number()),
                            ("nonzero_fraction_full_weighted", s::number()),
                        ],
                        &["n_nodes", "n_edges", "density", "avg_degree"],
                        true,
                    ),
                ),
                (
                    "component_morphology",
                    s::object(
                        &[
                            ("component_count", s::integer()),
                            ("top_component_sizes", s::array(s::integer())),
                            ("top_component_sizes_omitted", s::integer()),
                            ("largest_component_fraction", s::number()),
                            ("second_largest_ratio", s::number()),
                            ("singleton_count", s::integer()),
                            ("singleton_fraction", s::number()),
                            ("nontrivial_component_floor", s::number()),
                            ("nontrivial_component_count", s::integer()),
                            ("nontrivial_mass_fraction", s::number()),
                            ("multi_component_coverage", s::number()),
                            ("tail_component_count", s::integer()),
                            ("tail_mass_fraction", s::number()),
                            ("component_size_entropy", s::number()),
                            ("top_two_balance", s::number()),
                            ("component_sizes", s::array(s::integer())),
                        ],
                        &["component_count", "top_component_sizes", "largest_component_fraction"],
                        true,
                    ),
                ),
                (
                    "weight_distribution",
                    s::object(
                        &[
                            ("p25", s::number()),
                            ("p50", s::number()),
                            ("p95", s::number()),
                            ("sparkline", s::string()),
                            ("basis", s::string()),
                        ],
                        &["p25", "p50", "p95"],
                        true,
                    ),
                ),
                (
                    "sweep_support",
                    s::object(
                        &[
                            ("n_ball_maps", s::integer()),
                            ("grid_adequacy_status", s::string()),
                            ("grid_adequacy_note", s::string()),
                        ],
                        &["n_ball_maps", "grid_adequacy_status"],
                        true,
                    ),
                ),
                ("observed_patterns", s::array(s::string())),
                (
                    "risk_factors",
                    s::array(s::object(&[("code", s::string()), ("severity", s::string())], &["code"], true)),
                ),
                ("minhash_profile", s::open_object()),
            ],
            &[
                "status",
                "detail",
                "source",
                "graph_surface",
                "scale",
                "component_morphology",
                "weight_distribution",
                "sweep_support",
                "observed_patterns",
                "risk_factors",
            ],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

/// `_component_morphology`.
fn component_morphology(m: &GraphMetrics, detail: Detail) -> Value {
    let sizes = &m.component_sizes;
    let n = m.n_nodes.max(1) as f64;
    let (nontrivial, _) = significant_component_sizes(sizes);
    let tail = &sizes[nontrivial.len().min(sizes.len())..];
    let largest = sizes.first().copied().unwrap_or(0);
    let second = sizes.get(1).copied().unwrap_or(0);
    let total = sizes.iter().sum::<usize>().max(1) as f64;
    let entropy: f64 = if sizes.is_empty() {
        0.0
    } else {
        -sizes
            .iter()
            .map(|&s| {
                let p = s as f64 / total;
                p * (p + 1e-12).ln()
            })
            .sum::<f64>()
    };
    let normalized_entropy = if sizes.len() > 1 {
        entropy / (sizes.len() as f64).ln()
    } else {
        0.0
    };
    let balance = if second > 0 {
        1.0 - (largest as f64 - second as f64).abs() / (largest + second).max(1) as f64
    } else {
        0.0
    };
    let mut out = json!({
        "component_count": m.component_count,
        "top_component_sizes": sizes.iter().take(10).collect::<Vec<_>>(),
        "top_component_sizes_omitted": sizes.len().saturating_sub(10),
        "largest_component_fraction": m.giant_fraction,
        "second_largest_ratio": round_to(if largest > 0 { second as f64 / largest as f64 } else { 0.0 }, 4),
        "singleton_count": m.singleton_count,
        "singleton_fraction": m.singleton_fraction,
        "nontrivial_component_floor": useful_component_size_floor(m.n_nodes),
        "nontrivial_component_count": nontrivial.len(),
        "nontrivial_mass_fraction": round_to(nontrivial.iter().sum::<usize>() as f64 / n, 4),
        "multi_component_coverage": round_to(nontrivial.iter().skip(1).sum::<usize>() as f64 / n, 4),
        "tail_component_count": tail.len(),
        "tail_mass_fraction": round_to(tail.iter().sum::<usize>() as f64 / n, 4),
        "component_size_entropy": round_to(normalized_entropy, 4),
        "top_two_balance": round_to(balance, 4),
    });
    if detail == Detail::Full {
        out["component_sizes"] = json!(sizes);
    }
    out
}

fn observed_patterns(m: &GraphMetrics, morph: &Value) -> Vec<&'static str> {
    let mut out = Vec::new();
    if m.n_edges == 0 {
        out.push("empty_graph");
    }
    if morph["nontrivial_component_count"].as_u64().unwrap_or(0) >= 2 {
        out.push("multiple_nontrivial_components");
    }
    if m.giant_fraction >= 0.95 {
        out.push("dominant_component");
    } else {
        out.push("no_dominant_component");
    }
    if m.singleton_fraction >= 0.25 {
        out.push("singleton_residual_present");
    } else {
        out.push("low_singleton_residual");
    }
    if morph["top_two_balance"].as_f64().unwrap_or(0.0) >= 0.7 {
        out.push("high_top_component_balance");
    }
    if m.density > 0.8 {
        out.push("dense_connectivity");
    }
    match m.grid_adequacy_status {
        "sample_count_ok" => out.push("adequate_grid_sample"),
        "under_sampled" => out.push("under_sampled"),
        "thin_grid" => out.push("thin_grid"),
        _ => {}
    }
    out
}

fn risk_factors(m: &GraphMetrics) -> Vec<Value> {
    let mut out = Vec::new();
    if m.n_edges == 0 {
        out.push(json!({"code": "empty_graph", "severity": "error"}));
    }
    if m.density > 0.8 && m.n_edges > 0 {
        out.push(json!({"code": "dense_hairball", "severity": "warning"}));
    }
    if m.singleton_fraction > 0.8 {
        out.push(json!({"code": "singleton_heavy", "severity": "warning"}));
    }
    if m.giant_fraction > 0.95 {
        out.push(json!({"code": "dominant_component", "severity": "warning"}));
    }
    if matches!(m.grid_adequacy_status, "under_sampled" | "thin_grid") {
        out.push(json!({"code": m.grid_adequacy_status, "severity": "warning"}));
    }
    out
}

/// Metrics of a run's constructed graph (one component pass).
fn run_metrics(tool: &'static str, view: &RunView) -> Result<GraphMetrics, ToolError> {
    let graph = view.graph().map_err(|e| ToolError::from_store(tool, e))?;
    let affinity = view
        .affinity()
        .map_err(|e| ToolError::from_store(tool, e))?;
    let analysis = GraphAnalysis::new(&graph).map_err(|e| ToolError::from_analysis(tool, e))?;
    Ok(analysis.metrics(
        view.record.resolved_construction_threshold,
        view.record.n_ball_maps,
        Some(&affinity.edges),
        view.minhash_d(),
    ))
}

pub async fn diagnose(state: &ServerState, p: DiagnoseParams) -> Result<CallToolResult, ToolError> {
    if p.surface == Surface::Artifact {
        return Err(artifact_unavailable(DIAGNOSE));
    }
    let view = state.resolve_run(DIAGNOSE, &p.run_id)?;
    let m = run_metrics(DIAGNOSE, &view)?;
    let morph = component_morphology(&m, p.detail);
    let mut payload = json!({
        "status": "ok",
        "detail": p.detail.as_str(),
        "source": if view.live.is_some() { "live" } else { "persisted" },
        "run_id": view.record.run_id,
        "graph_surface": {
            "kind": "constructed_cosmic_graph",
            "construction_threshold": m.resolved_construction_threshold,
            "threshold_role": "construction",
            "preserves_current_fit": true,
        },
        "scale": {
            "n_nodes": m.n_nodes,
            "n_edges": m.n_edges,
            "density": m.density,
            "avg_degree": m.avg_degree,
            "nonzero_fraction_full_weighted": m.nonzero_fraction,
        },
        "component_morphology": morph,
        "weight_distribution": {
            "p25": m.weight_p25,
            "p50": m.weight_p50,
            "p95": m.weight_p95,
            "sparkline": m.weight_distribution_sparkline,
            "basis": "upper-triangle node-pair distribution; zeros include absent edges",
        },
        "sweep_support": {
            "n_ball_maps": m.n_ball_maps,
            "grid_adequacy_status": m.grid_adequacy_status,
            "grid_adequacy_note": m.grid_adequacy_note,
        },
        "observed_patterns": observed_patterns(&m, &payload_morph(&m, p.detail)),
        "risk_factors": risk_factors(&m),
    });
    if let Some(profile) = &m.minhash_profile {
        payload["minhash_profile"] = json!(profile);
    }
    let text = match p.response_format {
        Format::Markdown => Some(markdown::diagnosis(&payload)),
        Format::Json => None,
    };
    Ok(ok_result(payload, text))
}

fn payload_morph(m: &GraphMetrics, detail: Detail) -> Value {
    component_morphology(m, detail)
}

// ------------------------------------------------------------------- curve

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CurveSource {
    /// Recompute from the run's persisted full affinity (always available).
    #[default]
    Persisted,
    /// Same computation on the in-memory model when it is cached.
    Live,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PolicyParam {
    #[default]
    Balanced,
    ReportReady,
    DetailSeeking,
    OutlierMining,
}

impl From<PolicyParam> for Policy {
    fn from(p: PolicyParam) -> Self {
        match p {
            PolicyParam::Balanced => Policy::Balanced,
            PolicyParam::ReportReady => Policy::ReportReady,
            PolicyParam::DetailSeeking => Policy::DetailSeeking,
            PolicyParam::OutlierMining => Policy::OutlierMining,
        }
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema, Default)]
#[serde(default)]
pub struct CurveParams {
    /// 'summary' (default): bounded decision packet. 'full': adds plateaus, a curve sample and raw O(thresholds) arrays.
    pub detail: Detail,
    /// Run to read, from run_topological_sweep. Omit for the latest run. Any completed run works: the curve is recomputed from the persisted affinity.
    pub run_id: String,
    /// 'persisted' (default) or 'live'; both recompute the curve, 'live' only differs in reading the cached model.
    pub source: CurveSource,
    /// Lens for ranking candidates (not a score): 'balanced' (default), 'report_ready' (few large components), 'detail_seeking' (finer splits), 'outlier_mining' (singleton tails).
    pub threshold_candidate_policy: PolicyParam,
    /// Text view: 'markdown' (default) or 'json'. structured_content is identical.
    pub response_format: Format,
}

pub fn curve_spec() -> ToolSpec {
    let open = s::open_object();
    ToolSpec {
        name: CURVE,
        title: "Threshold Stability Curve",
        description: "Threshold morphology and candidate lenses along the H0 component-stability curve. Compares the fitted construction threshold with H0 plateaus and structural breakpoints. Use to judge or choose a construction threshold before editing the config; use diagnose_cosmic_graph for the graph already built.",
        input_schema: input_schema::<CurveParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::enumeration(&["ok"])),
                ("detail", s::enumeration(&["summary", "full"])),
                ("source", s::enumeration(&["live", "persisted"])),
                ("run_id", s::nullable(s::string())),
                ("resolved_construction_threshold", s::nullable(s::number())),
                ("h0_longest_plateau_threshold", s::nullable(s::number())),
                ("matches_current_threshold", s::boolean()),
                ("current_threshold_morphology", open.clone()),
                ("h0_longest_plateau_morphology", open.clone()),
                ("h0_plateau_readout", s::string()),
                ("threshold_guidance", s::string()),
                ("threshold_candidate_policy", s::string()),
                ("selection_strategy", s::any()),
                ("threshold_candidates", s::array(open.clone())),
                ("threshold_candidates_omitted", s::integer()),
                ("threshold_profiles", s::array(open.clone())),
                ("threshold_profiles_omitted", s::integer()),
                ("unthresholded_component_count", s::nullable(s::integer())),
                ("structural_breakpoints", s::array(open.clone())),
                ("structural_breakpoints_omitted", s::integer()),
                ("structural_breakpoints_filter", s::string()),
                ("structural_breakpoints_guidance", s::string()),
                ("plateaus_returned", s::integer()),
                ("plateaus_omitted", s::integer()),
                ("curve_point_count", s::integer()),
                ("full_detail_available", s::string()),
                ("agent_threshold_options", open.clone()),
                ("plateaus", s::array(open.clone())),
                ("curve_sample", s::array(open)),
                ("curve_sample_omitted", s::integer()),
                ("curve_sample_guidance", s::string()),
                ("thresholds", s::array(s::number())),
                ("component_counts", s::array(s::integer())),
                ("singleton_counts", s::array(s::integer())),
            ],
            &["status", "source"],
            true,
        )),
        read_only: true,
        idempotent: true,
    }
}

fn singleton_counts(n: usize, edges: &[(usize, usize, f64)], thresholds: &[f64]) -> Vec<usize> {
    let mut row_max = vec![0.0_f64; n];
    for &(i, j, w) in edges {
        row_max[i] = row_max[i].max(w);
        row_max[j] = row_max[j].max(w);
    }
    thresholds
        .iter()
        .map(|&t| row_max.iter().filter(|&&v| v <= t).count())
        .collect()
}

fn linspace_indices(len: usize, max_points: usize) -> Vec<usize> {
    if len <= max_points {
        return (0..len).collect();
    }
    let step = (len - 1) as f64 / (max_points - 1) as f64;
    let mut out: Vec<usize> = (0..max_points)
        .map(|i| {
            if i == max_points - 1 {
                len - 1
            } else {
                (i as f64 * step) as usize
            }
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn compact_candidate(candidate: &Candidate) -> Value {
    const KEEP: [&str; 15] = [
        "candidate_kind",
        "threshold",
        "midpoint",
        "start",
        "end",
        "component_count",
        "component_count_before",
        "component_count_after",
        "top_component_sizes",
        "singleton_fraction",
        "interpretability_tier",
        "best_for",
        "avoid_for",
        "why",
        "event",
    ];
    let full = serde_json::to_value(candidate).unwrap_or(json!({}));
    let mut out = Map::new();
    if let Some(obj) = full.as_object() {
        for key in KEEP {
            if let Some(v) = obj.get(key) {
                out.insert(key.to_owned(), v.clone());
            }
        }
        if let Some(side) = obj.get("side") {
            out.insert("side".into(), side.clone());
        }
    }
    Value::Object(out)
}

fn agent_readout(selected: Option<&Value>, breakpoints: &[Breakpoint]) -> String {
    let Some(profile) = selected else {
        return "No threshold mass profile was available.".into();
    };
    let largest_pct = profile["largest_component_fraction"]
        .as_f64()
        .unwrap_or(0.0)
        * 100.0;
    let small_pct = profile["small_component_mass_fraction"]
        .as_f64()
        .unwrap_or(0.0)
        * 100.0;
    let singleton_pct = profile["singleton_fraction"].as_f64().unwrap_or(0.0) * 100.0;
    let large = breakpoints
        .iter()
        .any(|b| b.event == "large_component_transition");
    if singleton_pct >= 75.0 {
        format!(
            "H0 longest plateau is singleton-dominated ({singleton_pct:.2}% singletons); treat it as an outlier/frontier lens, not a construction or cohort threshold."
        )
    } else if singleton_pct >= 35.0 {
        format!(
            "H0 longest plateau is singleton-rich ({singleton_pct:.2}% singletons); use only as an exploratory frontier lens unless the task is anomaly mining."
        )
    } else if largest_pct >= 95.0 {
        if large {
            format!(
                "Auto threshold is stable but giant-component dominated ({largest_pct:.2}% of rows in the largest component); inspect large structural breakpoints before naming cohorts."
            )
        } else {
            format!(
                "Auto threshold is stable but produces one giant component containing {largest_pct:.2}% of rows; only {small_pct:.2}% of rows sit in small components."
            )
        }
    } else if large {
        "Auto threshold exposes nontrivial component structure and large threshold transitions are available for parameter discussion.".into()
    } else {
        "Auto threshold exposes nontrivial component structure without large split markers.".into()
    }
}

fn plateau_row(
    graph: &ThresholdGraph,
    n: usize,
    edges: &[(usize, usize, f64)],
    p: &Plateau,
    with_profile: bool,
) -> Value {
    let midpoint = p.midpoint();
    let singleton_count = singleton_counts(n, edges, &[midpoint])[0];
    let mut row = json!({
        "start": p.start_threshold,
        "end": p.end_threshold,
        "component_count": p.component_count,
        "length": p.length(),
        "midpoint": midpoint,
        "singleton_count": singleton_count,
        "singleton_fraction": round_to(singleton_count as f64 / n.max(1) as f64, 4),
    });
    if with_profile {
        let profile = component_mass_profile(graph, midpoint, 8, 0.01);
        row["interpretation_hint"] = json!(mass_profile_hint(&profile));
        row["component_mass_profile"] = json!(profile);
    }
    row
}

/// `_threshold_curve_payload`.
pub fn curve_payload(
    n: usize,
    affinity: &[(usize, usize, f64)],
    resolved_construction_threshold: f64,
    stability: &StabilityResult,
    detail: Detail,
    policy: Policy,
) -> Value {
    let graph = ThresholdGraph::from_edges(n, affinity);
    let thresholds = &stability.thresholds;
    let counts = &stability.component_counts;
    let optimal = stability.optimal_threshold;
    let matches_current =
        (resolved_construction_threshold - optimal).abs() <= 1e-8 + 1e-5 * optimal.abs();
    let breakpoints = structural_breakpoints(&graph, thresholds, counts, 5, 16);
    let current_morph = threshold_morphology_profile(&graph, resolved_construction_threshold, 10);
    let plateau_morph = threshold_morphology_profile(&graph, optimal, 10);
    let profile_rows = |max_points: Option<usize>| -> Vec<Value> {
        let indices = match max_points {
            Some(m) => linspace_indices(thresholds.len(), m),
            None => (0..thresholds.len()).collect(),
        };
        indices
            .into_iter()
            .map(|i| json!(threshold_morphology_profile(&graph, thresholds[i], 10)))
            .collect()
    };
    let candidate_rows = |candidates: &[Candidate]| -> Vec<Value> {
        candidates
            .iter()
            .map(|c| {
                let mut row = compact_candidate(c);
                row["morphology"] = json!(threshold_morphology_profile(&graph, c.threshold(), 10));
                row
            })
            .collect()
    };
    let non_dust: Vec<&Breakpoint> = breakpoints
        .iter()
        .filter(|b| b.event != "small_component_absorption")
        .collect();

    if detail == Detail::Summary {
        let top: Vec<Plateau> = stability.plateaus.iter().take(4).cloned().collect();
        let selected = top
            .first()
            .map(|p| json!(component_mass_profile(&graph, p.midpoint(), 8, 0.01)));
        let ten: Vec<Plateau> = stability.plateaus.iter().take(10).cloned().collect();
        let options = agent_threshold_options(&graph, &ten, thresholds, counts, policy, 7);
        let candidates = candidate_rows(&options.candidates[..options.candidates.len().min(3)]);
        let profiles = profile_rows(Some(10));
        let summary_bps: Vec<&Breakpoint> = non_dust.iter().copied().take(3).collect();
        return json!({
            "status": "ok",
            "detail": "summary",
            "source": "live",
            "resolved_construction_threshold": resolved_construction_threshold,
            "h0_longest_plateau_threshold": optimal,
            "matches_current_threshold": matches_current,
            "current_threshold_morphology": current_morph,
            "h0_longest_plateau_morphology": plateau_morph,
            "h0_plateau_readout": agent_readout(selected.as_ref(), &breakpoints),
            "threshold_candidate_policy": policy.as_str(),
            "selection_strategy": options.selection_strategy,
            "threshold_candidates": candidates,
            "threshold_candidates_omitted": options.candidates.len().saturating_sub(3),
            "threshold_profiles_omitted": thresholds.len().saturating_sub(profiles.len()),
            "threshold_profiles": profiles,
            "unthresholded_component_count": counts.last().copied(),
            "structural_breakpoints": summary_bps,
            "structural_breakpoints_omitted": breakpoints.len().saturating_sub(summary_bps.len()),
            "structural_breakpoints_filter": "summary hides small_component_absorption dust events; use detail='full' for all breakpoints",
            "plateaus_returned": top.len().min(3),
            "plateaus_omitted": top.len().saturating_sub(3),
            "curve_point_count": thresholds.len(),
            "full_detail_available": "get_threshold_stability_curve(detail='full')",
        });
    }

    let singletons = singleton_counts(n, affinity, thresholds);
    let profiles = profile_rows(None);
    let plateaus: Vec<Value> = stability
        .plateaus
        .iter()
        .take(10)
        .map(|p| plateau_row(&graph, n, affinity, p, true))
        .collect();
    let selected = plateaus
        .first()
        .map(|p| p["component_mass_profile"].clone());
    let sample: Vec<Value> = linspace_indices(thresholds.len(), 25)
        .into_iter()
        .map(|i| json!({"threshold": thresholds[i], "component_count": counts[i], "singleton_count": singletons[i]}))
        .collect();
    let twenty: Vec<Plateau> = stability.plateaus.iter().take(20).cloned().collect();
    let options = agent_threshold_options(&graph, &twenty, thresholds, counts, policy, 12);
    let enriched = candidate_rows(&options.candidates);
    let mut audit = serde_json::to_value(&options).unwrap_or(json!({}));
    if let Some(obj) = audit.as_object_mut() {
        obj.remove("candidates");
    }
    json!({
        "status": "ok",
        "detail": "full",
        "source": "live",
        "resolved_construction_threshold": resolved_construction_threshold,
        "h0_longest_plateau_threshold": optimal,
        "matches_current_threshold": matches_current,
        "current_threshold_morphology": current_morph,
        "h0_longest_plateau_morphology": plateau_morph,
        "threshold_guidance": if matches_current {
            "Current graph construction matches the H0 longest plateau."
        } else {
            "H0 longest plateau differs from the current constructed graph; treat it as a rerun candidate, not a change to the fitted graph."
        },
        "h0_plateau_readout": agent_readout(selected.as_ref(), &breakpoints),
        "threshold_candidate_policy": policy.as_str(),
        "agent_threshold_options": audit,
        "threshold_candidates": enriched,
        "plateaus": plateaus,
        "structural_breakpoints": breakpoints,
        "structural_breakpoints_guidance": "Breakpoints are capped structural transition candidates ranked first by large-component transitions, then by the smaller mass that actually joins or splits.",
        "curve_point_count": thresholds.len(),
        "threshold_profiles": profiles,
        "threshold_profiles_omitted": 0,
        "curve_sample_omitted": thresholds.len().saturating_sub(sample.len()),
        "curve_sample": sample,
        "curve_sample_guidance": "curve_sample is an evenly spaced sketch of the full threshold curve. Use detail='full' only when exact per-threshold arrays are needed.",
        "thresholds": thresholds,
        "component_counts": counts,
        "singleton_counts": singletons,
    })
}

pub async fn curve(state: &ServerState, p: CurveParams) -> Result<CallToolResult, ToolError> {
    let view = state.resolve_run(CURVE, &p.run_id)?;
    let affinity = view
        .affinity()
        .map_err(|e| ToolError::from_store(CURVE, e))?;
    let resolved = view.record.resolved_construction_threshold;
    let policy: Policy = p.threshold_candidate_policy.into();
    let detail = p.detail;
    let run_id = view.record.run_id.clone();
    let live = view.live.is_some();
    let mut payload = tokio::task::spawn_blocking(move || -> Result<Value, ToolError> {
        let stability =
            find_stable_thresholds_from_edges(affinity.n, &affinity.edges, DEFAULT_NUM_BINS)
                .map_err(|e| {
                    ToolError::new(
                        CURVE,
                        "THRESHOLD_CURVE_FAILED",
                        e.to_string(),
                        "Check get_runtime_context for the session state, then retry.",
                    )
                })?;
        Ok(curve_payload(
            affinity.n,
            &affinity.edges,
            resolved,
            &stability,
            detail,
            policy,
        ))
    })
    .await
    .map_err(|e| {
        ToolError::new(
            CURVE,
            "THRESHOLD_CURVE_FAILED",
            e.to_string(),
            "Retry the call.",
        )
    })??;
    payload["source"] = json!(if live && p.source == CurveSource::Live {
        "live"
    } else {
        "persisted"
    });
    payload["run_id"] = json!(run_id);
    let text = match p.response_format {
        Format::Markdown => Some(markdown::threshold_curve(&payload)),
        Format::Json => None,
    };
    Ok(ok_result(payload, text))
}

// ---------------------------------------------------------------- skeleton

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SkeletonDetail {
    #[default]
    Summary,
    Nodes,
    Edges,
    Full,
    FullNodes,
}

impl SkeletonDetail {
    fn as_str(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::Nodes => "nodes",
            Self::Edges => "edges",
            Self::Full => "full",
            Self::FullNodes => "full_nodes",
        }
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct SkeletonParams {
    /// Run to read, from run_topological_sweep. Omit for the latest run. Reads the persisted run, so any stored run works on any transport.
    pub run_id: String,
    /// 'constructed' (default): the run's persisted cosmic graph. 'artifact' is reserved for graph artifacts (not available yet).
    pub surface: Surface,
    /// Artifact handle; only used when surface='artifact'.
    pub artifact_id: String,
    /// 'summary' (default): counts and components. 'nodes': per-node summary. 'edges': capped raw edges. 'full_nodes': capped raw nodes. 'full': nodes, edges and config YAML.
    pub detail: SkeletonDetail,
    /// Cap on edges returned in 'edges'/'full' detail (>= 1).
    pub max_edges: usize,
    /// Cap on nodes returned in 'nodes'/'full_nodes'/'full' detail (>= 1).
    pub max_nodes: usize,
}

impl Default for SkeletonParams {
    fn default() -> Self {
        Self {
            run_id: String::new(),
            surface: Surface::Constructed,
            artifact_id: String::new(),
            detail: SkeletonDetail::Summary,
            max_edges: 100,
            max_nodes: 100,
        }
    }
}

pub fn skeleton_spec() -> ToolSpec {
    ToolSpec {
        name: SKELETON,
        title: "Topological Skeleton",
        description: "Connectivity structure (components, nodes, edges) of a run's persisted graph. Summary-first: raw nodes and edges are capped by max_nodes / max_edges. Reads the stored run, not the live model. Use diagnose_cosmic_graph for risk-oriented measurements instead.",
        input_schema: input_schema::<SkeletonParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("run_id", s::string()),
                ("dataset_id", s::nullable(s::string())),
                ("config_yaml_omitted", s::boolean()),
                ("config_yaml", s::string()),
                ("resolved_construction_threshold", s::nullable(s::number())),
                ("graph_surface", s::string()),
                ("graph", s::open_object()),
                ("available_followup_tools", s::array(s::string())),
            ],
            &["run_id", "config_yaml_omitted", "graph_surface", "graph", "available_followup_tools"],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

pub async fn skeleton(state: &ServerState, p: SkeletonParams) -> Result<CallToolResult, ToolError> {
    if p.max_edges < 1 {
        return Err(ToolError::invalid_argument(
            SKELETON,
            format!("max_edges must be >= 1, got '{}'", p.max_edges),
            "Pass max_edges >= 1.",
        ));
    }
    if p.max_nodes < 1 {
        return Err(ToolError::invalid_argument(
            SKELETON,
            format!("max_nodes must be >= 1, got '{}'", p.max_nodes),
            "Pass max_nodes >= 1.",
        ));
    }
    if p.surface == Surface::Artifact {
        return Err(artifact_unavailable(SKELETON));
    }
    let view = state.resolve_run(SKELETON, &p.run_id)?;
    let graph = view
        .graph()
        .map_err(|e| ToolError::from_store(SKELETON, e))?;
    let summary = GraphAnalysis::new(&graph)
        .map_err(|e| ToolError::from_analysis(SKELETON, e))?
        .summary(view.record.resolved_construction_threshold);
    let record = &view.record;
    let mut payload = json!({
        "run_id": record.run_id,
        "dataset_id": record.dataset_id,
        "config_yaml_omitted": p.detail != SkeletonDetail::Full,
        "resolved_construction_threshold": record.resolved_construction_threshold,
        "graph_surface": format!(
            "persisted cosmic graph @ resolved_construction_threshold ({}); not an interpretation_edge_weight_threshold slice",
            record.resolved_construction_threshold
        ),
        "graph": skeleton_payload(&summary, p.detail.as_str(), p.max_edges, p.max_nodes),
        "available_followup_tools": ["diagnose_cosmic_graph", "get_threshold_stability_curve", "compare_sweeps"],
    });
    if p.detail == SkeletonDetail::Full {
        payload["config_yaml"] = json!(record.config_yaml);
    }
    Ok(ok_result(payload, None))
}
