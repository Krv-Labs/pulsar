//! `create_graph_artifact` (`pulsar/mcp/tools/diagnostics.py`): estimate or
//! build a spectral sparsifier of a run's cosmic graph. A built artifact gets
//! an `artifact_id` that `diagnose_cosmic_graph` and
//! `get_topological_skeleton` read with `surface='artifact'`; the fitted
//! graph never changes.
//!
//! Known divergences from Python:
//! - An estimate works for any stored run (counts come from the persisted
//!   affinity); Python needed the run's model in memory.
//! - The session keeps the last [`MAX_GRAPH_ARTIFACTS`] artifacts.

use std::sync::Arc;
use std::time::Instant;

use pulsar_analysis::pyround::round_to;
use pulsar_store::RunId;
use rmcp::model::CallToolResult;
use serde_json::{json, Value};

use super::diagnostics::artifact_staleness;
use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::schema::{self as s, input_schema};
use crate::state::{GraphArtifact, ServerState, MAX_GRAPH_ARTIFACTS};

pub const NAME: &str = "create_graph_artifact";

/// Rough throughput of the sparsifier inner loop (one preconditioned-CG
/// solve per sketch row over the edge set); an order-of-magnitude figure.
const SPARSIFY_OPS_PER_S: f64 = 4.5e6;

fn spectral_sparsifier() -> String {
    "spectral_sparsifier".into()
}

fn one() -> f64 {
    1.0
}

fn forty_two() -> u64 {
    42
}

fn pcg_tol() -> f64 {
    1e-6
}

fn thousand() -> usize {
    1000
}

fn yes() -> bool {
    true
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct Params {
    /// Artifact type. 'spectral_sparsifier' builds an edge-sparsified approximation of the weighted graph (the only supported kind).
    #[serde(default = "spectral_sparsifier")]
    pub kind: String,
    /// Run to use, from run_topological_sweep. Omit for the latest run. Any stored run works with estimate_only=True (a pure read); building publishes into the session, so it must be the latest run (else SESSION_REVISION_CONFLICT).
    #[serde(default)]
    pub run_id: String,
    /// Sparsifier accuracy in (0, inf). Smaller is more faithful and costs roughly 1/epsilon^2 more time and edges. Default 1.0.
    #[serde(default = "one")]
    pub epsilon: f64,
    /// Random seed for the sparsifier sketch. Default 42.
    #[serde(default = "forty_two")]
    pub seed: u64,
    /// Johnson-Lindenstrauss sketch rows. Omit to derive from epsilon; set a small value only for quick tests.
    #[serde(default)]
    pub sketch_dim: Option<usize>,
    /// Number of edge samples drawn. Omit to derive from epsilon and node count.
    #[serde(default)]
    pub sample_count: Option<usize>,
    /// Convergence tolerance of the preconditioned CG solves.
    #[serde(default = "pcg_tol")]
    pub pcg_tol: f64,
    /// Maximum CG iterations per solve.
    #[serde(default = "thousand")]
    pub max_iter: usize,
    /// True (default): return only a cost estimate (runtime, memory, edges). False: build and store the artifact (seconds to minutes).
    #[serde(default = "yes")]
    pub estimate_only: bool,
}

pub fn spec() -> ToolSpec {
    let open = s::open_object();
    ToolSpec {
        name: NAME,
        title: "Graph Artifact",
        description: "Estimate or build a spectral-sparsifier graph artifact (estimates for any stored run; builds for the latest). Expensive by design: estimate_only defaults true. A built artifact gets an artifact_id usable with diagnose_cosmic_graph and get_topological_skeleton (surface='artifact'). It never changes the fitted graph.",
        input_schema: input_schema::<Params>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::enumeration(&["ok"])),
                ("estimate_only", s::boolean()),
                ("kind", s::string()),
                ("run_id", s::string()),
                ("input", s::object(&[("n_nodes", s::integer()), ("edge_count", s::integer())], &[], true)),
                (
                    "estimated_cost",
                    s::object(
                        &[
                            ("runtime_s", s::number()),
                            ("runtime_basis", s::string()),
                            ("peak_mem_mb", s::number()),
                            ("edge_count", s::integer()),
                            ("sketch_dim", s::integer()),
                            ("sample_count", s::integer()),
                        ],
                        &[],
                        true,
                    ),
                ),
                (
                    "params",
                    s::object(
                        &[
                            ("epsilon", s::number()),
                            ("seed", s::integer()),
                            ("sketch_dim", s::nullable(s::integer())),
                            ("sample_count", s::nullable(s::integer())),
                            ("pcg_tol", s::number()),
                            ("max_iter", s::integer()),
                        ],
                        &[],
                        true,
                    ),
                ),
                ("build_call", s::object(&[("tool", s::string()), ("args", open)], &[], true)),
                ("artifact_id", s::string()),
                ("runtime_s", s::number()),
                ("input_edge_count", s::integer()),
                ("artifact_edge_count", s::integer()),
                (
                    "staleness",
                    s::object(
                        &[
                            ("stale", s::boolean()),
                            ("artifact_run_id", s::nullable(s::string())),
                            ("latest_run_id", s::nullable(s::string())),
                            ("warning", s::nullable(s::string())),
                        ],
                        &[],
                        true,
                    ),
                ),
            ],
            &["status", "estimate_only", "kind", "run_id", "params"],
            false,
        )),
        read_only: false,
        idempotent: false,
    }
}

/// Python `f"{x:g}"`: six significant digits, trailing zeros dropped.
fn py_g(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let exp = x.abs().log10().floor() as i32;
    if !(-4..6).contains(&exp) {
        let s = format!("{:.5e}", x);
        let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
        let m = m.trim_end_matches('0').trim_end_matches('.');
        let e: i32 = e.parse().unwrap_or(0);
        return format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    let decimals = (5 - exp).max(0) as usize;
    let s = format!("{x:.decimals$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        s
    }
}

/// `_artifact_estimate`: sketch and sample sizes, runtime and memory.
fn estimate(n_nodes: usize, edge_count: usize, epsilon: f64) -> Value {
    let eps = epsilon.max(1e-9);
    let log_n = (n_nodes.max(1) as f64).ln().max(1.0);
    let sketch_dim = ((24.0 * log_n / (eps * eps)).ceil() as usize).max(1);
    let sample_count = ((9.0 * n_nodes as f64 * log_n / (eps * eps)).ceil() as usize).max(1);
    json!({
        "runtime_s": round_to((sketch_dim * edge_count.max(1)) as f64 / SPARSIFY_OPS_PER_S, 2),
        "runtime_basis": format!(
            "rough projection; scales ~sketch_dim x edges (so ~1/epsilon^2). Estimated at epsilon={}.",
            py_g(eps)
        ),
        "peak_mem_mb": round_to((edge_count * 24 + n_nodes * 8 * 3) as f64 / (1024.0 * 1024.0), 2),
        "edge_count": edge_count.min(sample_count),
        "sketch_dim": sketch_dim,
        "sample_count": sample_count,
    })
}

fn conflict() -> ToolError {
    ToolError::new(
        NAME,
        "SESSION_REVISION_CONFLICT",
        "Refusing to publish a graph artifact for a run that is not the session's latest.",
        "Omit run_id or re-run the sweep; dossiers and artifacts can only be published for the session's latest run.",
    )
}

pub async fn call(state: &ServerState, p: Params) -> Result<CallToolResult, ToolError> {
    let view = state.resolve_run(NAME, &p.run_id)?;
    let run_id: RunId = view.run_id().clone();
    if p.kind != "spectral_sparsifier" {
        return Err(ToolError::new(
            NAME,
            "GRAPH_ARTIFACT_KIND_UNSUPPORTED",
            format!(
                "Unsupported graph artifact kind: {}",
                pulsar_analysis::plan::py_repr(&p.kind)
            ),
            "Use kind='spectral_sparsifier'. structural_backbone is planned but not callable.",
        ));
    }
    if !(p.epsilon.is_finite() && p.epsilon > 0.0) {
        return Err(ToolError::invalid_argument(
            NAME,
            "epsilon must be finite and positive.",
            "Pass a finite epsilon > 0 (default 1.0).",
        ));
    }
    let affinity = view
        .affinity()
        .map_err(|e| ToolError::from_store(NAME, e))?;
    let (n_nodes, edge_count) = (affinity.n, affinity.edges.len());
    let params = json!({
        "epsilon": p.epsilon,
        "seed": p.seed,
        "sketch_dim": p.sketch_dim,
        "sample_count": p.sample_count,
        "pcg_tol": p.pcg_tol,
        "max_iter": p.max_iter,
    });
    if p.estimate_only {
        return Ok(ok_result(
            json!({
                "status": "ok",
                "estimate_only": true,
                "kind": p.kind,
                "run_id": run_id,
                "input": {"n_nodes": n_nodes, "edge_count": edge_count},
                "estimated_cost": estimate(n_nodes, edge_count, p.epsilon),
                "params": params,
                "build_call": {
                    "tool": NAME,
                    "args": {"kind": p.kind, "run_id": run_id, "epsilon": p.epsilon, "seed": p.seed, "estimate_only": false},
                },
            }),
            None,
        ));
    }

    if state.session().latest_run_id.as_ref() != Some(&run_id) {
        return Err(conflict());
    }
    let live = view.live.clone().ok_or_else(|| {
        ToolError::new(
            NAME,
            "MODEL_NOT_LOADED",
            format!("The fitted model of run {run_id} is no longer in memory."),
            "Re-run run_topological_sweep (the same config reuses cached projections), then retry.",
        )
        .detail("run_id", json!(run_id))
    })?;
    let (eps, seed, sketch_dim, sample_count, tol, max_iter) = (
        p.epsilon,
        p.seed,
        p.sketch_dim,
        p.sample_count,
        p.pcg_tol,
        p.max_iter,
    );
    let (graph, metrics, elapsed) = tokio::task::spawn_blocking(move || {
        let started = Instant::now();
        let (graph, metrics) = pulsar_analysis::artifact::spectral_sparsifier_graph(
            &live.fitted.base_cosmic,
            eps,
            seed,
            sketch_dim,
            sample_count,
            tol,
            max_iter,
        )
        .map_err(|e| e.to_string())?;
        let elapsed = started.elapsed().as_secs_f64();
        Ok::<_, String>((graph, metrics, elapsed))
    })
    .await
    .map_err(|e| ToolError::worker_stopped(NAME, "GRAPH_ARTIFACT_FAILED", &e))?
    .map_err(|e| {
        ToolError::new(
            NAME,
            "GRAPH_ARTIFACT_FAILED",
            e,
            "Check the sparsifier parameters (epsilon, sketch_dim, sample_count), then retry.",
        )
    })?;

    let artifact_edge_count = graph.edges.len();
    let artifact = Arc::new(GraphArtifact {
        artifact_id: format!("graph_{}", &pulsar_store::RunId::random().as_str()[4..]),
        run_id: run_id.clone(),
        kind: p.kind.clone(),
        params: params.clone(),
        graph,
        metrics,
    });
    let staleness = {
        let mut session = state.session();
        if session.latest_run_id.as_ref() != Some(&run_id) {
            return Err(ToolError::new(
                NAME,
                "SESSION_REVISION_CONFLICT",
                "Active run changed while building the graph artifact; refusing to publish it.",
                "The artifact was not published. The live run changed while it was building; call get_runtime_context and rebuild against the current run.",
            )
            .detail("requested_run_id", json!(run_id)));
        }
        session.graph_artifacts.push(artifact.clone());
        let excess = session
            .graph_artifacts
            .len()
            .saturating_sub(MAX_GRAPH_ARTIFACTS);
        session.graph_artifacts.drain(..excess);
        artifact_staleness(&artifact, session.latest_run_id.as_ref())
    };
    Ok(ok_result(
        json!({
            "status": "ok",
            "estimate_only": false,
            "artifact_id": artifact.artifact_id,
            "kind": p.kind,
            "run_id": run_id,
            "runtime_s": round_to(elapsed, 3),
            "input_edge_count": edge_count,
            "artifact_edge_count": artifact_edge_count,
            "params": params,
            "staleness": staleness,
        }),
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_match_python() {
        // _artifact_estimate(800, 12000, 1.0) and (800, 12000, 0.3) from the reference.
        let e = estimate(800, 12000, 1.0);
        assert_eq!(e["sketch_dim"], 161);
        assert_eq!(e["sample_count"], 48130);
        assert_eq!(e["runtime_s"], 0.43);
        assert_eq!(e["edge_count"], 12000);
        assert_eq!(e["runtime_basis"], "rough projection; scales ~sketch_dim x edges (so ~1/epsilon^2). Estimated at epsilon=1.");
        let e = estimate(800, 12000, 0.3);
        assert_eq!(e["sketch_dim"], 1783);
        assert!(e["runtime_basis"]
            .as_str()
            .unwrap()
            .ends_with("epsilon=0.3."));
        assert_eq!(py_g(1e-9), "1e-09");
        assert_eq!(py_g(123456.0), "123456");
        assert_eq!(py_g(1234567.0), "1.23457e+06");
    }
}
