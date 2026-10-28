//! Graph diagnostics (`pulsar/mcp/diagnostics.py`): pure geometric signal
//! for the agent to interpret. One component pass ([`GraphAnalysis`])
//! serves the metrics, the summary and the skeleton.

use pulsar_pipeline::Graph;
use serde::Serialize;
use serde_json::{json, Value};

use crate::error::AnalysisError;
use crate::minhash::{depth_profile, DepthProfile, DEFAULT_DELTA};
use crate::payloads::size_summary;
use crate::pyround::round_int;
use crate::thresholds::{first_report_ready_candidate, Candidate};

/// Characters of the Unicode sparkline, index 0 (empty) to 8 (full).
const SPARK_CHARS: [char; 9] = [
    ' ', ' ', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}',
];
const MAX_EDGES_IN_SUMMARY: usize = 500;

/// `numpy.histogram(data, bins)` counts for equal-width bins over
/// `[min, max]` (widened by 0.5 each way when `min == max`).
pub fn histogram(data: &[f64], bins: usize) -> Vec<usize> {
    let mut counts = vec![0usize; bins];
    if data.is_empty() || bins == 0 {
        return counts;
    }
    let (mut first, mut last) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
    if first == last {
        first -= 0.5;
        last += 0.5;
    }
    let step = (last - first) / bins as f64;
    let edge = |i: usize| {
        if i == bins {
            last
        } else {
            first + i as f64 * step
        }
    };
    let norm = bins as f64 / (last - first);
    for &x in data {
        let mut idx = ((x - first) * norm) as usize;
        if idx == bins {
            idx = bins - 1;
        }
        if x < edge(idx) {
            idx -= 1;
        }
        if idx != bins - 1 && x >= edge(idx + 1) {
            idx += 1;
        }
        counts[idx] += 1;
    }
    counts
}

/// Unicode sparkline histogram of a distribution
/// (`generate_distribution_sparkline`).
pub fn distribution_sparkline(data: &[f64], bins: usize) -> String {
    if data.is_empty() {
        return String::new();
    }
    let counts = histogram(data, bins);
    let max = counts.iter().copied().max().unwrap_or(0);
    if max == 0 {
        return std::iter::repeat_n(SPARK_CHARS[0], bins).collect();
    }
    counts
        .iter()
        .map(|&c| SPARK_CHARS[(c as f64 / max as f64 * (SPARK_CHARS.len() - 1) as f64) as usize])
        .collect()
}

fn value_at_sparse_upper_index(sorted: &[f64], zero_count: usize, index: usize) -> f64 {
    if index < zero_count {
        return 0.0;
    }
    sorted.get(index - zero_count).copied().unwrap_or(0.0)
}

/// Percentile of the weight distribution over *all* node pairs, where the
/// pairs without an edge count as zeros (`_sparse_upper_percentile`).
pub fn sparse_upper_percentile(weights: &[f64], total_pairs: usize, percentile: f64) -> f64 {
    if total_pairs == 0 {
        return 0.0;
    }
    let mut sorted: Vec<f64> = weights.iter().copied().filter(|&w| w > 0.0).collect();
    sorted.sort_by(f64::total_cmp);
    let zero_count = total_pairs.saturating_sub(sorted.len());
    let pos = (total_pairs - 1) as f64 * (percentile / 100.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let lo_v = value_at_sparse_upper_index(&sorted, zero_count, lo);
    let hi_v = value_at_sparse_upper_index(&sorted, zero_count, hi);
    lo_v + (hi_v - lo_v) * (pos - lo as f64)
}

/// A bounded sample of the pair-weight distribution: up to 512 zeros plus
/// up to 1536 evenly spaced positive weights (`_sparse_upper_sample`).
pub fn sparse_upper_sample(weights: &[f64], total_pairs: usize) -> Vec<f64> {
    if total_pairs == 0 {
        return Vec::new();
    }
    let mut sorted: Vec<f64> = weights.iter().copied().filter(|&w| w > 0.0).collect();
    sorted.sort_by(f64::total_cmp);
    let zero_count = total_pairs.saturating_sub(sorted.len());
    let mut out = vec![0.0; zero_count.min(512)];
    if sorted.len() <= 1536 {
        out.extend_from_slice(&sorted);
    } else {
        let num = 1536;
        let step = (sorted.len() - 1) as f64 / (num - 1) as f64;
        for i in 0..num {
            let idx = if i == num - 1 {
                sorted.len() - 1
            } else {
                (i as f64 * step) as usize
            };
            out.push(sorted[idx]);
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Advisory {
    pub code: &'static str,
    pub severity: &'static str,
    pub message: String,
    pub diagnostic_interpretation: &'static str,
}

fn graph_advisories(
    n_edges: usize,
    singleton_fraction: f64,
    giant_fraction: f64,
    density: f64,
) -> Vec<Advisory> {
    let mut out = Vec::new();
    if density > 0.8 && n_edges > 0 {
        out.push(Advisory {
            code: "HAIRBALL_DENSITY",
            severity: "warning",
            message: format!(
                "Graph density is {:.0}% \u{2014} most node pairs share an edge.",
                density * 100.0
            ),
            diagnostic_interpretation:
                "A near-fully-connected graph cannot express readable topological structure.",
        });
    }
    if n_edges == 0 {
        out.push(Advisory {
            code: "EMPTY_GRAPH",
            severity: "error",
            message: "Construction threshold produced zero edges.".into(),
            diagnostic_interpretation:
                "The constructed graph has no connectivity at the selected construction threshold.",
        });
    }
    if singleton_fraction > 0.8 {
        out.push(Advisory {
            code: "HIGH_SINGLETONS",
            severity: "warning",
            message: format!(
                "{:.0}% of nodes are singletons.",
                singleton_fraction * 100.0
            ),
            diagnostic_interpretation:
                "Most points are isolated, so component labels are dominated by fragmentation.",
        });
    }
    if giant_fraction > 0.95 {
        out.push(Advisory {
            code: "DOMINANT_COMPONENT",
            severity: "warning",
            message: format!(
                "One component covers {:.0}% of nodes.",
                giant_fraction * 100.0
            ),
            diagnostic_interpretation:
                "The graph may be structurally real but has little visible component separation.",
        });
    }
    out
}

/// `(status, note)` for the number of ball maps in the sweep.
pub fn grid_adequacy(n_ball_maps: usize) -> (&'static str, &'static str) {
    if n_ball_maps == 0 {
        (
            "unknown",
            "Ball-map count is unavailable; run a standard sweep before judging grid adequacy.",
        )
    } else if n_ball_maps < 15 {
        (
            "under_sampled",
            "Grid has fewer than 15 ball maps. Add projection dimensions, seeds, or epsilon steps before trusting structural persistence.",
        )
    } else if n_ball_maps < 40 {
        (
            "thin_grid",
            "Grid has fewer than 40 ball maps. Usable for a quick probe, but compare against a wider sweep before interpreting clusters.",
        )
    } else {
        (
            "sample_count_ok",
            "Grid sample count is adequate for a baseline. Still judge graph quality from density, components, and sweep comparisons.",
        )
    }
}

/// Graph structure measurements (`GraphMetrics`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GraphMetrics {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub density: f64,
    pub avg_degree: f64,
    pub giant_fraction: f64,
    pub singleton_count: usize,
    pub singleton_fraction: f64,
    pub component_count: usize,
    pub resolved_construction_threshold: f64,
    pub nonzero_fraction: f64,
    pub weight_p25: f64,
    pub weight_p50: f64,
    pub weight_p95: f64,
    pub weight_distribution_sparkline: String,
    /// Descending.
    pub component_sizes: Vec<usize>,
    pub n_ball_maps: usize,
    pub grid_adequacy_status: &'static str,
    pub grid_adequacy_note: &'static str,
    pub advisories: Vec<Advisory>,
    /// Realized MinHash accuracy at the depth used; `None` for exact
    /// construction.
    pub minhash_profile: Option<DepthProfile>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NodeSummary {
    pub node: usize,
    pub component_id: usize,
    pub degree: usize,
    pub weighted_degree: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EdgeSummary {
    pub source: usize,
    pub target: usize,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DegreeDistribution {
    pub mean: f64,
    pub max: usize,
    pub sparkline: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TopologicalSummary {
    /// Up to three hubs per component, by weighted degree.
    pub key_hubs: Vec<NodeSummary>,
    pub overall_hubs: Vec<NodeSummary>,
    /// Articulation points (computed for graphs up to 1000 nodes and 25000
    /// edges), sorted.
    pub bridges: Vec<usize>,
    pub degree_distribution: DegreeDistribution,
}

/// The persisted constructed-graph summary (`_build_graph_summary`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GraphSummary {
    pub node_count: usize,
    pub edge_count: usize,
    pub resolved_construction_threshold: f64,
    pub component_count: usize,
    pub component_sizes: Vec<usize>,
    pub nodes: Vec<NodeSummary>,
    /// Heaviest first, at most 500.
    pub edges: Vec<EdgeSummary>,
    pub edges_truncated: bool,
    pub edges_shown: usize,
    pub topological_summary: TopologicalSummary,
}

/// One component pass over a constructed graph, reused by every view.
#[derive(Debug)]
pub struct GraphAnalysis<'a> {
    graph: &'a Graph,
    labels: Vec<usize>,
    sizes: Vec<usize>,
    degree: Vec<usize>,
    weighted_degree: Vec<f64>,
}

impl<'a> GraphAnalysis<'a> {
    pub fn new(graph: &'a Graph) -> Result<Self, AnalysisError> {
        if graph.n == 0 {
            return Err(AnalysisError::EmptyGraph);
        }
        let labels = graph.component_labels();
        let mut sizes = vec![0usize; labels.iter().copied().max().map_or(0, |m| m + 1)];
        for &l in &labels {
            sizes[l] += 1;
        }
        let mut sorted = sizes.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        Ok(Self {
            graph,
            labels,
            sizes: sorted,
            degree: graph.degree(),
            weighted_degree: graph.weighted_degree(),
        })
    }

    pub fn component_labels(&self) -> &[usize] {
        &self.labels
    }

    /// Descending.
    pub fn component_sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// Metrics of the constructed graph. `full_weight_edges` (the whole
    /// affinity) drives the weight percentiles when given; otherwise the
    /// constructed edges do.
    pub fn metrics(
        &self,
        resolved_construction_threshold: f64,
        n_ball_maps: usize,
        full_weight_edges: Option<&[(usize, usize, f64)]>,
        minhash_d: Option<usize>,
    ) -> GraphMetrics {
        let n = self.graph.n;
        let n_edges = self.graph.n_edges();
        let max_pairs = n * (n - 1) / 2;
        let density = if max_pairs > 0 {
            n_edges as f64 / max_pairs as f64
        } else {
            0.0
        };
        let avg_degree = 2.0 * n_edges as f64 / n as f64;
        let giant_fraction = self.sizes.first().map_or(0.0, |&s| s as f64 / n as f64);
        let singleton_count = self.sizes.iter().filter(|&&s| s == 1).count();
        let singleton_fraction = singleton_count as f64 / n as f64;

        let weights: Vec<f64> = match full_weight_edges {
            Some(edges) => edges.iter().map(|e| e.2).collect(),
            None => self.graph.edges.iter().map(|e| e.2).collect(),
        };
        let nonzero_fraction = if max_pairs > 0 {
            weights.len() as f64 / max_pairs as f64
        } else {
            0.0
        };
        let sample = sparse_upper_sample(&weights, max_pairs);
        let (status, note) = grid_adequacy(n_ball_maps);
        GraphMetrics {
            n_nodes: n,
            n_edges,
            density,
            avg_degree,
            giant_fraction,
            singleton_count,
            singleton_fraction,
            component_count: self.sizes.len(),
            resolved_construction_threshold,
            nonzero_fraction,
            weight_p25: sparse_upper_percentile(&weights, max_pairs, 25.0),
            weight_p50: sparse_upper_percentile(&weights, max_pairs, 50.0),
            weight_p95: sparse_upper_percentile(&weights, max_pairs, 95.0),
            weight_distribution_sparkline: distribution_sparkline(&sample, 10),
            component_sizes: self.sizes.clone(),
            n_ball_maps,
            grid_adequacy_status: status,
            grid_adequacy_note: note,
            advisories: graph_advisories(n_edges, singleton_fraction, giant_fraction, density),
            minhash_profile: minhash_d.map(|d| depth_profile(d, n, DEFAULT_DELTA)),
        }
    }

    /// The bounded graph summary with hubs, bridges and degree shape.
    pub fn summary(&self, resolved_construction_threshold: f64) -> GraphSummary {
        let n = self.graph.n;
        let nodes: Vec<NodeSummary> = (0..n)
            .map(|node| NodeSummary {
                node,
                component_id: self.labels[node],
                degree: self.degree[node],
                weighted_degree: self.weighted_degree[node],
            })
            .collect();
        let mut edges: Vec<EdgeSummary> = self
            .graph
            .edges
            .iter()
            .map(|&(i, j, w)| EdgeSummary {
                source: i,
                target: j,
                weight: w,
            })
            .collect();
        edges.sort_by(|a, b| {
            b.weight
                .total_cmp(&a.weight)
                .then_with(|| (a.source, a.target).cmp(&(b.source, b.target)))
        });
        let total_edges = edges.len();
        let truncated = total_edges > MAX_EDGES_IN_SUMMARY;
        edges.truncate(MAX_EDGES_IN_SUMMARY);

        let n_components = self.sizes.len();
        let mut key_hubs = Vec::new();
        for comp in 0..n_components {
            let mut members: Vec<&NodeSummary> =
                nodes.iter().filter(|x| x.component_id == comp).collect();
            members.sort_by(|a, b| b.weighted_degree.total_cmp(&a.weighted_degree));
            key_hubs.extend(members.into_iter().take(3).cloned());
        }
        let mut by_weight: Vec<&NodeSummary> = nodes.iter().collect();
        by_weight.sort_by(|a, b| b.weighted_degree.total_cmp(&a.weighted_degree));
        let overall_hubs: Vec<NodeSummary> = by_weight.into_iter().take(3).cloned().collect();

        let bridges = if n <= 1000 && total_edges <= 25_000 {
            articulation_points(self.graph)
        } else {
            Vec::new()
        };
        let degrees: Vec<f64> = self.degree.iter().map(|&d| d as f64).collect();
        let degree_distribution = DegreeDistribution {
            mean: degrees.iter().sum::<f64>() / n as f64,
            max: self.degree.iter().copied().max().unwrap_or(0),
            sparkline: distribution_sparkline(&degrees, 10),
        };

        GraphSummary {
            node_count: n,
            edge_count: total_edges,
            resolved_construction_threshold,
            component_count: n_components,
            component_sizes: self.sizes.clone(),
            nodes,
            edges_shown: edges.len(),
            edges,
            edges_truncated: truncated,
            topological_summary: TopologicalSummary {
                key_hubs,
                overall_hubs,
                bridges,
                degree_distribution,
            },
        }
    }
}

/// Articulation points of an undirected graph (iterative Tarjan), sorted.
pub fn articulation_points(graph: &Graph) -> Vec<usize> {
    let n = graph.n;
    let csr = graph.csr();
    let mut disc = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut is_ap = vec![false; n];
    let mut time = 0;
    for root in 0..n {
        if disc[root] != usize::MAX {
            continue;
        }
        // Stack of (node, parent, next neighbor index).
        let mut stack: Vec<(usize, usize, usize)> = vec![(root, usize::MAX, 0)];
        disc[root] = time;
        low[root] = time;
        time += 1;
        let mut root_children = 0;
        while let Some(&mut (u, parent, ref mut next)) = stack.last_mut() {
            let neighbors = csr.neighbors(u);
            if *next < neighbors.len() {
                let v = neighbors[*next];
                *next += 1;
                if v == parent {
                    continue;
                }
                if disc[v] == usize::MAX {
                    disc[v] = time;
                    low[v] = time;
                    time += 1;
                    if u == root {
                        root_children += 1;
                    }
                    stack.push((v, u, 0));
                } else {
                    low[u] = low[u].min(disc[v]);
                }
            } else {
                stack.pop();
                if let Some(&(p, _, _)) = stack.last() {
                    low[p] = low[p].min(low[u]);
                    if p != root && low[u] >= disc[p] {
                        is_ap[p] = true;
                    }
                }
            }
        }
        if root_children > 1 {
            is_ap[root] = true;
        }
    }
    (0..n).filter(|&v| is_ap[v]).collect()
}

/// The topological-skeleton payload for one `detail` level
/// (`_skeleton_graph_payload`).
pub fn skeleton_payload(
    summary: &GraphSummary,
    detail: &str,
    max_edges: usize,
    max_nodes: usize,
) -> Value {
    let include_nodes = matches!(detail, "full_nodes" | "full");
    let include_edges = matches!(detail, "edges" | "full");
    let nodes: &[NodeSummary] = if include_nodes {
        &summary.nodes[..summary.nodes.len().min(max_nodes)]
    } else {
        &[]
    };
    let edges: &[EdgeSummary] = if include_edges {
        &summary.edges[..summary.edges.len().min(max_edges)]
    } else {
        &[]
    };
    let mut payload = json!({
        "node_count": summary.node_count,
        "edge_count": summary.edge_count,
        "resolved_construction_threshold": summary.resolved_construction_threshold,
        "component_count": summary.component_count,
        "component_sizes_summary": size_summary(&summary.component_sizes, 20),
        "detail": detail,
        "nodes_returned": nodes.len(),
        "nodes_omitted": summary.node_count.saturating_sub(nodes.len()),
        "edges_returned": edges.len(),
        "edges_omitted": summary.edge_count.saturating_sub(edges.len()),
        "source_edges_truncated": summary.edges_truncated,
    });
    let obj = payload.as_object_mut().expect("object");
    if detail == "full" {
        obj.insert("component_sizes".into(), json!(summary.component_sizes));
    }
    if detail == "nodes" {
        obj.insert(
            "topological_summary".into(),
            json!(summary.topological_summary),
        );
    }
    if include_nodes {
        obj.insert("nodes".into(), json!(nodes));
    }
    if include_edges {
        obj.insert("edges".into(), json!(edges));
    }
    payload
}

/// Projection dimensions to zoom into after a dominant-component gate
/// (`_suggest_resolution_pca_dims`). `ceiling` bounds the suggestions: the
/// projection input's feature count, or the run's own largest dimension when
/// that is unknown (`0`). Python clamped at 16 instead. A ceiling below 2
/// leaves no valid projection dimension, so nothing is suggested.
pub fn suggest_resolution_dims(dims: &[usize], ceiling: usize) -> Vec<usize> {
    let mut dims: Vec<usize> = dims.iter().copied().filter(|&d| d > 1).collect();
    dims.sort_unstable();
    dims.dedup();
    let ceiling = match ceiling {
        0 => dims.last().copied().unwrap_or(usize::MAX),
        1 => return Vec::new(),
        c => c,
    };
    if dims.is_empty() {
        let mut out: Vec<usize> = [5, 8, 12, 16].map(|d| d.min(ceiling)).to_vec();
        out.dedup();
        return out;
    }
    if dims.len() == 1 {
        let center = dims[0].min(ceiling);
        let mut out = vec![
            center.saturating_sub(4).max(2),
            center.saturating_sub(2).max(2),
            center,
            (center + 2).min(ceiling),
        ];
        out.sort_unstable();
        out.dedup();
        return out;
    }
    let low = dims[0];
    let high = dims[dims.len() - 1].min(ceiling);
    if high <= low {
        dims.retain(|&d| d <= ceiling);
        return dims;
    }
    let mut step = round_int((high - low) as f64 / 5.0).max(1) as usize;
    if high - low >= 6 {
        step = step.max(2);
    }
    (low..=high).step_by(step).take(6).collect()
}

/// Whether clusters may be finalized on a dominant-component graph
/// (`_finalization_gate`). `candidates` are the threshold candidates of the
/// stability curve, if any. `n_features` is the projection input's feature
/// count (`0` when unknown); it bounds the suggested dimensions.
pub fn finalization_gate(
    metrics: &GraphMetrics,
    sweep_count: usize,
    projection_dims: &[usize],
    n_features: usize,
    candidates: Option<&[Candidate]>,
) -> Value {
    let needs_resolution = metrics.giant_fraction > 0.95
        && (sweep_count < 3 || metrics.density > 0.35 || metrics.n_ball_maps < 40);
    if !needs_resolution {
        return json!({ "status": "ok", "reason": "No dominant-component resolution gate triggered." });
    }
    if let Some(candidate) = candidates.and_then(first_report_ready_candidate) {
        return json!({
            "status": "caution",
            "code": "DOMINANT_COMPONENT_HAS_COMPONENT_LENS",
            "message": format!(
                "giant_fraction={:.1}%, but threshold stability found a stricter component lens suitable for cohort reading.",
                metrics.giant_fraction * 100.0
            ),
            "recommended_action": {
                "tool": "generate_cluster_dossier",
                "method": "components",
                "interpretation_edge_weight_threshold": candidate.threshold(),
                "reason": "Use the report-ready or balanced H0 component slice before falling back to spectral clustering.",
            },
        });
    }
    json!({
        "status": "blocked",
        "code": "UNRESOLVED_DOMINANT_COMPONENT",
        "message": format!(
            "giant_fraction={:.1}% after {sweep_count} sweep(s). Do not finalize clusters until one targeted resolution sweep is run or the dominant component is explicitly justified for this dataset.",
            metrics.giant_fraction * 100.0
        ),
        "suggested_refinement": {
            "projection_dimensions": suggest_resolution_dims(projection_dims, n_features),
            "projection_seeds": [42, 7, 13],
            "epsilon_steps_min": 24,
            "strategy": "Zoom into the useful projection band with multiple seeds. Shift/lower epsilon if density remains high; avoid narrowing to a single projection dimension or single seed.",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_and_sparkline_follow_numpy() {
        assert_eq!(
            histogram(&[0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 10),
            vec![2, 0, 0, 0, 0, 1, 0, 0, 0, 3]
        );
        assert_eq!(
            histogram(&[2.0, 2.0], 10),
            vec![0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
        );
        let spark = distribution_sparkline(&[0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 10);
        let chars: Vec<char> = spark.chars().collect();
        assert_eq!(chars.len(), 10);
        assert_eq!(chars[9], '\u{2588}');
        assert_eq!(chars[0], SPARK_CHARS[(2.0 / 3.0 * 8.0) as usize]);
        assert_eq!(chars[1], ' ');
        assert_eq!(distribution_sparkline(&[], 10), "");
    }

    #[test]
    fn sparse_percentiles_count_missing_pairs_as_zero() {
        // 6 pairs, 2 edges: positions 0..3 are zeros, then 0.4, 0.8.
        let weights = [0.8, 0.4];
        assert_eq!(sparse_upper_percentile(&weights, 6, 50.0), 0.0);
        let p95 = sparse_upper_percentile(&weights, 6, 95.0);
        // pos = 4.75 → between 0.4 (index 4) and 0.8 (index 5).
        assert!((p95 - (0.4 + 0.4 * 0.75)).abs() < 1e-12);
        assert_eq!(
            sparse_upper_sample(&weights, 6),
            vec![0.0, 0.0, 0.0, 0.0, 0.4, 0.8]
        );
        let many: Vec<f64> = (1..=3001).map(|i| i as f64).collect();
        let sample = sparse_upper_sample(&many, 1_000_000);
        assert_eq!(sample.len(), 512 + 1536);
        assert_eq!(&sample[512..517], &[1.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(sample[sample.len() - 1], 3001.0);
    }

    #[test]
    fn metrics_summary_and_bridges_from_one_pass() {
        // Triangle 0-1-2, bridge node 2-3, pendant 3-4, isolated 5.
        let g = Graph {
            n: 6,
            edges: vec![
                (0, 1, 0.9),
                (0, 2, 0.8),
                (1, 2, 0.7),
                (2, 3, 0.5),
                (3, 4, 0.4),
            ],
        };
        let a = GraphAnalysis::new(&g).unwrap();
        assert_eq!(a.component_sizes(), &[5, 1]);
        let m = a.metrics(0.3, 20, None, Some(128));
        assert_eq!(m.n_edges, 5);
        assert_eq!(m.component_count, 2);
        assert_eq!(m.singleton_count, 1);
        assert!((m.density - 5.0 / 15.0).abs() < 1e-12);
        assert_eq!(m.grid_adequacy_status, "thin_grid");
        assert!(m.advisories.is_empty());
        assert_eq!(m.minhash_profile.as_ref().unwrap().d, 128);
        assert_eq!(m.weight_distribution_sparkline.chars().count(), 10);

        let s = a.summary(0.3);
        assert_eq!(s.edges[0].weight, 0.9);
        assert!(!s.edges_truncated);
        assert_eq!(s.topological_summary.bridges, vec![2, 3]);
        assert_eq!(s.topological_summary.overall_hubs[0].node, 2);
        assert_eq!(s.topological_summary.key_hubs.len(), 4);
        assert_eq!(s.topological_summary.degree_distribution.max, 3);

        let payload = skeleton_payload(&s, "edges", 2, 10);
        assert_eq!(payload["edges_returned"], 2);
        assert_eq!(payload["edges_omitted"], 3);
        assert!(payload.get("nodes").is_none());
        let nodes = skeleton_payload(&s, "nodes", 2, 10);
        assert!(nodes["topological_summary"]["bridges"].is_array());

        let empty = Graph {
            n: 3,
            edges: vec![],
        };
        let m = GraphAnalysis::new(&empty)
            .unwrap()
            .metrics(0.5, 0, None, None);
        assert_eq!(
            m.advisories.iter().map(|a| a.code).collect::<Vec<_>>(),
            vec!["EMPTY_GRAPH", "HIGH_SINGLETONS"]
        );
        assert_eq!(
            GraphAnalysis::new(&Graph {
                n: 0,
                edges: vec![]
            })
            .unwrap_err()
            .code(),
            "EMPTY_GRAPH"
        );
    }

    #[test]
    fn gate_and_resolution_dims_follow_python() {
        // Grids up to 16 dims with at least 16 features match Python.
        assert_eq!(suggest_resolution_dims(&[], 16), vec![5, 8, 12, 16]);
        assert_eq!(suggest_resolution_dims(&[3], 30), vec![2, 3, 5]);
        assert_eq!(suggest_resolution_dims(&[2, 5, 8], 30), vec![2, 4, 6, 8]);
        assert_eq!(suggest_resolution_dims(&[16], 16), vec![12, 14, 16]);
        assert_eq!(
            suggest_resolution_dims(&[4, 16], 0),
            vec![4, 6, 8, 10, 12, 14]
        );
        // Python clamped at 16; the ceiling is now the feature count or the
        // run's own largest dimension.
        assert_eq!(
            suggest_resolution_dims(&[2, 20], 30),
            vec![2, 6, 10, 14, 18]
        );
        assert_eq!(suggest_resolution_dims(&[2, 20], 0), vec![2, 6, 10, 14, 18]);
        assert_eq!(
            suggest_resolution_dims(&[16, 20], 30),
            vec![16, 17, 18, 19, 20]
        );
        assert_eq!(suggest_resolution_dims(&[18], 19), vec![14, 16, 18, 19]);
        assert_eq!(suggest_resolution_dims(&[], 6), vec![5, 6]);
        // Never above the feature count, and nothing when it is below 2.
        assert!(suggest_resolution_dims(&[], 1).is_empty());
        assert!(suggest_resolution_dims(&[2, 3], 1).is_empty());
        assert_eq!(suggest_resolution_dims(&[5, 8], 4), Vec::<usize>::new());
        assert_eq!(suggest_resolution_dims(&[3, 8], 3), vec![3]);
        assert_eq!(suggest_resolution_dims(&[3], 2), vec![2]);

        let g = Graph {
            n: 40,
            edges: (0..39).map(|i| (i, i + 1, 1.0)).collect(),
        };
        let m = GraphAnalysis::new(&g).unwrap().metrics(0.1, 10, None, None);
        let gate = finalization_gate(&m, 1, &[2, 5], 0, None);
        assert_eq!(gate["status"], "blocked");
        assert_eq!(gate["code"], "UNRESOLVED_DOMINANT_COMPONENT");
        assert!(gate["message"]
            .as_str()
            .unwrap()
            .starts_with("giant_fraction=100.0% after 1 sweep(s)."));
        let ok = finalization_gate(&m, 3, &[2, 5], 0, None);
        assert_eq!(
            ok["status"], "blocked",
            "density 0.05 but n_ball_maps 10 < 40"
        );
        let m2 = GraphAnalysis::new(&g).unwrap().metrics(0.1, 50, None, None);
        assert_eq!(finalization_gate(&m2, 3, &[2], 0, None)["status"], "ok");
    }
}
