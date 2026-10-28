//! The standalone HTML research report (`pulsar/mcp/report/`): a summary,
//! the topology figure with its threshold transition map, the cohort signal
//! matrix, one evidence section per cluster and the run parameters. CSS, JS
//! and font links are Python's, verbatim, in `assets/`.
//!
//! Known divergences from Python: the topology figure's layout uses the
//! force method at every size (networkx uses an L-BFGS "energy" method from
//! 500 nodes); see [`crate::layout`].

use std::collections::{BTreeMap, HashMap, HashSet};

use pulsar_analysis::dossier::{ClusterProfile, TopologicalDossier};
use pulsar_analysis::pyround::round_to;
use pulsar_analysis::thresholds::{
    agent_threshold_options, component_mass_profile, structural_breakpoints, Policy,
};
use pulsar_analysis::ThresholdGraph;
use pulsar_core::{find_stable_thresholds_from_edges, StabilityResult, DEFAULT_NUM_BINS};
use pulsar_data::preprocess::py_float_repr;
use pulsar_pipeline::Graph;
use serde_json::{json, Value};

use crate::layout::spring_layout;

const STYLES: &str = include_str!("../assets/report.css");
const SCRIPT: &str = include_str!("../assets/report.js");
const FONTS: &str = include_str!("../assets/fonts.html");

const MAX_SIGNAL_MATRIX_NUMERIC: usize = 10;
const MAX_SIGNAL_MATRIX_CATEGORICAL: usize = 5;
const FIGURE_NODE_BUDGET: usize = 1000;
const PALETTE: [&str; 10] = [
    "#1a73e8", "#669df6", "#147d64", "#8ab4f8", "#f29900", "#5f6368", "#00acc1", "#7baaf7",
    "#c58a00", "#7c8b95",
];

/// The fitted graph surfaces the figure and the threshold map read.
pub struct ReportGraph<'a> {
    /// The constructed graph (`model.cosmic_graph`).
    pub graph: &'a Graph,
    /// Every weighted edge on `[0, 1]` above 0 (`weighted_edges(threshold=0)`).
    pub affinity: &'a [(usize, usize, f64)],
    /// The run's H0 stability result, when the construction threshold was auto.
    pub stability: Option<&'a StabilityResult>,
    pub resolved_threshold: f64,
}

// ---- formatting --------------------------------------------------------------------------

/// `html.escape(str(value), quote=True)`.
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// Python `str()` of a JSON scalar.
fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) if n.is_f64() => py_float_repr(n.as_f64().unwrap_or(f64::NAN)),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `_escape_html`: `None` is empty.
fn esc_value(v: &Value) -> String {
    if v.is_null() {
        String::new()
    } else {
        esc(&py_str(v))
    }
}

fn group_digits(int_part: &str) -> String {
    let (sign, digits) = int_part
        .strip_prefix('-')
        .map_or(("", int_part), |d| ("-", d));
    let mut out = String::new();
    for (k, c) in digits.chars().enumerate() {
        if k > 0 && (digits.len() - k) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    format!("{sign}{out}")
}

/// `f"{i:,}"`.
fn commas(i: i64) -> String {
    group_digits(&i.to_string())
}

/// `f"{x:.{d}f}"` (`nan`, `inf` spelled as Python spells them).
fn fixed(x: f64, d: usize) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.into()
    } else {
        format!("{x:.d$}")
    }
}

/// `f"{x:+.{d}f}"`.
fn signed(x: f64, d: usize) -> String {
    if x.is_nan() {
        "+nan".into()
    } else if x == f64::INFINITY {
        "+inf".into()
    } else if x.is_infinite() {
        "-inf".into()
    } else {
        format!("{x:+.d$}")
    }
}

/// `f"{x:,.{d}f}"`.
fn grouped(x: f64, d: usize) -> String {
    let s = fixed(x, d);
    match s.split_once('.') {
        Some((i, f)) => format!("{}.{f}", group_digits(i)),
        None if x.is_finite() => group_digits(&s),
        None => s,
    }
}

/// `f"{x:.{d}%}"`.
fn percent(x: f64, d: usize) -> String {
    format!("{}%", fixed(x * 100.0, d))
}

fn num(v: Option<&Value>) -> f64 {
    v.and_then(Value::as_f64).unwrap_or(f64::NAN)
}

/// `_format_value(value, digits)`; `null_text` is how a missing value prints.
fn format_value(v: Option<&Value>, digits: usize, null_text: &str) -> String {
    match v {
        None | Some(Value::Null) => null_text.into(),
        Some(Value::Bool(b)) => commas(i64::from(*b)),
        Some(Value::Number(n)) if n.is_i64() || n.is_u64() => {
            n.as_i64().map_or_else(|| n.to_string(), commas)
        }
        Some(Value::Number(n)) => {
            let x = n.as_f64().unwrap_or(f64::NAN);
            if x.is_nan() {
                "NaN".into()
            } else {
                fixed(x, digits)
            }
        }
        Some(other) => py_str(other),
    }
}

/// `_format_heatmap_value`.
fn heatmap_value(v: Option<&Value>) -> String {
    match v {
        None => "—".into(),
        Some(Value::Null) => "NaN".into(),
        Some(Value::Bool(b)) => commas(i64::from(*b)),
        Some(Value::Number(n)) if n.is_i64() || n.is_u64() => {
            n.as_i64().map_or_else(|| n.to_string(), commas)
        }
        Some(Value::Number(n)) => {
            let x = n.as_f64().unwrap_or(f64::NAN);
            if x.is_nan() {
                "NaN".into()
            } else if x.abs() >= 100.0 {
                grouped(x, 0)
            } else if x.abs() >= 10.0 {
                grouped(x, 1)
            } else {
                grouped(x, 2)
            }
        }
        Some(other) => esc_value(other),
    }
}

fn signal_tone(z: f64) -> (&'static str, &'static str) {
    if z > 1.0 {
        ("Enriched", "signal-pos")
    } else if z < -1.0 {
        ("Depleted", "signal-neg")
    } else {
        ("Baseline", "signal-neu")
    }
}

fn cid(id: i64) -> String {
    format!("{id:02}")
}

fn prevalence(f: &Value) -> f64 {
    num(f
        .get("in_cluster_prevalence")
        .or_else(|| f.get("concentration"))
        .or(Some(&json!(0.0))))
}

fn int_field(f: &Value, key: &str, default: i64) -> i64 {
    f.get(key)
        .and_then(Value::as_f64)
        .map_or(default, |x| x as i64)
}

// ---- sections ----------------------------------------------------------------------------

fn key_findings(d: &TopologicalDossier, gm: &Value) -> Vec<(&'static str, String)> {
    let mut out = vec![];
    if !d.clusters.is_empty() {
        let mut dominant = &d.clusters[0];
        for p in &d.clusters[1..] {
            if p.size > dominant.size {
                dominant = p;
            }
        }
        out.push((
            "Largest cohort",
            format!(
                "{} accounts for {}% of the population.",
                dominant.semantic_name,
                fixed(dominant.size_pct, 1)
            ),
        ));
        let mut best: Option<(&ClusterProfile, &Value)> = None;
        let mut best_score = -1.0;
        for p in &d.clusters {
            for f in &p.numeric_features {
                let score = num(f.get("z_score")).abs();
                if score > best_score {
                    best = Some((p, f));
                    best_score = score;
                }
            }
        }
        if let Some((p, f)) = best {
            out.push((
                "Sharpest shift",
                format!(
                    "{} shows the strongest deviation on {} ({} z).",
                    p.semantic_name,
                    py_str(&f["column"]),
                    signed(num(f.get("z_score")), 2)
                ),
            ));
        }
    }
    if gm.as_object().is_some_and(|m| !m.is_empty()) {
        let g = |k: &str| gm.get(k).map_or_else(|| "N/A".to_owned(), py_str);
        out.push((
            "Topology footprint",
            format!(
                "{} stable components across {} graph nodes and {} edges.",
                g("component_count"),
                g("n_nodes"),
                g("n_edges")
            ),
        ));
    }
    out.truncate(3);
    out
}

fn render_nav(d: &TopologicalDossier) -> String {
    let mut parts = vec![
        "<aside class='report-nav'>".to_owned(),
        "<div class='report-nav__brand'>".into(),
        "<span class='report-nav__eyebrow brand-mark'>Quick links</span>".into(),
        "<h1>Pulsar topological report</h1>".into(),
        format!(
            "<p>{} observations · {} cohorts</p>",
            commas(d.n_total as i64),
            d.n_clusters
        ),
        "</div>".into(),
        "<nav aria-label='Section navigation'>".into(),
        "<div class='nav-group'>".into(),
        "<div class='nav-label report-nav__eyebrow'>Sections</div>".into(),
        "<a class='nav-link is-active' href='#summary'>Summary</a>".into(),
        "<a class='nav-link' href='#topology'>Topology figure</a>".into(),
        "<a class='nav-link' href='#heatmap'>Feature drift</a>".into(),
        "<a class='nav-link' href='#config'>Run parameters</a>".into(),
        "</div>".into(),
        "<div class='nav-group'>".into(),
        "<div class='nav-label report-nav__eyebrow'>Clusters</div>".into(),
    ];
    for p in &d.clusters {
        parts.push(format!(
            "<a class='nav-link' href='#cluster-{0}' data-cluster-id='{0}'>C{1} — {2}</a>",
            p.cluster_id,
            cid(p.cluster_id),
            esc(&p.semantic_name)
        ));
    }
    parts.extend(["</div>".into(), "</nav>".into(), "</aside>".into()]);
    parts.concat()
}

fn render_summary(d: &TopologicalDossier, gm: &Value) -> String {
    let findings = key_findings(d, gm);
    let column_count = d
        .global_stats
        .get("columns")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let na = json!("N/A");
    let g = |k: &str| format_value(Some(gm.get(k).unwrap_or(&na)), 0, "N/A");
    let metrics = [
        ("Observations", commas(d.n_total as i64)),
        ("Cohorts", d.n_clusters.to_string()),
        ("Graph nodes", g("n_nodes")),
        ("Graph edges", g("n_edges")),
        ("Columns profiled", column_count.to_string()),
    ];
    let mut parts = vec![
        "<section id='summary' class='report-section'>".to_owned(),
        "<div class='hero-grid'>".into(),
        "<div class='hero-copy'>".into(),
        "<span class='section-eyebrow'>Summary</span>".into(),
        "<h2>Latent structure report</h2>".into(),
        format!(
            "<p>{} observations resolve into <strong>{}</strong> cohorts. The supporting topology contains <strong>{}</strong> nodes, <strong>{}</strong> edges, and <strong>{}</strong> stable components.</p>",
            commas(d.n_total as i64),
            d.n_clusters,
            esc(&g("n_nodes")),
            esc(&g("n_edges")),
            esc(&g("component_count"))
        ),
        "</div>".into(),
        "</div>".into(),
        "<div class='hero-metrics'>".into(),
    ];
    for (label, value) in metrics {
        parts.push(format!(
            "<div class='metric-card'><div class='metric-value'>{}</div><div class='metric-label'>{}</div></div>",
            esc(&value),
            esc(label)
        ));
    }
    parts.push("</div>".into());
    if !findings.is_empty() {
        parts.push("<div class='findings-grid'><ul class='hero-list'>".into());
        for (title, body) in findings {
            parts.push(format!(
                "<li><strong>{}.</strong> {}</li>",
                esc(title),
                esc(&body)
            ));
        }
        parts.push("</ul></div>".into());
    }
    parts.push("</section>".into());
    parts.concat()
}

/// A networkx `Graph` built by `add_weighted_edges_from`: per-node neighbour
/// lists in insertion order.
struct NxGraph {
    adj: Vec<Vec<(usize, f64)>>,
}

impl NxGraph {
    fn new(graph: &Graph) -> Self {
        let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; graph.n];
        let mut index: Vec<HashMap<usize, usize>> = vec![HashMap::new(); graph.n];
        let mut add =
            |u: usize, v: usize, w: f64, adj: &mut Vec<Vec<(usize, f64)>>| match index[u].get(&v) {
                Some(&k) => adj[u][k].1 = w,
                None => {
                    index[u].insert(v, adj[u].len());
                    adj[u].push((v, w));
                }
            };
        for &(i, j, w) in &graph.edges {
            add(i, j, w, &mut adj);
            if i != j {
                add(j, i, w, &mut adj);
            }
        }
        Self { adj }
    }

    fn degree(&self, u: usize) -> usize {
        self.adj[u].len() + self.adj[u].iter().filter(|(v, _)| *v == u).count()
    }
}

fn cosmic_graph_svg(graph: &Graph, labels: &[i64], palette: &BTreeMap<i64, &str>) -> String {
    let (width, height) = (920.0_f64, 500.0_f64);
    let n = graph.n;
    if n == 0 {
        return String::new();
    }
    let g = NxGraph::new(graph);
    let degree_desc =
        |nodes: &mut Vec<usize>| nodes.sort_by_key(|&u| std::cmp::Reverse(g.degree(u)));
    let nodes: Vec<usize> = if n > FIGURE_NODE_BUDGET {
        let mut sampled: HashSet<usize> = HashSet::new();
        let mut ids: Vec<i64> = labels.to_vec();
        ids.sort_unstable();
        ids.dedup();
        for c in ids {
            let mut members: Vec<usize> = (0..labels.len())
                .filter(|&i| labels[i] == c && i < n)
                .collect();
            if members.is_empty() {
                continue;
            }
            let take = 10.max((members.len() as f64 * 0.05) as usize);
            degree_desc(&mut members);
            sampled.extend(members.into_iter().take(take));
        }
        let mut remaining = FIGURE_NODE_BUDGET as i64 - sampled.len() as i64;
        if remaining > 0 {
            let mut all: Vec<usize> = (0..n).collect();
            degree_desc(&mut all);
            for node in all {
                if sampled.insert(node) {
                    remaining -= 1;
                }
                if remaining <= 0 {
                    break;
                }
            }
        }
        (0..n).filter(|u| sampled.contains(u)).collect()
    } else {
        (0..n).collect()
    };
    let local: HashMap<usize, usize> = nodes.iter().enumerate().map(|(k, &u)| (u, k)).collect();
    let sub_adj: Vec<Vec<(usize, f64)>> = nodes
        .iter()
        .map(|&u| {
            g.adj[u]
                .iter()
                .filter_map(|&(v, w)| local.get(&v).map(|&k| (k, w)))
                .collect()
        })
        .collect();
    let k = 1.5 / (nodes.len().max(1) as f64).sqrt();
    let pos = spring_layout(nodes.len(), &sub_adj, k, 42);
    let (xmin, xmax) = pos
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
            (lo.min(p[0]), hi.max(p[0]))
        });
    let (ymin, ymax) = pos
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
            (lo.min(p[1]), hi.max(p[1]))
        });
    let padding = 40.0;
    let scale = |v: f64, lo: f64, hi: f64, target: f64| {
        padding + (v - lo) / (hi - lo + 1e-9) * (target - 2.0 * padding)
    };

    let mut svg = vec![
        format!("<svg id='cosmicGraph' viewBox='0 0 {w} {h}' xmlns='http://www.w3.org/2000/svg' role='img' aria-label='Topological skeleton projection'>", w = width as i64, h = height as i64),
        format!("<rect x='0' y='0' width='{w}' height='{h}' fill='#ffffff' rx='24' />", w = width as i64, h = height as i64),
    ];
    // networkx edge iteration: nodes in order, neighbours in insertion order,
    // each undirected edge once (from its earlier node).
    let mut bands: Vec<(f64, Vec<String>)> = vec![];
    let mut seen = vec![false; nodes.len()];
    for (a, row) in sub_adj.iter().enumerate() {
        for &(b, w) in row {
            if seen[b] {
                continue;
            }
            let (x1, y1) = (
                scale(pos[a][0], xmin, xmax, width),
                scale(pos[a][1], ymin, ymax, height),
            );
            let (x2, y2) = (
                scale(pos[b][0], xmin, xmax, width),
                scale(pos[b][1], ymin, ymax, height),
            );
            let band = round_to(0.10_f64.max(0.08 + w * 0.22), 2);
            let seg = format!("M{x1:.1} {y1:.1} L{x2:.1} {y2:.1}");
            match bands.iter_mut().find(|(o, _)| *o == band) {
                Some((_, segs)) => segs.push(seg),
                None => bands.push((band, vec![seg])),
            }
        }
        seen[a] = true;
    }
    bands.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (opacity, segs) in bands {
        svg.push(format!(
            "<path d='{}' stroke='#a8afb8' stroke-width='0.65' stroke-opacity='{}' fill='none' />",
            segs.join(" "),
            py_float_repr(opacity)
        ));
    }
    for (a, &node) in nodes.iter().enumerate() {
        let x = scale(pos[a][0], xmin, xmax, width);
        let y = scale(pos[a][1], ymin, ymax, height);
        let degree = g.degree(node);
        let radius = 1.8 + (degree as f64 / n.max(1) as f64 * 7.5);
        let cluster = labels.get(node).copied();
        let fill = cluster
            .and_then(|c| palette.get(&c).copied())
            .unwrap_or("#2563eb");
        let c = cluster.map_or(String::new(), |c| c.to_string());
        svg.push(format!(
            "<circle cx='{x:.2}' cy='{y:.2}' r='{radius:.2}' fill='{fill}' fill-opacity='0.78' class='graph-node' data-cluster='{c}' data-base-radius='{radius:.2}' data-base-fill='{fill}' data-base-opacity='0.78'><title>Node {node} (degree {degree}, cluster {c})</title></circle>"
        ));
    }
    svg.push("</svg>".into());
    svg.concat()
}

fn component_sizes_text(sizes: &Value) -> String {
    let sizes: Vec<i64> = sizes
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_i64)
        .collect();
    if sizes.is_empty() {
        "none".into()
    } else {
        sizes
            .iter()
            .map(|&s| commas(s))
            .collect::<Vec<_>>()
            .join(" / ")
    }
}

struct ThresholdPayload {
    resolved: f64,
    auto: f64,
    plateaus: Vec<Value>,
    candidates: Vec<Value>,
    breakpoints: Vec<Value>,
}

fn threshold_payload(rg: &ReportGraph<'_>) -> Option<ThresholdPayload> {
    let n = rg.graph.n;
    let tgraph = ThresholdGraph::from_edges(n, rg.affinity);
    let computed;
    let stability = match rg.stability {
        Some(s) => s,
        None => {
            computed = find_stable_thresholds_from_edges(n, rg.affinity, DEFAULT_NUM_BINS).ok()?;
            &computed
        }
    };
    let thresholds = &stability.thresholds;
    let counts = &stability.component_counts;
    let plateaus = stability
        .plateaus
        .iter()
        .take(3)
        .map(|p| {
            let mid = p.midpoint();
            json!({
                "midpoint": mid,
                "start": p.start_threshold,
                "end": p.end_threshold,
                "component_count": p.component_count,
                "length": p.length(),
                "mass_profile": serde_json::to_value(component_mass_profile(&tgraph, mid, 5, 0.01)).unwrap_or(Value::Null),
            })
        })
        .collect();
    let ten: Vec<_> = stability.plateaus.iter().take(10).cloned().collect();
    let report_ready =
        agent_threshold_options(&tgraph, &ten, thresholds, counts, Policy::ReportReady, 1);
    let exploratory =
        agent_threshold_options(&tgraph, &ten, thresholds, counts, Policy::DetailSeeking, 1);
    let mut candidates: Vec<Value> = vec![];
    let mut seen: Vec<f64> = vec![];
    let pool = report_ready
        .stable_plateau_candidates
        .iter()
        .map(|c| serde_json::to_value(c).unwrap_or(Value::Null))
        .chain(
            exploratory
                .transition_adjacent_candidates
                .iter()
                .map(|c| serde_json::to_value(c).unwrap_or(Value::Null)),
        );
    for c in pool {
        let t = round_to(num(c.get("threshold")), 8);
        if seen.contains(&t) {
            continue;
        }
        candidates.push(c);
        seen.push(t);
        if candidates.len() >= 2 {
            break;
        }
    }
    let breakpoints = structural_breakpoints(&tgraph, thresholds, counts, 4, 16)
        .iter()
        .map(|b| serde_json::to_value(b).unwrap_or(Value::Null))
        .collect();
    Some(ThresholdPayload {
        resolved: rg.resolved_threshold,
        auto: stability.optimal_threshold,
        plateaus,
        candidates,
        breakpoints,
    })
}

fn threshold_axis(p: &ThresholdPayload) -> String {
    let (width, height) = (920.0, 180.0);
    let (axis_start, axis_end) = (70.0, width - 70.0);
    let x_pos = |t: f64| axis_start + (1.0 - t) * (axis_end - axis_start);
    let auto_x = x_pos(p.auto);
    let current_x = x_pos(p.resolved);
    let mut svg = vec![
        format!("<svg class='threshold-svg' viewBox='0 0 {} {}' role='img' aria-label='Threshold transition map'>", width as i64, height as i64),
        "<defs>".to_owned(),
        "<linearGradient id='thresholdAxisGradient' x1='0' x2='1' y1='0' y2='0'>".into(),
        "<stop offset='0%' stop-color='#1f1f1f' stop-opacity='0.82' />".into(),
        "<stop offset='55%' stop-color='#1a73e8' stop-opacity='0.62' />".into(),
        "<stop offset='100%' stop-color='#147d64' stop-opacity='0.72' />".into(),
        "</linearGradient>".into(),
        "</defs>".into(),
        format!("<line x1='{0}' y1='92' x2='{1}' y2='92' stroke='url(#thresholdAxisGradient)' stroke-width='5' stroke-linecap='round' />", axis_start as i64, axis_end as i64),
        format!("<text x='{}' y='130' text-anchor='start' class='threshold-axis-label'>Stricter: splits appear</text>", axis_start as i64),
        format!("<text x='{}' y='130' text-anchor='end' class='threshold-axis-label'>Looser: components merge</text>", axis_end as i64),
        format!("<line x1='{auto_x:.1}' y1='38' x2='{auto_x:.1}' y2='116' class='threshold-marker threshold-marker--auto' />"),
        format!("<circle cx='{auto_x:.1}' cy='38' r='6' class='threshold-dot threshold-dot--auto' />"),
        format!("<text x='{auto_x:.1}' y='24' text-anchor='middle' class='threshold-marker-label'>auto {}</text>", fixed(p.auto, 3)),
    ];
    // np.isclose(auto_x, current_x)
    if (auto_x - current_x).abs() > 1e-8 + 1e-5 * current_x.abs() {
        svg.extend([
            format!("<line x1='{current_x:.1}' y1='54' x2='{current_x:.1}' y2='118' class='threshold-marker threshold-marker--current' />"),
            format!("<circle cx='{current_x:.1}' cy='54' r='5' class='threshold-dot threshold-dot--current' />"),
            format!("<text x='{current_x:.1}' y='150' text-anchor='middle' class='threshold-marker-label'>current {}</text>", fixed(p.resolved, 3)),
        ]);
    }
    for bp in &p.breakpoints {
        let threshold = num(bp.get("threshold"));
        let mass = match bp.get("absorbed_mass_fraction") {
            Some(v) if !v.is_null() => num(Some(v)),
            _ => num(bp.get("affected_mass_fraction")),
        };
        let marker_x = x_pos(threshold);
        let top_y = 92.0 - (18.0 + 42.0 * mass.min(1.0));
        svg.extend([
            format!("<line x1='{marker_x:.1}' y1='{top_y:.1}' x2='{marker_x:.1}' y2='92' class='threshold-split-line' />"),
            format!("<circle cx='{marker_x:.1}' cy='{top_y:.1}' r='{:.1}' class='threshold-split-dot'>", 4.0 + 7.0 * mass.min(1.0)),
            format!("<title>{} of rows affected near threshold {}</title>", percent(mass, 0), fixed(threshold, 3)),
            "</circle>".into(),
        ]);
    }
    svg.push("</svg>".into());
    svg.concat()
}

fn render_threshold_section(rg: &ReportGraph<'_>) -> String {
    let Some(p) = threshold_payload(rg) else {
        return String::new();
    };
    let mut parts = vec![
        "<div class='threshold-panel'>".to_owned(),
        "<div class='threshold-panel__header'>".into(),
        "<div>".into(),
        "<div class='figure-title'>Threshold transition map</div>".into(),
        "<p class='figure-subtitle'>Large dots mark thresholds where substantial graph mass changes component membership.</p>".into(),
        "</div>".into(),
        "</div>".into(),
        threshold_axis(&p),
        "<div class='threshold-explainer'>".into(),
        "<strong>Auto threshold, simply:</strong> Pulsar sweeps possible edge thresholds and picks the midpoint of the longest flat region where the number of connected components does not change. That favors a stable graph cut, not a guaranteed “best” clustering.".into(),
        "</div>".into(),
    ];
    if p.candidates.is_empty() {
        parts.push("<div class='empty-state'>No report-ready or exploratory threshold alternatives were detected.</div>".into());
    } else {
        parts.push("<div class='threshold-breakpoints'>".into());
        for c in &p.candidates {
            let direct = c
                .get("top_component_sizes")
                .filter(|v| v.as_array().is_some_and(|a| !a.is_empty()));
            let empty = json!([]);
            let sizes = direct.unwrap_or_else(|| {
                c.get("component_mass_profile")
                    .and_then(|m| m.get("top_component_sizes"))
                    .unwrap_or(&empty)
            });
            let tier_raw = py_str(c.get("interpretability_tier").unwrap_or(&Value::Null));
            let mut class = "threshold-breakpoint-card".to_owned();
            if tier_raw != "report_ready" && tier_raw != "balanced" {
                class.push_str(" threshold-breakpoint-card--exploratory");
            }
            parts.push(format!(
                "<article class='{class}'><div class='threshold-card-kicker'>{}</div><h3>Threshold {}</h3><p>{} <span>top component sizes</span></p><small>{}</small></article>",
                esc(&tier_raw.replace('_', " ")),
                esc(&format_value(c.get("threshold"), 3, "N/A")),
                esc(&component_sizes_text(sizes)),
                esc_value(c.get("why").unwrap_or(&Value::Null)),
            ));
        }
        parts.push("</div>".into());
    }
    if let Some(best) = p.plateaus.first() {
        parts.push(format!(
            "<p class='figure-caption'><strong>Suggested cut.</strong> The auto threshold is {}, inside a plateau spanning {} to {}. At that cut, top component sizes are {}. Treat giant-dominated cuts as stable diagnostics, not final cohort cuts.</p>",
            esc(&fixed(p.auto, 3)),
            esc(&format_value(best.get("start"), 3, "N/A")),
            esc(&format_value(best.get("end"), 3, "N/A")),
            esc(&component_sizes_text(&best["mass_profile"]["top_component_sizes"])),
        ));
    }
    parts.push("</div>".into());
    parts.concat()
}

fn render_topology(
    d: &TopologicalDossier,
    labels: &[i64],
    graph: Option<&ReportGraph<'_>>,
    palette: &BTreeMap<i64, &str>,
) -> String {
    let mut parts = vec![
        "<section id='topology' class='report-section'>".to_owned(),
        "<div class='section-header'>".into(),
        "<div>".into(),
        "<span class='section-eyebrow'>Figure 1</span>".into(),
        "<h2>Topological skeleton projection</h2>".into(),
        "</div>".into(),
        "<p class='section-subtitle'>Sampled from the cosmic graph for responsiveness. Cohort colors stay consistent across the full report.</p>".into(),
        "</div>".into(),
        "<div class='figure-frame'>".into(),
        "<div class='figure-toolbar'>".into(),
        "<div>".into(),
        "<div class='figure-title'>Global manifold view</div>".into(),
        "<p class='figure-subtitle'>Edges are intentionally quiet. The emphasis stays on node structure and cohort support.</p>".into(),
        "</div>".into(),
        "<div class='legend'>".into(),
    ];
    for p in &d.clusters {
        parts.push(format!(
            "<span class='legend-chip' style='--cluster-accent:{}'>C{} · {}</span>",
            palette[&p.cluster_id],
            cid(p.cluster_id),
            esc(&p.semantic_name)
        ));
    }
    parts.extend([
        "</div>".into(),
        "</div>".into(),
        "<div class='graph-wrapper'>".into(),
    ]);
    match graph {
        Some(rg) => {
            parts.push(cosmic_graph_svg(rg.graph, labels, palette));
            parts.push("<p class='figure-caption'><strong>Figure note.</strong> Nodes are sampled manifold landmarks. Edges encode topological consensus. Hover a cohort row below to isolate its sampled support.</p>".into());
            parts.push(render_threshold_section(rg));
        }
        None => parts
            .push("<div class='empty-state'>Graph view unavailable for this report.</div>".into()),
    }
    parts.extend(["</div>".into(), "</div>".into(), "</section>".into()]);
    parts.concat()
}

/// The top `limit` names by score, descending, ties by name.
fn top_by_score(scores: Vec<(String, f64)>, limit: usize) -> Vec<String> {
    let mut v = scores;
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v.into_iter().take(limit).map(|(n, _)| n).collect()
}

fn accumulate(scores: &mut Vec<(String, f64)>, name: String, add: f64) {
    match scores.iter_mut().find(|(n, _)| *n == name) {
        Some((_, s)) => *s += add,
        None => scores.push((name, add)),
    }
}

fn render_heatmap(d: &TopologicalDossier) -> String {
    let mut numeric_scores = vec![];
    let mut categorical_scores = vec![];
    for p in &d.clusters {
        for f in &p.numeric_features {
            accumulate(
                &mut numeric_scores,
                py_str(&f["column"]),
                num(f.get("z_score").or(Some(&json!(0.0)))).abs(),
            );
        }
        for f in &p.categorical_features {
            let score = prevalence(f) + 0.25 * num(f.get("global_recall").or(Some(&json!(0.0))));
            accumulate(&mut categorical_scores, py_str(&f["column"]), score);
        }
    }
    let numeric = top_by_score(numeric_scores, MAX_SIGNAL_MATRIX_NUMERIC);
    let categorical = top_by_score(categorical_scores, MAX_SIGNAL_MATRIX_CATEGORICAL);
    if numeric.is_empty() && categorical.is_empty() {
        return String::new();
    }
    let mut parts = vec![
        "<section id='heatmap' class='report-section'>".to_owned(),
        "<div class='section-header'>".into(),
        "<div>".into(),
        "<span class='section-eyebrow'>Figure 2</span>".into(),
        "<h2>Cohort signal matrix</h2>".into(),
        "</div>".into(),
        format!("<p class='section-subtitle'>Bounded for readability: up to {MAX_SIGNAL_MATRIX_NUMERIC} numeric features and {MAX_SIGNAL_MATRIX_CATEGORICAL} categorical columns. Numeric values are cohort means, and the smaller line below each numeric value is the z-shift.</p>"),
        "</div>".into(),
        "<div class='heatmap-shell'>".into(),
        "<table class='heatmap-table'><thead>".into(),
        "<tr>".into(),
        "<th rowspan='2'>Cohort</th>".into(),
    ];
    if !numeric.is_empty() {
        parts.push(format!(
            "<th colspan='{}'>Numeric means</th>",
            numeric.len()
        ));
    }
    if !categorical.is_empty() {
        parts.push(format!(
            "<th colspan='{}'>Dominant categories</th>",
            categorical.len()
        ));
    }
    parts.push("</tr><tr>".into());
    for name in numeric.iter().chain(&categorical) {
        parts.push(format!("<th>{}</th>", esc(name)));
    }
    parts.push("</tr></thead><tbody>".into());
    for p in &d.clusters {
        parts.push(format!(
            "<tr data-cluster-id='{}'><td><div><strong>C{} — {}</strong></div><div class='cluster-card__meta'>n = {} · {}%</div></td>",
            p.cluster_id,
            cid(p.cluster_id),
            esc(&p.semantic_name),
            commas(p.size as i64),
            fixed(p.size_pct, 1)
        ));
        for name in &numeric {
            let Some(f) = p
                .numeric_features
                .iter()
                .rev()
                .find(|f| py_str(&f["column"]) == *name)
            else {
                parts
                    .push("<td class='heat-cell'><span class='heat-subvalue'>—</span></td>".into());
                continue;
            };
            let z = num(f.get("z_score").or(Some(&json!(0.0))));
            let alpha = if (z.abs() * 0.14) < 0.38 {
                z.abs() * 0.14
            } else {
                0.38
            };
            let tone = if z > 0.5 {
                "heat-positive"
            } else if z < -0.5 {
                "heat-negative"
            } else {
                ""
            };
            let style = if tone.is_empty() {
                String::new()
            } else {
                format!(" style='--heat-alpha:{}'", fixed(alpha, 2))
            };
            parts.push(format!(
                "<td class='heat-cell {tone}'{style}><span class='heat-value'>{}</span><span class='heat-subvalue'>z {}</span></td>",
                esc(&heatmap_value(f.get("mean"))),
                signed(z, 2)
            ));
        }
        for name in &categorical {
            let column: Vec<&Value> = p
                .categorical_features
                .iter()
                .filter(|f| py_str(&f["column"]) == *name)
                .collect();
            let no_match = "<td class='heat-cell'><span class='heat-value'>—</span><span class='heat-subvalue'>no match</span></td>";
            let Some(&first) = column.first() else {
                parts.push(no_match.into());
                continue;
            };
            let mut best = first;
            for &f in &column[1..] {
                let key = |f: &Value| (prevalence(f), int_field(f, "count", 0));
                let (bp, bc) = key(best);
                let (fp, fc) = key(f);
                if fp > bp || (fp == bp && fc > bc) {
                    best = f;
                }
            }
            let prev = prevalence(best);
            let count = int_field(best, "count", 0);
            if count <= 0 || prev <= 0.0 {
                parts.push(no_match.into());
                continue;
            }
            let alpha = if prev * 0.0038 < 0.38 {
                prev * 0.0038
            } else {
                0.38
            };
            let tone = if prev >= 50.0 { "heat-positive" } else { "" };
            let style = if prev > 0.0 {
                format!(" style='--heat-alpha:{}'", fixed(alpha, 2))
            } else {
                String::new()
            };
            parts.push(format!(
                "<td class='heat-cell {tone}'{style}><span class='heat-value'>{}</span><span class='heat-subvalue'>{count}/{} · {}%</span></td>",
                esc_value(&best["value"]),
                p.size,
                fixed(prev, 1)
            ));
        }
        parts.push("</tr>".into());
    }
    parts.extend([
        "</tbody></table>".into(),
        "</div>".into(),
        "</section>".into(),
    ]);
    parts.concat()
}

fn render_numeric_panel(p: &ClusterProfile) -> String {
    let mut parts = vec![
        "<section class='panel'>".to_owned(),
        "<h3>Dominant numeric signals</h3>".into(),
        "<p class='panel-note'>Top features are ranked by shift magnitude adjusted for within-cohort homogeneity.</p>".into(),
        "<table class='data-table'><thead><tr><th>Feature</th><th>Distribution</th><th>Mean</th><th>Z-score</th><th>Signal</th></tr></thead><tbody>".into(),
    ];
    for f in &p.numeric_features {
        let z = num(f.get("z_score"));
        let (label, tone) = signal_tone(z);
        parts.push(format!(
            "<tr><td><strong>{}</strong></td><td class='sparkline'>{}</td><td><code>{}</code></td><td><code>{}</code></td><td><span class='signal-badge {tone}'>{label}</span></td></tr>",
            esc_value(&f["column"]),
            esc_value(f.get("sparkline").unwrap_or(&json!(""))),
            fixed(num(f.get("mean")), 3),
            signed(z, 2)
        ));
    }
    parts.extend(["</tbody></table>".into(), "</section>".into()]);
    parts.concat()
}

fn render_categorical_panel(p: &ClusterProfile, accent: &str) -> String {
    if p.categorical_features.is_empty() {
        return "<section class='panel'><h3>Categorical concentrations</h3><div class='empty-state'>No categorical enrichment was captured for this cohort.</div></section>".into();
    }
    let mut parts = vec![
        "<section class='panel'>".to_owned(),
        "<h3>Categorical concentrations</h3>".into(),
        "<p class='panel-note'>Prevalence reports what share of this cohort has the value. Global recall reports how much of the overall value population this cohort captures.</p>".into(),
        "<table class='data-table'><thead><tr><th>Feature</th><th>Value</th><th>Prevalence</th><th>Global recall</th></tr></thead><tbody>".into(),
    ];
    for f in &p.categorical_features {
        let prev = prevalence(f);
        let recall = num(f.get("global_recall").or(Some(&json!(0.0))));
        let count = int_field(f, "count", 0);
        let cluster_size = int_field(f, "cluster_size", p.size as i64);
        let global_count = int_field(f, "global_count", count);
        parts.push(format!(
            "<tr><td>{}</td><td><code>{}</code></td><td>{}% <span class='cluster-card__meta'>({count}/{cluster_size})</span><div class='bar-track' style='margin-top:0.45rem;'><div class='bar-fill' style='width:{}%; --cluster-accent:{accent};'></div></div></td><td>{}% <span class='cluster-card__meta'>({count}/{global_count})</span></td></tr>",
            esc_value(&f["column"]),
            esc_value(&f["value"]),
            fixed(prev, 1),
            fixed(prev, 2),
            fixed(recall, 1)
        ));
    }
    parts.extend(["</tbody></table>".into(), "</section>".into()]);
    parts.concat()
}

fn render_instances_panel(p: &ClusterProfile) -> String {
    let mut parts = vec![
        "<section class='panel'>".to_owned(),
        "<h3>Representative instances</h3>".into(),
        "<p class='panel-note'>Rows chosen from the cohort core so the values stay close to the local manifold center.</p>".into(),
    ];
    if p.central_rows.is_empty() {
        parts.push("<div class='empty-state'>No representative rows available for this cohort.</div></section>".into());
        return parts.concat();
    }
    parts.push("<div class='instances-grid'>".into());
    for (index, row) in p.central_rows.iter().enumerate() {
        let details: String = row
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, v)| {
                // A missing cell reached the dossier as NaN.
                format!(
                    "<span class='detail-chip'><strong>{}</strong>: {}</span>",
                    esc(k),
                    esc(&format_value(Some(v), 3, "NaN"))
                )
            })
            .collect();
        parts.push(format!(
            "<article class='instance-card'><div class='instance-label'>Example {:02}</div><div class='instance-details'>{details}</div></article>",
            index + 1
        ));
    }
    parts.extend(["</div>".into(), "</section>".into()]);
    parts.concat()
}

fn render_cluster_section(p: &ClusterProfile, accent: &str) -> String {
    let traits: String = p
        .numeric_features
        .iter()
        .take(5)
        .map(|f| {
            format!(
                "<span class='trait-chip'>{}</span>",
                esc_value(&f["column"])
            )
        })
        .collect();
    [
        format!("<section id='cluster-{0}' class='report-section cluster-section' data-cluster-id='{0}' style='--cluster-accent:{accent}'>", p.cluster_id),
        "<div class='cluster-section__heading'>".into(),
        "<div class='cluster-section__topline'>".into(),
        format!("<span class='cluster-id'>C{}</span>", cid(p.cluster_id)),
        format!("<span class='legend-chip' style='--cluster-accent:{accent}'>n = {} · {}%</span>", commas(p.size as i64), fixed(p.size_pct, 1)),
        "</div>".into(),
        format!("<h2>C{} — {}</h2>", cid(p.cluster_id), esc(&p.semantic_name)),
        format!(
            "<p class='cluster-section__summary'>This cohort contains <strong>{}</strong> observations and represents <strong>{}%</strong> of the analyzed population. Treat the numeric and categorical tables below as the evidence sheet for this slice of the manifold.</p>",
            commas(p.size as i64),
            fixed(p.size_pct, 1)
        ),
        format!("<div class='cluster-section__traits'>{traits}</div>"),
        "</div>".into(),
        "<div class='cluster-detail-grid'>".into(),
        render_numeric_panel(p),
        "<div class='cluster-secondary-stack'>".into(),
        render_categorical_panel(p, accent),
        render_instances_panel(p),
        "</div>".into(),
        "</div>".into(),
        "</section>".into(),
    ]
    .concat()
}

fn render_config_appendix(config_yaml: Option<&str>) -> String {
    let Some(yaml) = config_yaml.filter(|y| !y.is_empty()) else {
        return String::new();
    };
    [
        "<section id='config' class='report-section config-appendix'>".to_owned(),
        "<div class='section-header'>".into(),
        "<div>".into(),
        "<span class='section-eyebrow'>Appendix</span>".into(),
        "<h2>Run parameters</h2>".into(),
        "</div>".into(),
        "<p class='section-subtitle'>Normalized params.yaml used for the exported run.</p>".into(),
        "</div>".into(),
        "<div class='config-toolbar'>".into(),
        "<p>Readonly, formatted for direct copy and reuse.</p>".into(),
        "<button type='button' class='copy-button' data-copy-target='paramsYaml'>Copy YAML</button>".into(),
        "</div>".into(),
        format!("<textarea id='paramsYaml' class='config-textarea' readonly spellcheck='false'>{}</textarea>", esc(yaml)),
        "</section>".into(),
    ]
    .concat()
}

/// `dossier_to_html`: the report as one standalone HTML document. `labels`
/// are the per-row cluster labels; without `graph` the figure is omitted.
pub fn dossier_to_html(
    dossier: &TopologicalDossier,
    labels: &[i64],
    graph: Option<&ReportGraph<'_>>,
    config_yaml: Option<&str>,
) -> String {
    let empty = json!({});
    let gm = dossier.global_stats.get("graph_metrics").unwrap_or(&empty);
    let palette: BTreeMap<i64, &str> = dossier
        .clusters
        .iter()
        .enumerate()
        .map(|(i, p)| (p.cluster_id, PALETTE[i % PALETTE.len()]))
        .collect();
    let mut sections = vec![
        render_summary(dossier, gm),
        render_topology(dossier, labels, graph, &palette),
        render_heatmap(dossier),
    ];
    sections.extend(
        dossier
            .clusters
            .iter()
            .map(|p| render_cluster_section(p, palette[&p.cluster_id])),
    );
    sections.push(render_config_appendix(config_yaml));

    let mut lines: Vec<String> = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>Pulsar Topological Dossier</title>",
        FONTS,
        "<style>",
        STYLES,
        "</style>",
        "</head>",
        "<body>",
        "<div class='report-shell'>",
        "<main class='report-main'>",
        "<div class='report-stack'>",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    lines.extend(sections.into_iter().filter(|s| !s.is_empty()));
    lines.extend(
        ["</div>", "</main>"]
            .iter()
            .map(|s| s.to_string())
            .chain([render_nav(dossier)])
            .chain(
                [
                    "</div>",
                    "<script>",
                    SCRIPT,
                    "</script>",
                    "</body>",
                    "</html>",
                ]
                .iter()
                .map(|s| s.to_string()),
            ),
    );
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_number_formatting() {
        assert_eq!(commas(1234567), "1,234,567");
        assert_eq!(commas(-1234), "-1,234");
        assert_eq!(grouped(12345.678, 1), "12,345.7");
        assert_eq!(grouped(-1500.0, 0), "-1,500");
        assert_eq!(signed(0.0, 2), "+0.00");
        assert_eq!(signed(f64::NAN, 2), "+nan");
        assert_eq!(percent(0.256, 0), "26%");
        assert_eq!(
            esc("a<'b'>&\"c\""),
            "a&lt;&#x27;b&#x27;&gt;&amp;&quot;c&quot;"
        );
        assert_eq!(format_value(Some(&json!(true)), 3, "N/A"), "1");
        assert_eq!(format_value(Some(&json!(2.0)), 3, "N/A"), "2.000");
        assert_eq!(format_value(None, 3, "N/A"), "N/A");
        assert_eq!(heatmap_value(Some(&json!(1234.5))), "1,234");
        assert_eq!(cid(3), "03");
    }
}
