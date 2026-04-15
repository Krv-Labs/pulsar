"""
Topological Interpreter Engine.

Translates the raw topological graph and clustered data into a high-signal
statistical dossier for LLM synthesis.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from pulsar._pulsar import find_stable_thresholds
from pulsar.pipeline import ThemaRS
from pulsar.runtime.utils import generate_distribution_sparkline, generate_proportion_bar
from pulsar.mcp.diagnostics import diagnose_model

logger = logging.getLogger(__name__)

# Clustering strategy constants
_MAX_COMPONENTS = 30  # Use component strategy if fewer than this
_MAX_SINGLETON_RATIO = 0.5  # Reject if >50% of nodes are singletons
_SPECTRAL_K_MIN = 2
_SPECTRAL_K_MAX = 20


@dataclass
class ClusterProfile:
    """Statistical profile of a single topological cluster."""

    cluster_id: int
    size: int
    size_pct: float
    semantic_name: str = "Unknown Cluster"
    numeric_features: List[Dict[str, Any]] = field(default_factory=list)
    categorical_features: List[Dict[str, Any]] = field(default_factory=list)
    central_rows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TopologicalDossier:
    """Complete collection of cluster profiles and global context."""

    n_total: int
    n_clusters: int
    clusters: List[ClusterProfile]
    global_stats: Dict[str, Any]
    cluster_labels: pd.Series | None = None


@dataclass
class ClusterResult:
    """Clustering result with provenance metadata."""

    labels: pd.Series
    method_used: str  # "threshold_stability" | "components" | "spectral"
    n_clusters: int
    silhouette_score: float | None
    failure_reason: str | None
    edge_weight_threshold_applied: float = 0.0
    stability_plateaus: list[dict] | None = None


def resolve_clusters(
    model: ThemaRS,
    method: str = "auto",
    max_k: int = 15,
    edge_weight_threshold: float = 0.0,
) -> ClusterResult:
    """Entry point for clustering. Supports auto-thresholding and spectral fallback."""
    W = model.weighted_adjacency
    n = W.shape[0]

    # 1. Component Strategy (if very obvious gaps)
    if edge_weight_threshold > 0:
        adj = (W > edge_weight_threshold).astype(np.int64)
        G = nx.from_numpy_array(adj)
        labels = np.zeros(n, dtype=int)
        comps = list(nx.connected_components(G))
        for i, comp in enumerate(comps):
            for node in comp:
                labels[node] = i
        return ClusterResult(
            labels=pd.Series(labels, name="cluster"),
            method_used="components",
            n_clusters=len(comps),
            silhouette_score=None,
            failure_reason=None,
            edge_weight_threshold_applied=edge_weight_threshold,
        )

    # 2. Threshold Stability (PH-based)
    if method in ("auto", "threshold_stability"):
        result = _cluster_by_threshold_stability(W, n)
        if result:
            return result

    # 3. Spectral Fallback
    if method in ("auto", "spectral"):
        try:
            return _cluster_spectral(W, n, max_k)
        except Exception as e:
            return ClusterResult(
                labels=pd.Series(np.zeros(n, dtype=int), name="cluster"),
                method_used="none",
                n_clusters=1,
                silhouette_score=None,
                failure_reason=str(e),
            )

    raise ValueError(f"Unknown clustering method: {method}")


def _cluster_by_threshold_stability(adj: np.ndarray, n: int) -> ClusterResult | None:
    """Find clusters via H0 persistent homology on edge weights."""
    stability = find_stable_thresholds(adj)
    plateau_dicts = [
        {
            "start": float(p.start_threshold),
            "end": float(p.end_threshold),
            "component_count": int(p.component_count),
            "length": float(p.length),
        }
        for p in stability.top_k_plateaus(5)
    ]

    for plateau in stability.plateaus:
        comp_count = int(plateau.component_count)
        if comp_count <= 1 or comp_count >= _MAX_COMPONENTS:
            continue

        # Check singleton ratio
        thresh = float(plateau.midpoint)
        binary_adj = (adj > thresh).astype(np.int64)
        G = nx.from_numpy_array(binary_adj)
        singletons = sum(1 for node in G.nodes() if G.degree(node) == 0)
        if (singletons / n) > _MAX_SINGLETON_RATIO:
            continue

        # Success: valid stable split
        labels = np.zeros(n, dtype=int)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        for i, comp in enumerate(comps):
            for node in comp:
                labels[node] = i

        return ClusterResult(
            labels=pd.Series(labels, name="cluster"),
            method_used="threshold_stability",
            n_clusters=len(comps),
            silhouette_score=None,
            failure_reason=None,
            edge_weight_threshold_applied=thresh,
            stability_plateaus=plateau_dicts,
        )
    return None


def _cluster_spectral(adj: np.ndarray, n: int, max_k: int) -> ClusterResult:
    """Fall back to spectral clustering if no stable threshold split exists."""
    # Distance = 1 - Affinity
    affinity = adj.copy()
    np.fill_diagonal(affinity, 1.0)
    distance = 1.0 - affinity

    best_score = -1.0
    best_labels = None
    best_k = _SPECTRAL_K_MIN
    k_range = range(_SPECTRAL_K_MIN, min(max_k + 1, n))
    scores_by_k = {}

    for k_test in k_range:
        sc = SpectralClustering(
            n_clusters=k_test,
            affinity="precomputed",
            assign_labels="discretize",
            random_state=42,
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                labels = sc.fit_predict(affinity)
        except Exception:
            continue

        if len(np.unique(labels)) > 1:
            score = float(silhouette_score(distance, labels, metric="precomputed"))
            scores_by_k[k_test] = score
            if score > best_score:
                best_score = score
                best_k = k_test
                best_labels = labels

    if best_labels is not None and best_score > 0.05:
        return ClusterResult(
            labels=pd.Series(best_labels, name="cluster"),
            method_used="spectral",
            n_clusters=best_k,
            silhouette_score=best_score,
            failure_reason=None,
        )

    raise ValueError("No stable cluster cut found.")


def build_dossier(
    model: ThemaRS, data: pd.DataFrame, clusters: pd.Series, exclude_columns: list[str] | None = None
) -> TopologicalDossier:
    """Computes per-cluster statistical profiles."""
    if len(clusters) != len(data):
        raise ValueError(f"Alignment error: clusters({len(clusters)}) != data({len(data)})")

    if exclude_columns:
        to_drop = [c for c in exclude_columns if c in data.columns]
        if to_drop:
            data = data.drop(columns=to_drop)
            logger.info("build_dossier: excluded columns from report: %s", to_drop)

    n_total = len(data)
    n_clusters = len(clusters.unique())
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    global_means = data[numeric_cols].mean()
    global_vars = data[numeric_cols].var().replace(0, 1e-10)

    cluster_profiles = []
    for cid in sorted(clusters.unique()):
        c_mask = clusters == cid
        c_data = data[c_mask]
        c_size = len(c_data)
        c_means = c_data[numeric_cols].mean()

        profile = ClusterProfile(cluster_id=int(cid), size=c_size, size_pct=(c_size / n_total) * 100)
        c_numeric = []
        for col in numeric_cols:
            c_mean = c_means[col]
            g_mean = global_means[col]
            g_std = np.sqrt(global_vars[col])
            z_score = (c_mean - g_mean) / g_std if g_std > 0 else 0.0
            homogeneity = c_data[col].std() / g_std if g_std > 0 else 0.0
            importance = abs(z_score) / (homogeneity + 0.1)
            spark = generate_distribution_sparkline(c_data[col].dropna())

            c_numeric.append({
                "column": col, "mean": float(c_mean), "global_mean": float(g_mean),
                "z_score": float(z_score), "homogeneity": float(homogeneity),
                "importance": float(importance), "sparkline": spark
            })

        profile.numeric_features = sorted(c_numeric, key=lambda x: x["importance"], reverse=True)[:10]

        # Semantic Naming (Fallbacks)
        if len(profile.numeric_features) >= 2:
            f1, f2 = profile.numeric_features[0], profile.numeric_features[1]
            def desc(f):
                adj = "High" if f["z_score"] > 0.5 else "Low" if f["z_score"] < -0.5 else "Normal"
                return f"{adj} {f['column']}"
            profile.semantic_name = f"[Auto] {desc(f1)}, {desc(f2)}"
        elif len(profile.numeric_features) == 1:
            f1 = profile.numeric_features[0]
            adj = "High" if f1["z_score"] > 0.5 else "Low" if f1["z_score"] < -0.5 else "Normal"
            profile.semantic_name = f"[Auto] {adj} {f1['column']}"
        else:
            profile.semantic_name = f"Cluster {cid}"

        # Categorical
        c_categorical = []
        for col in categorical_cols:
            counts = c_data[col].value_counts()
            global_counts = data[col].value_counts()
            for val, count in counts.head(5).items():
                concentration = (count / global_counts[val]) * 100
                c_categorical.append({
                    "column": col, "value": str(val), "count": int(count), "concentration": float(concentration)
                })
        profile.categorical_features = sorted(c_categorical, key=lambda x: x["concentration"], reverse=True)[:10]

        # Central Rows (Compressed)
        try:
            sub_nodes = list(np.where(clusters.values == cid)[0])
            sub_graph = model.cosmic_graph.subgraph(sub_nodes)
            if len(sub_graph) > 0:
                pagerank = nx.pagerank(sub_graph, weight="weight")
                central_ids = sorted(pagerank, key=pagerank.get, reverse=True)[:3]
                top_cols = [f["column"] for f in profile.numeric_features[:10]]
                if profile.categorical_features:
                    top_cols += [f["column"] for f in profile.categorical_features[:5]]
                top_cols = list(set(top_cols)) if top_cols else data.columns[:10].tolist()
                profile.central_rows = data.iloc[central_ids][top_cols].to_dict("records")
        except Exception:
            top_cols = [f["column"] for f in profile.numeric_features[:10]]
            top_cols = list(set(top_cols)) if top_cols else data.columns[:10].tolist()
            profile.central_rows = c_data[top_cols].head(3).to_dict("records")

        cluster_profiles.append(profile)

    try:
        from dataclasses import asdict
        graph_metrics = asdict(diagnose_model(model))
    except Exception:
        graph_metrics = {}

    return TopologicalDossier(
        n_total=n_total, n_clusters=n_clusters, clusters=cluster_profiles,
        global_stats={"numeric": global_means.to_dict(), "columns": data.columns.tolist(), "graph_metrics": graph_metrics},
        cluster_labels=clusters
    )


def _generate_cosmic_graph_svg(model: ThemaRS, width: int = 800, height: int = 400, cluster_labels: pd.Series | None = None) -> str:
    """Generates an optimized Google Research style SVG with topological coverage sampling."""
    G = model.cosmic_graph
    if G.number_of_nodes() == 0: return ""
    
    budget = 1000
    if G.number_of_nodes() > budget:
        sampled = set()
        if cluster_labels is not None:
            for cid in sorted(cluster_labels.unique()):
                c_mask = cluster_labels == cid
                c_nodes = cluster_labels.index[c_mask].tolist()
                c_nodes = [n for n in c_nodes if G.has_node(n)]
                if not c_nodes: continue
                n_sample = max(10, int(len(c_nodes) * 0.05))
                sampled.update(sorted(c_nodes, key=lambda n: G.degree(n), reverse=True)[:n_sample])
        
        remaining = budget - len(sampled)
        if remaining > 0:
            global_sorted = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
            for n in global_sorted:
                if n not in sampled:
                    sampled.add(n)
                    remaining -= 1
                if remaining <= 0: break
        G_sub = G.subgraph(sampled)
    else:
        G_sub = G

    pos = nx.spring_layout(G_sub, seed=42, k=1.5/np.sqrt(max(1, G_sub.number_of_nodes())))
    x_vals = [p[0] for p in pos.values()]; y_vals = [p[1] for p in pos.values()]
    if not x_vals: return ""
    xmin, xmax, ymin, ymax = min(x_vals), max(x_vals), min(y_vals), max(y_vals)
    padding = 40
    def sc(v, v_min, v_max, t_max):
        return padding + (v - v_min) / (v_max - v_min + 1e-9) * (t_max - 2 * padding)

    svg = [f'<svg id="cosmicGraph" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#fff; width:100%; height:auto;">']
    opacity_bands = {}
    for u, v, data in G_sub.edges(data=True):
        x1, y1 = sc(pos[u][0], xmin, xmax, width), sc(pos[u][1], ymin, ymax, height)
        x2, y2 = sc(pos[v][0], xmin, xmax, width), sc(pos[v][1], ymin, ymax, height)
        band = round(max(0.1, 0.1 + data.get("weight", 0.1) * 0.3), 1)
        opacity_bands.setdefault(band, []).append(f"M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f}")
    
    for op, segs in opacity_bands.items():
        svg.append(f'<path d="{" ".join(segs)}" stroke="#dadce0" stroke-width="0.5" stroke-opacity="{op}" fill="none" />')
    
    for node in G_sub.nodes():
        x, y = sc(pos[node][0], xmin, xmax, width), sc(pos[node][1], ymin, ymax, height)
        deg = G.degree(node)
        radius = 1.5 + (deg / max(1, G.number_of_nodes()) * 8)
        cid = str(cluster_labels[node]) if cluster_labels is not None and node in cluster_labels.index else ""
        svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="#4285F4" fill-opacity="0.6" class="graph-node" data-cluster="{cid}" style="transition: all 0.2s ease;"><title>Node {node} (deg: {deg}, cluster: {cid})</title></circle>')
    
    svg.append("</svg>")
    return "".join(svg)


def dossier_to_html(dossier: TopologicalDossier, model: ThemaRS | None = None) -> str:
    """Renders the dossier as a rich, high-fidelity HTML report with Google Research aesthetic."""
    gm = dossier.global_stats.get("graph_metrics", {})
    
    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>Pulsar Cosmic Dossier</title>",
        "<style>",
        """
        :root {
          --google-blue: #4285F4; --google-red: #EA4335; --google-yellow: #FBBC05; --google-green: #34A853;
          --text-primary: #202124; --text-secondary: #5f6368; --border-color: #dadce0; --bg-color: #ffffff;
          --surface-color: #f8f9fa;
          --font-main: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          --font-mono: ui-monospace, 'JetBrains Mono', 'Cascadia Code', 'Segoe UI Mono', Menlo, Monaco, Consolas, monospace;
        }
        html { scroll-behavior: smooth; }
        body { font-family: var(--font-main); background-color: var(--bg-color); color: var(--text-primary); margin: 0; display: flex; line-height: 1.6; }
        aside { width: 300px; height: 100vh; position: sticky; top: 0; background: var(--bg-color); border-right: 1px solid var(--border-color); padding: 2.5rem 1.5rem; box-sizing: border-box; overflow-y: auto; }
        main { flex: 1; padding: 4rem 5rem; max-width: 1100px; margin: 0; }
        h1, h2, h3, h4 { font-weight: 700; color: var(--text-primary); margin-top: 1.5em; letter-spacing: -0.02em; }
        h1 { font-size: 2.5rem; margin-top: 0; }
        nav ul { list-style: none; padding: 0; margin: 2rem 0; }
        nav li { margin-bottom: 0.25rem; }
        nav a { display: block; padding: 8px 12px; color: var(--text-secondary); text-decoration: none; font-size: 0.9rem; border-left: 3px solid transparent; transition: all 0.15s ease; border-radius: 0 4px 4px 0; }
        nav a:hover { background: var(--surface-color); color: var(--google-blue); }
        nav a.active { color: var(--google-blue); border-left-color: var(--google-blue); background: rgba(66, 133, 244, 0.05); font-weight: 600; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin: 3rem 0; }
        .stat-card { background: var(--bg-color); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border-color); }
        .stat-value { font-size: 1.75rem; font-weight: 700; color: var(--google-blue); font-family: var(--font-mono); margin-bottom: 0.25rem; }
        .stat-label { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
        table { width: 100%; border-collapse: collapse; margin: 1.5rem 0; border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden; }
        th { background: var(--surface-color); padding: 12px 16px; text-align: left; font-size: 0.75rem; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 2px solid var(--border-color); }
        td { padding: 16px; border-bottom: 1px solid var(--border-color); font-size: 0.95rem; }
        tr:last-child td { border-bottom: none; }
        .sparkline { font-family: var(--font-mono); color: var(--google-blue); font-size: 1.25rem; letter-spacing: -0.05em; line-height: 1; opacity: 0.8; }
        .bar-outer { background: #e8eaed; width: 80px; height: 6px; border-radius: 3px; display: inline-block; vertical-align: middle; margin-left: 10px; }
        .bar-inner { background: var(--google-blue); height: 100%; border-radius: 3px; }
        .tag { padding: 4px 10px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }
        .tag-pos { background: rgba(52, 168, 83, 0.1); color: var(--google-green); }
        .tag-neg { background: rgba(234, 67, 53, 0.1); color: var(--google-red); }
        .tag-neu { background: rgba(95, 99, 104, 0.1); color: var(--text-secondary); }
        .central-row { font-family: var(--font-mono); font-size: 0.8rem; padding: 1.25rem; background: var(--surface-color); border-radius: 8px; margin-bottom: 0.75rem; color: var(--text-secondary); border: 1px solid transparent; transition: border-color 0.2s; }
        .central-row:hover { border-color: var(--border-color); color: var(--text-primary); }
        code { font-family: var(--font-mono); background: var(--surface-color); padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.85em; color: var(--google-blue); }
        hr { border: 0; border-top: 1px solid var(--border-color); margin: 4rem 0; }
        .graph-container { border: 1px solid var(--border-color); border-radius: 12px; margin: 3rem 0; padding: 2.5rem; background: #ffffff; text-align: center; }
        .cluster-header { display: flex; align-items: baseline; gap: 1.5rem; border-bottom: 1px solid var(--border-color); padding-bottom: 1rem; margin-bottom: 2rem; }
        .cluster-name { font-size: 1.5rem; font-weight: 500; color: var(--text-secondary); }
        .hero-text { font-size: 1.25rem; color: var(--text-secondary); max-width: 800px; margin-bottom: 3rem; }
        section { scroll-margin-top: 2rem; }
        .highlight-row { background-color: transparent; cursor: pointer; transition: background-color 0.2s; }
        .highlight-row:hover { background-color: rgba(66, 133, 244, 0.05); }
        """,
        "</style>",
        "</head>",
        "<body>",
        "<aside>",
        "<div style='margin-bottom: 3rem;'><svg width='40' height='40' viewBox='0 0 40 40' fill='none' xmlns='http://www.w3.org/2000/svg'><circle cx='20' cy='20' r='18' stroke='#4285F4' stroke-width='4'/><path d='M20 10V30M10 20H30' stroke='#4285F4' stroke-width='4' stroke-linecap='round'/></svg><h2 style='margin-top: 1rem; font-size: 1.2rem;'>Pulsar Analysis</h2></div>",
        "<nav><ul>",
        "<li><a href='#summary' class='active'>Executive Summary</a></li>",
        "<li><a href='#heatmap'>Feature Drift</a></li>",
        "<li><a href='#clusters'>Cluster Matrix</a></li>",
        "<li><hr style='margin: 1.5rem 0;'></li>",
    ]

    for p in dossier.clusters:
        html.append(f"<li><a href='#cluster-{p.cluster_id}'>Cluster {p.cluster_id}</a></li>")
    
    html.extend([
        "</ul></nav>",
        "<div style='margin-top: auto; font-size: 0.7rem; color: var(--text-secondary); opacity: 0.6;'>Generated by Pulsar Topological Engine</div>",
        "</aside>",
        "<main>",
        "<section id='summary'>",
        "<h1>Cosmic Dossier</h1>",
        "<p class='hero-text'>Topological manifold decomposition revealing the latent structure of the dataset through multi-scale persistence aggregation.</p>",
        "<div class='stats-grid'>",
        f"<div class='stat-card'><div class='stat-value'>{dossier.n_total:,}</div><div class='stat-label'>Total Observations</div></div>",
        f"<div class='stat-card'><div class='stat-value'>{gm.get('n_nodes', 'N/A')}</div><div class='stat-label'>Graph Nodes</div></div>",
        f"<div class='stat-card'><div class='stat-value'>{gm.get('n_edges', 'N/A')}</div><div class='stat-label'>Graph Edges</div></div>",
        f"<div class='stat-card'><div class='stat-value'>{gm.get('component_count', 'N/A')}</div><div class='stat-label'>Stable Components</div></div>",
        "</div>",
    ])

    if model:
        html.append("<div class='graph-container'>")
        html.append("<h4 style='margin-top: 0; margin-bottom: 2rem; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.75rem; color: var(--text-secondary);'>Topological Skeleton Projection</h4>")
        html.append(_generate_cosmic_graph_svg(model, cluster_labels=dossier.cluster_labels))
        html.append("<p style='font-size: 0.75rem; color: var(--text-secondary); margin-top: 2rem; font-style: italic;'>Geometric projection of the high-dimensional manifold. Edges represent topological consensus across the ensemble.</p>")
        html.append("</div>")

    # Feature Drift Heatmap
    all_features = set()
    for p in dossier.clusters:
        for f in p.numeric_features[:5]:
            all_features.add(f["column"])
    sorted_features = sorted(list(all_features))

    if sorted_features:
        html.append("<section id='heatmap'>")
        html.append("<h2>Feature Drift Heatmap</h2>")
        html.append("<p style='color: var(--text-secondary); margin-bottom: 2rem;'>Comparative Z-score intensity across cohorts. Green indicates Enrichment, Red indicates Depletion.</p>")
        html.append("<div style='overflow-x: auto;'>")
        html.append("<table><thead><tr><th>Cohort</th>")
        for feat in sorted_features:
            html.append(f"<th>{feat}</th>")
        html.append("</tr></thead><tbody>")
        for p in dossier.clusters:
            html.append(f"<tr><td><strong>{p.semantic_name}</strong></td>")
            feat_map = {f["column"]: f["z_score"] for f in p.numeric_features}
            for feat in sorted_features:
                z = feat_map.get(feat, 0.0)
                opacity = min(0.4, abs(z) * 0.15)
                bg = f"rgba(52, 168, 83, {opacity})" if z > 0.5 else f"rgba(234, 67, 53, {opacity})" if z < -0.5 else "transparent"
                html.append(f"<td style='background-color: {bg}; text-align: center; font-family: var(--font-mono); font-size: 0.85rem;'>{z:+.2f}</td>")
            html.append("</tr>")
        html.append("</tbody></table></div></section>")

    html.extend([
        "</section>",
        "<section id='clusters'>",
        "<h2>Cluster Matrix</h2>",
        "<div style='margin-bottom: 2rem;'>",
        "<input type='text' id='clusterSearch' placeholder='Search clusters...' style='width: 100%; padding: 12px 16px; border: 1px solid var(--border-color); border-radius: 8px; font-family: var(--font-main); font-size: 0.95rem;'>",
        "</div>",
        "<table id='clusterTable'><thead><tr><th>ID</th><th>Semantic Cohort Name</th><th>Population Share</th><th>Defining Characteristics</th></tr></thead><tbody>",
    ])

    for p in dossier.clusters:
        traits = ", ".join([f["column"] for f in p.numeric_features[:3]])
        html.append(f"<tr class='highlight-row' data-cluster-id='{p.cluster_id}'><td><code>{p.cluster_id:02d}</code></td>")
        html.append(f"<td><div style='font-weight: 600; color: var(--google-blue); margin-bottom: 4px;'>{p.semantic_name}</div></td>")
        html.append(f"<td><span style='font-family: var(--font-mono); font-weight: 500;'>{p.size_pct:.1f}%</span><div class='bar-outer'><div class='bar-inner' style='width: {p.size_pct}%'></div></div></td>")
        html.append(f"<td><div style='font-size: 0.85rem; color: var(--text-secondary);'>{traits}</div></td></tr>")
    
    html.append("</tbody></table></section>")

    for p in dossier.clusters:
        html.append(f"<hr><section id='cluster-{p.cluster_id}'>")
        html.append("<div class='cluster-header'>")
        html.append(f"<h2 style='margin: 0;'>Cluster {p.cluster_id}</h2>")
        html.append(f"<div class='cluster-name'>{p.semantic_name}</div>")
        html.append("</div>")
        html.append(f"<p>This cohort comprises <strong>{p.size:,}</strong> points, representing <strong>{p.size_pct:.1f}%</strong> of total population.</p>")
        html.append("<h3>Numeric Profile</h3>")
        html.append("<table><thead><tr><th>Feature</th><th>Distribution</th><th>Mean</th><th>Z-Score</th><th>Signal</th></tr></thead><tbody>")
        for f in p.numeric_features:
            z = f["z_score"]
            z_class = "tag-pos" if z > 1.0 else "tag-neg" if z < -1.0 else "tag-neu"
            h_text = "Enriched" if z > 1.0 else "Depleted" if z < -1.0 else "Baseline"
            html.append(f"<tr><td><strong>{f['column']}</strong></td>")
            html.append(f"<td class='sparkline'>{f.get('sparkline', '')}</td>")
            html.append(f"<td><code>{f['mean']:.3f}</code></td><td><code>{z:+.2f}</code></td>")
            html.append(f"<td><span class='tag {z_class}'>{h_text}</span></td></tr>")
        html.append("</tbody></table>")

        if p.categorical_features:
            html.append("<h3>Categorical Concentrations</h3>")
            html.append("<table><thead><tr><th>Feature</th><th>Value</th><th>Concentration</th></tr></thead><tbody>")
            for f in p.categorical_features:
                html.append(f"<tr><td>{f['column']}</td><td><code>{f['value']}</code></td>")
                html.append(f"<td>{f['concentration']:.1f}% <div class='bar-outer'><div class='bar-inner' style='background: var(--google-green); width: {f['concentration']}%'></div></div></td></tr>")
            html.append("</tbody></table>")

        html.append("<h3>Representative Instances</h3>")
        for i, row in enumerate(p.central_rows):
            summary = ", ".join([f"<span style='font-weight: 500;'>{k}</span>: {v}" for k, v in row.items()])
            html.append(f"<div class='central-row'>#{i+1:02d} &mdash; {summary}...</div>")
        html.append("</section>")

    html.append("</main>")
    html.append("<script>")
    html.append("""
    window.addEventListener('DOMContentLoaded', () => {
        const observer = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                const id = entry.target.getAttribute('id');
                if (entry.isIntersecting) {
                    document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
                    const targetLink = document.querySelector(`nav a[href="#${id}"]`);
                    if (targetLink) targetLink.classList.add('active');
                }
            });
        }, { threshold: 0.1 });
        document.querySelectorAll('section').forEach(s => observer.observe(s));

        const searchInput = document.getElementById('clusterSearch');
        searchInput.addEventListener('keyup', (e) => {
            const query = e.target.value.toLowerCase();
            document.querySelectorAll('#clusterTable tbody tr').forEach(row => {
                row.style.display = row.textContent.toLowerCase().includes(query) ? '' : 'none';
            });
        });

        document.querySelectorAll('.highlight-row').forEach(row => {
            row.addEventListener('mouseenter', () => {
                const cid = row.getAttribute('data-cluster-id');
                document.querySelectorAll('.graph-node').forEach(node => {
                    if (node.getAttribute('data-cluster') === cid) {
                        node.setAttribute('fill-opacity', '1');
                        node.setAttribute('r', '5');
                        node.setAttribute('stroke', 'var(--google-blue)');
                        node.setAttribute('stroke-width', '1');
                    } else {
                        node.setAttribute('fill-opacity', '0.1');
                    }
                });
            });
            row.addEventListener('mouseleave', () => {
                document.querySelectorAll('.graph-node').forEach(node => {
                    node.setAttribute('fill-opacity', '0.6');
                    node.setAttribute('r', '2');
                    node.removeAttribute('stroke');
                });
            });
        });
    });
    """)
    html.append("</script></body></html>")
    return "\n".join(html)


def dossier_to_markdown(dossier: TopologicalDossier) -> str:
    """Renders the dossier as a high-signal Markdown report for LLM consumption."""
    md = [
        "# Topological Analysis Dossier",
        f"**Dataset Size**: {dossier.n_total} points",
        f"**Clusters Found**: {dossier.n_clusters}",
        "",
        "## Cluster Profiles",
        "",
    ]

    for p in dossier.clusters:
        size_bar = generate_proportion_bar(p.size, dossier.n_total, length=12)
        md.append(f"### {p.semantic_name}")
        md.append(f"- **Size**: {p.size} points ({p.size_pct:.1f}%) {size_bar}")

        md.append("\n#### Top Defining Numeric Features (Shifted & Homogeneous)")
        md.append("| Feature | Distribution | Cluster Mean | Global Mean | Z-Score | Homogeneity |")
        md.append("|---|:---|---|---|---|---|")
        for f in p.numeric_features:
            h_desc = "Tight" if f["homogeneity"] < 0.5 else "Divergent" if f["homogeneity"] > 1.5 else "Normal"
            md.append(f"| {f['column']} | `{f.get('sparkline', '')}` | {f['mean']:.3f} | {f['global_mean']:.3f} | {f['z_score']:.2f} | {h_desc} ({f['homogeneity']:.2f}) |")

        if p.categorical_features:
            md.append("\n#### Distinctive Categories (Concentration)")
            md.append("| Feature | Value | Concentration (% of all instances) |")
            md.append("|---|---|---|")
            for f in p.categorical_features:
                md.append(f"| {f['column']} | {f['value']} | {f['concentration']:.1f}% |")

        md.append("\n#### Central Representative Rows")
        for i, row in enumerate(p.central_rows):
            row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
            md.append(f"{i + 1}. {row_str}...")

        md.append("\n---\n")

    return "\n".join(md)


def comparison_to_markdown(id_a: int, id_b: int, results: List[Dict[str, Any]]) -> str:
    """Renders the pairwise comparison results as a Markdown table."""
    md = [
        f"# Cluster Comparison: {id_a} vs {id_b}",
        "",
        "| Feature | Mean A | Mean B | Diff | Cohen's d | T-test (p) | KS-test (p) |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in results:
        diff = r["mean_a"] - r["mean_b"]
        md.append(
            f"| {r['column']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {diff:.3f} | "
            f"{r['cohens_d']:.2f} | {r['p_val_t']:.4f} | {r['p_val_ks']:.4f} |"
        )

    return "\n".join(md)


def compare_clusters(
    data: pd.DataFrame, clusters: pd.Series, id_a: int, id_b: int
) -> List[Dict[str, Any]]:
    """
    Perform pairwise statistical tests between two clusters.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    results = []

    mask_a = clusters.values == id_a
    mask_b = clusters.values == id_b

    data_a = data[mask_a]
    data_b = data[mask_b]

    if len(data_a) < 2 or len(data_b) < 2:
        return []

    for col in numeric_cols:
        vals_a = data_a[col].dropna()
        vals_b = data_b[col].dropna()

        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        mean_a = vals_a.mean()
        mean_b = vals_b.mean()
        std_a = vals_a.std()
        std_b = vals_b.std()

        t_stat, p_val_t = stats.ttest_ind(vals_a, vals_b, equal_var=False)
        _, p_val_ks = stats.ks_2samp(vals_a, vals_b)

        n_a, n_b = len(vals_a), len(vals_b)
        pooled_std = np.sqrt(
            ((n_a - 1) * (std_a**2) + (n_b - 1) * (std_b**2)) / (n_a + n_b - 2)
        )
        cohens_d = (mean_a - mean_b) / max(pooled_std, 1e-10)

        results.append(
            {
                "column": col,
                "mean_a": float(mean_a),
                "mean_b": float(mean_b),
                "p_val_t": float(p_val_t),
                "p_val_ks": float(p_val_ks),
                "cohens_d": float(cohens_d),
            }
        )

    results.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)
    return results
