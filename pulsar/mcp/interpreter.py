"""
Topological Interpreter Engine.

Translates the raw topological graph and clustered data into a high-signal
statistical dossier for LLM synthesis.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from html import escape
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from pulsar._pulsar import find_stable_thresholds
from pulsar.pipeline import ThemaRS
from pulsar.runtime.utils import (
    generate_distribution_sparkline,
    generate_proportion_bar,
)
from pulsar.mcp.diagnostics import diagnose_model

logger = logging.getLogger(__name__)

# Clustering strategy constants
_MAX_COMPONENTS = 30  # Use component strategy if fewer than this
_MAX_SINGLETON_RATIO = 0.5  # Reject if >50% of nodes are singletons
_SPECTRAL_K_MIN = 2
_SPECTRAL_K_MAX = 20
_MAX_SIGNAL_MATRIX_NUMERIC = 10
_MAX_SIGNAL_MATRIX_CATEGORICAL = 6


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

    # 1. Component Strategy
    if method == "components" or edge_weight_threshold > 0:
        thresh = edge_weight_threshold if edge_weight_threshold > 0 else 0.0
        adj = (W > thresh).astype(np.int64)
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
            edge_weight_threshold_applied=thresh,
        )

    # 2. Threshold Stability (PH-based)
    if method in ("auto", "threshold_stability"):
        result = _cluster_by_threshold_stability(W, n)
        if result:
            return result

    # 3. Spectral Fallback
    if method in ("auto", "spectral"):
        if method == "spectral":
            # Explicit spectral request — let errors propagate to the caller.
            return _cluster_spectral(W, n, max_k)
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
    G = nx.from_numpy_array((adj > 0).astype(np.int64))
    if not nx.is_connected(G):
        raise ValueError(
            "Graph is disconnected — spectral clustering requires a connected affinity graph. "
            "Use method='components' or increase epsilon to connect the graph."
        )

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
    model: ThemaRS,
    data: pd.DataFrame,
    clusters: pd.Series,
    exclude_columns: list[str] | None = None,
) -> TopologicalDossier:
    """Computes per-cluster statistical profiles."""
    if len(clusters) != len(data):
        raise ValueError(
            f"Alignment error: clusters({len(clusters)}) != data({len(data)})"
        )

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

        profile = ClusterProfile(
            cluster_id=int(cid), size=c_size, size_pct=(c_size / n_total) * 100
        )
        c_numeric = []
        for col in numeric_cols:
            c_mean = c_means[col]
            g_mean = global_means[col]
            g_std = np.sqrt(global_vars[col])
            z_score = (c_mean - g_mean) / g_std if g_std > 0 else 0.0
            homogeneity = c_data[col].std() / g_std if g_std > 0 else 0.0
            importance = abs(z_score) / (homogeneity + 0.1)
            spark = generate_distribution_sparkline(c_data[col].dropna())

            c_numeric.append(
                {
                    "column": col,
                    "mean": float(c_mean),
                    "global_mean": float(g_mean),
                    "z_score": float(z_score),
                    "homogeneity": float(homogeneity),
                    "importance": float(importance),
                    "sparkline": spark,
                }
            )

        profile.numeric_features = sorted(
            c_numeric, key=lambda x: x["importance"], reverse=True
        )[:10]

        # Semantic Naming (Fallbacks)
        if len(profile.numeric_features) >= 2:
            f1, f2 = profile.numeric_features[0], profile.numeric_features[1]

            def desc(f):
                adj = (
                    "High"
                    if f["z_score"] > 0.5
                    else "Low"
                    if f["z_score"] < -0.5
                    else "Normal"
                )
                return f"{adj} {f['column']}"

            profile.semantic_name = f"[Auto] {desc(f1)}, {desc(f2)}"
        elif len(profile.numeric_features) == 1:
            f1 = profile.numeric_features[0]
            adj = (
                "High"
                if f1["z_score"] > 0.5
                else "Low"
                if f1["z_score"] < -0.5
                else "Normal"
            )
            profile.semantic_name = f"[Auto] {adj} {f1['column']}"
        else:
            profile.semantic_name = f"Cluster {cid}"

        # Categorical
        c_categorical = []
        for col in categorical_cols:
            counts = c_data[col].value_counts()
            global_counts = data[col].value_counts()
            for val, count in counts.head(5).items():
                in_cluster_prevalence = (count / c_size) * 100 if c_size > 0 else 0.0
                global_recall = (
                    (count / global_counts[val]) * 100
                    if global_counts[val] > 0
                    else 0.0
                )
                c_categorical.append(
                    {
                        "column": col,
                        "value": str(val),
                        "count": int(count),
                        # Backward-compatible alias: concentration now reflects the
                        # primary user-facing signal, namely in-cluster prevalence.
                        "concentration": float(in_cluster_prevalence),
                        "in_cluster_prevalence": float(in_cluster_prevalence),
                        "global_recall": float(global_recall),
                        "cluster_size": int(c_size),
                        "global_count": int(global_counts[val]),
                    }
                )
        profile.categorical_features = sorted(
            c_categorical,
            key=lambda x: (
                x["in_cluster_prevalence"],
                x["global_recall"],
                x["count"],
            ),
            reverse=True,
        )[:10]

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
                top_cols = (
                    list(set(top_cols)) if top_cols else data.columns[:10].tolist()
                )
                profile.central_rows = data.iloc[central_ids][top_cols].to_dict(
                    "records"
                )
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
        n_total=n_total,
        n_clusters=n_clusters,
        clusters=cluster_profiles,
        global_stats={
            "numeric": global_means.to_dict(),
            "columns": data.columns.tolist(),
            "graph_metrics": graph_metrics,
        },
        cluster_labels=clusters,
    )


def _escape_html(value: Any) -> str:
    """Escape arbitrary content before inserting it into HTML."""
    if value is None:
        return ""
    return escape(str(value), quote=True)


def _format_value(value: Any, digits: int = 3) -> str:
    """Format scalar values consistently for report display."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NaN"
        return f"{float(value):.{digits}f}"
    return str(value)


def _format_heatmap_value(value: Any) -> str:
    """Format heatmap values with restrained precision."""
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isnan(value):
            return "NaN"
        abs_value = abs(value)
        if abs_value >= 100:
            return f"{value:,.0f}"
        if abs_value >= 10:
            return f"{value:,.1f}"
        return f"{value:,.2f}"
    return _escape_html(value)


def _signal_tone(z_score: float) -> tuple[str, str]:
    """Return semantic label and CSS class for a z-score."""
    if z_score > 1.0:
        return "Enriched", "signal-pos"
    if z_score < -1.0:
        return "Depleted", "signal-neg"
    return "Baseline", "signal-neu"


def _cluster_palette(clusters: List[ClusterProfile]) -> dict[int, str]:
    """Muted palette for cohort accents in figures and cards."""
    colors = [
        "#1a73e8",
        "#669df6",
        "#147d64",
        "#8ab4f8",
        "#f29900",
        "#5f6368",
        "#00acc1",
        "#7baaf7",
        "#c58a00",
        "#7c8b95",
    ]
    return {
        profile.cluster_id: colors[i % len(colors)]
        for i, profile in enumerate(clusters)
    }


def _cluster_trait_summary(profile: ClusterProfile) -> str:
    """Create a short textual summary for overview cards."""
    fragments = []
    for feature in profile.numeric_features[:3]:
        direction = (
            "high"
            if feature["z_score"] > 0.5
            else "low"
            if feature["z_score"] < -0.5
            else "stable"
        )
        fragments.append(f"{direction} {feature['column']}")
    return ", ".join(fragments) if fragments else "No dominant numeric signal captured."


def _build_key_findings(
    dossier: TopologicalDossier, gm: Dict[str, Any]
) -> list[tuple[str, str]]:
    """Generate concise top-level findings for the hero panel."""
    findings: list[tuple[str, str]] = []
    if dossier.clusters:
        dominant = max(dossier.clusters, key=lambda profile: profile.size)
        findings.append(
            (
                "Largest cohort",
                f"{dominant.semantic_name} accounts for {dominant.size_pct:.1f}% of the population.",
            )
        )

        strongest_feature = None
        strongest_cluster = None
        strongest_score = -1.0
        for profile in dossier.clusters:
            for feature in profile.numeric_features:
                score = abs(float(feature["z_score"]))
                if score > strongest_score:
                    strongest_feature = feature
                    strongest_cluster = profile
                    strongest_score = score
        if strongest_feature is not None and strongest_cluster is not None:
            findings.append(
                (
                    "Sharpest shift",
                    f"{strongest_cluster.semantic_name} shows the strongest deviation on "
                    f"{strongest_feature['column']} ({strongest_feature['z_score']:+.2f} z).",
                )
            )

    if gm:
        findings.append(
            (
                "Topology footprint",
                f"{gm.get('component_count', 'N/A')} stable components across "
                f"{gm.get('n_nodes', 'N/A')} graph nodes and {gm.get('n_edges', 'N/A')} edges.",
            )
        )
    return findings[:3]


def _render_report_styles() -> str:
    """Return the standalone report CSS."""
    return """
    :root {
      --bg: #ffffff;
      --surface: #f8f9fa;
      --surface-strong: #f1f3f4;
      --ink: #1f1f1f;
      --muted: #5f6368;
      --border: rgba(60, 64, 67, 0.14);
      --hairline: rgba(60, 64, 67, 0.10);
      --accent: #1a73e8;
      --accent-soft: rgba(26, 115, 232, 0.08);
      --positive: #147d64;
      --negative: #c2482b;
      --neutral: #6b7280;
      --radius-lg: 18px;
      --radius-md: 12px;
      --radius-sm: 999px;
      --font-heading: "Google Sans", "Google Sans Text", "Product Sans", Roboto, Arial, sans-serif;
      --font-sans: Roboto, Arial, sans-serif;
      --font-mono: "SFMono-Regular", "JetBrains Mono", "Cascadia Code", "Menlo", monospace;
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--bg);
      font-family: var(--font-sans);
      line-height: 1.62;
      -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility;
    }

    a {
      color: var(--accent);
      text-decoration: none;
    }

    .report-shell {
      width: min(100%, 1320px);
      margin: 0 auto;
      padding: 28px 44px 112px;
      display: grid;
      grid-template-columns: minmax(0, 940px) 260px;
      gap: 68px;
      align-items: start;
    }

    .report-main {
      min-width: 0;
      max-width: 100%;
    }

    .report-nav {
      position: sticky;
      top: 24px;
      max-height: calc(100vh - 48px);
      overflow-y: auto;
      padding-right: 8px;
    }

    .report-nav__brand {
      margin-bottom: 18px;
    }

    .report-nav p,
    .section-subtitle,
    .hero-copy p,
    .figure-caption,
    .finding-card p,
    .cluster-card__summary,
    .cluster-card__meta,
    .panel-note,
    .instance-card,
    .cluster-section__summary,
    .hero-list li,
    .cluster-lede,
    .overview-note {
      color: var(--muted);
    }

    .report-nav nav {
      display: grid;
      gap: 2px;
    }

    .report-nav__eyebrow,
    .section-eyebrow {
      display: inline-block;
      font-size: 0.72rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 600;
    }

    .brand-mark {
      margin-bottom: 10px;
    }

    .report-nav h1,
    .hero-copy h1,
    .section-header h2,
    .cluster-section__heading h2 {
      font-family: var(--font-heading);
    }

    .report-nav h1 {
      margin: 0 0 6px;
      font-size: 1.04rem;
      line-height: 1.2;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: var(--ink);
    }

    .nav-group {
      display: grid;
      gap: 2px;
      margin-bottom: 18px;
    }

    .nav-label {
      margin-bottom: 8px;
    }

    .nav-link {
      display: block;
      padding: 6px 10px;
      border-radius: 10px;
      color: #6f7275;
      font-size: 0.88rem;
      font-weight: 500;
      line-height: 1.35;
      transition: color 0.18s ease, background-color 0.18s ease;
    }

    .nav-link:hover,
    .nav-link:focus-visible {
      outline: none;
      color: var(--accent);
      background: var(--accent-soft);
    }

    .nav-link.is-active {
      color: var(--accent);
      background: var(--accent-soft);
    }

    .report-nav__footer {
      display: none;
    }

    .report-stack {
      display: grid;
      gap: 64px;
      min-width: 0;
      max-width: 100%;
    }

    .report-section {
      scroll-margin-top: 28px;
      min-width: 0;
      max-width: 100%;
    }

    .hero-grid {
      display: grid;
      gap: 22px;
    }

    .hero-copy h2 {
      font-family: var(--font-heading);
      margin: 10px 0 14px;
      font-size: clamp(2.7rem, 8vw, 4.8rem);
      line-height: 0.98;
      letter-spacing: -0.045em;
      font-weight: 750;
    }

    .hero-copy p {
      font-size: 1.08rem;
      margin: 0;
    }

    .hero-metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 18px;
      margin-top: 12px;
    }

    .metric-card {
      padding: 14px 0 16px;
      border-bottom: 1px solid var(--hairline);
    }

    .metric-value {
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .metric-label {
      margin-top: 6px;
      color: var(--muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 600;
    }

    .findings-grid {
      margin-top: 8px;
    }

    .hero-list {
      margin: 0;
      padding-left: 18px;
      display: grid;
      gap: 8px;
    }

    .hero-list strong {
      color: var(--ink);
    }

    .section-header {
      display: grid;
      gap: 10px;
      margin-bottom: 22px;
    }

    .section-header h2,
    .cluster-section__heading h2 {
      margin: 0;
      font-size: clamp(1.65rem, 4vw, 2.35rem);
      line-height: 1.04;
      letter-spacing: -0.035em;
      font-weight: 720;
    }

    .figure-frame {
      margin: 0;
      padding: 18px 18px 14px;
      background: var(--surface);
      border-radius: var(--radius-lg);
      min-width: 0;
      max-width: 100%;
      overflow: hidden;
    }

    .figure-toolbar {
      display: grid;
      gap: 12px;
      justify-content: start;
      margin-bottom: 14px;
    }

    .figure-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--ink);
    }

    .figure-subtitle {
      font-size: 0.9rem;
      color: var(--muted);
      margin: 0;
    }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .legend-chip,
    .trait-chip,
    .detail-chip,
    .signal-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      border-radius: var(--radius-sm);
      font-size: 0.78rem;
      font-weight: 600;
    }

    .legend-chip {
      padding: 0;
      background: transparent;
      color: var(--muted);
    }

    .legend-chip::before,
    .detail-dot {
      content: "";
      width: 0.65rem;
      height: 0.65rem;
      border-radius: 999px;
      background: var(--cluster-accent, var(--accent));
      flex: 0 0 auto;
    }

    .figure-caption {
      margin: 14px auto 0;
      max-width: 640px;
      font-size: 0.9rem;
      text-align: center;
    }

    .figure-caption strong {
      color: var(--ink);
    }

    .graph-wrapper svg {
      display: block;
      width: 100%;
      height: auto;
      background: transparent;
    }

    .search-shell {
      display: grid;
      gap: 10px;
      margin-bottom: 24px;
    }

    .cluster-search {
      width: 100%;
      padding: 12px 14px;
      border-radius: var(--radius-md);
      border: none;
      background: var(--surface);
      color: var(--ink);
      font: inherit;
    }

    .cluster-search:focus-visible {
      outline: 2px solid rgba(37, 99, 235, 0.16);
    }

    .cluster-grid {
      display: grid;
      gap: 0;
    }

    .cluster-card {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      padding: 18px 0;
      color: inherit;
      border-bottom: 1px solid var(--hairline);
      transition: color 0.18s ease, background-color 0.18s ease;
    }

    .cluster-card:hover,
    .cluster-card:focus-visible {
      outline: none;
      color: var(--accent);
      background: linear-gradient(90deg, rgba(248, 249, 250, 0.95), rgba(248, 249, 250, 0));
    }

    .cluster-card__topline,
    .cluster-section__topline {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }

    .cluster-id {
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 600;
    }

    .cluster-share {
      font: 600 0.95rem/1 var(--font-mono);
      color: var(--ink);
    }

    .cluster-card h3,
    .cluster-section__heading h2 {
      margin: 0;
    }

    .cluster-card h3 {
      font-family: var(--font-heading);
      font-size: 1.22rem;
      line-height: 1.12;
      letter-spacing: -0.02em;
      font-weight: 700;
      color: var(--ink);
    }

    .cluster-card__summary {
      margin: 8px 0 12px;
      font-size: 0.95rem;
    }

    .bar-track {
      width: 100%;
      height: 4px;
      border-radius: 999px;
      background: rgba(95, 99, 104, 0.12);
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      border-radius: inherit;
      background: var(--cluster-accent, var(--accent));
    }

    .cluster-card__traits,
    .cluster-section__traits,
    .instance-details {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .trait-chip,
    .detail-chip {
      padding: 0;
      background: transparent;
      color: var(--ink);
      font-size: 0.84rem;
    }

    .detail-chip strong {
      font-weight: 700;
    }

    .heatmap-shell {
      overflow-x: auto;
      border-radius: var(--radius-lg);
      background: var(--surface);
      padding: 8px 0;
      min-width: 0;
      max-width: 100%;
    }

    .heatmap-table,
    .data-table {
      width: 100%;
      border-collapse: collapse;
      min-width: 42rem;
    }

    .heatmap-table th,
    .heatmap-table td,
    .data-table th,
    .data-table td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--hairline);
      text-align: left;
      vertical-align: top;
    }

    .heatmap-table th,
    .data-table th {
      font-size: 0.73rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      font-weight: 600;
    }

    .heatmap-table tbody tr:last-child td,
    .data-table tbody tr:last-child td {
      border-bottom: none;
    }

    .heatmap-table td:first-child,
    .heatmap-table th:first-child {
      min-width: 14rem;
    }

    .heat-cell {
      text-align: center;
      white-space: nowrap;
    }

    .heat-value {
      display: block;
      font-family: var(--font-sans);
      font-size: 0.88rem;
      color: var(--ink);
      font-weight: 500;
    }

    .heat-subvalue {
      display: block;
      margin-top: 4px;
      font-size: 0.72rem;
      color: var(--muted);
      letter-spacing: 0.01em;
    }

    .heat-positive {
      background: rgba(20, 125, 100, var(--heat-alpha, 0));
    }

    .heat-negative {
      background: rgba(242, 153, 0, var(--heat-alpha, 0));
    }

    .heatmap-subsection {
      margin-top: 18px;
    }

    .heatmap-subsection h3 {
      margin: 0 0 8px;
      font-size: 0.98rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }

    .heatmap-subsection p {
      margin: 0 0 12px;
      font-size: 0.9rem;
      color: var(--muted);
    }

    .cluster-section {
      padding-top: 6px;
      border-top: 1px solid var(--hairline);
    }

    .cluster-section__heading {
      display: grid;
      gap: 12px;
      margin-bottom: 22px;
    }

    .cluster-section__summary {
      margin: 0;
      font-size: 1rem;
      max-width: 700px;
    }

    .cluster-lede {
      font-size: 1rem;
      margin: 0;
    }

    .overview-note {
      margin: 0;
      font-size: 1rem;
    }

    .cluster-detail-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 26px;
      align-items: start;
    }

    .cluster-secondary-stack {
      display: grid;
      gap: 26px;
    }

    .panel {
      padding: 0;
    }

    .panel h3 {
      margin: 0 0 10px;
      font-size: 1rem;
      line-height: 1.25;
      letter-spacing: -0.01em;
      font-weight: 700;
    }

    .panel-note {
      margin: 0 0 12px;
      font-size: 0.9rem;
    }

    .data-table {
      min-width: 100%;
    }

    .sparkline {
      font-family: var(--font-mono);
      color: var(--accent);
      font-size: 1.1rem;
      letter-spacing: -0.08em;
    }

    code {
      font-family: var(--font-mono);
      font-size: 0.86em;
      color: var(--ink);
      background: var(--surface);
      padding: 0.16rem 0.38rem;
      border-radius: 0.45rem;
    }

    .signal-badge {
      padding: 0.34rem 0.62rem;
      background: transparent;
    }

    .signal-pos {
      color: var(--positive);
    }

    .signal-neg {
      color: var(--negative);
    }

    .signal-neu {
      color: var(--neutral);
    }

    .instances-grid {
      display: grid;
      gap: 12px;
    }

    .instance-card {
      padding: 14px 16px;
      background: var(--surface);
      border-radius: var(--radius-md);
    }

    .instance-label {
      margin-bottom: 10px;
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
      font-weight: 700;
    }

    .empty-state {
      padding: 14px 16px;
      border-radius: var(--radius-md);
      background: var(--surface);
      color: var(--muted);
      font-size: 0.92rem;
    }

    .config-appendix {
      display: grid;
      gap: 14px;
      padding-top: 8px;
      border-top: 1px solid var(--hairline);
      min-width: 0;
      max-width: 100%;
    }

    .config-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }

    .config-toolbar p {
      margin: 0;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .copy-button {
      appearance: none;
      border: none;
      background: var(--surface);
      color: var(--ink);
      padding: 10px 14px;
      border-radius: 10px;
      font: 500 0.9rem/1 var(--font-sans);
      cursor: pointer;
      transition: background-color 0.18s ease, color 0.18s ease;
    }

    .copy-button:hover,
    .copy-button:focus-visible {
      outline: none;
      background: var(--accent-soft);
      color: var(--accent);
    }

    .config-textarea {
      width: 100%;
      min-height: 320px;
      max-width: 100%;
      resize: vertical;
      border: none;
      border-radius: var(--radius-md);
      background: var(--surface);
      color: var(--ink);
      padding: 18px;
      font: 0.88rem/1.55 "Roboto Mono", var(--font-mono);
      white-space: pre;
      overflow: auto;
    }

    .config-textarea:focus-visible {
      outline: 2px solid rgba(26, 115, 232, 0.16);
    }

    @media (max-width: 1180px) {
      .report-shell {
        grid-template-columns: minmax(0, 1fr);
        gap: 36px;
      }

      .report-nav {
        position: static;
        max-height: none;
        overflow: visible;
        order: -1;
      }

      .report-nav nav {
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      }

      .hero-metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }

    @media (max-width: 720px) {
      .report-shell {
        padding: 20px 20px 72px;
      }

      .report-stack {
        gap: 44px;
      }

      .hero-copy h2 {
        font-size: 2.55rem;
      }

      .hero-metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .cluster-card {
        grid-template-columns: 1fr;
        gap: 12px;
      }
    }
    """


def _render_report_script() -> str:
    """Return the standalone report JS."""
    return """
    window.addEventListener('DOMContentLoaded', () => {
      const navLinks = Array.from(document.querySelectorAll('.nav-link'));
      const sections = Array.from(document.querySelectorAll('.report-section[id], .cluster-section[id]'));

      const setActiveNav = (id) => {
        navLinks.forEach((link) => {
          link.classList.toggle('is-active', link.getAttribute('href') === `#${id}`);
        });
      };

      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const id = entry.target.getAttribute('id');
            if (id) setActiveNav(id);
          }
        });
      }, { rootMargin: '-20% 0px -65% 0px', threshold: 0.05 });

      sections.forEach((section) => observer.observe(section));

      const resetGraphNodes = () => {
        document.querySelectorAll('.graph-node').forEach((node) => {
          node.setAttribute('fill-opacity', node.dataset.baseOpacity || '0.78');
          node.setAttribute('r', node.dataset.baseRadius || '2');
          node.setAttribute('fill', node.dataset.baseFill || '#2563eb');
          node.removeAttribute('stroke');
          node.removeAttribute('stroke-width');
        });
      };

      const highlightCluster = (clusterId) => {
        document.querySelectorAll('.graph-node').forEach((node) => {
          const isMatch = node.dataset.cluster === clusterId;
          if (isMatch) {
            node.setAttribute('fill-opacity', '1');
            node.setAttribute('r', String(Math.max(Number(node.dataset.baseRadius || '2'), 4.4)));
            node.setAttribute('stroke', node.dataset.baseFill || '#2563eb');
            node.setAttribute('stroke-width', '1.1');
          } else {
            node.setAttribute('fill-opacity', '0.12');
          }
        });
      };

      document.querySelectorAll('[data-cluster-id]').forEach((target) => {
        target.addEventListener('mouseenter', () => highlightCluster(target.dataset.clusterId));
        target.addEventListener('focus', () => highlightCluster(target.dataset.clusterId), true);
        target.addEventListener('mouseleave', resetGraphNodes);
        target.addEventListener('blur', resetGraphNodes, true);
      });

      const searchInput = document.getElementById('clusterSearch');
      if (searchInput) {
        searchInput.addEventListener('input', (event) => {
          const query = event.target.value.trim().toLowerCase();
          document.querySelectorAll('.cluster-card').forEach((card) => {
            const matches = card.textContent.toLowerCase().includes(query);
            card.hidden = !matches;
          });
        });
      }

      document.querySelectorAll('[data-copy-target]').forEach((button) => {
        button.addEventListener('click', async () => {
          const target = document.getElementById(button.dataset.copyTarget);
          if (!target) return;

          const text = target.value || target.textContent || '';
          try {
            if (navigator.clipboard && window.isSecureContext) {
              await navigator.clipboard.writeText(text);
            } else {
              target.focus();
              target.select();
              document.execCommand('copy');
            }
            const original = button.textContent;
            button.textContent = 'Copied';
            window.setTimeout(() => {
              button.textContent = original;
            }, 1400);
          } catch (_error) {
            target.focus();
            target.select();
          }
        });
      });
    });
    """


def _render_nav(dossier: TopologicalDossier) -> str:
    """Render the persistent navigation rail."""
    parts = [
        "<aside class='report-nav'>",
        "<div class='report-nav__brand'>",
        "<span class='report-nav__eyebrow brand-mark'>Quick links</span>",
        "<h1>Pulsar topological report</h1>",
        f"<p>{dossier.n_total:,} observations · {dossier.n_clusters} cohorts</p>",
        "</div>",
        "<nav aria-label='Section navigation'>",
        "<div class='nav-group'>",
        "<div class='nav-label report-nav__eyebrow'>Sections</div>",
        "<a class='nav-link is-active' href='#summary'>Summary</a>",
        "<a class='nav-link' href='#topology'>Topology figure</a>",
        "<a class='nav-link' href='#heatmap'>Feature drift</a>",
        "<a class='nav-link' href='#config'>Run parameters</a>",
        "</div>",
        "<div class='nav-group'>",
        "<div class='nav-label report-nav__eyebrow'>Clusters</div>",
    ]
    for profile in dossier.clusters:
        parts.append(
            f"<a class='nav-link' href='#cluster-{profile.cluster_id}' data-cluster-id='{profile.cluster_id}'>"
            f"C{profile.cluster_id:02d} — {_escape_html(profile.semantic_name)}</a>"
        )
    parts.extend(
        [
            "</div>",
            "</nav>",
            "</aside>",
        ]
    )
    return "".join(parts)


def _render_summary_section(dossier: TopologicalDossier, gm: Dict[str, Any]) -> str:
    """Render the executive summary section."""
    findings = _build_key_findings(dossier, gm)
    column_count = len(dossier.global_stats.get("columns", []))
    metrics = [
        ("Observations", f"{dossier.n_total:,}"),
        ("Cohorts", f"{dossier.n_clusters}"),
        ("Graph nodes", _format_value(gm.get("n_nodes", "N/A"), digits=0)),
        ("Graph edges", _format_value(gm.get("n_edges", "N/A"), digits=0)),
        ("Columns profiled", f"{column_count}"),
    ]
    parts = [
        "<section id='summary' class='report-section'>",
        "<div class='hero-grid'>",
        "<div class='hero-copy'>",
        "<span class='section-eyebrow'>Summary</span>",
        "<h2>Latent structure report</h2>",
        (
            f"<p>{dossier.n_total:,} observations resolve into <strong>{dossier.n_clusters}</strong> cohorts. "
            f"The supporting topology contains <strong>{_escape_html(_format_value(gm.get('n_nodes', 'N/A'), digits=0))}</strong> nodes, "
            f"<strong>{_escape_html(_format_value(gm.get('n_edges', 'N/A'), digits=0))}</strong> edges, and "
            f"<strong>{_escape_html(_format_value(gm.get('component_count', 'N/A'), digits=0))}</strong> stable components.</p>"
        ),
        "</div>",
        "</div>",
        "<div class='hero-metrics'>",
    ]
    for label, value in metrics:
        parts.append(
            "<div class='metric-card'>"
            f"<div class='metric-value'>{_escape_html(value)}</div>"
            f"<div class='metric-label'>{_escape_html(label)}</div>"
            "</div>"
        )
    parts.append("</div>")
    if findings:
        parts.append("<div class='findings-grid'><ul class='hero-list'>")
        for title, body in findings:
            parts.append(
                f"<li><strong>{_escape_html(title)}.</strong> {_escape_html(body)}</li>"
            )
        parts.append("</ul></div>")
    parts.append("</section>")
    return "".join(parts)


def _generate_cosmic_graph_svg(
    model: ThemaRS,
    width: int = 920,
    height: int = 500,
    cluster_labels: pd.Series | None = None,
    palette: dict[int, str] | None = None,
) -> str:
    """Generate a sampled SVG topology figure with persistent node state for interaction."""
    G = model.cosmic_graph
    if G.number_of_nodes() == 0:
        return ""

    budget = 1000
    if G.number_of_nodes() > budget:
        sampled = set()
        if cluster_labels is not None:
            for cid in sorted(cluster_labels.unique()):
                c_mask = cluster_labels == cid
                c_nodes = cluster_labels.index[c_mask].tolist()
                c_nodes = [node for node in c_nodes if G.has_node(node)]
                if not c_nodes:
                    continue
                n_sample = max(10, int(len(c_nodes) * 0.05))
                sampled.update(
                    sorted(c_nodes, key=lambda node: G.degree(node), reverse=True)[
                        :n_sample
                    ]
                )

        remaining = budget - len(sampled)
        if remaining > 0:
            global_sorted = sorted(
                G.nodes(), key=lambda node: G.degree(node), reverse=True
            )
            for node in global_sorted:
                if node not in sampled:
                    sampled.add(node)
                    remaining -= 1
                if remaining <= 0:
                    break
        G_sub = G.subgraph(sampled)
    else:
        G_sub = G

    pos = nx.spring_layout(
        G_sub, seed=42, k=1.5 / np.sqrt(max(1, G_sub.number_of_nodes()))
    )
    x_vals = [point[0] for point in pos.values()]
    y_vals = [point[1] for point in pos.values()]
    if not x_vals:
        return ""

    xmin, xmax, ymin, ymax = min(x_vals), max(x_vals), min(y_vals), max(y_vals)
    padding = 40

    def scale(
        value: float, value_min: float, value_max: float, target_max: int
    ) -> float:
        return padding + (value - value_min) / (value_max - value_min + 1e-9) * (
            target_max - 2 * padding
        )

    svg = [
        f"<svg id='cosmicGraph' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg' role='img' aria-label='Topological skeleton projection'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff' rx='24' />",
    ]

    opacity_bands: dict[float, list[str]] = {}
    for u, v, data in G_sub.edges(data=True):
        x1 = scale(pos[u][0], xmin, xmax, width)
        y1 = scale(pos[u][1], ymin, ymax, height)
        x2 = scale(pos[v][0], xmin, xmax, width)
        y2 = scale(pos[v][1], ymin, ymax, height)
        band = round(max(0.10, 0.08 + data.get("weight", 0.1) * 0.22), 2)
        opacity_bands.setdefault(band, []).append(
            f"M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f}"
        )

    for opacity, segments in sorted(opacity_bands.items()):
        svg.append(
            f"<path d='{' '.join(segments)}' stroke='#a8afb8' stroke-width='0.65' "
            f"stroke-opacity='{opacity}' fill='none' />"
        )

    for node in G_sub.nodes():
        x = scale(pos[node][0], xmin, xmax, width)
        y = scale(pos[node][1], ymin, ymax, height)
        degree = G.degree(node)
        radius = 1.8 + (degree / max(1, G.number_of_nodes()) * 7.5)
        cluster_id = None
        if cluster_labels is not None and node in cluster_labels.index:
            cluster_id = int(cluster_labels[node])
        base_fill = palette.get(cluster_id, "#2563eb") if palette else "#2563eb"
        svg.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{radius:.2f}' fill='{base_fill}' fill-opacity='0.78' "
            f"class='graph-node' data-cluster='{'' if cluster_id is None else cluster_id}' "
            f"data-base-radius='{radius:.2f}' data-base-fill='{base_fill}' data-base-opacity='0.78'>"
            f"<title>Node {node} (degree {degree}, cluster {'' if cluster_id is None else cluster_id})</title>"
            "</circle>"
        )

    svg.append("</svg>")
    return "".join(svg)


def _render_topology_section(
    dossier: TopologicalDossier,
    model: ThemaRS | None,
    palette: dict[int, str],
) -> str:
    """Render the topology figure and legend."""
    parts = [
        "<section id='topology' class='report-section'>",
        "<div class='section-header'>",
        "<div>",
        "<span class='section-eyebrow'>Figure 1</span>",
        "<h2>Topological skeleton projection</h2>",
        "</div>",
        "<p class='section-subtitle'>Sampled from the cosmic graph for responsiveness. Cohort colors stay consistent across the full report.</p>",
        "</div>",
        "<div class='figure-frame'>",
        "<div class='figure-toolbar'>",
        "<div>",
        "<div class='figure-title'>Global manifold view</div>",
        "<p class='figure-subtitle'>Edges are intentionally quiet. The emphasis stays on node structure and cohort support.</p>",
        "</div>",
        "<div class='legend'>",
    ]
    for profile in dossier.clusters:
        parts.append(
            f"<span class='legend-chip' style='--cluster-accent:{palette[profile.cluster_id]}'>"
            f"C{profile.cluster_id:02d} · {_escape_html(profile.semantic_name)}</span>"
        )
    parts.extend(["</div>", "</div>", "<div class='graph-wrapper'>"])
    if model is not None:
        parts.append(
            _generate_cosmic_graph_svg(
                model,
                cluster_labels=dossier.cluster_labels,
                palette=palette,
            )
        )
        parts.append(
            "<p class='figure-caption'><strong>Figure note.</strong> Nodes are sampled manifold landmarks. Edges encode topological consensus. Hover a cohort row below to isolate its sampled support.</p>"
        )
    else:
        parts.append(
            "<div class='empty-state'>Graph view unavailable for this report.</div>"
        )
    parts.extend(["</div>", "</div>", "</section>"])
    return "".join(parts)


def _render_heatmap_section(dossier: TopologicalDossier) -> str:
    """Render a bounded cohort signal matrix with numeric and categorical groups."""
    numeric_scores: dict[str, float] = {}
    categorical_scores: dict[tuple[str, str], float] = {}

    for profile in dossier.clusters:
        for feature in profile.numeric_features:
            numeric_scores[feature["column"]] = numeric_scores.get(
                feature["column"], 0.0
            ) + abs(float(feature.get("z_score", 0.0)))
        for feature in profile.categorical_features:
            key = (str(feature["column"]), str(feature["value"]))
            categorical_scores[key] = categorical_scores.get(key, 0.0) + float(
                feature.get("in_cluster_prevalence", feature.get("concentration", 0.0))
            )

    numeric_features = [
        name
        for name, _score in sorted(
            numeric_scores.items(), key=lambda item: (-item[1], item[0])
        )[:_MAX_SIGNAL_MATRIX_NUMERIC]
    ]
    categorical_features = [
        key
        for key, _score in sorted(
            categorical_scores.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )[:_MAX_SIGNAL_MATRIX_CATEGORICAL]
    ]

    if not numeric_features and not categorical_features:
        return ""

    parts = [
        "<section id='heatmap' class='report-section'>",
        "<div class='section-header'>",
        "<div>",
        "<span class='section-eyebrow'>Figure 2</span>",
        "<h2>Cohort signal matrix</h2>",
        "</div>",
        f"<p class='section-subtitle'>Bounded for readability: up to {_MAX_SIGNAL_MATRIX_NUMERIC} numeric features and {_MAX_SIGNAL_MATRIX_CATEGORICAL} categorical signals. Numeric values are cohort means, and the smaller line below each numeric value is the z-shift.</p>",
        "</div>",
        "<div class='heatmap-shell'>",
        "<table class='heatmap-table'><thead>",
        "<tr>",
        "<th rowspan='2'>Cohort</th>",
    ]
    if numeric_features:
        parts.append(f"<th colspan='{len(numeric_features)}'>Numeric means</th>")
    if categorical_features:
        parts.append(
            f"<th colspan='{len(categorical_features)}'>Categorical prevalence</th>"
        )
    parts.append("</tr><tr>")
    for feature_name in numeric_features:
        parts.append(f"<th>{_escape_html(feature_name)}</th>")
    for column_name, value in categorical_features:
        parts.append(f"<th>{_escape_html(column_name)}={_escape_html(value)}</th>")
    parts.append("</tr></thead><tbody>")
    for profile in dossier.clusters:
        parts.append(
            f"<tr data-cluster-id='{profile.cluster_id}'>"
            "<td>"
            f"<div><strong>C{profile.cluster_id:02d} — {_escape_html(profile.semantic_name)}</strong></div>"
            f"<div class='cluster-card__meta'>n = {profile.size:,} · {profile.size_pct:.1f}%</div>"
            "</td>"
        )
        numeric_map = {
            feature["column"]: feature for feature in profile.numeric_features
        }
        for feature_name in numeric_features:
            feature = numeric_map.get(feature_name)
            if feature is None:
                parts.append(
                    "<td class='heat-cell'><span class='heat-subvalue'>—</span></td>"
                )
                continue
            z_score = float(feature.get("z_score", 0.0))
            mean_value = feature.get("mean")
            alpha = min(0.38, abs(z_score) * 0.14)
            tone_class = (
                "heat-positive"
                if z_score > 0.5
                else "heat-negative"
                if z_score < -0.5
                else ""
            )
            style = f" style='--heat-alpha:{alpha:.2f}'" if tone_class else ""
            parts.append(
                f"<td class='heat-cell {tone_class}'{style}>"
                f"<span class='heat-value'>{_escape_html(_format_heatmap_value(mean_value))}</span>"
                f"<span class='heat-subvalue'>z {z_score:+.2f}</span>"
                "</td>"
            )
        categorical_map = {
            (str(feature["column"]), str(feature["value"])): feature
            for feature in profile.categorical_features
        }
        for key in categorical_features:
            feature = categorical_map.get(key)
            prevalence = (
                float(
                    feature.get(
                        "in_cluster_prevalence", feature.get("concentration", 0.0)
                    )
                )
                if feature is not None
                else 0.0
            )
            count = int(feature.get("count", 0)) if feature is not None else 0
            alpha = min(0.38, prevalence * 0.0038)
            tone_class = "heat-positive" if prevalence >= 50.0 else ""
            style = f" style='--heat-alpha:{alpha:.2f}'" if prevalence > 0 else ""
            parts.append(
                f"<td class='heat-cell {tone_class}'{style}>"
                f"<span class='heat-value'>{_escape_html(key[1])}</span>"
                f"<span class='heat-subvalue'>{count}/{profile.size} · {prevalence:.1f}%</span>"
                "</td>"
            )
        parts.append("</tr>")
    parts.extend(["</tbody></table>", "</div>", "</section>"])
    return "".join(parts)


def _render_cluster_overview_section(
    dossier: TopologicalDossier, palette: dict[int, str]
) -> str:
    """Render the cohort overview cards."""
    parts = [
        "<section id='clusters' class='report-section'>",
        "<div class='section-header'>",
        "<div>",
        "<span class='section-eyebrow'>Cohort overview</span>",
        "<h2>Population structure, reduced to the essentials</h2>",
        "</div>",
        "<p class='section-subtitle'>Each row is a compact abstract: cohort identity, share, and the few traits most worth carrying into the detailed read.</p>",
        "</div>",
        "<div class='search-shell'>",
        "<input id='clusterSearch' class='cluster-search' type='search' placeholder='Search clusters, signals, or traits…' aria-label='Search cohorts'>",
        "<p class='overview-note'>The overview is deliberately spare. Click through only when a cohort looks meaningfully distinct.</p>",
        "</div>",
        "<div class='cluster-grid'>",
    ]
    for profile in dossier.clusters:
        trait_chips = []
        for feature in profile.numeric_features[:3]:
            trait_chips.append(
                f"<span class='trait-chip'>{_escape_html(feature['column'])}</span>"
            )
        parts.append(
            f"<a class='cluster-card' href='#cluster-{profile.cluster_id}' data-cluster-id='{profile.cluster_id}' "
            f"style='--cluster-accent:{palette[profile.cluster_id]}'>"
            "<div>"
            "<div class='cluster-card__topline'>"
            f"<span class='cluster-id'>Cohort {profile.cluster_id:02d}</span>"
            f"<span class='legend-chip' style='--cluster-accent:{palette[profile.cluster_id]}'>"
            f"{_escape_html(profile.semantic_name)}</span>"
            "</div>"
            f"<p class='cluster-card__summary'>{_escape_html(_cluster_trait_summary(profile))}</p>"
            f"<div class='cluster-card__traits'>{''.join(trait_chips)}</div>"
            "</div>"
            "<div>"
            f"<div class='cluster-share'>{profile.size_pct:.1f}%</div>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{profile.size_pct:.2f}%;'></div></div>"
            f"<p class='cluster-card__meta'>{profile.size:,} rows</p>"
            "</div>"
            "</a>"
        )
    parts.extend(["</div>", "</section>"])
    return "".join(parts)


def _render_numeric_panel(profile: ClusterProfile) -> str:
    """Render the numeric signal table for a cluster."""
    parts = [
        "<section class='panel'>",
        "<h3>Dominant numeric signals</h3>",
        "<p class='panel-note'>Top features are ranked by shift magnitude adjusted for within-cohort homogeneity.</p>",
        "<table class='data-table'><thead><tr><th>Feature</th><th>Distribution</th><th>Mean</th><th>Z-score</th><th>Signal</th></tr></thead><tbody>",
    ]
    for feature in profile.numeric_features:
        label, tone_class = _signal_tone(float(feature["z_score"]))
        parts.append(
            "<tr>"
            f"<td><strong>{_escape_html(feature['column'])}</strong></td>"
            f"<td class='sparkline'>{_escape_html(feature.get('sparkline', ''))}</td>"
            f"<td><code>{float(feature['mean']):.3f}</code></td>"
            f"<td><code>{float(feature['z_score']):+.2f}</code></td>"
            f"<td><span class='signal-badge {tone_class}'>{label}</span></td>"
            "</tr>"
        )
    parts.extend(["</tbody></table>", "</section>"])
    return "".join(parts)


def _render_categorical_panel(profile: ClusterProfile, accent_color: str) -> str:
    """Render categorical concentrations for a cluster."""
    if not profile.categorical_features:
        return "<section class='panel'><h3>Categorical concentrations</h3><div class='empty-state'>No categorical enrichment was captured for this cohort.</div></section>"

    parts = [
        "<section class='panel'>",
        "<h3>Categorical concentrations</h3>",
        "<p class='panel-note'>Prevalence reports what share of this cohort has the value. Global recall reports how much of the overall value population this cohort captures.</p>",
        "<table class='data-table'><thead><tr><th>Feature</th><th>Value</th><th>Prevalence</th><th>Global recall</th></tr></thead><tbody>",
    ]
    for feature in profile.categorical_features:
        prevalence = float(
            feature.get("in_cluster_prevalence", feature.get("concentration", 0.0))
        )
        global_recall = float(feature.get("global_recall", 0.0))
        count = int(feature.get("count", 0))
        cluster_size = int(feature.get("cluster_size", profile.size))
        global_count = int(feature.get("global_count", count))
        parts.append(
            "<tr>"
            f"<td>{_escape_html(feature['column'])}</td>"
            f"<td><code>{_escape_html(feature['value'])}</code></td>"
            "<td>"
            f"{prevalence:.1f}% <span class='cluster-card__meta'>({count}/{cluster_size})</span>"
            "<div class='bar-track' style='margin-top:0.45rem;'>"
            f"<div class='bar-fill' style='width:{prevalence:.2f}%; --cluster-accent:{accent_color};'></div>"
            "</div>"
            "</td>"
            f"<td>{global_recall:.1f}% <span class='cluster-card__meta'>({count}/{global_count})</span></td>"
            "</tr>"
        )
    parts.extend(["</tbody></table>", "</section>"])
    return "".join(parts)


def _render_instances_panel(profile: ClusterProfile) -> str:
    """Render representative instances as compact cards."""
    parts = [
        "<section class='panel'>",
        "<h3>Representative instances</h3>",
        "<p class='panel-note'>Rows chosen from the cohort core so the values stay close to the local manifold center.</p>",
    ]
    if not profile.central_rows:
        parts.append(
            "<div class='empty-state'>No representative rows available for this cohort.</div></section>"
        )
        return "".join(parts)

    parts.append("<div class='instances-grid'>")
    for index, row in enumerate(profile.central_rows, start=1):
        details = []
        for key, value in row.items():
            details.append(
                f"<span class='detail-chip'><strong>{_escape_html(key)}</strong>: {_escape_html(_format_value(value))}</span>"
            )
        parts.append(
            "<article class='instance-card'>"
            f"<div class='instance-label'>Example {index:02d}</div>"
            f"<div class='instance-details'>{''.join(details)}</div>"
            "</article>"
        )
    parts.extend(["</div>", "</section>"])
    return "".join(parts)


def _render_cluster_section(profile: ClusterProfile, accent_color: str) -> str:
    """Render a detailed cluster section."""
    traits = "".join(
        f"<span class='trait-chip'>{_escape_html(feature['column'])}</span>"
        for feature in profile.numeric_features[:5]
    )
    return "".join(
        [
            f"<section id='cluster-{profile.cluster_id}' class='report-section cluster-section' "
            f"data-cluster-id='{profile.cluster_id}' style='--cluster-accent:{accent_color}'>",
            "<div class='cluster-section__heading'>",
            "<div class='cluster-section__topline'>",
            f"<span class='cluster-id'>C{profile.cluster_id:02d}</span>",
            f"<span class='legend-chip' style='--cluster-accent:{accent_color}'>n = {profile.size:,} · {profile.size_pct:.1f}%</span>",
            "</div>",
            f"<h2>C{profile.cluster_id:02d} — {_escape_html(profile.semantic_name)}</h2>",
            (
                f"<p class='cluster-section__summary'>This cohort contains <strong>{profile.size:,}</strong> observations "
                f"and represents <strong>{profile.size_pct:.1f}%</strong> of the analyzed population. "
                "Treat the numeric and categorical tables below as the evidence sheet for this slice of the manifold.</p>"
            ),
            f"<div class='cluster-section__traits'>{traits}</div>",
            "</div>",
            "<div class='cluster-detail-grid'>",
            _render_numeric_panel(profile),
            "<div class='cluster-secondary-stack'>",
            _render_categorical_panel(profile, accent_color),
            _render_instances_panel(profile),
            "</div>",
            "</div>",
            "</section>",
        ]
    )


def _render_config_appendix(config_yaml: str | None) -> str:
    """Render a copyable params appendix at the bottom of the report."""
    if not config_yaml:
        return ""

    return "".join(
        [
            "<section id='config' class='report-section config-appendix'>",
            "<div class='section-header'>",
            "<div>",
            "<span class='section-eyebrow'>Appendix</span>",
            "<h2>Run parameters</h2>",
            "</div>",
            "<p class='section-subtitle'>Normalized params.yaml used for the exported run.</p>",
            "</div>",
            "<div class='config-toolbar'>",
            "<p>Readonly, formatted for direct copy and reuse.</p>",
            "<button type='button' class='copy-button' data-copy-target='paramsYaml'>Copy YAML</button>",
            "</div>",
            f"<textarea id='paramsYaml' class='config-textarea' readonly spellcheck='false'>{_escape_html(config_yaml)}</textarea>",
            "</section>",
        ]
    )


def _render_font_links() -> str:
    """Load Google-hosted body fonts with immediate local fallbacks when offline."""
    return "\n".join(
        [
            "<link rel='preconnect' href='https://fonts.googleapis.com'>",
            "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>",
            "<link href='https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Roboto+Mono:wght@400;500;700&display=swap' rel='stylesheet'>",
        ]
    )


def dossier_to_html(
    dossier: TopologicalDossier,
    model: ThemaRS | None = None,
    config_yaml: str | None = None,
) -> str:
    """Render the dossier as a polished standalone HTML research report."""
    graph_metrics = dossier.global_stats.get("graph_metrics", {})
    palette = _cluster_palette(dossier.clusters)
    sections = [
        _render_summary_section(dossier, graph_metrics),
        _render_topology_section(dossier, model, palette),
        _render_heatmap_section(dossier),
    ]
    sections.extend(
        _render_cluster_section(profile, palette[profile.cluster_id])
        for profile in dossier.clusters
    )
    sections.append(_render_config_appendix(config_yaml))
    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "<title>Pulsar Topological Dossier</title>",
            _render_font_links(),
            "<style>",
            _render_report_styles(),
            "</style>",
            "</head>",
            "<body>",
            "<div class='report-shell'>",
            "<main class='report-main'>",
            "<div class='report-stack'>",
            *[section for section in sections if section],
            "</div>",
            "</main>",
            _render_nav(dossier),
            "</div>",
            "<script>",
            _render_report_script(),
            "</script>",
            "</body>",
            "</html>",
        ]
    )


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
        md.append(
            "| Feature | Distribution | Cluster Mean | Global Mean | Z-Score | Homogeneity |"
        )
        md.append("|---|:---|---|---|---|---|")
        for f in p.numeric_features:
            h_desc = (
                "Tight"
                if f["homogeneity"] < 0.5
                else "Divergent"
                if f["homogeneity"] > 1.5
                else "Normal"
            )
            md.append(
                f"| {f['column']} | `{f.get('sparkline', '')}` | {f['mean']:.3f} | {f['global_mean']:.3f} | {f['z_score']:.2f} | {h_desc} ({f['homogeneity']:.2f}) |"
            )

        if p.categorical_features:
            md.append("\n#### Distinctive Categories (Concentration)")
            md.append("| Feature | Value | In-cluster prevalence | Global recall |")
            md.append("|---|---|---|---|")
            for f in p.categorical_features:
                prevalence = f.get("in_cluster_prevalence", f.get("concentration", 0.0))
                global_recall = f.get("global_recall", 0.0)
                count = f.get("count", 0)
                cluster_size = f.get("cluster_size", p.size)
                global_count = f.get("global_count", count)
                md.append(
                    f"| {f['column']} | {f['value']} | {prevalence:.1f}% ({count}/{cluster_size}) | {global_recall:.1f}% ({count}/{global_count}) |"
                )

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
