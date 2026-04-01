"""
Topological Interpreter Engine.

Translates the raw topological graph and clustered data into a high-signal
statistical dossier for LLM synthesis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from pulsar.pipeline import ThemaRS

logger = logging.getLogger(__name__)

# Clustering strategy constants
_MAX_COMPONENTS = 50            # Use component strategy if fewer than this
_MAX_SINGLETON_RATIO = 0.5      # Reject if >50% of nodes are singletons
_SPECTRAL_K_MIN = 2
_SPECTRAL_K_MAX = 15


@dataclass
class ClusterProfile:
    """Statistical profile of a single topological cluster."""
    cluster_id: int
    size: int
    size_pct: float
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


def resolve_clusters(
    model: ThemaRS,
    method: str = "auto",
    max_k: int = _SPECTRAL_K_MAX,
) -> pd.Series:
    """
    Finds clusters in the cosmic graph using the specified method.

    Args:
        model: Fitted ThemaRS instance.
        method: Clustering strategy.
            - "auto": Connected components first (if balanced), spectral fallback.
            - "spectral": Force spectral clustering with silhouette-optimized k.
            - "components": Force connected components only.
        max_k: Maximum k for spectral clustering search (default 15).

    Returns:
        pd.Series with integer cluster labels.
    """
    graph = model.cosmic_graph
    adj = model.weighted_adjacency
    n = adj.shape[0]

    if method == "components":
        return _cluster_by_components(graph, n)

    if method == "spectral":
        result = _cluster_by_spectral(adj, n, max_k)
        if result is not None:
            return result
        logger.warning("resolve_clusters: spectral failed, falling back to single cluster")
        return pd.Series(np.zeros(n, dtype=int), name="cluster")

    # method == "auto": try components first, then spectral
    components = list(nx.connected_components(graph))
    n_comp = len(components)

    if 1 < n_comp < _MAX_COMPONENTS:
        singleton_count = sum(1 for c in components if len(c) == 1)
        if (singleton_count / n) < _MAX_SINGLETON_RATIO:
            labels = np.zeros(n, dtype=int)
            for i, comp in enumerate(components):
                for node in comp:
                    labels[node] = i
            logger.debug(
                "resolve_clusters: using connected components (n_comp=%d)", n_comp
            )
            return pd.Series(labels, name="cluster")

    # Spectral fallback
    result = _cluster_by_spectral(adj, n, max_k)
    if result is not None:
        return result

    logger.warning("resolve_clusters: falling back to single cluster")
    return pd.Series(np.zeros(n, dtype=int), name="cluster")


def _cluster_by_components(graph, n: int) -> pd.Series:
    """Cluster by connected components."""
    components = list(nx.connected_components(graph))
    labels = np.zeros(n, dtype=int)
    for i, comp in enumerate(components):
        for node in comp:
            labels[node] = i
    logger.debug("resolve_clusters: components method (n_comp=%d)", len(components))
    return pd.Series(labels, name="cluster")


def _cluster_by_spectral(adj, n: int, max_k: int) -> pd.Series | None:
    """Spectral clustering with silhouette-optimized k. Returns None if it fails."""
    best_score = -1.0
    best_labels = None
    best_k = _SPECTRAL_K_MIN

    k_range = range(_SPECTRAL_K_MIN, min(max_k + 1, n))
    for k_test in k_range:
        sc = SpectralClustering(
            n_clusters=k_test,
            affinity='precomputed',
            assign_labels='discretize',
            random_state=42,
        )
        labels = sc.fit_predict(adj)

        if len(np.unique(labels)) > 1:
            score = silhouette_score(adj, labels, metric='precomputed')
            if score > best_score:
                best_score = score
                best_k = k_test
                best_labels = labels

    if best_labels is not None:
        logger.debug(
            "resolve_clusters: spectral, best_k=%d score=%.3f", best_k, best_score,
        )
        return pd.Series(best_labels, name="cluster")
    return None


def build_dossier(
    model: ThemaRS, data: pd.DataFrame, clusters: pd.Series
) -> TopologicalDossier:
    """
    Computes per-cluster statistical profiles.

    Calculates shifts, homogeneity, and categorical concentration for each cluster.

    Args:
        model: Fitted ThemaRS instance
        data: DataFrame with original data (must match clusters length)
        clusters: pd.Series with cluster assignments for each row

    Returns:
        TopologicalDossier with cluster profiles and global statistics
    """
    # Validate alignment
    if len(clusters) != len(data):
        msg = (
            f"clusters length ({len(clusters)}) "
            f"does not match data length ({len(data)})"
        )
        raise ValueError(msg)

    n_total = len(data)
    n_clusters = len(clusters.unique())
    
    # 1. Global Baseline
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    
    global_means = data[numeric_cols].mean()
    global_vars = data[numeric_cols].var().replace(0, 1e-10)
    
    # 2. Compute per-cluster stats
    cluster_profiles = []
    
    # To calculate 'Relative Rank', we need all cluster means first
    all_cluster_means = {}
    for col in numeric_cols:
        # Use .values to ensure alignment if indices differ
        all_cluster_means[col] = data.groupby(clusters.values, observed=True)[col].mean()

    for cid in sorted(clusters.unique()):
        # Use boolean array directly to avoid index label mismatch
        c_mask = (clusters.values == cid)
        c_data = data.iloc[c_mask]
        c_size = len(c_data)
        
        profile = ClusterProfile(
            cluster_id=int(cid),
            size=c_size,
            size_pct=(c_size / n_total) * 100
        )
        
        # Numeric Features Analysis
        c_numeric = []
        for col in numeric_cols:
            c_mean = c_data[col].mean()
            if pd.isna(c_mean):
                continue  # All NaN for this cluster in this column — no signal
            global_mean_val = global_means[col]
            if pd.isna(global_mean_val):
                continue  # Column is entirely NaN globally
            c_var = c_data[col].var() if c_size > 1 else 0.0

            # Z-score of cluster mean relative to global dist
            z_score = (c_mean - global_mean_val) / np.sqrt(global_vars[col])

            # Relative Rank among clusters
            rank = all_cluster_means[col].rank(ascending=False).get(cid, np.nan)
            if pd.isna(rank):
                continue

            # Homogeneity Index (Variance Ratio)
            # < 1.0 means more homogeneous than global; > 1.0 means more diverse
            homogeneity = c_var / global_vars[col] if global_vars[col] > 0 else 1.0

            # Importance Score: Combine Shift and Tightness
            # High score means it's both far from mean and very tight
            importance = abs(z_score) / (homogeneity + 0.1)

            c_numeric.append({
                "column": col,
                "mean": float(c_mean),
                "global_mean": float(global_mean_val),
                "z_score": float(z_score),
                "rank": int(rank),
                "homogeneity": float(homogeneity),
                "importance": float(importance)
            })
            
        # Sort by importance and take top 10
        profile.numeric_features = sorted(c_numeric, key=lambda x: x['importance'], reverse=True)[:10]
        
        # Categorical Concentration
        c_categorical = []
        for col in categorical_cols:
            # Most common values in this cluster
            counts = c_data[col].value_counts()
            global_counts = data[col].value_counts()
            
            for val, count in counts.head(5).items():
                # What % of the total 'val' rows are in this cluster?
                concentration = (count / global_counts[val]) * 100
                c_categorical.append({
                    "column": col,
                    "value": str(val),
                    "count": int(count),
                    "concentration": float(concentration)
                })
        
        profile.categorical_features = sorted(c_categorical, key=lambda x: x['concentration'], reverse=True)[:10]
        
        # Central Nodes (Sampling)
        # We use PageRank on the cluster's sub-graph if possible
        try:
            sub_nodes = list(np.where(clusters.values == cid)[0])
            sub_graph = model.cosmic_graph.subgraph(sub_nodes)
            if len(sub_graph) > 0:
                pagerank = nx.pagerank(sub_graph, weight='weight')
                central_ids = sorted(pagerank, key=pagerank.get, reverse=True)[:3]
                profile.central_rows = data.iloc[central_ids].to_dict('records')
        except (nx.NetworkXError, nx.NetworkXUnfeasible, ValueError, KeyError) as e:
            # Fallback to simple head if PageRank fails
            logger.debug("PageRank failed for cluster %d: %s, using head", cid, e)
            profile.central_rows = c_data.head(3).to_dict('records')
            
        cluster_profiles.append(profile)

    return TopologicalDossier(
        n_total=n_total,
        n_clusters=n_clusters,
        clusters=cluster_profiles,
        global_stats={
            "numeric": global_means.to_dict(),
            "columns": data.columns.tolist()
        }
    )


def comparison_to_markdown(
    id_a: int, id_b: int, results: List[Dict[str, Any]]
) -> str:
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

    Compares all numeric features using:
    1. Welch's T-test (unequal variance) for mean differences.
    2. Kolmogorov-Smirnov test for overall distribution shifts.
    3. Cohen's d for effect size.

    Results are sorted by the absolute magnitude of Cohen's d.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    results = []

    # Use boolean arrays directly to avoid index label mismatch
    mask_a = clusters.values == id_a
    mask_b = clusters.values == id_b

    data_a = data.iloc[mask_a]
    data_b = data.iloc[mask_b]

    if len(data_a) < 2 or len(data_b) < 2:
        logger.warning(
            "compare_clusters: clusters %d or %d too small for t-test", id_a, id_b
        )
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

        # Welch's T-test (equal_var=False)
        t_stat, p_val_t = stats.ttest_ind(vals_a, vals_b, equal_var=False)

        # KS-test
        _, p_val_ks = stats.ks_2samp(vals_a, vals_b)

        # Cohen's d
        # Pooled standard deviation
        n_a, n_b = len(vals_a), len(vals_b)
        pooled_std = np.sqrt(
            ((n_a - 1) * (std_a**2) + (n_b - 1) * (std_b**2)) / (n_a + n_b - 2)
        )
        cohens_d = (mean_a - mean_b) / max(pooled_std, 1e-10)

        results.append({
            "column": col,
            "mean_a": float(mean_a),
            "mean_b": float(mean_b),
            "p_val_t": float(p_val_t),
            "p_val_ks": float(p_val_ks),
            "cohens_d": float(cohens_d),
        })

    # Sort by magnitude of Cohen's d
    results.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)
    return results


def dossier_to_markdown(dossier: TopologicalDossier) -> str:
    """Renders the dossier as a high-signal Markdown report for LLM consumption."""
    md = [
        "# Topological Analysis Dossier",
        f"**Dataset Size**: {dossier.n_total} points",
        f"**Clusters Found**: {dossier.n_clusters}",
        "",
        "## Cluster Profiles",
        ""
    ]
    
    for p in dossier.clusters:
        md.append(f"### Cluster {p.cluster_id}")
        md.append(f"- **Size**: {p.size} points ({p.size_pct:.1f}%)")
        
        md.append("\n#### Top Defining Numeric Features (Shifted & Homogeneous)")
        md.append("| Feature | Cluster Mean | Global Mean | Rank / {0} | Z-Score | Homogeneity |".format(dossier.n_clusters))
        md.append("|---|---|---|---|---|---|")
        for f in p.numeric_features:
            h_desc = "Tight" if f['homogeneity'] < 0.5 else "Divergent" if f['homogeneity'] > 1.5 else "Normal"
            md.append(f"| {f['column']} | {f['mean']:.3f} | {f['global_mean']:.3f} | {f['rank']} | {f['z_score']:.2f} | {h_desc} ({f['homogeneity']:.2f}) |")
            
        if p.categorical_features:
            md.append("\n#### Distinctive Categories (Concentration)")
            md.append("| Feature | Value | Concentration (% of all instances) |")
            md.append("|---|---|---|")
            for f in p.categorical_features:
                md.append(f"| {f['column']} | {f['value']} | {f['concentration']:.1f}% |")
        
        md.append("\n#### Central Representative Rows")
        for i, row in enumerate(p.central_rows):
            # Format row as a compact string
            row_str = ", ".join([f"{k}: {v}" for k, v in row.items() if k in dossier.global_stats['columns'][:10]])
            md.append(f"{i+1}. {row_str}...")
            
        md.append("\n---\n")
        
    return "\n".join(md)
