"""
Topological Interpreter Engine.

Translates the raw topological graph and clustered data into a high-signal 
statistical dossier for LLM synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from pulsar.pipeline import ThemaRS


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


def resolve_clusters(model: ThemaRS, k: Optional[int] = None) -> pd.Series:
    """
    Finds the most informative clusters in the cosmic graph.
    
    1. If threshold was 'auto' and graph has components, use components.
    2. Otherwise, fall back to Spectral Clustering with Silhouette analysis.
    """
    graph = model.cosmic_graph
    adj = model.weighted_adjacency
    n = adj.shape[0]

    # Strategy 1: Connected Components (if graph is not too sparse)
    components = list(nx.connected_components(graph))
    n_comp = len(components)
    
    # If we have a reasonable number of components that aren't all singletons
    if 1 < n_comp < 50:
        # Check density or singleton ratio
        singleton_count = sum(1 for c in components if len(c) == 1)
        if (singleton_count / n) < 0.5:
            # Map nodes to component IDs
            labels = np.zeros(n, dtype=int)
            for i, comp in enumerate(components):
                for node in comp:
                    labels[node] = i
            return pd.Series(labels, name="cluster")

    # Strategy 2: Spectral Clustering Fallback (The MMLU Route)
    # We sweep k from 2 to 15 and pick the best silhouette score
    best_k = 2
    best_score = -1.0
    best_labels = None

    k_range = range(2, min(16, n))
    for k_test in k_range:
        sc = SpectralClustering(
            n_clusters=k_test, 
            affinity='precomputed', 
            assign_labels='discretize', 
            random_state=42
        )
        labels = sc.fit_predict(adj)
        
        # Silhouette requires at least 2 clusters and < n points
        if len(np.unique(labels)) > 1:
            score = silhouette_score(adj, labels, metric='precomputed')
            if score > best_score:
                best_score = score
                best_k = k_test
                best_labels = labels

    if best_labels is not None:
        return pd.Series(best_labels, name="cluster")
    
    # Ultimate fallback: Everyone is cluster 0
    return pd.Series(np.zeros(n, dtype=int), name="cluster")


def build_dossier(model: ThemaRS, data: pd.DataFrame, clusters: pd.Series) -> TopologicalDossier:
    """
    Computes the Enhanced Contextual Distillation (Ranked shifts, Homogeneity, etc.).
    """
    # Align data with clusters (ThemaRS may have dropped rows during preprocessing)
    # For now, we assume the user passed the same DataFrame that ThemaRS.fit() used.
    # We can refine this by checking row counts.
    
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
            c_var = c_data[col].var() if c_size > 1 else 0.0
            
            # Z-score of cluster mean relative to global dist
            z_score = (c_mean - global_means[col]) / np.sqrt(global_vars[col])
            
            # Relative Rank among clusters
            rank = all_cluster_means[col].rank(ascending=False)[cid]
            
            # Homogeneity Index (Variance Ratio)
            # < 1.0 means more homogeneous than global; > 1.0 means more diverse
            homogeneity = c_var / global_vars[col] if global_vars[col] > 0 else 1.0
            
            # Importance Score: Combine Shift and Tightness
            # High score means it's both far from mean and very tight
            importance = abs(z_score) / (homogeneity + 0.1)
            
            c_numeric.append({
                "column": col,
                "mean": float(c_mean),
                "global_mean": float(global_means[col]),
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
            sub_nodes = [i for i, val in enumerate(clusters) if val == cid]
            sub_graph = model.cosmic_graph.subgraph(sub_nodes)
            if len(sub_graph) > 0:
                pagerank = nx.pagerank(sub_graph, weight='weight')
                central_ids = sorted(pagerank, key=pagerank.get, reverse=True)[:3]
                profile.central_rows = data.iloc[central_ids].to_dict('records')
        except:
            # Fallback to simple head
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


def dossier_to_markdown(dossier: TopologicalDossier) -> str:
    """Renders the dossier as a high-signal Markdown report for LLM consumption."""
    md = [
        f"# Topological Analysis Dossier",
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
