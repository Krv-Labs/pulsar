"""
Analysis hooks — pure Python utilities that work on the outputs of the Rust layer.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def label_points(ball_mapper, n: int) -> np.ndarray:
    """
    Return an (n,) int64 array: for each data point, the ID of its
    first ball assignment (-1 if not covered by any ball).
    """
    labels = np.full(n, -1, dtype=np.int64)
    for ball_id, members in enumerate(ball_mapper.nodes):
        for pt in members:
            if labels[pt] == -1:
                labels[pt] = ball_id
    return labels


def membership_matrix(ball_mapper, n: int) -> np.ndarray:
    """
    Return a dense (n, n_balls) binary uint8 matrix.
    M[i, b] = 1 if point i belongs to ball b.
    """
    n_balls = ball_mapper.n_nodes()
    m = np.zeros((n, n_balls), dtype=np.uint8)
    for ball_id, members in enumerate(ball_mapper.nodes):
        for pt in members:
            m[pt, ball_id] = 1
    return m


def cosmic_clusters(
    cosmic_graph: nx.Graph,
    method: str = "agglomerative",
    n_clusters: int = 5,
) -> np.ndarray:
    """
    Run clustering on the cosmic graph adjacency matrix.
    Returns an (n,) int array of cluster labels.

    method: "agglomerative" | "spectral"
    """
    adj = nx.to_numpy_array(cosmic_graph)

    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        ).fit_predict(1.0 - adj)

    if method == "spectral":
        from sklearn.cluster import SpectralClustering

        return SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
        ).fit_predict(adj)

    raise ValueError(f"Unknown clustering method: {method!r}")


def graph_to_dataframe(ball_mapper, data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with one row per ball node, including:
    node_id, size (member count), centroid coordinates,
    mean/std of each original feature for members in that node.
    """
    rows = []
    for ball_id, members in enumerate(ball_mapper.nodes):
        subset = data.iloc[list(members)]
        row: dict = {"node_id": ball_id, "size": len(members)}
        row.update(subset.mean(numeric_only=True).add_prefix("mean_").to_dict())
        row.update(subset.std(numeric_only=True).add_prefix("std_").to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def unclustered_points(ball_mapper, n: int) -> list[int]:
    """Return list of point indices not covered by any ball."""
    covered = {pt for members in ball_mapper.nodes for pt in members}
    return [i for i in range(n) if i not in covered]


def cosmic_to_networkx(cg) -> nx.Graph:
    """Convert a CosmicGraph Rust object to a NetworkX graph with 'weight' attributes."""
    adj = np.array(cg.adj, dtype=np.float64)
    wadj = np.array(cg.weighted_adj, dtype=np.float64)
    # Use np.where to vectorize weight assignment: keep wadj values where adj > 0, else 0.0
    # This eliminates the O(E) Python loop while preserving threshold mask semantics.
    return nx.from_numpy_array(np.where(adj > 0, wadj, 0.0))
