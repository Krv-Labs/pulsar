"""
Topological Interpreter Engine.

Translates the raw topological graph and clustered data into a high-signal
statistical dossier for LLM synthesis.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from html import escape
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.cluster import KMeans, SpectralClustering
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
_MAX_SIGNAL_MATRIX_CATEGORICAL = 5
_EPS = 1e-9


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
    numeric_tiers: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    categorical_tiers: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    topological_neighbors: List[Dict[str, Any]] = field(default_factory=list)


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


@dataclass
class FeatureEvidenceIndex:
    """Cached feature evidence derived from a clustering assignment."""

    cluster_bundles: Dict[int, Dict[str, Any]]
    numeric_global_ranking: List[str]
    categorical_global_ranking: List[Dict[str, Any]]
    signal_matrix: Dict[str, Any]
    metadata: Dict[str, Any]


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
    if method == "auto":
        connectivity_graph = nx.from_numpy_array((W > 0).astype(np.int64))
        if not nx.is_connected(connectivity_graph):
            return resolve_clusters(
                model,
                method="components",
                max_k=max_k,
                edge_weight_threshold=edge_weight_threshold,
            )
    if method in ("auto", "spectral"):
        return _cluster_spectral(W, n, max_k)

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


def _bh_fdr(p_values: list[float | None]) -> list[float | None]:
    """Benjamini-Hochberg FDR correction preserving None entries."""
    indexed = [
        (idx, float(p))
        for idx, p in enumerate(p_values)
        if p is not None and np.isfinite(p)
    ]
    corrected: list[float | None] = [None] * len(p_values)
    if not indexed:
        return corrected

    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running = 1.0
    for rank, (idx, p_value) in reversed(list(enumerate(indexed, start=1))):
        adjusted = min(1.0, p_value * m / rank)
        running = min(running, adjusted)
        corrected[idx] = running
    return corrected


def _empirical_percentiles(values: list[float]) -> list[float]:
    """Return empirical percentiles in [0, 1], collapsing constants to 0."""
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1 or np.allclose(arr, arr[0], equal_nan=True):
        return [0.0] * len(values)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    return (ranks / max(len(arr) - 1, 1)).tolist()


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _safe_var(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.var(values, ddof=1))


def _safe_mad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    if a.size <= 1 or b.size <= 1:
        return max(_safe_std(a), _safe_std(b), _EPS)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = (((a.size - 1) * var_a) + ((b.size - 1) * var_b)) / max(
        a.size + b.size - 2, 1
    )
    return float(np.sqrt(max(pooled, _EPS)))


def _normalize_numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)


def _direction_from_effects(*effects: float) -> str:
    positives = sum(1 for effect in effects if effect > 0)
    negatives = sum(1 for effect in effects if effect < 0)
    if positives and not negatives:
        return "up"
    if negatives and not positives:
        return "down"
    return "mixed"


def _graph_cluster_adjacency(
    model: ThemaRS,
    clusters: pd.Series,
) -> dict[int, list[dict[str, Any]]]:
    """Aggregate the cosmic graph into cluster-to-cluster bridge strengths."""
    graph = model.cosmic_graph
    cluster_lookup = {int(idx): int(label) for idx, label in clusters.items()}
    cluster_sizes = clusters.value_counts().to_dict()
    pair_weights: dict[tuple[int, int], float] = {}
    internal_weights: dict[int, float] = {int(cid): 0.0 for cid in cluster_sizes}

    for source, target, data in graph.edges(data=True):
        source_cluster = cluster_lookup.get(int(source))
        target_cluster = cluster_lookup.get(int(target))
        if source_cluster is None or target_cluster is None:
            continue
        weight = float(data.get("weight", 0.0))
        if source_cluster == target_cluster:
            internal_weights[source_cluster] += weight
            continue
        pair = tuple(sorted((source_cluster, target_cluster)))
        pair_weights[pair] = pair_weights.get(pair, 0.0) + weight

    adjacency: dict[int, list[dict[str, Any]]] = {
        int(cid): [] for cid in sorted(cluster_sizes)
    }
    for (cluster_a, cluster_b), bridge_weight in pair_weights.items():
        denom = math.sqrt(
            max(cluster_sizes.get(cluster_a, 1), 1)
            * max(cluster_sizes.get(cluster_b, 1), 1)
        )
        normalized_weight = bridge_weight / max(denom, 1.0)
        bridge_strength = bridge_weight / max(
            bridge_weight
            + internal_weights.get(cluster_a, 0.0)
            + internal_weights.get(cluster_b, 0.0),
            _EPS,
        )
        edge_payload = {
            "cluster_id": int(cluster_b),
            "bridge_weight": float(bridge_weight),
            "normalized_weight": float(normalized_weight),
            "bridge_strength": float(bridge_strength),
        }
        adjacency[int(cluster_a)].append(edge_payload)
        adjacency[int(cluster_b)].append(
            {
                **edge_payload,
                "cluster_id": int(cluster_a),
            }
        )

    for cluster_id, neighbors in adjacency.items():
        neighbors.sort(
            key=lambda item: (
                -item["normalized_weight"],
                -item["bridge_strength"],
                item["cluster_id"],
            )
        )
    return adjacency


def _strongest_neighbor(
    adjacency: dict[int, list[dict[str, Any]]], cluster_id: int
) -> int | None:
    neighbors = adjacency.get(int(cluster_id), [])
    if not neighbors:
        return None
    return int(neighbors[0]["cluster_id"])


def _apply_numeric_scores(rows: list[dict[str, Any]]) -> None:
    metric_names = [
        ("effect_mean_std", lambda row: abs(float(row.get("effect_mean_std", 0.0)))),
        (
            "effect_median_mad",
            lambda row: abs(float(row.get("effect_median_mad", 0.0))),
        ),
        ("ks_stat", lambda row: float(row.get("ks_stat", 0.0))),
        ("wasserstein_norm", lambda row: float(row.get("wasserstein_norm", 0.0))),
        (
            "concentration_gain",
            lambda row: max(float(row.get("concentration_gain", 0.0)), 0.0),
        ),
        ("specificity_score", lambda row: float(row.get("specificity_score", 0.0))),
        (
            "one_vs_rest_sig",
            lambda row: -math.log10(float(row.get("one_vs_rest_q", 1.0)) + _EPS),
        ),
        ("global_sig", lambda row: -math.log10(float(row.get("global_q", 1.0)) + _EPS)),
        (
            "neighbor_effect",
            lambda row: abs(float(row.get("neighbor_effect", 0.0))),
        ),
    ]
    component_percentiles: dict[str, list[float]] = {}
    for metric_name, getter in metric_names:
        component_percentiles[metric_name] = _empirical_percentiles(
            [getter(row) for row in rows]
        )

    aggregate_components: list[float] = []
    for row_index, row in enumerate(rows):
        evidence_vector = {
            metric_name: component_percentiles[metric_name][row_index]
            for metric_name, _getter in metric_names
        }
        positive = [max(value, 0.0) + 1e-6 for value in evidence_vector.values()]
        aggregate = (
            float(math.exp(np.mean(np.log(positive))) - 1e-6) if positive else 0.0
        )
        row["aggregate_score"] = max(aggregate, 0.0)
        row["evidence_vector"] = evidence_vector
        aggregate_components.append(row["aggregate_score"])

    aggregate_percentiles = _empirical_percentiles(aggregate_components)
    for row_index, row in enumerate(rows):
        row["percentile_score"] = aggregate_percentiles[row_index]


def _apply_categorical_scores(rows: list[dict[str, Any]]) -> None:
    metric_names = [
        ("prevalence_cluster", lambda row: float(row.get("prevalence_cluster", 0.0))),
        ("log_lift", lambda row: abs(float(row.get("log_lift", 0.0)))),
        ("specificity", lambda row: abs(float(row.get("specificity", 0.0)))),
        ("global_recall", lambda row: float(row.get("global_recall", 0.0))),
        (
            "neighbor_specificity",
            lambda row: abs(float(row.get("neighbor_specificity", 0.0))),
        ),
        ("mi_contrib", lambda row: abs(float(row.get("mi_contrib", 0.0)))),
        ("fisher_sig", lambda row: -math.log10(float(row.get("fisher_q", 1.0)) + _EPS)),
        ("global_sig", lambda row: -math.log10(float(row.get("global_q", 1.0)) + _EPS)),
    ]
    component_percentiles: dict[str, list[float]] = {}
    for metric_name, getter in metric_names:
        component_percentiles[metric_name] = _empirical_percentiles(
            [getter(row) for row in rows]
        )

    aggregates: list[float] = []
    for row_index, row in enumerate(rows):
        evidence_vector = {
            metric_name: component_percentiles[metric_name][row_index]
            for metric_name, _getter in metric_names
        }
        positive = [max(value, 0.0) + 1e-6 for value in evidence_vector.values()]
        aggregate = (
            float(math.exp(np.mean(np.log(positive))) - 1e-6) if positive else 0.0
        )
        row["aggregate_score"] = max(aggregate, 0.0)
        row["evidence_vector"] = evidence_vector
        aggregates.append(row["aggregate_score"])

    aggregate_percentiles = _empirical_percentiles(aggregates)
    for row_index, row in enumerate(rows):
        row["percentile_score"] = aggregate_percentiles[row_index]


def _assign_signal_tiers(rows: list[dict[str, Any]]) -> None:
    """Assign adaptive tiers using the row score distribution itself."""
    for row in rows:
        row["signal_tier"] = "noise"

    positives = [row for row in rows if float(row.get("aggregate_score", 0.0)) > 0.0]
    if not positives:
        return

    positive_scores = np.array(
        [float(row["aggregate_score"]) for row in positives], dtype=float
    ).reshape(-1, 1)
    unique_scores = np.unique(positive_scores)
    if unique_scores.size == 1:
        for row in positives:
            row["signal_tier"] = "core"
        return

    cluster_count = min(3, unique_scores.size, len(positives))
    labels = KMeans(n_clusters=cluster_count, n_init=10, random_state=42).fit_predict(
        positive_scores
    )
    centroids = {
        cluster_id: float(np.mean(positive_scores[labels == cluster_id]))
        for cluster_id in range(cluster_count)
    }
    ordered_clusters = [
        cluster_id
        for cluster_id, _score in sorted(
            centroids.items(), key=lambda item: item[1], reverse=True
        )
    ]
    tier_names = ["core", "supporting", "context"]
    label_to_tier = {
        cluster_id: tier_names[min(rank, len(tier_names) - 1)]
        for rank, cluster_id in enumerate(ordered_clusters)
    }
    for row, label in zip(positives, labels):
        row["signal_tier"] = label_to_tier[int(label)]


def _compact_numeric_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "column": row["column"],
        "direction": row["direction"],
        "mean": row["mean"],
        "global_mean": row["global_mean"],
        "mean_rest": row["mean_rest"],
        "median": row["median"],
        "median_rest": row["median_rest"],
        "z_score": row["z_score"],
        "homogeneity": row["homogeneity"],
        "aggregate_score": row["aggregate_score"],
        "percentile_score": row["percentile_score"],
        "signal_tier": row["signal_tier"],
        "one_vs_rest_q": row["one_vs_rest_q"],
        "global_q": row["global_q"],
        "neighbor_effect": row["neighbor_effect"],
        "sparkline": row["sparkline"],
    }


def _compact_categorical_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "column": row["column"],
        "value": row["value"],
        "count": row["count"],
        "prevalence_cluster": row["prevalence_cluster"],
        "prevalence_rest": row["prevalence_rest"],
        "global_recall": row["global_recall"],
        "lift": row["lift"],
        "aggregate_score": row["aggregate_score"],
        "percentile_score": row["percentile_score"],
        "signal_tier": row["signal_tier"],
        "fisher_q": row["fisher_q"],
        "global_q": row["global_q"],
        "neighbor_specificity": row["neighbor_specificity"],
        "concentration": row["in_cluster_prevalence"],
        "in_cluster_prevalence": row["in_cluster_prevalence"],
        "cluster_size": row["cluster_size"],
        "global_count": row["global_count"],
    }


def _detail_numeric_rows(
    rows: list[dict[str, Any]], detail: str
) -> list[dict[str, Any]]:
    core = [row for row in rows if row["signal_tier"] == "core"]
    supporting = [row for row in rows if row["signal_tier"] == "supporting"]
    context = [row for row in rows if row["signal_tier"] == "context"]
    if detail == "summary":
        return [_compact_numeric_row(row) for row in core + supporting]
    if detail == "standard":
        detailed = core + supporting
        return detailed + [_compact_numeric_row(row) for row in context]
    return rows


def _detail_categorical_rows(
    rows: list[dict[str, Any]], detail: str
) -> list[dict[str, Any]]:
    core = [row for row in rows if row["signal_tier"] == "core"]
    supporting = [row for row in rows if row["signal_tier"] == "supporting"]
    context = [row for row in rows if row["signal_tier"] == "context"]
    if detail == "summary":
        return [_compact_categorical_row(row) for row in core + supporting]
    if detail == "standard":
        detailed = core + supporting
        return detailed + [_compact_categorical_row(row) for row in context]
    return rows


def _categorical_ranking_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "column": row["column"],
        "value": row["value"],
        "cluster_id": row["cluster_id"],
        "aggregate_score": row["aggregate_score"],
        "signal_tier": row["signal_tier"],
    }


def _compute_numeric_rows(
    data: pd.DataFrame,
    numeric_cols: list[str],
    clusters: pd.Series,
    adjacency: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    global_numeric_stats: dict[str, dict[str, float]] = {}
    cluster_values = sorted(int(cid) for cid in clusters.unique())
    for column in numeric_cols:
        grouped = [
            _normalize_numeric(data.loc[clusters == cid, column])
            for cid in cluster_values
        ]
        grouped = [values for values in grouped if values.size > 0]
        p_value = 1.0
        if len(grouped) >= 2:
            try:
                p_value = float(stats.kruskal(*grouped).pvalue)
            except ValueError:
                p_value = 1.0
        global_scale = np.std(_normalize_numeric(data[column]))
        mean_dispersion = (
            np.std([_safe_mean(values) for values in grouped]) if grouped else 0.0
        )
        global_numeric_stats[column] = {
            "p_value": p_value,
            "effect": float(mean_dispersion / max(global_scale, _EPS)),
        }

    global_qs = _bh_fdr(
        [global_numeric_stats[column]["p_value"] for column in numeric_cols]
    )
    for column, q_value in zip(numeric_cols, global_qs):
        global_numeric_stats[column]["q_value"] = (
            1.0 if q_value is None else float(q_value)
        )

    rows: list[dict[str, Any]] = []
    for cluster_id in cluster_values:
        cluster_mask = clusters == cluster_id
        rest_mask = clusters != cluster_id
        neighbor_id = _strongest_neighbor(adjacency, cluster_id)
        for column in numeric_cols:
            cluster_values_arr = _normalize_numeric(data.loc[cluster_mask, column])
            rest_values_arr = _normalize_numeric(data.loc[rest_mask, column])
            if cluster_values_arr.size == 0 or rest_values_arr.size == 0:
                continue
            global_values_arr = _normalize_numeric(data[column])
            neighbor_values_arr = (
                _normalize_numeric(data.loc[clusters == neighbor_id, column])
                if neighbor_id is not None
                else np.array([], dtype=float)
            )
            pooled = _pooled_std(cluster_values_arr, rest_values_arr)
            global_std = max(_safe_std(global_values_arr), _EPS)
            global_mad = max(_safe_mad(global_values_arr), _EPS)
            global_iqr = max(float(stats.iqr(global_values_arr)), _EPS)
            std_cluster = _safe_std(cluster_values_arr)
            std_rest = _safe_std(rest_values_arr)
            mad_cluster = _safe_mad(cluster_values_arr)
            mad_rest = _safe_mad(rest_values_arr)
            effect_mean_std = (
                _safe_mean(cluster_values_arr) - _safe_mean(rest_values_arr)
            ) / max(pooled, _EPS)
            effect_median_mad = (
                _safe_median(cluster_values_arr) - _safe_median(rest_values_arr)
            ) / max(1.4826 * max(mad_rest, global_mad), _EPS)
            try:
                ks_stat = float(
                    stats.ks_2samp(cluster_values_arr, rest_values_arr).statistic
                )
            except ValueError:
                ks_stat = 0.0
            wasserstein_norm = float(
                stats.wasserstein_distance(cluster_values_arr, rest_values_arr)
                / max(global_iqr, _EPS)
            )
            variance_ratio_log = float(
                math.log(
                    (_safe_var(cluster_values_arr) + _EPS)
                    / (_safe_var(rest_values_arr) + _EPS)
                )
            )
            concentration_gain = float(
                max(0.0, 1.0 - ((std_cluster + _EPS) / max(global_std, _EPS)))
            )
            try:
                one_vs_rest_p = float(
                    stats.mannwhitneyu(
                        cluster_values_arr,
                        rest_values_arr,
                        alternative="two-sided",
                    ).pvalue
                )
            except ValueError:
                one_vs_rest_p = 1.0

            neighbor_effect = 0.0
            neighbor_p = 1.0
            if neighbor_values_arr.size:
                neighbor_effect = (
                    _safe_mean(cluster_values_arr) - _safe_mean(neighbor_values_arr)
                ) / max(_pooled_std(cluster_values_arr, neighbor_values_arr), _EPS)
                try:
                    neighbor_p = float(
                        stats.mannwhitneyu(
                            cluster_values_arr,
                            neighbor_values_arr,
                            alternative="two-sided",
                        ).pvalue
                    )
                except ValueError:
                    neighbor_p = 1.0

            row = {
                "column": column,
                "cluster_id": int(cluster_id),
                "direction": _direction_from_effects(
                    effect_mean_std,
                    effect_median_mad,
                    neighbor_effect,
                ),
                "mean": _safe_mean(cluster_values_arr),
                "mean_rest": _safe_mean(rest_values_arr),
                "median": _safe_median(cluster_values_arr),
                "median_rest": _safe_median(rest_values_arr),
                "std_cluster": std_cluster,
                "std_rest": std_rest,
                "mad_cluster": mad_cluster,
                "mad_rest": mad_rest,
                "global_mean": _safe_mean(global_values_arr),
                "global_mean_rest": _safe_mean(rest_values_arr),
                "z_score": (
                    _safe_mean(cluster_values_arr) - _safe_mean(global_values_arr)
                )
                / max(global_std, _EPS),
                "homogeneity": std_cluster / max(global_std, _EPS),
                "effect_mean_std": float(effect_mean_std),
                "effect_median_mad": float(effect_median_mad),
                "ks_stat": ks_stat,
                "wasserstein_norm": wasserstein_norm,
                "variance_ratio_log": variance_ratio_log,
                "concentration_gain": concentration_gain,
                "one_vs_rest_p": one_vs_rest_p,
                "one_vs_rest_q": 1.0,
                "global_p": float(global_numeric_stats[column]["p_value"]),
                "global_q": float(global_numeric_stats[column]["q_value"]),
                "global_cluster_signal": float(global_numeric_stats[column]["effect"]),
                "neighbor_cluster_id": neighbor_id,
                "neighbor_effect": float(neighbor_effect),
                "neighbor_p": float(neighbor_p),
                "specificity_score": float(
                    np.mean(
                        [
                            abs(effect_mean_std),
                            abs(effect_median_mad),
                            abs(neighbor_effect),
                            ks_stat,
                            wasserstein_norm,
                        ]
                    )
                ),
                "aggregate_score": 0.0,
                "percentile_score": 0.0,
                "signal_tier": "noise",
                "sparkline": generate_distribution_sparkline(cluster_values_arr),
            }
            rows.append(row)

    one_vs_rest_qs = _bh_fdr([row["one_vs_rest_p"] for row in rows])
    for row, q_value in zip(rows, one_vs_rest_qs):
        row["one_vs_rest_q"] = 1.0 if q_value is None else float(q_value)

    _apply_numeric_scores(rows)
    return rows, global_numeric_stats


def _compute_categorical_rows(
    data: pd.DataFrame,
    categorical_cols: list[str],
    clusters: pd.Series,
    adjacency: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    global_categorical_stats: dict[str, dict[str, float]] = {}
    cluster_values = sorted(int(cid) for cid in clusters.unique())
    for column in categorical_cols:
        encoded = data[column].fillna("__MISSING__").astype(str)
        contingency = pd.crosstab(clusters, encoded)
        p_value = 1.0
        association = 0.0
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            try:
                chi2, p_value, _dof, _expected = stats.chi2_contingency(contingency)
                association = float(chi2 / max(contingency.values.sum(), 1))
            except ValueError:
                p_value = 1.0
        global_categorical_stats[column] = {
            "p_value": float(p_value),
            "association": association,
        }

    global_qs = _bh_fdr(
        [global_categorical_stats[column]["p_value"] for column in categorical_cols]
    )
    for column, q_value in zip(categorical_cols, global_qs):
        global_categorical_stats[column]["q_value"] = (
            1.0 if q_value is None else float(q_value)
        )

    rows: list[dict[str, Any]] = []
    n_total = len(data)
    for cluster_id in cluster_values:
        cluster_mask = clusters == cluster_id
        rest_mask = clusters != cluster_id
        cluster_size = int(cluster_mask.sum())
        rest_size = int(rest_mask.sum())
        neighbor_id = _strongest_neighbor(adjacency, cluster_id)
        neighbor_mask = clusters == neighbor_id if neighbor_id is not None else None
        neighbor_size = int(neighbor_mask.sum()) if neighbor_mask is not None else 0
        for column in categorical_cols:
            global_values = data[column].fillna("__MISSING__").astype(str)
            cluster_values_series = global_values.loc[cluster_mask]
            rest_values_series = global_values.loc[rest_mask]
            cluster_counts = cluster_values_series.value_counts()
            for value, count in cluster_counts.items():
                global_count = int((global_values == value).sum())
                count_rest = int((rest_values_series == value).sum())
                prevalence_cluster = (count / cluster_size) if cluster_size else 0.0
                prevalence_rest = (count_rest / rest_size) if rest_size else 0.0
                prevalence_global = (global_count / n_total) if n_total else 0.0
                lift = prevalence_cluster / max(prevalence_global, _EPS)
                log_lift = float(math.log(max(lift, _EPS)))
                neighbor_prevalence = 0.0
                if neighbor_mask is not None and neighbor_size:
                    neighbor_prevalence = float(
                        ((global_values.loc[neighbor_mask] == value).sum())
                        / neighbor_size
                    )
                p_cv = count / max(n_total, 1)
                p_c = cluster_size / max(n_total, 1)
                p_v = global_count / max(n_total, 1)
                mi_contrib = float(
                    p_cv * math.log(max(p_cv, _EPS) / max(p_c * p_v, _EPS))
                )
                fisher_p = 1.0
                try:
                    fisher_p = float(
                        stats.fisher_exact(
                            [
                                [count, max(cluster_size - count, 0)],
                                [count_rest, max(rest_size - count_rest, 0)],
                            ]
                        )[1]
                    )
                except ValueError:
                    fisher_p = 1.0

                rows.append(
                    {
                        "column": column,
                        "value": str(value),
                        "cluster_id": int(cluster_id),
                        "count": int(count),
                        "count_cluster": int(count),
                        "count_rest": int(count_rest),
                        "cluster_size": int(cluster_size),
                        "global_count": int(global_count),
                        "prevalence_cluster": float(prevalence_cluster * 100.0),
                        "prevalence_rest": float(prevalence_rest * 100.0),
                        "prevalence_global": float(prevalence_global * 100.0),
                        "lift": float(lift),
                        "log_lift": float(log_lift),
                        "specificity": float(prevalence_cluster - prevalence_rest),
                        "global_recall": float(
                            (count / global_count) * 100.0 if global_count else 0.0
                        ),
                        "neighbor_specificity": float(
                            prevalence_cluster - neighbor_prevalence
                        ),
                        "mi_contrib": float(mi_contrib),
                        "fisher_p": float(fisher_p),
                        "fisher_q": 1.0,
                        "global_p": float(global_categorical_stats[column]["p_value"]),
                        "global_q": float(global_categorical_stats[column]["q_value"]),
                        "aggregate_score": 0.0,
                        "percentile_score": 0.0,
                        "signal_tier": "noise",
                        "in_cluster_prevalence": float(prevalence_cluster * 100.0),
                        "concentration": float(prevalence_cluster * 100.0),
                    }
                )

    fisher_qs = _bh_fdr([row["fisher_p"] for row in rows])
    for row, q_value in zip(rows, fisher_qs):
        row["fisher_q"] = 1.0 if q_value is None else float(q_value)

    _apply_categorical_scores(rows)
    return rows, global_categorical_stats


def _rank_numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    scores: dict[str, float] = {}
    for row in rows:
        scores[row["column"]] = max(
            scores.get(row["column"], 0.0), row["aggregate_score"]
        )
    return [
        column
        for column, _score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def _rank_categorical_values(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["aggregate_score"],
            row["column"],
            row["value"],
            row["cluster_id"],
        ),
    )
    return [_categorical_ranking_payload(row) for row in ranked]


def _assemble_signal_matrix(
    cluster_bundles: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    numeric_columns = sorted(
        {
            row["column"]
            for bundle in cluster_bundles.values()
            for row in bundle["numeric"]
            if row["signal_tier"] in {"core", "supporting"}
        }
    )
    categorical_pairs = sorted(
        {
            (row["column"], row["value"])
            for bundle in cluster_bundles.values()
            for row in bundle["categorical"]
            if row["signal_tier"] in {"core", "supporting"}
        }
    )

    numeric_rows = []
    categorical_rows = []
    for cluster_id in sorted(cluster_bundles):
        bundle = cluster_bundles[cluster_id]
        numeric_map = {row["column"]: row for row in bundle["numeric"]}
        categorical_map = {
            (row["column"], row["value"]): row for row in bundle["categorical"]
        }
        numeric_rows.append(
            {
                "cluster_id": cluster_id,
                "values": {
                    column: {
                        "z_score": numeric_map[column]["z_score"],
                        "aggregate_score": numeric_map[column]["aggregate_score"],
                        "signal_tier": numeric_map[column]["signal_tier"],
                    }
                    for column in numeric_columns
                    if column in numeric_map
                },
            }
        )
        categorical_rows.append(
            {
                "cluster_id": cluster_id,
                "values": {
                    f"{column}={value}": {
                        "prevalence": categorical_map[(column, value)][
                            "in_cluster_prevalence"
                        ],
                        "aggregate_score": categorical_map[(column, value)][
                            "aggregate_score"
                        ],
                        "signal_tier": categorical_map[(column, value)]["signal_tier"],
                    }
                    for column, value in categorical_pairs
                    if (column, value) in categorical_map
                },
            }
        )

    return {
        "numeric_columns": numeric_columns,
        "categorical_values": [
            {"column": column, "value": value} for column, value in categorical_pairs
        ],
        "numeric_rows": numeric_rows,
        "categorical_rows": categorical_rows,
    }


def _auto_semantic_name(bundle: dict[str, Any], cluster_id: int) -> str:
    numeric_candidates = [
        row for row in bundle["numeric"] if row["signal_tier"] in {"core", "supporting"}
    ]
    categorical_candidates = [
        row for row in bundle["categorical"] if row["signal_tier"] == "core"
    ]
    fragments: list[str] = []
    for row in numeric_candidates[:2]:
        direction = (
            "High"
            if row["z_score"] > 0.5
            else "Low"
            if row["z_score"] < -0.5
            else "Shifted"
        )
        fragments.append(f"{direction} {row['column']}")
    if categorical_candidates:
        lead_cat = categorical_candidates[0]
        fragments.append(f"{lead_cat['value']} {lead_cat['column']}")
    if not fragments:
        return f"Cluster {cluster_id}"
    return "[Auto] " + ", ".join(fragments[:3])


def build_feature_evidence_index(
    model: ThemaRS,
    data: pd.DataFrame,
    clusters: pd.Series,
    exclude_columns: list[str] | None = None,
) -> FeatureEvidenceIndex:
    """Compute distribution-aware evidence for numeric and categorical signals."""
    if len(clusters) != len(data):
        raise ValueError(
            f"Alignment error: clusters({len(clusters)}) != data({len(data)})"
        )

    working = data.copy()
    if exclude_columns:
        to_drop = [column for column in exclude_columns if column in working.columns]
        if to_drop:
            working = working.drop(columns=to_drop)

    numeric_cols = working.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = working.select_dtypes(exclude=[np.number]).columns.tolist()
    adjacency = _graph_cluster_adjacency(model, clusters)
    numeric_rows, global_numeric_stats = _compute_numeric_rows(
        working, numeric_cols, clusters, adjacency
    )
    categorical_rows, global_categorical_stats = _compute_categorical_rows(
        working, categorical_cols, clusters, adjacency
    )

    cluster_bundles: dict[int, dict[str, Any]] = {}
    for cluster_id in sorted(int(cid) for cid in clusters.unique()):
        bundle_numeric = sorted(
            [row for row in numeric_rows if row["cluster_id"] == cluster_id],
            key=lambda row: (-row["aggregate_score"], row["column"]),
        )
        bundle_categorical = sorted(
            [row for row in categorical_rows if row["cluster_id"] == cluster_id],
            key=lambda row: (-row["aggregate_score"], row["column"], row["value"]),
        )
        _assign_signal_tiers(bundle_numeric)
        _assign_signal_tiers(bundle_categorical)
        bundle = {
            "cluster_id": cluster_id,
            "size": int((clusters == cluster_id).sum()),
            "size_pct": float(((clusters == cluster_id).sum() / len(working)) * 100.0),
            "numeric": bundle_numeric,
            "categorical": bundle_categorical,
            "topological_neighbors": adjacency.get(cluster_id, []),
        }
        bundle["semantic_name"] = _auto_semantic_name(bundle, cluster_id)
        cluster_bundles[cluster_id] = bundle

    signal_matrix = _assemble_signal_matrix(cluster_bundles)
    return FeatureEvidenceIndex(
        cluster_bundles=cluster_bundles,
        numeric_global_ranking=_rank_numeric_columns(numeric_rows),
        categorical_global_ranking=_rank_categorical_values(categorical_rows),
        signal_matrix=signal_matrix,
        metadata={
            "n_total": len(working),
            "n_clusters": int(clusters.nunique()),
            "columns": working.columns.tolist(),
            "excluded_columns": exclude_columns or [],
            "numeric_features_screened": len(numeric_cols),
            "categorical_columns_screened": len(categorical_cols),
            "categorical_values_screened": len(categorical_rows),
            "tiering_method": "adaptive_kmeans_on_aggregate_percentiles",
            "neighbor_contrast_enabled": True,
            "global_numeric_stats": global_numeric_stats,
            "global_categorical_stats": global_categorical_stats,
            "cluster_adjacency": adjacency,
        },
    )


def _select_top_columns(
    numeric_rows: list[dict[str, Any]], categorical_rows: list[dict[str, Any]]
) -> list[str]:
    columns = [row["column"] for row in numeric_rows[:10]]
    columns.extend(row["column"] for row in categorical_rows[:5])
    return list(dict.fromkeys(columns))


def build_dossier(
    model: ThemaRS,
    data: pd.DataFrame,
    clusters: pd.Series,
    exclude_columns: list[str] | None = None,
    *,
    detail: str = "standard",
    evidence_index: FeatureEvidenceIndex | None = None,
) -> TopologicalDossier:
    """Build a tiered dossier from cached distribution-aware feature evidence."""
    if detail not in {"summary", "standard", "full"}:
        raise ValueError("detail must be 'summary', 'standard', or 'full'")

    if evidence_index is None:
        evidence_index = build_feature_evidence_index(
            model,
            data,
            clusters,
            exclude_columns=exclude_columns,
        )

    working = data.copy()
    excluded = evidence_index.metadata.get("excluded_columns", [])
    if excluded:
        to_drop = [column for column in excluded if column in working.columns]
        if to_drop:
            working = working.drop(columns=to_drop)

    cluster_profiles = []
    for cluster_id in sorted(evidence_index.cluster_bundles):
        bundle = evidence_index.cluster_bundles[cluster_id]
        profile = ClusterProfile(
            cluster_id=int(cluster_id),
            size=int(bundle["size"]),
            size_pct=float(bundle["size_pct"]),
            semantic_name=str(bundle["semantic_name"]),
            numeric_tiers={
                "core": [
                    row for row in bundle["numeric"] if row["signal_tier"] == "core"
                ],
                "supporting": [
                    row
                    for row in bundle["numeric"]
                    if row["signal_tier"] == "supporting"
                ],
                "context": [
                    row for row in bundle["numeric"] if row["signal_tier"] == "context"
                ],
            },
            categorical_tiers={
                "core": [
                    row for row in bundle["categorical"] if row["signal_tier"] == "core"
                ],
                "supporting": [
                    row
                    for row in bundle["categorical"]
                    if row["signal_tier"] == "supporting"
                ],
                "context": [
                    row
                    for row in bundle["categorical"]
                    if row["signal_tier"] == "context"
                ],
            },
            topological_neighbors=list(bundle["topological_neighbors"]),
        )
        profile.numeric_features = _detail_numeric_rows(bundle["numeric"], detail)
        profile.categorical_features = _detail_categorical_rows(
            bundle["categorical"], detail
        )

        cluster_mask = clusters == cluster_id
        cluster_data = working.loc[cluster_mask]
        try:
            cluster_nodes = list(np.where(clusters.values == cluster_id)[0])
            sub_graph = model.cosmic_graph.subgraph(cluster_nodes)
            if len(sub_graph) > 0:
                pagerank = nx.pagerank(sub_graph, weight="weight")
                central_ids = sorted(pagerank, key=pagerank.get, reverse=True)[:3]
                top_cols = _select_top_columns(bundle["numeric"], bundle["categorical"])
                top_cols = top_cols or working.columns[:10].tolist()
                profile.central_rows = working.iloc[central_ids][top_cols].to_dict(
                    "records"
                )
        except Exception:
            top_cols = _select_top_columns(bundle["numeric"], bundle["categorical"])
            top_cols = top_cols or working.columns[:10].tolist()
            profile.central_rows = cluster_data[top_cols].head(3).to_dict("records")

        cluster_profiles.append(profile)

    try:
        from dataclasses import asdict

        graph_metrics = asdict(diagnose_model(model))
    except Exception:
        graph_metrics = {}

    return TopologicalDossier(
        n_total=int(evidence_index.metadata["n_total"]),
        n_clusters=int(evidence_index.metadata["n_clusters"]),
        clusters=cluster_profiles,
        global_stats={
            "numeric": {
                column: stats_payload["effect"]
                for column, stats_payload in evidence_index.metadata[
                    "global_numeric_stats"
                ].items()
            },
            "columns": evidence_index.metadata["columns"],
            "graph_metrics": graph_metrics,
            "evidence_metadata": {
                key: value
                for key, value in evidence_index.metadata.items()
                if key
                in {
                    "numeric_features_screened",
                    "categorical_columns_screened",
                    "categorical_values_screened",
                    "tiering_method",
                    "neighbor_contrast_enabled",
                    "excluded_columns",
                }
            },
            "cluster_adjacency": evidence_index.metadata["cluster_adjacency"],
            "signal_matrix": evidence_index.signal_matrix,
            "numeric_global_ranking": evidence_index.numeric_global_ranking,
            "categorical_global_ranking": evidence_index.categorical_global_ranking,
            "detail": detail,
        },
        cluster_labels=clusters,
    )


def cluster_profile_payload(
    evidence_index: FeatureEvidenceIndex,
    cluster_id: int,
    *,
    detail: str = "standard",
) -> dict[str, Any]:
    bundle = evidence_index.cluster_bundles.get(int(cluster_id))
    if bundle is None:
        raise KeyError(cluster_id)
    return {
        "cluster_id": int(cluster_id),
        "size": int(bundle["size"]),
        "size_pct": float(bundle["size_pct"]),
        "semantic_name": str(bundle["semantic_name"]),
        "topological_neighbors": list(bundle["topological_neighbors"]),
        "numeric_features": _detail_numeric_rows(bundle["numeric"], detail),
        "categorical_features": _detail_categorical_rows(bundle["categorical"], detail),
        "tier_counts": {
            "numeric": {
                tier: sum(1 for row in bundle["numeric"] if row["signal_tier"] == tier)
                for tier in ("core", "supporting", "context", "noise")
            },
            "categorical": {
                tier: sum(
                    1 for row in bundle["categorical"] if row["signal_tier"] == tier
                )
                for tier in ("core", "supporting", "context", "noise")
            },
        },
    }


def feature_signal_payload(
    evidence_index: FeatureEvidenceIndex,
    feature_names: list[str],
    *,
    cluster_ids: list[int] | None = None,
) -> dict[str, Any]:
    requested_clusters = (
        {int(cluster_id) for cluster_id in cluster_ids}
        if cluster_ids is not None
        else set(evidence_index.cluster_bundles.keys())
    )
    features = [feature for feature in feature_names if feature]
    categorical_pairs = set()
    bare_columns = set()
    for feature in features:
        if "=" in feature:
            column, value = feature.split("=", 1)
            categorical_pairs.add((column, value))
        bare_columns.add(feature.split("=", 1)[0])
    cluster_payloads = []
    for cluster_id in sorted(requested_clusters):
        bundle = evidence_index.cluster_bundles.get(cluster_id)
        if bundle is None:
            continue
        numeric_rows = [
            row for row in bundle["numeric"] if row["column"] in bare_columns
        ]
        categorical_rows = [
            row
            for row in bundle["categorical"]
            if row["column"] in bare_columns
            or (row["column"], row["value"]) in categorical_pairs
        ]
        if not numeric_rows and not categorical_rows:
            continue
        cluster_payloads.append(
            {
                "cluster_id": cluster_id,
                "semantic_name": bundle["semantic_name"],
                "numeric_features": numeric_rows,
                "categorical_features": categorical_rows,
            }
        )
    return {
        "feature_names": features,
        "clusters": cluster_payloads,
    }


def signal_matrix_payload(
    evidence_index: FeatureEvidenceIndex,
    *,
    cluster_ids: list[int] | None = None,
    include_context: bool = False,
) -> dict[str, Any]:
    requested_clusters = (
        {int(cluster_id) for cluster_id in cluster_ids}
        if cluster_ids is not None
        else set(evidence_index.cluster_bundles.keys())
    )
    matrix = evidence_index.signal_matrix
    numeric_rows = [
        {
            **row,
            "values": dict(row["values"]),
        }
        for row in matrix["numeric_rows"]
        if row["cluster_id"] in requested_clusters
    ]
    categorical_rows = [
        {
            **row,
            "values": dict(row["values"]),
        }
        for row in matrix["categorical_rows"]
        if row["cluster_id"] in requested_clusters
    ]
    if not include_context:
        for row_set in (numeric_rows, categorical_rows):
            for row in row_set:
                row["values"] = {
                    key: value
                    for key, value in row["values"].items()
                    if value.get("signal_tier") in {"core", "supporting"}
                }
    return {
        "numeric_columns": matrix["numeric_columns"],
        "categorical_values": matrix["categorical_values"],
        "numeric_rows": numeric_rows,
        "categorical_rows": categorical_rows,
    }


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
      padding: 4px 0;
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
      padding: 8px 10px;
      border-bottom: 1px solid var(--hairline);
      text-align: left;
      vertical-align: top;
    }

    .heatmap-table th,
    .data-table th {
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
      font-weight: 600;
    }

    .heatmap-table tbody tr:last-child td,
    .data-table tbody tr:last-child td {
      border-bottom: none;
    }

    .heatmap-table td:first-child,
    .heatmap-table th:first-child {
      min-width: 12.5rem;
    }

    .heat-cell {
      text-align: center;
      white-space: nowrap;
    }

    .heat-value {
      display: block;
      font-family: var(--font-sans);
      font-size: 0.8rem;
      color: var(--ink);
      font-weight: 500;
      line-height: 1.15;
    }

    .heat-subvalue {
      display: block;
      margin-top: 2px;
      font-size: 0.64rem;
      color: var(--muted);
      letter-spacing: 0.01em;
      line-height: 1.15;
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
    categorical_scores: dict[str, float] = {}

    for profile in dossier.clusters:
        for feature in profile.numeric_features:
            numeric_scores[feature["column"]] = numeric_scores.get(
                feature["column"], 0.0
            ) + abs(float(feature.get("z_score", 0.0)))
        for feature in profile.categorical_features:
            column_name = str(feature["column"])
            prevalence = float(
                feature.get("in_cluster_prevalence", feature.get("concentration", 0.0))
            )
            global_recall = float(feature.get("global_recall", 0.0))
            score = prevalence + (0.25 * global_recall)
            categorical_scores[column_name] = (
                categorical_scores.get(column_name, 0.0) + score
            )

    numeric_features = [
        name
        for name, _score in sorted(
            numeric_scores.items(), key=lambda item: (-item[1], item[0])
        )[:_MAX_SIGNAL_MATRIX_NUMERIC]
    ]
    categorical_columns = [
        name
        for name, _score in sorted(
            categorical_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:_MAX_SIGNAL_MATRIX_CATEGORICAL]
    ]

    if not numeric_features and not categorical_columns:
        return ""

    parts = [
        "<section id='heatmap' class='report-section'>",
        "<div class='section-header'>",
        "<div>",
        "<span class='section-eyebrow'>Figure 2</span>",
        "<h2>Cohort signal matrix</h2>",
        "</div>",
        f"<p class='section-subtitle'>Bounded for readability: up to {_MAX_SIGNAL_MATRIX_NUMERIC} numeric features and {_MAX_SIGNAL_MATRIX_CATEGORICAL} categorical columns. Numeric values are cohort means, and the smaller line below each numeric value is the z-shift.</p>",
        "</div>",
        "<div class='heatmap-shell'>",
        "<table class='heatmap-table'><thead>",
        "<tr>",
        "<th rowspan='2'>Cohort</th>",
    ]
    if numeric_features:
        parts.append(f"<th colspan='{len(numeric_features)}'>Numeric means</th>")
    if categorical_columns:
        parts.append(
            f"<th colspan='{len(categorical_columns)}'>Dominant categories</th>"
        )
    parts.append("</tr><tr>")
    for feature_name in numeric_features:
        parts.append(f"<th>{_escape_html(feature_name)}</th>")
    for column_name in categorical_columns:
        parts.append(f"<th>{_escape_html(column_name)}</th>")
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
        categorical_map: dict[str, list[dict[str, Any]]] = {}
        for feature in profile.categorical_features:
            categorical_map.setdefault(str(feature["column"]), []).append(feature)
        for column_name in categorical_columns:
            column_features = categorical_map.get(column_name, [])
            if not column_features:
                parts.append(
                    "<td class='heat-cell'>"
                    "<span class='heat-value'>—</span>"
                    "<span class='heat-subvalue'>no match</span>"
                    "</td>"
                )
                continue
            feature = max(
                column_features,
                key=lambda item: (
                    float(
                        item.get(
                            "in_cluster_prevalence", item.get("concentration", 0.0)
                        )
                    ),
                    int(item.get("count", 0)),
                ),
            )
            prevalence = float(
                feature.get("in_cluster_prevalence", feature.get("concentration", 0.0))
            )
            count = int(feature.get("count", 0))
            if count <= 0 or prevalence <= 0:
                parts.append(
                    "<td class='heat-cell'>"
                    "<span class='heat-value'>—</span>"
                    "<span class='heat-subvalue'>no match</span>"
                    "</td>"
                )
                continue
            alpha = min(0.38, prevalence * 0.0038)
            tone_class = "heat-positive" if prevalence >= 50.0 else ""
            style = f" style='--heat-alpha:{alpha:.2f}'" if prevalence > 0 else ""
            parts.append(
                f"<td class='heat-cell {tone_class}'{style}>"
                f"<span class='heat-value'>{_escape_html(feature['value'])}</span>"
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
