"""
Topological Interpreter Engine.

Translates the raw topological graph and clustered data into a high-signal
statistical dossier for LLM synthesis. HTML report rendering lives in
``pulsar.mcp.report``.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

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
    working_columns: List[str] = field(default_factory=list)


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


MetricExtractor = tuple[str, "Callable[[dict[str, Any]], float]"]


def _apply_percentile_aggregate(
    rows: list[dict[str, Any]],
    extractors: list[MetricExtractor],
) -> None:
    """Populate aggregate_score, percentile_score, and evidence_vector on each row.

    For each metric in *extractors*, compute empirical percentiles across rows.
    The per-row aggregate is the geometric mean of the component percentiles
    (with a small positivity offset). The per-row percentile_score is the
    empirical percentile of that aggregate.
    """
    component_percentiles = {
        name: _empirical_percentiles([extract(row) for row in rows])
        for name, extract in extractors
    }
    names = [name for name, _ in extractors]
    aggregates: list[float] = []
    for row_index, row in enumerate(rows):
        evidence_vector = {
            name: component_percentiles[name][row_index] for name in names
        }
        positive = [max(value, 0.0) + 1e-6 for value in evidence_vector.values()]
        aggregate = (
            float(math.exp(np.mean(np.log(positive))) - 1e-6) if positive else 0.0
        )
        row["aggregate_score"] = max(aggregate, 0.0)
        row["evidence_vector"] = evidence_vector
        aggregates.append(row["aggregate_score"])

    for row, percentile in zip(rows, _empirical_percentiles(aggregates)):
        row["percentile_score"] = percentile


_NUMERIC_METRIC_EXTRACTORS: list[MetricExtractor] = [
    ("effect_mean_std", lambda row: abs(float(row.get("effect_mean_std", 0.0)))),
    ("effect_median_mad", lambda row: abs(float(row.get("effect_median_mad", 0.0)))),
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
    ("neighbor_effect", lambda row: abs(float(row.get("neighbor_effect", 0.0)))),
]


_CATEGORICAL_METRIC_EXTRACTORS: list[MetricExtractor] = [
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


def _apply_numeric_scores(rows: list[dict[str, Any]]) -> None:
    _apply_percentile_aggregate(rows, _NUMERIC_METRIC_EXTRACTORS)


def _apply_categorical_scores(rows: list[dict[str, Any]]) -> None:
    _apply_percentile_aggregate(rows, _CATEGORICAL_METRIC_EXTRACTORS)


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


def _tier_filter(
    rows: list[dict[str, Any]],
    detail: str,
    compact_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter rows by detail tier, compacting the low-priority subset."""
    core = [row for row in rows if row["signal_tier"] == "core"]
    supporting = [row for row in rows if row["signal_tier"] == "supporting"]
    context = [row for row in rows if row["signal_tier"] == "context"]
    if detail == "summary":
        return [compact_fn(row) for row in core + supporting]
    if detail == "standard":
        return core + supporting + [compact_fn(row) for row in context]
    return rows


def _detail_numeric_rows(
    rows: list[dict[str, Any]], detail: str
) -> list[dict[str, Any]]:
    return _tier_filter(rows, detail, _compact_numeric_row)


def _detail_categorical_rows(
    rows: list[dict[str, Any]], detail: str
) -> list[dict[str, Any]]:
    return _tier_filter(rows, detail, _compact_categorical_row)


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
    global_numeric_stats: dict[str, dict[str, Any]] = {}
    cluster_values = sorted(int(cid) for cid in clusters.unique())
    for column in numeric_cols:
        grouped = [
            _normalize_numeric(data.loc[clusters == cid, column])
            for cid in cluster_values
        ]
        grouped = [values for values in grouped if values.size > 0]
        p_value = 1.0
        column_failures: list[str] = []
        if len(grouped) >= 2:
            try:
                p_value = float(stats.kruskal(*grouped).pvalue)
            except ValueError as exc:
                column_failures.append(f"kruskal: {exc}")
                logger.warning("kruskal failed for column %s: %s", column, exc)
        global_scale = np.std(_normalize_numeric(data[column]))
        mean_dispersion = (
            np.std([_safe_mean(values) for values in grouped]) if grouped else 0.0
        )
        global_numeric_stats[column] = {
            "p_value": p_value,
            "effect": float(mean_dispersion / max(global_scale, _EPS)),
            "failure_reasons": column_failures,
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
            row_failures: list[str] = []
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
            except ValueError as exc:
                ks_stat = 0.0
                row_failures.append(f"ks_2samp: {exc}")
                logger.warning(
                    "ks_2samp failed for cluster=%s column=%s: %s",
                    cluster_id,
                    column,
                    exc,
                )
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
            except ValueError as exc:
                one_vs_rest_p = 1.0
                row_failures.append(f"mannwhitneyu(one_vs_rest): {exc}")
                logger.warning(
                    "mannwhitneyu(one_vs_rest) failed for cluster=%s column=%s: %s",
                    cluster_id,
                    column,
                    exc,
                )

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
                except ValueError as exc:
                    neighbor_p = 1.0
                    row_failures.append(f"mannwhitneyu(neighbor): {exc}")
                    logger.warning(
                        "mannwhitneyu(neighbor) failed for cluster=%s column=%s: %s",
                        cluster_id,
                        column,
                        exc,
                    )

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
                "failure_reasons": row_failures,
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
    global_categorical_stats: dict[str, dict[str, Any]] = {}
    cluster_values = sorted(int(cid) for cid in clusters.unique())
    for column in categorical_cols:
        encoded = data[column].fillna("__MISSING__").astype(str)
        contingency = pd.crosstab(clusters, encoded)
        p_value = 1.0
        association = 0.0
        column_failures: list[str] = []
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            try:
                chi2, p_value, _dof, _expected = stats.chi2_contingency(contingency)
                association = float(chi2 / max(contingency.values.sum(), 1))
            except ValueError as exc:
                column_failures.append(f"chi2_contingency: {exc}")
                logger.warning("chi2_contingency failed for column %s: %s", column, exc)
        global_categorical_stats[column] = {
            "p_value": float(p_value),
            "association": association,
            "failure_reasons": column_failures,
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
                row_failures: list[str] = []
                try:
                    fisher_p = float(
                        stats.fisher_exact(
                            [
                                [count, max(cluster_size - count, 0)],
                                [count_rest, max(rest_size - count_rest, 0)],
                            ]
                        )[1]
                    )
                except ValueError as exc:
                    row_failures.append(f"fisher_exact: {exc}")
                    logger.warning(
                        "fisher_exact failed for cluster=%s column=%s value=%s: %s",
                        cluster_id,
                        column,
                        value,
                        exc,
                    )

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
                        "failure_reasons": row_failures,
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


def _collect_stats_failures(
    *,
    numeric_rows: list[dict[str, Any]],
    categorical_rows: list[dict[str, Any]],
    global_numeric_stats: dict[str, dict[str, Any]],
    global_categorical_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate statistical test failures for surfacing to the dossier."""
    numeric_column_level = {
        column: list(payload.get("failure_reasons", []))
        for column, payload in global_numeric_stats.items()
        if payload.get("failure_reasons")
    }
    categorical_column_level = {
        column: list(payload.get("failure_reasons", []))
        for column, payload in global_categorical_stats.items()
        if payload.get("failure_reasons")
    }
    numeric_row_level = [
        {
            "cluster_id": int(row["cluster_id"]),
            "column": row["column"],
            "reasons": list(row["failure_reasons"]),
        }
        for row in numeric_rows
        if row.get("failure_reasons")
    ]
    categorical_row_level = [
        {
            "cluster_id": int(row["cluster_id"]),
            "column": row["column"],
            "value": row["value"],
            "reasons": list(row["failure_reasons"]),
        }
        for row in categorical_rows
        if row.get("failure_reasons")
    ]
    return {
        "numeric_column_level": numeric_column_level,
        "categorical_column_level": categorical_column_level,
        "numeric_row_level": numeric_row_level,
        "categorical_row_level": categorical_row_level,
    }


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
    stats_failures = _collect_stats_failures(
        numeric_rows=numeric_rows,
        categorical_rows=categorical_rows,
        global_numeric_stats=global_numeric_stats,
        global_categorical_stats=global_categorical_stats,
    )
    working_columns = working.columns.tolist()
    return FeatureEvidenceIndex(
        cluster_bundles=cluster_bundles,
        numeric_global_ranking=_rank_numeric_columns(numeric_rows),
        categorical_global_ranking=_rank_categorical_values(categorical_rows),
        signal_matrix=signal_matrix,
        metadata={
            "n_total": len(working),
            "n_clusters": int(clusters.nunique()),
            "columns": working_columns,
            "excluded_columns": exclude_columns or [],
            "numeric_features_screened": len(numeric_cols),
            "categorical_columns_screened": len(categorical_cols),
            "categorical_values_screened": len(categorical_rows),
            "tiering_method": "adaptive_kmeans_on_aggregate_percentiles",
            "neighbor_contrast_enabled": True,
            "global_numeric_stats": global_numeric_stats,
            "global_categorical_stats": global_categorical_stats,
            "cluster_adjacency": adjacency,
            "stats_failures": stats_failures,
        },
        working_columns=working_columns,
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
    *,
    detail: str = "standard",
    evidence_index: FeatureEvidenceIndex | None = None,
) -> TopologicalDossier:
    """Build a tiered dossier from cached distribution-aware feature evidence."""
    if detail not in {"summary", "standard", "full"}:
        raise ValueError("detail must be 'summary', 'standard', or 'full'")

    if evidence_index is None:
        evidence_index = build_feature_evidence_index(model, data, clusters)

    working_columns = evidence_index.working_columns or [
        col
        for col in data.columns
        if col not in evidence_index.metadata.get("excluded_columns", [])
    ]

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
        try:
            cluster_nodes = list(np.where(clusters.values == cluster_id)[0])
            sub_graph = model.cosmic_graph.subgraph(cluster_nodes)
            if len(sub_graph) > 0:
                pagerank = nx.pagerank(sub_graph, weight="weight")
                central_ids = sorted(pagerank, key=pagerank.get, reverse=True)[:3]
                top_cols = _select_top_columns(bundle["numeric"], bundle["categorical"])
                top_cols = top_cols or working_columns[:10]
                profile.central_rows = data.iloc[central_ids][top_cols].to_dict(
                    "records"
                )
        except (
            KeyError,
            IndexError,
            nx.NetworkXError,
            nx.PowerIterationFailedConvergence,
        ) as exc:
            logger.warning(
                "central row selection via pagerank failed for cluster %s: %s",
                cluster_id,
                exc,
            )
            top_cols = _select_top_columns(bundle["numeric"], bundle["categorical"])
            top_cols = top_cols or working_columns[:10]
            profile.central_rows = (
                data.loc[cluster_mask, top_cols].head(3).to_dict("records")
            )

        cluster_profiles.append(profile)

    try:
        from dataclasses import asdict

        graph_metrics = asdict(diagnose_model(model))
    except (RuntimeError, AttributeError) as exc:
        logger.warning("diagnose_model failed: %s", exc)
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
                    "stats_failures",
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
