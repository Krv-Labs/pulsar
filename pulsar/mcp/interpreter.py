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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sparse_connected_components
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

from pulsar._pulsar import find_stable_thresholds, find_stable_thresholds_sparse
from pulsar.pipeline import ThemaRS
from pulsar.runtime.utils import (
    generate_distribution_sparkline,
    generate_proportion_bar,
)
from pulsar.mcp.diagnostics import diagnose_model
from pulsar.mcp.thresholds import (
    component_mass_profile,
    component_state_at_threshold,
    mass_profile_hint,
    useful_component_size_floor,
)

logger = logging.getLogger(__name__)

# Clustering strategy constants
_SPECTRAL_K_MIN = 2
_SPECTRAL_K_MAX = 20
# Reject a plateau only if >half its nodes are singletons — a scale-free dust
# guard. There is intentionally no absolute cap on component count (Pulsar must
# stay dataset-agnostic: large datasets can have many genuine cohorts).
_MAX_SINGLETON_RATIO = 0.5
_MAX_SIGNAL_MATRIX_NUMERIC = 10
_MAX_SIGNAL_MATRIX_CATEGORICAL = 5
_EPS = 1e-9
# Cochran's rule: chi-squared is reliable only when every expected 2x2 cell
# count is at least this large; below it we fall back to Fisher's exact test.
_CHI2_MIN_EXPECTED_CELL = 5.0
# A detail="full" dossier with more feature-cluster rows than this warns the
# reader to use targeted profiling tools instead of consuming the whole report.
_DOSSIER_OVERSIZE_ROW_WARNING = 500
# Categorical columns whose unique-value count exceeds
# max(_MIN_CARDINALITY_FLOOR, _MAX_CARDINALITY_RATIO * n_rows) are dropped from
# evidence as high-cardinality noise (IDs, free text). The floor protects
# small-cardinality columns (e.g. binary fields) at small dataset sizes.
_MAX_CARDINALITY_RATIO = 0.05
_MIN_CARDINALITY_FLOOR = 10


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
    interpretation_edge_weight_threshold_applied: float = 0.0
    stability_plateaus: list[dict] | None = None


class SpectralClusterCutError(ValueError):
    """Spectral search completed, but no evaluated k passed the cut floor."""

    def __init__(self, diagnostics: dict[str, Any]):
        super().__init__("No stable spectral cut found.")
        self.diagnostics = diagnostics

    def payload(self, *, detail: str = "summary") -> dict[str, Any]:
        diagnostics = dict(self.diagnostics)
        candidate_scores = diagnostics.pop("candidate_scores", [])
        if detail == "full":
            diagnostics["candidate_scores"] = candidate_scores
        return {
            "status": "error",
            "tool": "generate_cluster_dossier",
            "reason": "No stable spectral cut found.",
            "error_code": "NO_STABLE_SPECTRAL_CUT",
            "details": diagnostics,
        }


@dataclass
class FeatureEvidenceIndex:
    """Cached feature evidence derived from a clustering assignment."""

    cluster_bundles: Dict[int, Dict[str, Any]]
    numeric_global_ranking: List[str]
    categorical_global_ranking: List[Dict[str, Any]]
    signal_matrix: Dict[str, Any]
    metadata: Dict[str, Any]
    working_columns: List[str] = field(default_factory=list)
    categorical_columns_gated: List[Dict[str, Any]] = field(default_factory=list)


# Float tolerance for declaring that the interpretation threshold equals the
# fitted construction threshold (mirrors np.isclose defaults used elsewhere).
_THRESHOLD_MATCH_ATOL = 1e-8
_THRESHOLD_MATCH_RTOL = 1e-5


def cluster_provenance(
    result: "ClusterResult",
    resolved_construction_threshold: float,
) -> dict[str, Any]:
    """Self-describing provenance for a ``resolve_clusters`` partition.

    Emits the snake_case ``cluster_provenance`` contract (mirrors
    ``ClusterProvenance`` in isomorph ``packages/contracts/src/index.ts``) so a
    consumer never confuses a clustering count with the fitted cosmic-graph
    component count. Different methods cut a DIFFERENT matrix at a DIFFERENT
    threshold; this object says which, so divergent counts are expected, not a
    paradox.
    """
    construction = float(resolved_construction_threshold)
    labels = result.labels
    n_groups = int(labels.nunique())
    sizes = labels.value_counts()
    n_singletons = int((sizes == 1).sum())

    if result.method_used == "spectral":
        # Spectral labels every node into a community; there is no edge-weight
        # cut and no singleton notion.
        unit = "spectral_community"
        threshold_applied: float | None = None
        threshold_source = "none"
        base_matrix = "weighted_adjacency"
        matches_construction = False
        n_singletons = 0
    else:
        unit = "connected_component"
        threshold_applied = float(result.interpretation_edge_weight_threshold_applied)
        base_matrix = "weighted_adjacency"
        matches_construction = bool(
            math.isclose(
                threshold_applied,
                construction,
                rel_tol=_THRESHOLD_MATCH_RTOL,
                abs_tol=_THRESHOLD_MATCH_ATOL,
            )
        )
        if result.method_used == "threshold_stability":
            threshold_source = "stability_plateau_midpoint"
        elif threshold_applied == 0.0:
            threshold_source = "full_affinity"
        else:
            threshold_source = "explicit"

    comparable = unit == "connected_component" and matches_construction
    return {
        "unit": unit,
        "method_used": result.method_used,
        "threshold_applied": threshold_applied,
        "threshold_source": threshold_source,
        "base_matrix": base_matrix,
        "resolved_construction_threshold": construction,
        "matches_construction_threshold": matches_construction,
        "n_groups": n_groups,
        "n_singletons": n_singletons,
        "comparable_to_component_count": bool(comparable),
    }


def component_count_provenance(
    *,
    resolved_construction_threshold: float,
    component_count: int,
    singleton_count: int,
) -> dict[str, Any]:
    """Provenance for the fitted cosmic-graph connected-component count.

    This is the reference partition: connected components of the cosmic graph AT
    the construction threshold. It is the ONE count that is 1:1-comparable to
    itself, so ``comparable_to_component_count`` is true by definition.
    """
    construction = float(resolved_construction_threshold)
    return {
        "unit": "connected_component",
        "method_used": "components",
        "threshold_applied": construction,
        "threshold_source": "construction_threshold",
        "base_matrix": "cosmic_graph",
        "resolved_construction_threshold": construction,
        "matches_construction_threshold": True,
        "n_groups": int(component_count),
        "n_singletons": int(singleton_count),
        "comparable_to_component_count": True,
    }


def resolve_clusters(
    model: ThemaRS,
    method: str = "auto",
    max_k: int = 15,
    interpretation_edge_weight_threshold: float = 0.0,
) -> ClusterResult:
    """Entry point for clustering. Respects explicit method selection."""
    sparse_graph = _model_sparse_graph(model)
    W: np.ndarray | None = None
    if sparse_graph is None:
        W = model.weighted_adjacency
        n = W.shape[0]
    else:
        n, edges = sparse_graph

    # 1. Component Strategy
    if method == "components" or (
        method == "auto" and interpretation_edge_weight_threshold > 0
    ):
        thresh = max(float(interpretation_edge_weight_threshold), 0.0)
        if sparse_graph is not None:
            labels, n_clusters = _component_labels_from_edges(n, edges, thresh)
            return ClusterResult(
                labels=pd.Series(labels, name="cluster"),
                method_used="components",
                n_clusters=n_clusters,
                silhouette_score=None,
                failure_reason=None,
                interpretation_edge_weight_threshold_applied=thresh,
            )
        assert W is not None
        binary = (W > thresh).astype(np.int64)
        adj = W * binary
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
            interpretation_edge_weight_threshold_applied=thresh,
        )

    # 2. Threshold Stability (PH-based)
    if method in ("auto", "threshold_stability"):
        if sparse_graph is not None:
            result = _cluster_by_threshold_stability_sparse(
                n,
                edges,
                getattr(model, "_stability_result", None)
                or getattr(model, "stability_result", None),
            )
        else:
            assert W is not None
            result = _cluster_by_threshold_stability(W, n)
        if result:
            return result

    # 3. Spectral Fallback
    if method == "auto":
        if sparse_graph is not None:
            _labels, n_components = _component_labels_from_edges(n, edges, 0.0)
            if n_components > 1:
                return resolve_clusters(
                    model,
                    method="components",
                    max_k=max_k,
                    interpretation_edge_weight_threshold=interpretation_edge_weight_threshold,
                )
        else:
            assert W is not None
            connectivity_graph = nx.from_numpy_array((W > 0).astype(np.int64))
            if not nx.is_connected(connectivity_graph):
                return resolve_clusters(
                    model,
                    method="components",
                    max_k=max_k,
                    interpretation_edge_weight_threshold=interpretation_edge_weight_threshold,
                )
    if method in ("auto", "spectral"):
        if sparse_graph is not None:
            return _cluster_spectral_from_edges(
                n,
                edges,
                max_k,
                interpretation_edge_weight_threshold=max(
                    float(interpretation_edge_weight_threshold), 0.0
                ),
            )
        if W is None:
            W = model.weighted_adjacency
        thresh = max(float(interpretation_edge_weight_threshold), 0.0)
        spectral_W = W * (W > thresh)
        return _cluster_spectral(
            spectral_W,
            n,
            max_k,
            interpretation_edge_weight_threshold=thresh,
        )

    raise ValueError(f"Unknown clustering method: {method}")


def _has_reliable_mass_split(adj: np.ndarray, threshold: float) -> bool:
    """Accept stable plateaus when the component mass is not singleton-dominated.

    This keeps the reliability gate dataset-relative: use a size floor tied to n,
    then compare meaningful component mass against singleton dust instead of raw
    component count. If we later need a softer policy, split this into tiers.
    """
    sizes, _ = component_state_at_threshold(adj, threshold)
    if len(sizes) <= 1:
        return False

    hint = mass_profile_hint(component_mass_profile(adj, threshold, top_k=5))
    if hint == "singleton-heavy fragmentation":
        return False

    floor = useful_component_size_floor(int(adj.shape[0]))
    nontrivial_sizes = [size for size in sizes if size >= floor]
    if len(nontrivial_sizes) < 2:
        return False

    total = float(sum(sizes))
    singleton_mass = sum(size for size in sizes if size == 1) / total
    nontrivial_mass = sum(nontrivial_sizes) / total
    return (
        hint
        in {
            "stable nontrivial multi-component structure",
            "mostly connected graph with a small tail",
            "giant component with small-tail fragmentation",
        }
        and nontrivial_mass > singleton_mass
    )


def _model_sparse_graph(
    model: ThemaRS,
) -> tuple[int, list[tuple[int, int, float]]] | None:
    weighted_edges = getattr(model, "weighted_edges", None)
    if not callable(weighted_edges):
        return None
    try:
        n = int(model.cosmic_rust.n)
        edges = weighted_edges(threshold=0.0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return n, edges


def _component_labels_from_edges(
    n: int,
    edges: list[tuple[int, int, float]],
    threshold: float,
    *,
    sort_by_size: bool = False,
) -> tuple[np.ndarray, int]:
    rows: list[int] = []
    cols: list[int] = []
    for i, j, weight in edges:
        if float(weight) > threshold:
            rows.extend([int(i), int(j)])
            cols.extend([int(j), int(i)])

    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n, n),
    )
    n_components, labels = sparse_connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    labels = np.asarray(labels, dtype=int)

    if sort_by_size and n_components > 1:
        sizes = np.bincount(labels, minlength=n_components)
        order = sorted(
            range(n_components), key=lambda label: sizes[label], reverse=True
        )
        remap = np.empty(n_components, dtype=int)
        for new_label, old_label in enumerate(order):
            remap[old_label] = new_label
        labels = remap[labels]

    return labels, int(n_components)


def _cluster_by_threshold_stability_sparse(
    n: int,
    edges: list[tuple[int, int, float]],
    stability: Any | None = None,
) -> ClusterResult | None:
    """Find clusters via H0 persistence without materializing weighted_adjacency."""
    if stability is None:
        stability = find_stable_thresholds_sparse(n, edges)

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
        # Dataset-agnostic plateau gate: skip only the degenerate ends — a fully
        # connected cut (nothing to separate) and a dust-dominated cut (>half the
        # nodes are singletons). There is deliberately NO absolute cap on the
        # component count: a large dataset may have many genuine cohorts, and the
        # singleton fraction is scale-free, so this stays performant on any size.
        if int(plateau.component_count) <= 1:
            continue

        thresh = float(plateau.midpoint)
        labels, n_clusters = _component_labels_from_edges(
            n,
            edges,
            thresh,
            sort_by_size=True,
        )
        sizes = np.bincount(labels, minlength=n_clusters)
        singletons = int(np.count_nonzero(sizes == 1))
        if (singletons / n) > _MAX_SINGLETON_RATIO:
            continue

        return ClusterResult(
            labels=pd.Series(labels, name="cluster"),
            method_used="threshold_stability",
            n_clusters=n_clusters,
            silhouette_score=None,
            failure_reason=None,
            interpretation_edge_weight_threshold_applied=thresh,
            stability_plateaus=plateau_dicts,
        )
    return None


def _cluster_spectral_from_edges(
    n: int,
    edges: list[tuple[int, int, float]],
    max_k: int,
    interpretation_edge_weight_threshold: float = 0.0,
) -> ClusterResult:
    labels, n_components = _component_labels_from_edges(
        n,
        edges,
        interpretation_edge_weight_threshold,
        sort_by_size=True,
    )
    if n_components <= 1:
        adj = np.zeros((n, n), dtype=float)
        for i, j, weight in edges:
            if float(weight) > interpretation_edge_weight_threshold:
                adj[int(i), int(j)] = float(weight)
                adj[int(j), int(i)] = float(weight)
        return _cluster_spectral(
            adj,
            n,
            max_k,
            interpretation_edge_weight_threshold=interpretation_edge_weight_threshold,
        )

    components = [
        set(np.flatnonzero(labels == component_id))
        for component_id in range(n_components)
    ]
    giant = sorted(components[0])
    giant_lookup = {node: idx for idx, node in enumerate(giant)}
    affinity_sub = np.zeros((len(giant), len(giant)), dtype=float)
    for i, j, weight in edges:
        if float(weight) <= interpretation_edge_weight_threshold:
            continue
        local_i = giant_lookup.get(int(i))
        local_j = giant_lookup.get(int(j))
        if local_i is None or local_j is None:
            continue
        affinity_sub[local_i, local_j] = float(weight)
        affinity_sub[local_j, local_i] = float(weight)

    best_labels, best_k, best_score, candidate_scores = _spectral_best_cut(
        affinity_sub,
        max_k,
    )
    if best_labels is None or best_score <= 0.05:
        raise SpectralClusterCutError(
            _spectral_failure_diagnostics(
                n=n,
                max_k=max_k,
                interpretation_edge_weight_threshold=interpretation_edge_weight_threshold,
                components=components,
                giant_size=len(giant),
                candidate_scores=candidate_scores,
                best_k=best_k,
                best_score=best_score,
            )
        )

    result_labels = np.empty(n, dtype=int)
    result_labels[np.asarray(giant, dtype=int)] = best_labels
    next_id = best_k
    residual_nodes = 0
    for comp in components[1:]:
        for node in sorted(comp):
            result_labels[node] = next_id
            next_id += 1
            residual_nodes += 1

    return ClusterResult(
        labels=pd.Series(result_labels, name="cluster"),
        method_used="spectral",
        n_clusters=next_id,
        silhouette_score=best_score,
        failure_reason=(
            f"Graph disconnected: clustered giant component "
            f"({len(giant)} of {n} nodes) spectrally; "
            f"{residual_nodes} residual node(s) in {len(components) - 1} "
            f"smaller component(s) isolated as singleton clusters."
        ),
        interpretation_edge_weight_threshold_applied=(
            interpretation_edge_weight_threshold
        ),
    )


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
        thresh = float(plateau.midpoint)
        if not _has_reliable_mass_split(adj, thresh):
            continue

        binary_adj = (adj > thresh).astype(np.int64)
        G = nx.from_numpy_array(binary_adj)

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
            interpretation_edge_weight_threshold_applied=thresh,
            stability_plateaus=plateau_dicts,
        )
    return None


def _spectral_best_cut(
    affinity_sub: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray | None, int, float, list[dict[str, Any]]]:
    """Sweep k on a connected affinity submatrix; return best (labels, k, score).

    Operates on a self-contained affinity matrix (no global node ids); callers
    are responsible for mapping the returned local labels back to global
    positions. Returns ``(None, _SPECTRAL_K_MIN, -1.0)`` when no cut with more
    than one label could be scored.
    """
    sub_n = affinity_sub.shape[0]

    # Distance = 1 - affinity. Spectral sparsification can reweight edges above
    # 1.0, so clip only for the silhouette distance matrix.
    affinity = affinity_sub.copy()
    np.fill_diagonal(affinity, 1.0)
    distance = 1.0 - np.clip(affinity, 0.0, 1.0)

    best_score = -1.0
    best_labels: np.ndarray | None = None
    best_k = _SPECTRAL_K_MIN
    candidate_scores: list[dict[str, Any]] = []

    for k_test in range(_SPECTRAL_K_MIN, min(max_k + 1, sub_n)):
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
        except Exception as exc:
            candidate_scores.append(
                {
                    "k": k_test,
                    "status": "solver_error",
                    "error": str(exc),
                }
            )
            continue

        if len(np.unique(labels)) > 1:
            score = float(silhouette_score(distance, labels, metric="precomputed"))
            candidate_scores.append(
                {
                    "k": k_test,
                    "status": "scored",
                    "silhouette_score": round(score, 6),
                    "n_labels": int(len(np.unique(labels))),
                }
            )
            if score > best_score:
                best_score = score
                best_k = k_test
                best_labels = labels
        else:
            candidate_scores.append(
                {
                    "k": k_test,
                    "status": "single_label",
                    "n_labels": int(len(np.unique(labels))),
                }
            )

    return best_labels, best_k, best_score, candidate_scores


def _spectral_failure_diagnostics(
    *,
    n: int,
    max_k: int,
    interpretation_edge_weight_threshold: float,
    components: list[set[int]],
    giant_size: int,
    candidate_scores: list[dict[str, Any]],
    best_k: int,
    best_score: float,
) -> dict[str, Any]:
    return {
        "method": "spectral",
        "interpretation_edge_weight_threshold": round(
            float(interpretation_edge_weight_threshold), 6
        ),
        "affinity_component_count": len(components),
        "giant_component_size": int(giant_size),
        "giant_component_fraction": round(giant_size / max(n, 1), 6),
        "residual_node_count": int(n - giant_size),
        "k_min": _SPECTRAL_K_MIN,
        "max_k": int(max_k),
        "best_k": int(best_k) if best_score >= 0 else None,
        "best_silhouette_score": round(float(best_score), 6)
        if best_score >= 0
        else None,
        "accepted_silhouette_min": 0.05,
        "candidate_scores": candidate_scores,
    }


def _cluster_spectral(
    adj: np.ndarray,
    n: int,
    max_k: int,
    interpretation_edge_weight_threshold: float = 0.0,
) -> ClusterResult:
    """Run spectral clustering on a weighted affinity matrix.

    When the affinity graph is disconnected (the common case for real EHR data
    with natural outliers/singletons), spectral clustering is run on the giant
    (largest) connected component only. Every off-giant node — smaller
    components and singletons — is then assigned its own cluster id continuing
    the giant-component numbering, so each node receives a label and outliers
    are surfaced as a residual rather than blocking the whole call. A fully
    connected graph takes the original single-pass path with identical results.
    """
    G = nx.from_numpy_array((adj > 0).astype(np.int64))
    components = sorted(nx.connected_components(G), key=len, reverse=True)

    if len(components) <= 1:
        # Connected graph: cluster all nodes directly (unchanged behavior).
        best_labels, best_k, best_score, candidate_scores = _spectral_best_cut(
            adj, max_k
        )
        if best_labels is not None and best_score > 0.05:
            return ClusterResult(
                labels=pd.Series(best_labels, name="cluster"),
                method_used="spectral",
                n_clusters=best_k,
                silhouette_score=best_score,
                failure_reason=None,
                interpretation_edge_weight_threshold_applied=(
                    interpretation_edge_weight_threshold
                ),
            )
        raise SpectralClusterCutError(
            _spectral_failure_diagnostics(
                n=n,
                max_k=max_k,
                interpretation_edge_weight_threshold=interpretation_edge_weight_threshold,
                components=components,
                giant_size=n,
                candidate_scores=candidate_scores,
                best_k=best_k,
                best_score=best_score,
            )
        )

    # Disconnected graph: cluster the giant component, isolate the residual.
    giant = sorted(components[0])
    giant_idx = np.asarray(giant, dtype=int)
    affinity_sub = adj[np.ix_(giant_idx, giant_idx)]

    best_labels, best_k, best_score, candidate_scores = _spectral_best_cut(
        affinity_sub, max_k
    )
    if best_labels is None or best_score <= 0.05:
        raise SpectralClusterCutError(
            _spectral_failure_diagnostics(
                n=n,
                max_k=max_k,
                interpretation_edge_weight_threshold=interpretation_edge_weight_threshold,
                components=components,
                giant_size=len(giant),
                candidate_scores=candidate_scores,
                best_k=best_k,
                best_score=best_score,
            )
        )

    labels = np.empty(n, dtype=int)
    labels[giant_idx] = best_labels
    # Off-giant nodes (smaller components + singletons) each get their own
    # cluster id continuing the numbering. Keeping every off-giant node
    # distinct (rather than collapsing them into one bucket) preserves the
    # outlier structure for downstream profiling; the residual count below
    # makes the split auditable.
    next_id = best_k
    residual_nodes = 0
    for comp in components[1:]:
        for node in sorted(comp):
            labels[node] = next_id
            next_id += 1
            residual_nodes += 1

    return ClusterResult(
        labels=pd.Series(labels, name="cluster"),
        method_used="spectral",
        n_clusters=next_id,
        silhouette_score=best_score,
        failure_reason=(
            f"Graph disconnected: clustered giant component "
            f"({len(giant)} of {n} nodes) spectrally; "
            f"{residual_nodes} residual node(s) in {len(components) - 1} "
            f"smaller component(s) isolated as singleton clusters."
        ),
        interpretation_edge_weight_threshold_applied=(
            interpretation_edge_weight_threshold
        ),
    )


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


@dataclass(frozen=True)
class _NumericPrecompute:
    """Batched per-column statistics for ``_compute_numeric_rows``.

    Every field is shape ``(K,)`` (per column) or ``(N, K)`` (per row × column).
    All NaN-aware: ``valid`` masks the rows that survived numeric coercion in
    each column independently. Anything that takes an ``N`` in a formula
    (Kruskal denominator, MWU variance) must use the per-column ``n_total_j``
    rather than the raw row count.

    Two std flavors are stored deliberately. The existing implementation uses
    ``ddof=0`` for the column-level "global_scale" feeding ``mean_dispersion /
    global_scale`` (interpreter.py:707) and ``ddof=1`` everywhere else via
    ``_safe_std``. The precompute mirrors that split so the vectorized rewrite
    can pick the right one at each call site.
    """

    column_names: list[str]
    X: np.ndarray  # (N, K) float64
    valid: np.ndarray  # (N, K) bool
    n_total_j: np.ndarray  # (K,) int — per-column valid count
    col_mean: np.ndarray  # (K,) float64 — np.nanmean
    col_std_pop: np.ndarray  # (K,) float64 — np.nanstd(ddof=0); for global_scale
    col_std_sample: np.ndarray  # (K,) float64 — np.nanstd(ddof=1); for row-level
    col_var_sample: np.ndarray  # (K,) float64 — np.nanvar(ddof=1)
    col_median: np.ndarray  # (K,) float64
    col_mad: np.ndarray  # (K,) float64 — median(|x - col_median|)
    col_iqr: np.ndarray  # (K,) float64 — q75 - q25
    sort_idx: np.ndarray  # (N, K) int — np.argsort per column; NaN entries trail
    ranks: np.ndarray  # (N, K) float64 — scipy.rankdata(omit); NaN at NaN rows
    tie_correction: (
        np.ndarray
    )  # (K,) float64 — Kruskal 1 - Σ(t³-t)/(N³-N); 1.0 when ties absent or undefined


def _column_tie_correction(sorted_col: np.ndarray, n_valid: int) -> float:
    """Kruskal-Wallis tie correction factor.

    Matches scipy.stats.kruskal: ``1 - sum(t**3 - t) / (N**3 - N)`` where ``t``
    is the size of each tie group across valid (non-NaN) values, and ``N`` is
    the per-column valid count. Returns 1.0 when undefined (``N <= 1`` or all
    values identical), preserving scipy's "no correction" baseline.
    """
    if n_valid <= 1:
        return 1.0
    finite = sorted_col[:n_valid]
    # Count consecutive runs of equal values in the sorted vector.
    if finite.size == 0:
        return 1.0
    diffs = np.diff(finite)
    # Run lengths via boundary detection.
    run_starts = np.concatenate(([0], np.where(diffs != 0)[0] + 1, [finite.size]))
    run_lengths = np.diff(run_starts)
    tie_groups = run_lengths[run_lengths > 1].astype(np.float64)
    denom = float(n_valid) ** 3 - float(n_valid)
    if denom == 0.0:
        return 1.0
    correction = 1.0 - float(np.sum(tie_groups**3 - tie_groups)) / denom
    if correction <= 0.0:
        # Matches scipy behavior: when every value is identical the correction
        # collapses to 0; downstream consumers must propagate NaN, not divide.
        return 0.0
    return correction


def _build_numeric_precompute(
    data: pd.DataFrame, numeric_cols: list[str]
) -> _NumericPrecompute:
    """Build ``_NumericPrecompute`` from a raw DataFrame slice.

    Coerces each requested column via ``pd.to_numeric(errors='coerce')``, so
    non-numeric strings become NaN — matching ``_normalize_numeric``'s
    behavior — but rows are preserved (not dropped) so cluster alignment by
    position remains valid.
    """
    if not numeric_cols:
        empty = np.empty((len(data), 0), dtype=np.float64)
        empty_int = np.empty((len(data), 0), dtype=np.int64)
        empty_k = np.empty(0, dtype=np.float64)
        empty_k_int = np.empty(0, dtype=np.int64)
        return _NumericPrecompute(
            column_names=[],
            X=empty,
            valid=empty.astype(bool),
            n_total_j=empty_k_int,
            col_mean=empty_k,
            col_std_pop=empty_k,
            col_std_sample=empty_k,
            col_var_sample=empty_k,
            col_median=empty_k,
            col_mad=empty_k,
            col_iqr=empty_k,
            sort_idx=empty_int,
            ranks=empty,
            tie_correction=empty_k,
        )

    coerced = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    X = np.ascontiguousarray(coerced.to_numpy(dtype=np.float64))
    valid = ~np.isnan(X)
    n_total_j = valid.sum(axis=0).astype(np.int64)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN slice encountered")
        warnings.filterwarnings("ignore", r"Mean of empty slice")
        warnings.filterwarnings("ignore", r"Degrees of freedom <= 0 for slice")
        col_mean = np.nanmean(X, axis=0)
        col_std_pop = np.nanstd(X, axis=0, ddof=0)
        col_std_sample = np.nanstd(X, axis=0, ddof=1)
        col_var_sample = np.nanvar(X, axis=0, ddof=1)
        col_median = np.nanmedian(X, axis=0)
        abs_dev = np.abs(X - col_median[np.newaxis, :])
        col_mad = np.nanmedian(abs_dev, axis=0)
        col_q25 = np.nanquantile(X, 0.25, axis=0)
        col_q75 = np.nanquantile(X, 0.75, axis=0)
        col_iqr = col_q75 - col_q25

    # Per-column ranks. scipy preserves NaN in-place when nan_policy="omit".
    ranks = stats.rankdata(X, axis=0, nan_policy="omit")

    # Sort each column; NaN values sort to the end with np.argsort.
    sort_idx = np.argsort(X, axis=0)

    # Tie correction: walk each sorted column over its non-NaN prefix.
    tie_correction = np.empty(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        sorted_col = X[sort_idx[:, j], j]
        tie_correction[j] = _column_tie_correction(sorted_col, int(n_total_j[j]))

    return _NumericPrecompute(
        column_names=list(numeric_cols),
        X=X,
        valid=valid,
        n_total_j=n_total_j,
        col_mean=col_mean,
        col_std_pop=col_std_pop,
        col_std_sample=col_std_sample,
        col_var_sample=col_var_sample,
        col_median=col_median,
        col_mad=col_mad,
        col_iqr=col_iqr,
        sort_idx=sort_idx,
        ranks=ranks,
        tie_correction=tie_correction,
    )


def _mwu_asymptotic_two_sided_pvalue(
    u1: float, n1: int, n2: int, tie_sum: float
) -> float:
    """Asymptotic two-sided Mann-Whitney U p-value with continuity correction.

    Matches ``scipy.stats.mannwhitneyu(..., alternative='two-sided',
    method='asymptotic').pvalue`` bit-for-bit. ``tie_sum`` is
    ``Σ(tᵢ³ − tᵢ)`` over tie-group sizes in the combined sample.
    """
    from scipy.special import ndtr  # local import keeps top-level surface clean

    n = n1 + n2
    u2 = n1 * n2 - u1
    u = max(u1, u2)
    mu = n1 * n2 / 2.0
    if n <= 1 or n1 == 0 or n2 == 0:
        return 1.0
    variance = n1 * n2 / 12.0 * ((n + 1) - tie_sum / (n * (n - 1)))
    if variance <= 0.0:
        # Combined sample is fully tied → MWU is undefined; scipy returns 1.0
        # after p clamping. Mirror that exactly.
        return 1.0
    s = math.sqrt(variance)
    z = (u - mu - 0.5) / s  # continuity correction matches scipy
    p = float(ndtr(-z)) * 2.0
    return float(np.clip(p, 0.0, 1.0))


def _mwu_one_vs_rest_pvalue(
    cluster_mask: np.ndarray,
    column_index: int,
    pre: _NumericPrecompute,
) -> float:
    """Two-sided MWU p-value for cluster-vs-rest using precomputed ranks.

    Mirrors ``scipy.stats.mannwhitneyu`` with ``method='auto'`` — uses the
    asymptotic z-test (with tie correction) when scipy would (n₁>8 and n₂>8,
    OR ties present in the combined sample) and falls back to scipy for the
    small-no-tie exact branch. Combined sample is the full column so the
    precomputed column ranks apply directly.
    """
    j = column_index
    valid_j = pre.valid[:, j]
    cluster_valid = cluster_mask & valid_j
    rest_valid = (~cluster_mask) & valid_j
    n1 = int(cluster_valid.sum())
    n2 = int(rest_valid.sum())
    if n1 == 0 or n2 == 0:
        raise ValueError("mannwhitneyu: at least one input has size 0")
    n_total = n1 + n2
    tie_correction = float(pre.tie_correction[j])
    has_ties = tie_correction < 1.0
    if (n1 > 8 and n2 > 8) or has_ties:
        r1 = float(pre.ranks[cluster_valid, j].sum())
        u1 = r1 - n1 * (n1 + 1) / 2.0
        tie_sum = (n_total**3 - n_total) * (1.0 - tie_correction)
        return _mwu_asymptotic_two_sided_pvalue(u1, n1, n2, tie_sum)
    # Exact branch: scipy handles small-no-tie inputs. Defer to scipy so the
    # exact-distribution path is preserved bit-for-bit.
    cluster_arr = pre.X[cluster_valid, j]
    rest_arr = pre.X[rest_valid, j]
    return float(
        stats.mannwhitneyu(cluster_arr, rest_arr, alternative="two-sided").pvalue
    )


def _mwu_pair_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided MWU p-value for an arbitrary pair (cluster vs neighbor).

    Same method-selection logic as ``_mwu_one_vs_rest_pvalue``. Computes
    ranks fresh on the combined sample because cluster ∪ neighbor is a
    subset of the column, so the column-level precompute does not apply.
    """
    if x.size == 0 or y.size == 0:
        raise ValueError("mannwhitneyu: at least one input has size 0")
    n1, n2 = x.size, y.size
    n = n1 + n2
    xy = np.concatenate([x, y])
    order = np.argsort(xy, kind="mergesort")
    sorted_xy = xy[order]
    # Tie-group boundaries: a new group starts where consecutive values differ.
    diff = np.diff(sorted_xy)
    boundaries = np.concatenate(
        ([0], (np.where(diff != 0)[0] + 1).astype(np.int64), [n])
    )
    group_sizes = np.diff(boundaries).astype(np.float64)
    has_ties = bool(np.any(group_sizes > 1))
    if (n1 > 8 and n2 > 8) or has_ties:
        # Average ranks per tie group (1-based).
        avg_ranks_per_group = (
            boundaries[:-1].astype(np.float64) + (group_sizes - 1) / 2.0 + 1.0
        )
        sorted_ranks = np.repeat(avg_ranks_per_group, group_sizes.astype(np.int64))
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = sorted_ranks
        r1 = float(ranks[:n1].sum())
        u1 = r1 - n1 * (n1 + 1) / 2.0
        tie_sum = float(np.sum(group_sizes**3 - group_sizes))
        return _mwu_asymptotic_two_sided_pvalue(u1, n1, n2, tie_sum)
    # Small-no-tie exact branch.
    return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)


def _kruskal_wallis_pvalue(
    ranks_column: np.ndarray,
    group_masks: list[np.ndarray],
    tie_correction: float,
) -> float:
    """Kruskal-Wallis p-value from precomputed ranks + tie correction.

    Equivalent to ``scipy.stats.kruskal(*groups).pvalue`` where ``groups`` are
    the value arrays extracted by ``group_masks`` from the original column.
    ``ranks_column`` is the column slice of the precomputed rank matrix; NaN
    positions in the source data are NaN here too and must be excluded via the
    masks before summation.

    Raises ``ValueError`` when fewer than two groups have data — mirroring
    ``scipy.stats.kruskal``'s behavior. Returns ``nan`` when the tie
    correction collapses to zero (every value in the column is tied), again
    matching scipy's silent divide-by-zero output on that input.
    """
    rank_sums: list[float] = []
    group_sizes: list[float] = []
    for mask in group_masks:
        if not mask.any():
            continue
        group_ranks = ranks_column[mask]
        rank_sums.append(float(np.sum(group_ranks)))
        group_sizes.append(float(group_ranks.size))
    if len(rank_sums) < 2:
        raise ValueError("Kruskal-Wallis requires at least two non-empty groups")
    n_total = float(sum(group_sizes))
    h_uncorrected = (12.0 / (n_total * (n_total + 1.0))) * sum(
        rs * rs / gs for rs, gs in zip(rank_sums, group_sizes)
    ) - 3.0 * (n_total + 1.0)
    if tie_correction == 0.0:
        # Matches scipy's ``h /= ties`` behavior on all-identical input:
        # returns nan (with a RuntimeWarning suppressed here).
        return float("nan")
    h = h_uncorrected / tie_correction
    degrees_of_freedom = len(rank_sums) - 1
    return float(stats.chi2.sf(h, degrees_of_freedom))


def _ks_two_sample_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic via combined-sort ECDFs.

    Equivalent to ``scipy.stats.ks_2samp(a, b).statistic`` for finite inputs.
    Raises ``ValueError`` on empty inputs (mirroring scipy's behavior). Failure
    strings emitted by callers retain the legacy ``"ks_2samp: ..."`` label so
    downstream consumers and the ``stats_failures`` contract are unaffected.

    Acts as the monkeypatch seam for the failure-injection test in
    ``tests/test_mcp_feature_evidence.py``.
    """
    if a.size == 0 or b.size == 0:
        raise ValueError("ks_2samp: at least one input has size 0")
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    combined = np.concatenate([a_sorted, b_sorted])
    n1, n2 = a.size, b.size
    cdf_a = np.searchsorted(a_sorted, combined, side="right") / n1
    cdf_b = np.searchsorted(b_sorted, combined, side="right") / n2
    cdf_diff = cdf_a - cdf_b
    max_pos = float(np.clip(np.max(cdf_diff), 0.0, 1.0))
    max_neg = float(np.clip(-np.min(cdf_diff), 0.0, 1.0))
    d = max(max_pos, max_neg)
    # Rationalize to the lcm-denominator. scipy.stats.ks_2samp does this in
    # its default ``exact`` mode (samples ≤ 10000) — the KS two-sample
    # statistic is exactly rational with denominator ``lcm(n1, n2)``, so
    # rounding ``d * lcm`` recovers the exact float scipy returns and
    # eliminates last-bit drift that would otherwise perturb tie-breaking
    # in downstream empirical percentile ranks.
    if max(n1, n2) <= 10_000:
        g = math.gcd(n1, n2)
        lcm = (n1 // g) * n2
        h = int(round(d * lcm))
        d = h / lcm
    return d


def _wasserstein_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1D Wasserstein (earth-mover) distance via ECDF integration.

    Equivalent to ``scipy.stats.wasserstein_distance(a, b)``. Closed-form on
    sorted union: ``Σ |F_a(x_i) − F_b(x_i)| · (x_{i+1} − x_i)``.
    """
    if a.size == 0 or b.size == 0:
        raise ValueError("wasserstein: at least one input has size 0")
    all_values = np.concatenate([a, b])
    all_values.sort()
    deltas = np.diff(all_values)
    if deltas.size == 0:
        return 0.0
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    cdf_a = np.searchsorted(a_sorted, all_values[:-1], side="right") / a.size
    cdf_b = np.searchsorted(b_sorted, all_values[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * deltas))


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


def _feature_signal_sort_key(row: dict[str, Any], kind: str) -> tuple[Any, ...]:
    aggregate = abs(float(row.get("aggregate_score", 0.0)))
    if kind == "numeric":
        secondary = (
            abs(float(row.get("effect_mean_std", 0.0))),
            abs(float(row.get("z_score", 0.0))),
            abs(float(row.get("neighbor_effect", 0.0))),
        )
    else:
        secondary = (
            abs(float(row.get("log_lift", 0.0))),
            abs(float(row.get("neighbor_specificity", 0.0))),
            abs(float(row.get("lift", 1.0)) - 1.0),
        )
    return (
        -aggregate,
        *(-value for value in secondary),
        str(row.get("column", "")),
        str(row.get("value", "")),
    )


def _limit_cluster_feature_rows(
    numeric_rows: list[dict[str, Any]],
    categorical_rows: list[dict[str, Any]],
    max_features: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ranked = [
        ("numeric", row)
        for row in sorted(
            numeric_rows, key=lambda row: _feature_signal_sort_key(row, "numeric")
        )
    ]
    ranked.extend(
        ("categorical", row)
        for row in sorted(
            categorical_rows,
            key=lambda row: _feature_signal_sort_key(row, "categorical"),
        )
    )
    ranked.sort(key=lambda item: _feature_signal_sort_key(item[1], item[0]))
    selected = ranked[:max_features]

    limited_numeric = [row for kind, row in selected if kind == "numeric"]
    limited_categorical = [row for kind, row in selected if kind == "categorical"]
    omitted_numeric = max(len(numeric_rows) - len(limited_numeric), 0)
    omitted_categorical = max(len(categorical_rows) - len(limited_categorical), 0)
    limit_metadata = {
        "feature_limit": int(max_features),
        "features_returned": {
            "numeric": len(limited_numeric),
            "categorical": len(limited_categorical),
            "total": len(selected),
        },
        "features_omitted": {
            "numeric": omitted_numeric,
            "categorical": omitted_categorical,
            "total": omitted_numeric + omitted_categorical,
        },
    }
    return limited_numeric, limited_categorical, limit_metadata


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
    target_clusters: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    if not numeric_cols:
        return [], {}

    # Build the numeric matrix once. pandas .loc / .iloc and per-cell
    # _normalize_numeric / _safe_X calls are replaced by precomputed
    # per-column arrays and batched numpy operations downstream.
    pre = _build_numeric_precompute(data, numeric_cols)
    clusters_array = clusters.to_numpy()
    cluster_values = sorted(int(cid) for cid in clusters.unique())

    if target_clusters is not None:
        cluster_values = [cid for cid in cluster_values if cid in target_clusters]

    # Per-column arrays of NaN-dropped values; one extraction per column,
    # reused for every (cluster, column) cell. Replaces the O(C*K) calls to
    # `_normalize_numeric(data[column])` and `data.loc[mask, column]`.
    column_values: list[np.ndarray] = [
        pre.X[pre.valid[:, j], j] for j in range(len(numeric_cols))
    ]

    # ------------------------------------------------------------------
    # Column-level Kruskal + global effect (one pass per column)
    # ------------------------------------------------------------------
    global_numeric_stats: dict[str, dict[str, Any]] = {}
    for j, column in enumerate(numeric_cols):
        valid_j = pre.valid[:, j]
        group_masks = [(clusters_array == cid) & valid_j for cid in cluster_values]
        grouped: list[np.ndarray] = [pre.X[m, j] for m in group_masks if m.any()]
        p_value = 1.0
        column_failures: list[str] = []
        if len(grouped) >= 2:
            try:
                p_value = _kruskal_wallis_pvalue(
                    pre.ranks[:, j], group_masks, float(pre.tie_correction[j])
                )
            except ValueError as exc:
                column_failures.append(f"kruskal: {exc}")
                logger.warning("kruskal failed for column %s: %s", column, exc)
        # ``global_scale`` mirrors the legacy ``np.std(...)`` (ddof=0). Compute
        # against the column's NaN-dropped array so float ordering matches the
        # previous implementation bit-for-bit.
        global_scale = float(np.std(column_values[j])) if column_values[j].size else 0.0
        mean_dispersion = (
            float(np.std([float(np.mean(values)) for values in grouped]))
            if grouped
            else 0.0
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

    # ------------------------------------------------------------------
    # Hoisted per-column "global" stats (formerly recomputed per cluster)
    # ------------------------------------------------------------------
    global_std_per_col: list[float] = [
        max(_safe_std(values), _EPS) for values in column_values
    ]
    global_mad_per_col: list[float] = [
        max(_safe_mad(values), _EPS) for values in column_values
    ]
    global_iqr_per_col: list[float] = [
        max(float(stats.iqr(values)), _EPS) if values.size else _EPS
        for values in column_values
    ]
    global_mean_per_col: list[float] = [_safe_mean(values) for values in column_values]

    # ------------------------------------------------------------------
    # Per-cluster row generation
    # ------------------------------------------------------------------
    rows: list[dict[str, Any]] = []
    for cluster_id in cluster_values:
        cluster_mask = clusters_array == cluster_id
        rest_mask = ~cluster_mask
        neighbor_id = _strongest_neighbor(adjacency, cluster_id)
        neighbor_mask = (
            (clusters_array == neighbor_id) if neighbor_id is not None else None
        )
        for j, column in enumerate(numeric_cols):
            valid_j = pre.valid[:, j]
            cluster_values_arr = pre.X[cluster_mask & valid_j, j]
            rest_values_arr = pre.X[rest_mask & valid_j, j]
            if cluster_values_arr.size == 0 or rest_values_arr.size == 0:
                continue
            neighbor_values_arr = (
                pre.X[neighbor_mask & valid_j, j]
                if neighbor_mask is not None
                else np.array([], dtype=float)
            )
            row_failures: list[str] = []
            pooled = _pooled_std(cluster_values_arr, rest_values_arr)
            global_std = global_std_per_col[j]
            global_mad = global_mad_per_col[j]
            global_iqr = global_iqr_per_col[j]
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
                ks_stat = _ks_two_sample_stat(cluster_values_arr, rest_values_arr)
            except ValueError as exc:
                ks_stat = 0.0
                row_failures.append(f"ks_2samp: {exc}")
                logger.warning(
                    "ks_2samp failed for cluster=%s column=%s: %s",
                    cluster_id,
                    column,
                    exc,
                )
            try:
                wasserstein_norm = _wasserstein_distance_1d(
                    cluster_values_arr, rest_values_arr
                ) / max(global_iqr, _EPS)
            except ValueError as exc:
                wasserstein_norm = 0.0
                row_failures.append(f"wasserstein: {exc}")
                logger.warning(
                    "wasserstein failed for cluster=%s column=%s: %s",
                    cluster_id,
                    column,
                    exc,
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
                one_vs_rest_p = _mwu_one_vs_rest_pvalue(cluster_mask, j, pre)
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
                    neighbor_p = _mwu_pair_pvalue(
                        cluster_values_arr, neighbor_values_arr
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
                "global_mean": global_mean_per_col[j],
                "global_mean_rest": _safe_mean(rest_values_arr),
                "z_score": (_safe_mean(cluster_values_arr) - global_mean_per_col[j])
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
    target_clusters: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], list[dict[str, Any]]]:
    n_total = len(data)
    gated_cols: list[dict[str, Any]] = []
    active_categorical_cols: list[str] = []

    # Early-Stage Cardinality Gating
    for col in categorical_cols:
        nunique = data[col].nunique()
        if nunique > max(_MIN_CARDINALITY_FLOOR, _MAX_CARDINALITY_RATIO * n_total):
            logger.warning(
                "Categorical column '%s' gated (cardinality: %d, ratio: %.4f) to prevent context bloat.",
                col,
                nunique,
                nunique / n_total if n_total else 0.0,
            )
            gated_cols.append(
                {
                    "column": col,
                    "cardinality": int(nunique),
                    "reason": "high_cardinality",
                }
            )
        else:
            active_categorical_cols.append(col)

    global_categorical_stats: dict[str, dict[str, Any]] = {}
    cluster_values = sorted(int(cid) for cid in clusters.unique())
    if target_clusters is not None:
        cluster_values = [cid for cid in cluster_values if cid in target_clusters]

    # Precompute encoded columns, global value counts, and per-column
    # association stats in a single pass over the active columns.
    encoded_cols: dict[str, pd.Series] = {}
    global_value_counts: dict[str, dict[str, int]] = {}
    for column in active_categorical_cols:
        encoded = data[column].fillna("__MISSING__").astype(str)
        encoded_cols[column] = encoded
        global_value_counts[column] = encoded.value_counts().to_dict()

        if target_clusters is not None:
            mask = clusters.isin(target_clusters)
            contingency = pd.crosstab(clusters[mask], encoded[mask])
        else:
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
        [
            global_categorical_stats[column]["p_value"]
            for column in active_categorical_cols
        ]
    )
    for column, q_value in zip(active_categorical_cols, global_qs):
        global_categorical_stats[column]["q_value"] = (
            1.0 if q_value is None else float(q_value)
        )

    rows: list[dict[str, Any]] = []
    for cluster_id in cluster_values:
        cluster_mask = clusters == cluster_id
        cluster_size = int(cluster_mask.sum())
        # rest_size = n_total - cluster_size is mathematically valid only because len(clusters) == len(data) is strictly enforced.
        rest_size = n_total - cluster_size
        neighbor_id = _strongest_neighbor(adjacency, cluster_id)
        neighbor_mask = clusters == neighbor_id if neighbor_id is not None else None
        neighbor_size = int(neighbor_mask.sum()) if neighbor_mask is not None else 0

        for column in active_categorical_cols:
            global_values = encoded_cols[column]
            col_global_counts = global_value_counts[column]
            cluster_values_series = global_values.loc[cluster_mask]
            cluster_counts = cluster_values_series.value_counts().to_dict()

            # Precompute neighbor value counts for O(1) lookups inside the category value loop
            neighbor_counts = {}
            if neighbor_mask is not None and neighbor_size > 0:
                neighbor_counts = (
                    global_values.loc[neighbor_mask].value_counts().to_dict()
                )

            for value, count in cluster_counts.items():
                global_count = col_global_counts.get(value, 0)
                count_rest = global_count - count
                prevalence_cluster = (count / cluster_size) if cluster_size else 0.0
                prevalence_rest = (count_rest / rest_size) if rest_size else 0.0
                prevalence_global = (global_count / n_total) if n_total else 0.0
                lift = prevalence_cluster / max(prevalence_global, _EPS)
                log_lift = float(math.log(max(lift, _EPS)))
                neighbor_prevalence = 0.0
                if neighbor_mask is not None and neighbor_size > 0:
                    neighbor_prevalence = float(
                        neighbor_counts.get(value, 0) / neighbor_size
                    )
                p_cv = count / max(n_total, 1)
                p_c = cluster_size / max(n_total, 1)
                p_v = global_count / max(n_total, 1)
                mi_contrib = float(
                    p_cv * math.log(max(p_cv, _EPS) / max(p_c * p_v, _EPS))
                )

                fisher_p = 1.0
                test_method = "fisher"
                row_failures: list[str] = []

                contingency_table = [
                    [count, max(cluster_size - count, 0)],
                    [count_rest, max(rest_size - count_rest, 0)],
                ]

                # Arithmetic expected cell size guard for the 2x2 contingency table
                if n_total > 0:
                    e00 = cluster_size * global_count / n_total
                    e01 = cluster_size * (n_total - global_count) / n_total
                    e10 = rest_size * global_count / n_total
                    e11 = rest_size * (n_total - global_count) / n_total
                else:
                    e00 = e01 = e10 = e11 = 0.0

                if min(e00, e01, e10, e11) >= _CHI2_MIN_EXPECTED_CELL:
                    try:
                        _, chi2_p, _, _ = stats.chi2_contingency(
                            contingency_table,
                            correction=True,
                        )
                        fisher_p = float(chi2_p)
                        test_method = "chi2"
                    except ValueError as exc:
                        # chi2 rejected the table (e.g. a zero margin); fall
                        # through to the exact test rather than fail silently.
                        logger.debug(
                            "chi2_contingency fell back to fisher for "
                            "cluster=%s column=%s value=%s: %s",
                            cluster_id,
                            column,
                            value,
                            exc,
                        )

                if test_method == "fisher":
                    try:
                        fisher_p = float(stats.fisher_exact(contingency_table)[1])
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
                        "test_method": test_method,
                        "failure_reasons": row_failures,
                    }
                )

    # `fisher_p` may hold either an exact (Fisher) or asymptotic (chi2) p-value
    # depending on each row's expected cell counts; both are valid p-values, so
    # pooling them into a single BH-FDR correction is intentional.
    fisher_qs = _bh_fdr([row["fisher_p"] for row in rows])
    for row, q_value in zip(rows, fisher_qs):
        row["fisher_q"] = 1.0 if q_value is None else float(q_value)

    _apply_categorical_scores(rows)
    return rows, global_categorical_stats, gated_cols


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
                        "value": numeric_map[column]["z_score"],
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
                        "value": categorical_map[(column, value)][
                            "in_cluster_prevalence"
                        ],
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
    max_clusters_to_characterize: int | None = None,
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

    cluster_sizes = clusters.value_counts()
    n_unique_clusters = len(cluster_sizes)
    target_clusters = None
    omitted_clusters_count = 0
    if (
        max_clusters_to_characterize is not None
        and n_unique_clusters > max_clusters_to_characterize
    ):
        target_clusters = set(
            cluster_sizes.nlargest(max_clusters_to_characterize).index.astype(int)
        )
        omitted_clusters_count = n_unique_clusters - max_clusters_to_characterize

    adjacency = _graph_cluster_adjacency(model, clusters)
    numeric_rows, global_numeric_stats = _compute_numeric_rows(
        working, numeric_cols, clusters, adjacency, target_clusters=target_clusters
    )
    categorical_rows, global_categorical_stats, gated_cols = _compute_categorical_rows(
        working,
        categorical_cols,
        clusters,
        adjacency,
        target_clusters=target_clusters,
    )

    cluster_bundles: dict[int, dict[str, Any]] = {}
    # We only bundle clusters that were actually characterized
    characterized_cluster_ids = sorted(
        int(cid)
        for cid in clusters.unique()
        if target_clusters is None or cid in target_clusters
    )
    for cluster_id in characterized_cluster_ids:
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
            "n_clusters": n_unique_clusters,
            "n_characterized": len(characterized_cluster_ids),
            "omitted_clusters_count": omitted_clusters_count,
            "max_clusters_to_characterize": max_clusters_to_characterize,
            "columns": working_columns,
            "excluded_columns": exclude_columns or [],
            "numeric_features_screened": len(numeric_cols),
            "categorical_columns_screened": len(categorical_cols) - len(gated_cols),
            "categorical_values_screened": len(categorical_rows),
            "categorical_columns_gated": gated_cols,
            "tiering_method": "adaptive_kmeans_on_aggregate_percentiles",
            "neighbor_contrast_enabled": True,
            "global_numeric_stats": global_numeric_stats,
            "global_categorical_stats": global_categorical_stats,
            "cluster_adjacency": adjacency,
            "stats_failures": stats_failures,
        },
        working_columns=working_columns,
        categorical_columns_gated=gated_cols,
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
                    "categorical_columns_gated",
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
    max_features: int = 16,
) -> dict[str, Any]:
    bundle = evidence_index.cluster_bundles.get(int(cluster_id))
    if bundle is None:
        raise KeyError(cluster_id)
    numeric_features = _detail_numeric_rows(bundle["numeric"], detail)
    categorical_features = _detail_categorical_rows(bundle["categorical"], detail)
    numeric_features, categorical_features, limit_metadata = (
        _limit_cluster_feature_rows(
            numeric_features,
            categorical_features,
            max_features,
        )
    )
    return {
        "cluster_id": int(cluster_id),
        "size": int(bundle["size"]),
        "size_pct": float(bundle["size_pct"]),
        "semantic_name": str(bundle["semantic_name"]),
        "topological_neighbors": list(bundle["topological_neighbors"]),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        **limit_metadata,
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
    detail: str = "summary",
    max_clusters: int | None = 8,
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
    raw_payloads: list[dict[str, Any]] = []
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
        signal_score = max(
            (
                abs(float(row.get("aggregate_score", 0.0)))
                for row in (*numeric_rows, *categorical_rows)
            ),
            default=0.0,
        )
        raw_payloads.append(
            {
                "cluster_id": cluster_id,
                "semantic_name": bundle["semantic_name"],
                "cluster_size": int(bundle["size"]),
                "_signal_score": signal_score,
                "_numeric_rows": numeric_rows,
                "_categorical_rows": categorical_rows,
            }
        )

    if cluster_ids is None and max_clusters is not None:
        raw_payloads.sort(key=lambda c: c["_signal_score"], reverse=True)
        kept = raw_payloads[:max_clusters]
        omitted = len(raw_payloads) - len(kept)
        kept.sort(key=lambda c: c["cluster_id"])
    else:
        kept = raw_payloads
        omitted = 0

    cluster_payloads = []
    for entry in kept:
        cluster_payloads.append(
            {
                "cluster_id": entry["cluster_id"],
                "semantic_name": entry["semantic_name"],
                "cluster_size": entry["cluster_size"],
                "numeric_features": _detail_numeric_rows(
                    entry["_numeric_rows"], detail
                ),
                "categorical_features": _detail_categorical_rows(
                    entry["_categorical_rows"], detail
                ),
            }
        )
    return {
        "feature_names": features,
        "detail": detail,
        "clusters_returned": len(cluster_payloads),
        "clusters_omitted": omitted,
        "clusters": cluster_payloads,
    }


def safe_float(val: Any) -> float:
    try:
        return float(val) if val is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def signal_matrix_payload(
    evidence_index: FeatureEvidenceIndex,
    *,
    cluster_ids: list[int] | None = None,
    include_context_tier: bool = False,
    max_clusters: int | None = 8,
    return_markdown: bool = True,
) -> dict[str, Any] | str:
    # 1. Parse and sanitize cluster IDs
    parsed_requested_ids: set[int] = set()
    if cluster_ids is not None:
        for cid in cluster_ids:
            try:
                parsed_requested_ids.add(int(cid))
            except (ValueError, TypeError):
                logger.warning(f"Skipping unparseable cluster_id: {cid}")
    else:
        bundles = getattr(evidence_index, "cluster_bundles", {}) or {}
        parsed_requested_ids = set(bundles.keys())

    # 2. Pre-calculate cluster signal scores to avoid duplicate sorting loops
    cluster_scores: dict[int, float] = {}
    for cid in parsed_requested_ids:
        bundle = evidence_index.cluster_bundles.get(cid, {})
        rows = [
            *bundle.get("numeric", []),
            *bundle.get("categorical", []),
        ]
        max_score = 0.0
        for row in rows:
            try:
                score_val = abs(safe_float(row.get("aggregate_score", 0.0)))
                if score_val > max_score:
                    max_score = score_val
            except Exception:
                continue
        cluster_scores[cid] = max_score

    # 3. Handle sorting and truncation
    ranked_ids = sorted(
        parsed_requested_ids, key=lambda x: cluster_scores.get(x, 0.0), reverse=True
    )
    if max_clusters is not None and len(ranked_ids) > max_clusters:
        kept = set(ranked_ids[:max_clusters])
        omitted_ids = ranked_ids[max_clusters:]
    else:
        kept = set(ranked_ids)
        omitted_ids = []

    # Map omitted cluster IDs to their maximum aggregate signal score
    omitted_telemetry = [
        {"cluster_id": cid, "max_signal": round(cluster_scores.get(cid, 0.0), 3)}
        for cid in omitted_ids
    ]

    matrix = getattr(evidence_index, "signal_matrix", {}) or {}

    # 4. Group rows by cluster_id to avoid O(R) global scans
    numeric_by_cluster: dict[int, list[dict[str, Any]]] = {}
    categorical_by_cluster: dict[int, list[dict[str, Any]]] = {}

    for row in matrix.get("numeric_rows", []):
        cid = row.get("cluster_id")
        if cid is not None:
            numeric_by_cluster.setdefault(cid, []).append(row)

    for row in matrix.get("categorical_rows", []):
        cid = row.get("cluster_id")
        if cid is not None:
            categorical_by_cluster.setdefault(cid, []).append(row)

    # 5. Extract only kept rows and filter tiers on-the-fly
    allowed_tiers = (
        {"core", "supporting"}
        if not include_context_tier
        else {"core", "supporting", "context"}
    )

    numeric_rows = []
    categorical_rows = []
    numeric_feature_scores: dict[str, float] = {}
    categorical_feature_scores: dict[str, float] = {}

    for cid in kept:
        for row in numeric_by_cluster.get(cid, []):
            for key, value in (row.get("values", {}) or {}).items():
                if (
                    isinstance(value, dict)
                    and value.get("signal_tier") not in allowed_tiers
                ):
                    continue
                score = (
                    safe_float(value.get("aggregate_score"))
                    if isinstance(value, dict)
                    else safe_float(value)
                )
                numeric_feature_scores[key] = max(
                    numeric_feature_scores.get(key, 0.0), abs(score)
                )
        for row in categorical_by_cluster.get(cid, []):
            for key, value in (row.get("values", {}) or {}).items():
                if (
                    isinstance(value, dict)
                    and value.get("signal_tier") not in allowed_tiers
                ):
                    continue
                score = (
                    safe_float(value.get("aggregate_score"))
                    if isinstance(value, dict)
                    else safe_float(value)
                )
                categorical_feature_scores[key] = max(
                    categorical_feature_scores.get(key, 0.0), abs(score)
                )

    selected_numeric_features = _rank_matrix_features(
        numeric_feature_scores, _MAX_SIGNAL_MATRIX_NUMERIC
    )
    selected_categorical_features = _rank_matrix_features(
        categorical_feature_scores, _MAX_SIGNAL_MATRIX_CATEGORICAL
    )
    selected_numeric_set = set(selected_numeric_features)
    selected_categorical_set = set(selected_categorical_features)
    numeric_features_omitted = max(
        len(numeric_feature_scores) - len(selected_numeric_features), 0
    )
    categorical_features_omitted = max(
        len(categorical_feature_scores) - len(selected_categorical_features), 0
    )

    for cid in kept:
        for row in numeric_by_cluster.get(cid, []):
            raw_vals = row.get("values", {}) or {}
            filtered_vals = {
                k: safe_float(v.get("value")) if isinstance(v, dict) else safe_float(v)
                for k, v in raw_vals.items()
                if k in selected_numeric_set
                and (not isinstance(v, dict) or v.get("signal_tier") in allowed_tiers)
            }
            if filtered_vals:
                numeric_rows.append({"cluster_id": cid, "values": filtered_vals})

        for row in categorical_by_cluster.get(cid, []):
            raw_vals = row.get("values", {}) or {}
            filtered_vals = {
                k: v.get("value") if isinstance(v, dict) else v
                for k, v in raw_vals.items()
                if k in selected_categorical_set
                and (not isinstance(v, dict) or v.get("signal_tier") in allowed_tiers)
            }
            if filtered_vals:
                categorical_rows.append({"cluster_id": cid, "values": filtered_vals})

    # 6. Return high-density Markdown representation for the LLM
    if return_markdown:
        import io

        report = io.StringIO()
        report.write("## Topological Signal Matrix\n\n")
        report.write("### Telemetry\n")
        report.write(f"- Clusters returned: {len(kept)}\n")
        report.write(f"- Clusters omitted: {len(omitted_ids)}\n")
        report.write(
            f"- Numeric columns shown: {len(selected_numeric_features)}"
            f" ({numeric_features_omitted} omitted)\n"
        )
        report.write(
            f"- Categorical values shown: {len(selected_categorical_features)}"
            f" ({categorical_features_omitted} omitted)\n"
        )
        report.write(
            "- Cell meanings: numeric = z-score; categorical = in-cluster prevalence percent.\n"
        )

        if omitted_telemetry:
            report.write("\n#### Omitted Clusters\n")
            for item in omitted_telemetry:
                score = item["max_signal"]
                rec = (
                    "Recommended for targeted inspection"
                    if score >= 2.0
                    else "Low Signal: Ignorable"
                )
                report.write(
                    f"- Cluster {item['cluster_id']}: max signal {score}; {rec}\n"
                )
        report.write("\n")

        report.write("### Numeric Feature Matrix\n")
        report.write(
            _render_markdown_table(
                numeric_rows, feature_order=selected_numeric_features
            )
            + "\n\n"
        )

        report.write("### Categorical Feature Matrix\n")
        report.write(
            _render_markdown_table(
                categorical_rows, feature_order=selected_categorical_features
            )
            + "\n"
        )

        return {
            "clusters_returned": len(kept),
            "clusters_omitted": len(omitted_ids),
            "omitted_clusters": omitted_telemetry,
            "numeric_features_returned": len(selected_numeric_features),
            "numeric_features_omitted": numeric_features_omitted,
            "categorical_features_returned": len(selected_categorical_features),
            "categorical_features_omitted": categorical_features_omitted,
            "markdown_report": report.getvalue(),
        }

    return {
        "numeric_columns": selected_numeric_features,
        "categorical_values": [
            item
            for item in matrix.get("categorical_values", [])
            if f"{item.get('column')}={item.get('value')}" in selected_categorical_set
        ],
        "clusters_returned": len(kept),
        "clusters_omitted": len(omitted_ids),
        "omitted_clusters": omitted_telemetry,
        "numeric_features_omitted": numeric_features_omitted,
        "categorical_features_omitted": categorical_features_omitted,
        "numeric_rows": numeric_rows,
        "categorical_rows": categorical_rows,
    }


def _rank_matrix_features(scores: dict[str, float], limit: int) -> list[str]:
    return [
        feature
        for feature, _score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:limit]
    ]


def _render_markdown_table(
    rows: list[dict[str, Any]], *, feature_order: list[str] | None = None
) -> str:
    if not rows:
        return "*No signals found in the selected tiers.*"

    available = {k for r in rows for k in r["values"].keys()}
    all_features = (
        [feature for feature in feature_order if feature in available]
        if feature_order is not None
        else sorted(available)
    )
    if not all_features:
        return "*No features available in selected tiers.*"

    import io

    output = io.StringIO()
    output.write("| Cluster ID | " + " | ".join(all_features) + " |\n")
    output.write("| --- |" + " | ".join(["---"] * len(all_features)) + " |\n")

    for row in rows:
        vals = []
        for feat in all_features:
            val = row["values"].get(feat, "-")
            vals.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        output.write(f"| {row['cluster_id']} | " + " | ".join(vals) + " |\n")

    return output.getvalue()


def dossier_to_markdown(dossier: TopologicalDossier) -> str:
    """Renders the dossier as a high-signal Markdown report for LLM consumption."""
    total_rows = sum(
        len(p.numeric_features) + len(p.categorical_features) for p in dossier.clusters
    )

    md = [
        "# Topological Analysis Dossier",
        f"**Dataset Size**: {dossier.n_total} points",
        f"**Clusters Found**: {dossier.n_clusters}",
        "",
    ]

    if total_rows >= _DOSSIER_OVERSIZE_ROW_WARNING:
        md.extend(
            [
                "> [!WARNING]",
                f"> This dossier contains {total_rows} feature-cluster rows, which is extremely large and may cause substantial context bloat.",
                "> It is highly recommended to use targeted profiling tools like `get_cluster_profile` or `get_feature_signal` to inspect specific clusters or features of interest instead of analyzing this full report.",
                "",
            ]
        )

    gated = dossier.global_stats.get("evidence_metadata", {}).get(
        "categorical_columns_gated", []
    )
    if gated:
        gated_desc = ", ".join(
            f"{g['column']} (cardinality {g['cardinality']})" for g in gated
        )
        md.append(
            f"**Categorical columns gated as high-cardinality noise**: {gated_desc}"
        )
        md.append("")

    md.extend(
        [
            "## Cluster Profiles",
            "",
        ]
    )

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
