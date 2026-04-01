"""
Graph diagnostics and quality classification for MCP agentic loops.

Analyzes the fitted cosmic graph to classify quality (good/hairball/singletons/etc)
and returns concrete epsilon corrections for retry logic in <5-shot loops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import networkx as nx

from pulsar.config import config_to_yaml

if TYPE_CHECKING:
    from pulsar.pipeline import ThemaRS

logger = logging.getLogger(__name__)


@dataclass
class SweepHistoryEntry:
    """Record of one sweep attempt's outcome for history-aware retry."""

    quality: str         # "hairball", "singletons", "fragmented", etc.
    epsilon_min: float
    epsilon_max: float
    pca_dims: list[int]


@dataclass
class GraphMetrics:
    """Graph structure measurements."""

    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    giant_fraction: float
    singleton_count: int
    singleton_fraction: float
    component_count: int
    resolved_threshold: float
    nonzero_fraction: float
    weight_p50: float
    weight_p95: float
    component_sizes: list[int]  # sorted descending — agent sees balance at a glance


@dataclass
class DiagnosisResult:
    """Diagnosis outcome with concrete corrective suggestions."""

    quality: str  # "good" | "hairball" | "singletons" | "fragmented" | "sparse_connected"
    metrics: GraphMetrics
    diagnosis: str
    epsilon_factor: float  # informational: multiplier applied
    suggested_epsilon_min: float  # concrete: agent pastes into YAML
    suggested_epsilon_max: float
    suggested_epsilon_steps: int
    suggestions: list[str]
    suggested_config_yaml: str  # complete corrected YAML, pass to run_topological_sweep
    clustering_notes: list[str]  # factual observations about component balance


def diagnose_model(
    model: ThemaRS,
    history: list[SweepHistoryEntry] | None = None,
) -> DiagnosisResult:
    """
    Classify cosmic graph quality and return concrete epsilon corrections.

    Uses sweep history (if provided) to narrow epsilon via binary search
    instead of blind multiplicative correction. Detects oscillation and
    escalates to PCA dimension reduction when needed.

    Args:
        model: Fitted ThemaRS instance
        history: Prior sweep attempts from this session (optional)

    Returns:
        DiagnosisResult with classification, metrics, and corrective suggestions

    Raises:
        RuntimeError: If model has not been fitted
    """
    # Extract graph and weights
    G = model.cosmic_graph
    W = model.weighted_adjacency
    n = G.number_of_nodes()

    if n == 0:
        raise RuntimeError("Empty graph: model may not have been fitted or data is empty")

    # Compute metrics
    n_edges = G.number_of_edges()
    max_pairs = n * (n - 1) // 2
    density = float(n_edges / max_pairs) if max_pairs > 0 else 0.0
    avg_degree = float(2 * n_edges / n) if n > 0 else 0.0

    # Component analysis
    components = list(nx.connected_components(G))
    sizes = sorted((len(c) for c in components), reverse=True)
    giant_fraction = float(sizes[0] / n) if sizes and n > 0 else 0.0
    singleton_count = sum(1 for s in sizes if s == 1)
    singleton_fraction = float(singleton_count / n) if n > 0 else 0.0
    component_count = len(components)

    # Weight analysis
    upper = W[np.triu_indices(n, k=1)]
    nonzero = upper[upper > 0]
    nonzero_fraction = float(len(nonzero) / len(upper)) if len(upper) > 0 else 0.0
    weight_p50 = float(np.percentile(upper, 50)) if len(upper) > 0 else 0.0
    weight_p95 = float(np.percentile(upper, 95)) if len(upper) > 0 else 0.0

    # Component sizes for balance analysis (already computed as `sizes`)
    component_sizes = sizes  # sorted descending

    metrics = GraphMetrics(
        n_nodes=n,
        n_edges=n_edges,
        density=density,
        avg_degree=avg_degree,
        giant_fraction=giant_fraction,
        singleton_count=singleton_count,
        singleton_fraction=singleton_fraction,
        component_count=component_count,
        resolved_threshold=model.resolved_threshold,
        nonzero_fraction=nonzero_fraction,
        weight_p50=weight_p50,
        weight_p95=weight_p95,
        component_sizes=component_sizes,
    )

    # Build factual clustering notes — no judgments, just observations
    clustering_notes = _build_clustering_notes(n, component_sizes, singleton_count)

    # Classify and get epsilon correction factor
    quality, epsilon_factor = _classify(metrics)

    # Build concrete epsilon suggestion from fitted ball maps
    current_epsilons = [bm.eps for bm in model.ball_maps]
    if current_epsilons:
        cur_min, cur_max = min(current_epsilons), max(current_epsilons)
    else:
        cur_min, cur_max = 0.5, 1.5  # safe fallback

    suggested_epsilon_steps = 15

    # History-aware epsilon correction (binary search between known bounds)
    reduce_pca = False
    if history and len(history) >= 2:
        suggested_epsilon_min, suggested_epsilon_max, reduce_pca = (
            _history_aware_epsilon(quality, cur_min, cur_max, epsilon_factor, history)
        )
    else:
        # No history or single attempt: use blind multiplier
        suggested_epsilon_min = round(cur_min * epsilon_factor, 3)
        suggested_epsilon_max = round(cur_max * epsilon_factor, 3)

    # Build diagnostic message and suggestions
    diagnosis, suggestions = _build_diagnosis(quality, metrics, epsilon_factor)

    if reduce_pca:
        suggestions.append(
            "Oscillating between sparse and dense: reducing PCA dimensions "
            "in suggested config to widen the viable epsilon window."
        )

    # Build corrected config YAML with updated epsilon range
    from dataclasses import replace as dc_replace
    from pulsar.config import BallMapperSpec, PCASpec

    corrected_epsilons = np.linspace(
        suggested_epsilon_min, suggested_epsilon_max, suggested_epsilon_steps,
    ).tolist()
    corrected_cfg = dc_replace(
        model.config,
        ball_mapper=BallMapperSpec(epsilons=corrected_epsilons),
    )

    if reduce_pca:
        current_dims = list(model.config.pca.dimensions)
        reduced = sorted(set(max(2, int(d * 0.7)) for d in current_dims))
        corrected_cfg = dc_replace(
            corrected_cfg,
            pca=PCASpec(dimensions=reduced, seeds=list(model.config.pca.seeds)),
        )

    suggested_config_yaml = config_to_yaml(corrected_cfg)

    logger.info(
        "diagnose_model: quality=%s, epsilon_factor=%.2f, nodes=%d, edges=%d, "
        "history_len=%d, reduce_pca=%s",
        quality,
        epsilon_factor,
        n,
        n_edges,
        len(history) if history else 0,
        reduce_pca,
    )

    return DiagnosisResult(
        quality=quality,
        metrics=metrics,
        diagnosis=diagnosis,
        epsilon_factor=epsilon_factor,
        suggested_epsilon_min=suggested_epsilon_min,
        suggested_epsilon_max=suggested_epsilon_max,
        suggested_epsilon_steps=suggested_epsilon_steps,
        suggestions=suggestions,
        suggested_config_yaml=suggested_config_yaml,
        clustering_notes=clustering_notes,
    )


def _build_clustering_notes(
    n: int, component_sizes: list[int], singleton_count: int,
) -> list[str]:
    """Build factual observations about component balance for the agent."""
    notes: list[str] = []
    if component_sizes:
        largest = component_sizes[0]
        largest_pct = largest / n * 100 if n > 0 else 0
        notes.append(
            f"Largest component contains {largest_pct:.0f}% of points "
            f"({largest}/{n})."
        )
    multi_node = sum(1 for s in component_sizes if s >= 2)
    notes.append(f"{multi_node} components have 2+ nodes.")
    if singleton_count > 0:
        notes.append(
            f"{singleton_count} singleton components detected."
        )
    return notes


def _history_aware_epsilon(
    quality: str,
    cur_min: float,
    cur_max: float,
    epsilon_factor: float,
    history: list[SweepHistoryEntry],
) -> tuple[float, float, bool]:
    """Compute epsilon range using binary search on sweep history.

    Returns (suggested_min, suggested_max, reduce_pca).
    """
    sparse_entries = [h for h in history if h.quality in ("singletons", "fragmented")]
    dense_entries = [h for h in history if h.quality == "hairball"]

    # Find tightest known bounds
    lower_bound = max((h.epsilon_max for h in sparse_entries), default=None)
    upper_bound = min((h.epsilon_min for h in dense_entries), default=None)

    if lower_bound is not None and upper_bound is not None:
        # Binary search: midpoint of known bounds ± 15%
        midpoint = (lower_bound + upper_bound) / 2.0
        eps_min = round(midpoint * 0.85, 3)
        eps_max = round(midpoint * 1.15, 3)
    elif lower_bound is not None and quality in ("singletons", "fragmented"):
        # Only sparse history: step up cautiously from known upper bound
        eps_min = round(lower_bound * 1.1, 3)
        eps_max = round(lower_bound * 1.5, 3)
    elif upper_bound is not None and quality == "hairball":
        # Only dense history: step down cautiously from known lower bound
        eps_min = round(upper_bound * 0.4, 3)
        eps_max = round(upper_bound * 0.8, 3)
    else:
        # Fallback to blind multiplier
        eps_min = round(cur_min * epsilon_factor, 3)
        eps_max = round(cur_max * epsilon_factor, 3)

    # Detect oscillation: count direction changes
    directions = []
    for h in history:
        if h.quality in ("singletons", "fragmented"):
            directions.append("sparse")
        elif h.quality == "hairball":
            directions.append("dense")
    oscillation_count = sum(
        1 for i in range(1, len(directions)) if directions[i] != directions[i - 1]
    )
    reduce_pca = oscillation_count >= 2

    return eps_min, eps_max, reduce_pca


def _classify(m: GraphMetrics) -> tuple[str, float]:
    """
    Classify graph quality and return epsilon correction factor.

    Returns:
        (quality_string, epsilon_factor)
        - Hairball: over-connected, reduce epsilon by 0.5x
        - Singletons: all isolated, increase epsilon by 2.0x
        - Fragmented: multiple disconnected clusters, increase by 1.4x
        - Sparse: connected but near-zero weights, increase by 1.2x
        - Good: balanced structure, no change (1.0x)
    """
    # Priority-ordered: worst cases first
    if m.density > 0.3 or m.avg_degree > 100:
        return "hairball", 0.5
    if m.n_edges == 0 or m.giant_fraction < 0.1:
        return "singletons", 2.0
    if m.giant_fraction < 0.5:
        return "fragmented", 1.4
    if m.nonzero_fraction < 0.02 and m.weight_p50 < 0.01:
        # Connected but near-zero weights: no signal
        return "sparse_connected", 1.2
    return "good", 1.0


def _build_diagnosis(
    quality: str, m: GraphMetrics, epsilon_factor: float
) -> tuple[str, list[str]]:
    """Build human-readable diagnostic message and suggestions."""
    suggestions = []

    if quality == "good":
        diagnosis = (
            f"✓ Graph quality is good: {m.n_nodes} nodes, {m.n_edges} edges, "
            f"density={m.density:.4f}, giant_fraction={m.giant_fraction:.2%}. "
            f"Proceed to cluster dossier."
        )
    elif quality == "hairball":
        diagnosis = (
            f"✗ Over-connected hairball: density={m.density:.4f}, "
            f"avg_degree={m.avg_degree:.1f}. Too many edges at current epsilon."
        )
        suggestions = [
            f"Reduce epsilon by {epsilon_factor:.1f}x "
            f"(suggested: [{m.weight_p50:.3f}, {m.weight_p95:.3f}])",
            "Try lower PCA dimensions to reduce false neighbor connections",
        ]
    elif quality == "singletons":
        diagnosis = (
            f"✗ All singletons: {m.singleton_count}/{m.n_nodes} isolated nodes, "
            f"0 edges. Epsilon is too small."
        )
        suggestions = [
            f"Increase epsilon by {epsilon_factor:.1f}x",
            "Ensure data has intrinsic geometry (not random noise)",
            "Try lower PCA dimensions to widen the viable epsilon window",
        ]
    elif quality == "fragmented":
        diagnosis = (
            f"✗ Fragmented graph: giant_fraction={m.giant_fraction:.2%}, "
            f"{m.component_count} connected components. "
            f"Epsilon is too small for global connectivity."
        )
        suggestions = [
            f"Increase epsilon by {epsilon_factor:.1f}x to connect clusters",
            "Review data: may have natural separated populations",
            f"Consider threshold=0.0 to include weak edges "
            f"(current: {m.resolved_threshold:.4f})",
        ]
    else:  # sparse_connected
        diagnosis = (
            f"✗ Sparse connected: {m.n_edges} edges but "
            f"nonzero_fraction={m.nonzero_fraction:.2%}, weight_p50={m.weight_p50:.4f}. "
            f"Graph has no signal."
        )
        suggestions = [
            f"Increase epsilon by {epsilon_factor:.1f}x to strengthen edges",
            "Check data preprocessing: may need different scaling or imputation",
            "Verify Ball Mapper parameters and PCA configuration",
        ]

    return diagnosis, suggestions
