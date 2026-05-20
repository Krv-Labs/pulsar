"""
Graph diagnostics for MCP agentic loops.

Analyzes the fitted cosmic graph to return pure geometric signal (metrics).
The agent is responsible for interpreting these metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import networkx as nx

from pulsar.runtime.utils import generate_distribution_sparkline

if TYPE_CHECKING:
    from pulsar.pipeline import ThemaRS

logger = logging.getLogger(__name__)


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
    resolved_construction_threshold: float
    nonzero_fraction: float
    weight_p50: float
    weight_p95: float
    weight_distribution_sparkline: str
    component_sizes: list[int]  # sorted descending — agent sees balance at a glance
    n_ball_maps: int = 0
    grid_adequacy_status: str = "unknown"
    grid_adequacy_note: str = ""
    advisories: list[dict] = field(default_factory=list)


def _grid_adequacy(n_ball_maps: int) -> tuple[str, str]:
    if n_ball_maps <= 0:
        return (
            "unknown",
            "Ball-map count is unavailable; run a standard sweep before judging grid adequacy.",
        )
    if n_ball_maps < 15:
        return (
            "under_sampled",
            "Grid has fewer than 15 ball maps. Add PCA dimensions, seeds, or epsilon steps before trusting structural persistence.",
        )
    if n_ball_maps < 40:
        return (
            "thin_grid",
            "Grid has fewer than 40 ball maps. Usable for a quick probe, but compare against a wider sweep before interpreting clusters.",
        )
    return (
        "sample_count_ok",
        "Grid sample count is adequate for a baseline. Still judge graph quality from density, components, and sweep comparisons.",
    )


def _graph_advisories(
    *,
    n_edges: int,
    singleton_fraction: float,
    giant_fraction: float,
    density: float,
) -> list[dict]:
    """Advisory codes for degenerate graph regimes.

    Advisories are data on the success payload — they do not raise. Each
    carries an ``agent_action`` consistent with the ``mcp_error`` pattern so
    agents can react with a single decision rule.
    """
    out: list[dict] = []
    if density > 0.8 and n_edges > 0:
        out.append(
            {
                "code": "HAIRBALL_DENSITY",
                "severity": "warning",
                "message": f"Graph density is {density:.0%} — most node pairs share an edge.",
                "agent_action": (
                    "Raise construction_threshold or shift the PCA grid upward "
                    "(drop low dims) before clustering. A near-fully-connected "
                    "graph cannot express topological structure."
                ),
            }
        )
    if n_edges == 0:
        out.append(
            {
                "code": "EMPTY_GRAPH",
                "severity": "error",
                "message": "Construction threshold produced zero edges.",
                "agent_action": (
                    "Lower construction_threshold to a plateau midpoint from "
                    "threshold_stability_summary, then re-run the sweep."
                ),
            }
        )
    if singleton_fraction > 0.8:
        out.append(
            {
                "code": "HIGH_SINGLETONS",
                "severity": "warning",
                "message": f"{singleton_fraction:.0%} of nodes are singletons.",
                "agent_action": (
                    "Lower construction_threshold, or pick a lower-θ plateau "
                    "from threshold_stability_summary, then re-run the sweep."
                ),
            }
        )
    if giant_fraction > 0.95:
        out.append(
            {
                "code": "DOMINANT_COMPONENT",
                "severity": "info",
                "message": f"One component covers {giant_fraction:.0%} of nodes.",
                "agent_action": (
                    "Call generate_cluster_dossier with "
                    "interpretation_edge_weight_threshold > construction_threshold "
                    "to split the giant component before rebuilding the graph."
                ),
            }
        )
    return out


def diagnose_model(model: ThemaRS) -> GraphMetrics:
    """
    Extract pure graph metrics from the fitted model.

    Args:
        model: Fitted ThemaRS instance

    Returns:
        GraphMetrics with raw topological signal

    Raises:
        RuntimeError: If model has not been fitted
    """
    G = model.cosmic_graph
    W = model.weighted_adjacency
    n = G.number_of_nodes()

    if n == 0:
        raise RuntimeError(
            "Empty graph: model may not have been fitted or data is empty"
        )

    n_edges = G.number_of_edges()
    max_pairs = n * (n - 1) // 2
    density = float(n_edges / max_pairs) if max_pairs > 0 else 0.0
    avg_degree = float(2 * n_edges / n) if n > 0 else 0.0

    components = list(nx.connected_components(G))
    sizes = sorted((len(c) for c in components), reverse=True)
    giant_fraction = float(sizes[0] / n) if sizes and n > 0 else 0.0
    singleton_count = sum(1 for s in sizes if s == 1)
    singleton_fraction = float(singleton_count / n) if n > 0 else 0.0
    component_count = len(components)

    upper = W[np.triu_indices(n, k=1)]
    nonzero = upper[upper > 0]
    nonzero_fraction = float(len(nonzero) / len(upper)) if len(upper) > 0 else 0.0
    weight_p50 = float(np.percentile(upper, 50)) if len(upper) > 0 else 0.0
    weight_p95 = float(np.percentile(upper, 95)) if len(upper) > 0 else 0.0
    weight_dist_spark = generate_distribution_sparkline(upper) if len(upper) > 0 else ""
    n_ball_maps = len(getattr(model, "_ball_maps", []) or [])
    grid_status, grid_note = _grid_adequacy(n_ball_maps)
    advisories = _graph_advisories(
        n_edges=n_edges,
        singleton_fraction=singleton_fraction,
        giant_fraction=giant_fraction,
        density=density,
    )

    logger.info(
        "diagnose_model: nodes=%d, edges=%d, components=%d, ball_maps=%d",
        n,
        n_edges,
        component_count,
        n_ball_maps,
    )

    return GraphMetrics(
        n_nodes=n,
        n_edges=n_edges,
        density=density,
        avg_degree=avg_degree,
        giant_fraction=giant_fraction,
        singleton_count=singleton_count,
        singleton_fraction=singleton_fraction,
        component_count=component_count,
        resolved_construction_threshold=model.resolved_construction_threshold,
        nonzero_fraction=nonzero_fraction,
        weight_p50=weight_p50,
        weight_p95=weight_p95,
        weight_distribution_sparkline=weight_dist_spark,
        component_sizes=sizes,
        n_ball_maps=n_ball_maps,
        grid_adequacy_status=grid_status,
        grid_adequacy_note=grid_note,
        advisories=advisories,
    )
