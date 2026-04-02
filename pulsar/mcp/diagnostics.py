"""
Graph diagnostics for MCP agentic loops.

Analyzes the fitted cosmic graph to return pure geometric signal (metrics).
The agent is responsible for interpreting these metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import networkx as nx

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
    resolved_threshold: float
    nonzero_fraction: float
    weight_p50: float
    weight_p95: float
    component_sizes: list[int]  # sorted descending — agent sees balance at a glance


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

    logger.info(
        "diagnose_model: nodes=%d, edges=%d, components=%d",
        n,
        n_edges,
        component_count,
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
        resolved_threshold=model.resolved_threshold,
        nonzero_fraction=nonzero_fraction,
        weight_p50=weight_p50,
        weight_p95=weight_p95,
        component_sizes=sizes,
    )
