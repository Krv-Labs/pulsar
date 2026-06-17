"""
Graph diagnostics for MCP agentic loops.

Analyzes the fitted cosmic graph to return pure geometric signal (metrics).
The agent is responsible for interpreting these metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
    weight_p25: float
    weight_p50: float
    weight_p95: float
    weight_distribution_sparkline: str
    component_sizes: list[int]  # sorted descending — agent sees balance at a glance
    n_ball_maps: int = 0
    grid_adequacy_status: str = "unknown"
    grid_adequacy_note: str = ""
    advisories: list[dict] = field(default_factory=list)
    # Self-describing provenance for the component_count above: it is the
    # connected-component count of the FITTED cosmic graph at the construction
    # threshold — the reference partition both sides can compare against.
    cluster_provenance: dict[str, Any] = field(default_factory=dict)


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
                "severity": "warning",
                "message": f"One component covers {giant_fraction:.0%} of nodes.",
                "agent_action": (
                    "Do not treat this as final unless a targeted resolution "
                    "sweep has already shown the giant component is stable. "
                    "Use generate_cluster_dossier only after either refining "
                    "the PCA/epsilon grid or explicitly justifying that the "
                    "dominant component is clinically expected."
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
    weight_p25 = float(np.percentile(upper, 25)) if len(upper) > 0 else 0.0
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

    from pulsar.mcp.interpreter import component_count_provenance

    provenance = component_count_provenance(
        resolved_construction_threshold=float(model.resolved_construction_threshold),
        component_count=component_count,
        singleton_count=singleton_count,
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
        weight_p25=weight_p25,
        weight_p50=weight_p50,
        weight_p95=weight_p95,
        weight_distribution_sparkline=weight_dist_spark,
        component_sizes=sizes,
        n_ball_maps=n_ball_maps,
        grid_adequacy_status=grid_status,
        grid_adequacy_note=grid_note,
        advisories=advisories,
        cluster_provenance=provenance,
    )


_MAX_EDGES_IN_SUMMARY = 500


def _build_graph_summary(model: Any) -> dict[str, Any]:
    graph = model.cosmic_graph
    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))
    component_sizes = sorted((len(component) for component in components), reverse=True)

    component_by_node: dict[int, int] = {}
    for component_id, component in enumerate(components):
        for node in component:
            component_by_node[int(node)] = component_id

    nodes = []
    for node in sorted(graph.nodes()):
        nodes.append(
            {
                "node": int(node),
                "component_id": component_by_node[int(node)],
                "degree": int(graph.degree(node)),
                "weighted_degree": float(graph.degree(node, weight="weight")),
            }
        )

    edges = []
    for source, target, data in graph.edges(data=True):
        edges.append(
            {
                "source": int(source),
                "target": int(target),
                "weight": float(data.get("weight", 0.0)),
            }
        )
    edges.sort(key=lambda edge: (-edge["weight"], edge["source"], edge["target"]))

    total_edges = len(edges)
    truncated = total_edges > _MAX_EDGES_IN_SUMMARY
    edges = edges[:_MAX_EDGES_IN_SUMMARY]

    # Precompute bounded graph signals for detail="nodes" without returning raw nodes.
    hubs = []
    for comp_id, comp in enumerate(components):
        comp_nodes = sorted(
            [n for n in nodes if n["component_id"] == comp_id],
            key=lambda x: x["weighted_degree"],
            reverse=True,
        )
        for hn in comp_nodes[:3]:
            hubs.append(
                {
                    "node": hn["node"],
                    "component_id": comp_id,
                    "degree": hn["degree"],
                    "weighted_degree": hn["weighted_degree"],
                }
            )

    overall_hubs = sorted(nodes, key=lambda x: x["weighted_degree"], reverse=True)[:3]

    bridges = []
    n_nodes = graph.number_of_nodes()
    if not graph.is_directed() and n_nodes <= 1000 and total_edges <= 25000:
        try:
            bridges = [int(node) for node in nx.articulation_points(graph)]
        except Exception as e:
            logger.warning(f"Error computing articulation points: {e}")

    degrees = [n["degree"] for n in nodes]
    avg_degree = float(np.mean(degrees)) if degrees else 0.0
    max_degree = int(np.max(degrees)) if degrees else 0
    degree_sparkline = (
        generate_distribution_sparkline(np.array(degrees, dtype=float))
        if degrees
        else ""
    )

    topological_summary = {
        "key_hubs": hubs,
        "overall_hubs": overall_hubs,
        "bridges": sorted(bridges),
        "degree_distribution": {
            "mean": avg_degree,
            "max": max_degree,
            "sparkline": degree_sparkline,
        },
    }

    return {
        "node_count": n_nodes,
        "edge_count": total_edges,
        "resolved_construction_threshold": float(model.resolved_construction_threshold),
        "component_count": len(components),
        "component_sizes": component_sizes,
        "nodes": nodes,
        "edges": edges,
        "edges_truncated": truncated,
        "edges_shown": len(edges),
        "topological_summary": topological_summary,
    }


def _skeleton_graph_payload(
    graph: dict[str, Any],
    *,
    detail: str,
    max_edges: int,
    max_nodes: int,
) -> dict[str, Any]:
    all_nodes = list(graph.get("nodes", []))
    all_edges = list(graph.get("edges", []))
    node_count = int(graph.get("node_count", len(all_nodes)))
    edge_count = int(graph.get("edge_count", len(all_edges)))
    include_nodes = detail in {"full_nodes", "full"}
    include_edges = detail in {"edges", "full"}
    nodes = all_nodes[:max_nodes] if include_nodes else []
    edges = all_edges[:max_edges] if include_edges else []

    payload = {
        "node_count": node_count,
        "edge_count": edge_count,
        "resolved_construction_threshold": graph.get("resolved_construction_threshold"),
        "component_count": graph.get("component_count"),
        "component_sizes": graph.get("component_sizes", []),
        "detail": detail,
        "nodes_returned": len(nodes),
        "nodes_omitted": max(node_count - len(nodes), 0),
        "edges_returned": len(edges),
        "edges_omitted": max(edge_count - len(edges), 0),
        "source_edges_truncated": bool(graph.get("edges_truncated", False)),
    }
    if detail == "nodes":
        payload["topological_summary"] = graph.get("topological_summary", {})
    if include_nodes:
        payload["nodes"] = nodes
    if include_edges:
        payload["edges"] = edges
    return payload


def _finalization_gate(
    metrics: dict[str, Any],
    *,
    sweep_count: int,
    config_yaml: str,
) -> dict[str, Any]:
    import yaml
    from pulsar.mcp.config_tools import _suggest_resolution_pca_dims

    giant_fraction = float(metrics.get("giant_fraction", 0.0))
    density = float(metrics.get("density", 0.0))
    n_ball_maps = int(metrics.get("n_ball_maps", 0) or 0)
    needs_resolution = giant_fraction > 0.95 and (
        sweep_count < 3 or density > 0.35 or n_ball_maps < 40
    )
    if not needs_resolution:
        return {
            "status": "ok",
            "reason": "No dominant-component resolution gate triggered.",
        }

    cfg = yaml.safe_load(config_yaml) or {}
    pca_dims = (
        cfg.get("sweep", {}).get("pca", {}).get("dimensions", {}).get("values", [])
    )
    return {
        "status": "blocked",
        "code": "UNRESOLVED_DOMINANT_COMPONENT",
        "message": (
            f"giant_fraction={giant_fraction:.1%} after {sweep_count} sweep(s). "
            "Do not finalize clusters until one targeted resolution sweep is run "
            "or the dominant component is explicitly justified as clinically expected."
        ),
        "suggested_refinement": {
            "pca_dims": _suggest_resolution_pca_dims(pca_dims),
            "pca_seeds": [42, 7, 13],
            "epsilon_steps_min": 24,
            "strategy": (
                "Zoom into the useful PCA band with multiple seeds. Shift/lower "
                "epsilon if density remains high; avoid narrowing to a single "
                "PCA dimension or single seed."
            ),
        },
    }


def _threshold_stability_summary(
    model: Any, metrics: dict[str, Any]
) -> dict[str, Any] | None:
    """Compact stability block for run_topological_sweep when auto threshold ran.

    Returns ``None`` when the run used a fixed construction_threshold (no
    stability analysis was performed).
    """
    from pulsar.mcp.payloads import singleton_count_at_threshold

    stability = getattr(model, "_stability_result", None)
    if stability is None:
        return None
    adj = model.weighted_adjacency
    n_nodes = int(adj.shape[0])
    top = stability.top_k_plateaus(3)
    plateaus = []
    for p in top:
        singleton_count = singleton_count_at_threshold(adj, float(p.midpoint))
        plateaus.append(
            {
                "start": float(p.start_threshold),
                "end": float(p.end_threshold),
                "midpoint": float(p.midpoint),
                "length": float(p.length),
                "component_count": int(p.component_count),
                "singleton_count": singleton_count,
                "singleton_fraction": round(singleton_count / max(n_nodes, 1), 4),
            }
        )
    warning: str | None = None
    if int(metrics.get("n_edges", 0)) == 0:
        warning = (
            "Auto threshold landed in an empty-graph regime. Pick a lower "
            "plateau midpoint and re-run, or fix construction_threshold."
        )
    elif float(metrics.get("singleton_fraction", 0.0)) > 0.8:
        warning = (
            "Auto threshold left >80% of nodes as singletons. Try a lower "
            "plateau midpoint to retain more edges."
        )
    return {
        "selected_threshold": float(stability.optimal_threshold),
        "top_plateaus": plateaus,
        "warning": warning,
    }
