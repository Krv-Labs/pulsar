"""Compact MCP payload builders for cluster interpretation tools."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def singleton_count_at_threshold(adj: np.ndarray, threshold: float) -> int:
    row_max = adj.max(axis=1)
    return int((row_max <= threshold).sum())


_CLUSTER_ID_STABILITY_NOTE = (
    "Cluster IDs are stable across tools; subsets differ only by ranking basis. "
    "ID lists are capped previews with exact omitted counts."
)


def bounded_list(items: Any, *, preview_limit: int = 20) -> dict[str, Any]:
    """Canonical bounded collection: ``{total, preview, omitted}``.

    The single shape for any collection too large to inline in a summary
    payload. Size collections extend this core via ``size_summary``; never
    serialize a raw O(n) list into a summary.
    """
    seq = list(items)
    preview = seq[:preview_limit]
    return {
        "total": len(seq),
        "preview": preview,
        "omitted": max(len(seq) - len(preview), 0),
    }


def size_summary(sizes: list[int], *, preview_limit: int = 20) -> dict[str, Any]:
    """Bounded size telemetry: the :func:`bounded_list` core plus size stats.

    Summaries must not leak O(n) component/cluster size arrays on shattered
    graphs. This keeps exact counts while exposing the shape agents need.
    """
    values = [int(size) for size in sizes if int(size) > 0]
    sorted_values = sorted(values, reverse=True)
    singleton_count = sum(1 for size in values if size == 1)
    summary = bounded_list(sorted_values, preview_limit=preview_limit)
    summary.update(
        {
            "largest": sorted_values[:5],
            "singleton_count": singleton_count,
            "size_buckets": {
                "1": singleton_count,
                "2-5": sum(1 for size in values if 2 <= size <= 5),
                "6-20": sum(1 for size in values if 6 <= size <= 20),
                ">20": sum(1 for size in values if size > 20),
            },
        }
    )
    return summary


def compact_cluster_result_payload(
    cluster_meta: dict[str, Any],
    *,
    include_full_sizes: bool = False,
) -> dict[str, Any]:
    """Cluster-result header with bounded size telemetry by default."""
    out = dict(cluster_meta)
    cluster_sizes = list(cluster_meta.get("cluster_sizes", []) or [])
    size_values = [int(row.get("n", 0)) for row in cluster_sizes]
    out["cluster_sizes_summary"] = size_summary(size_values)
    if not include_full_sizes:
        out.pop("cluster_sizes", None)
        stability_plateaus = out.get("stability_plateaus")
        if stability_plateaus is None:
            out.pop("stability_plateaus", None)
    return out


def build_cluster_selection(
    cluster_meta: dict[str, Any],
    shown_ids: Any,
    *,
    ranked_by: str,
) -> dict[str, Any]:
    """Uniform disclosure of which clusters a truncated view actually shows.

    Sibling interpretation tools truncate the same partition by different
    criteria (size vs. signal). This block makes the ranking basis, the total,
    and the shown/omitted IDs explicit so a partial view is never mistaken for a
    divergent partition. IDs are shared across tools — only the displayed subset
    differs.
    """
    all_ids = [int(c["cluster_id"]) for c in cluster_meta.get("cluster_sizes", [])]
    total = int(cluster_meta.get("n_clusters", len(all_ids)))
    shown = sorted({int(c) for c in shown_ids})
    shown_set = set(shown)
    omitted = sorted(cid for cid in all_ids if cid not in shown_set)
    return {
        "total_clusters": total,
        "ranked_by": ranked_by,
        "shown_ids": bounded_list(shown),
        "omitted_ids": bounded_list(omitted),
        "note": (
            f"Showing {len(shown)} of {total} clusters, ranked by {ranked_by}. "
            + _CLUSTER_ID_STABILITY_NOTE
        ),
    }


def cluster_selection_to_markdown(selection: dict[str, Any]) -> str:
    """Compact, prominent markdown header for a cluster-selection disclosure."""
    shown = selection.get("shown_ids", {})
    omitted = selection.get("omitted_ids", {})
    return "\n".join(
        [
            "## Cluster Selection",
            "",
            f"- Showing {shown.get('total', 0)} of "
            f"{selection.get('total_clusters', 0)} clusters, ranked by "
            f"`{selection.get('ranked_by', 'unknown')}`",
            f"- Shown IDs: {shown.get('preview', [])} (+{shown.get('omitted', 0)} more)",
            f"- Omitted IDs: {omitted.get('preview', [])} "
            f"(+{omitted.get('omitted', 0)} more)",
            f"- {selection.get('note', '')}",
            "",
        ]
    )


def cluster_result_payload(result: Any) -> dict[str, Any]:
    sizes = result.labels.value_counts().sort_index()
    total = len(result.labels)
    fragmentation = _cluster_fragmentation_payload(sizes, total)
    return {
        "method_used": result.method_used,
        "n_clusters": result.n_clusters,
        "cluster_sizes": [
            {"cluster_id": int(cid), "n": int(n), "pct": round(n / total * 100, 1)}
            for cid, n in sizes.items()
        ],
        "cluster_fragmentation": fragmentation,
        "interpretation_readiness": _interpretation_readiness(fragmentation),
        "silhouette_score": result.silhouette_score,
        "interpretation_edge_weight_threshold_applied": result.interpretation_edge_weight_threshold_applied,
        "stability_plateaus": result.stability_plateaus,
        "failure_reason": result.failure_reason,
    }


def build_evidence_payload(
    dossier: Any,
    cluster_meta: dict[str, Any],
    *,
    detail: str = "standard",
    max_clusters: int | None = None,
    feature_preview_limit: int = 5,
) -> dict[str, Any]:
    clusters = _cluster_payload_from_dossier(
        dossier,
        detail=detail,
        feature_preview_limit=feature_preview_limit,
    )
    clusters.sort(key=lambda c: c["size"], reverse=True)
    clusters_omitted = 0
    if max_clusters is not None and len(clusters) > max_clusters:
        clusters_omitted = len(clusters) - max_clusters
        clusters = clusters[:max_clusters]
    evidence_metadata = dossier.global_stats.get("evidence_metadata", {})
    payload: dict[str, Any] = {
        "status": "ok",
        "cluster_result": compact_cluster_result_payload(
            cluster_meta,
            include_full_sizes=detail == "full",
        ),
        "detail": dossier.global_stats.get("detail", "standard"),
        "graph_metrics": dossier.global_stats.get("graph_metrics", {}),
        "clusters_returned": len(clusters),
        "clusters_omitted": clusters_omitted,
        "clusters": clusters,
    }
    if detail == "summary":
        payload["evidence_metadata_summary"] = _evidence_metadata_summary(
            evidence_metadata
        )
        payload["numeric_global_ranking_preview"] = dossier.global_stats.get(
            "numeric_global_ranking", []
        )[:10]
        payload["categorical_global_ranking_preview"] = dossier.global_stats.get(
            "categorical_global_ranking", []
        )[:10]
        signal_matrix = dossier.global_stats.get("signal_matrix", {})
        payload["signal_matrix_summary"] = {
            "n_numeric_columns": len(signal_matrix.get("numeric_columns", [])),
            "n_categorical_values": len(signal_matrix.get("categorical_values", [])),
            "n_numeric_rows": len(signal_matrix.get("numeric_rows", [])),
            "n_categorical_rows": len(signal_matrix.get("categorical_rows", [])),
        }
    else:
        payload["evidence_metadata"] = evidence_metadata
        payload["numeric_global_ranking"] = dossier.global_stats.get(
            "numeric_global_ranking", []
        )
        payload["categorical_global_ranking"] = dossier.global_stats.get(
            "categorical_global_ranking", []
        )
        payload["signal_matrix"] = dossier.global_stats.get("signal_matrix", {})
    return payload


def build_summary_evidence_payload(
    evidence_index: Any,
    cluster_meta: dict[str, Any],
    graph_metrics: dict[str, Any],
    *,
    max_clusters: int | None = 8,
    feature_preview_limit: int = 5,
    global_preview_limit: int = 5,
) -> dict[str, Any]:
    clusters = []
    bundles = sorted(
        evidence_index.cluster_bundles.values(),
        key=lambda bundle: int(bundle["size"]),
        reverse=True,
    )
    clusters_omitted = 0
    if max_clusters is not None and len(bundles) > max_clusters:
        clusters_omitted = len(bundles) - max_clusters
        bundles = bundles[:max_clusters]

    for bundle in bundles:
        clusters.append(_cluster_payload_from_bundle(bundle, feature_preview_limit))

    signal_matrix = evidence_index.signal_matrix or {}
    return {
        "status": "ok",
        "cluster_result": compact_cluster_result_payload(cluster_meta),
        "detail": "summary",
        "graph_metrics": graph_metrics,
        "clusters_returned": len(clusters),
        "clusters_omitted": clusters_omitted,
        "clusters": clusters,
        "evidence_metadata_summary": _evidence_metadata_summary(
            evidence_index.metadata
        ),
        "numeric_global_ranking_preview": evidence_index.numeric_global_ranking[
            :global_preview_limit
        ],
        "categorical_global_ranking_preview": evidence_index.categorical_global_ranking[
            :global_preview_limit
        ],
        "signal_matrix_summary": {
            "n_numeric_columns": len(signal_matrix.get("numeric_columns", [])),
            "n_categorical_values": len(signal_matrix.get("categorical_values", [])),
            "n_numeric_rows": len(signal_matrix.get("numeric_rows", [])),
            "n_categorical_rows": len(signal_matrix.get("categorical_rows", [])),
        },
    }


def summary_evidence_payload_to_markdown(payload: dict[str, Any]) -> str:
    cluster_result = payload.get("cluster_result", {})
    readiness = cluster_result.get("interpretation_readiness", {})
    lines = [
        "# Cluster Dossier Summary",
        "",
        f"- Method: {cluster_result.get('method_used', 'unknown')}",
        f"- Clusters: {cluster_result.get('n_clusters', 0)}",
        f"- Returned: {payload.get('clusters_returned', 0)}",
        f"- Omitted: {payload.get('clusters_omitted', 0)}",
    ]
    if readiness:
        lines.append(f"- Readiness: {readiness.get('status', 'unknown')}")
        reason_codes = readiness.get("reason_codes") or []
        if reason_codes:
            lines.append(f"- Readiness reasons: {', '.join(reason_codes)}")
        message = readiness.get("message")
        if message:
            lines.append(f"- Guidance: {message}")

    signal_summary = payload.get("signal_matrix_summary", {})
    lines.extend(
        [
            "",
            "## Signal Inventory",
            "",
            f"- Numeric signal columns: {signal_summary.get('n_numeric_columns', 0)}",
            f"- Categorical signal values: {signal_summary.get('n_categorical_values', 0)}",
        ]
    )

    gated = payload.get("evidence_metadata_summary", {}).get(
        "categorical_columns_gated", []
    )
    if gated:
        gated_desc = ", ".join(
            f"{g['column']} (cardinality {g['cardinality']})" for g in gated
        )
        lines.append(
            f"- Categorical columns gated as high-cardinality noise: {gated_desc}"
        )

    lines.extend(["", "## Clusters", ""])

    for cluster in payload.get("clusters", []):
        lines.append(f"### Cluster {cluster['cluster_id']}: {cluster['semantic_name']}")
        lines.append(f"- Size: {cluster['size']} ({cluster['size_pct']:.1f}%)")
        lines.append(
            "- Numeric tiers: "
            + _format_tier_counts(cluster.get("numeric_tier_counts", {}))
        )
        lines.append(
            "- Categorical tiers: "
            + _format_tier_counts(cluster.get("categorical_tier_counts", {}))
        )
        preview = cluster.get("feature_preview", [])
        if preview:
            lines.append("- Feature preview (top capped signals):")
            for row in preview:
                lines.append(f"  - {_format_feature_preview(row)}")
        else:
            lines.append("- Feature preview: none in selected tiers")
        lines.append(f"- Feature rows omitted: {cluster.get('features_omitted', 0)}")
        lines.append("")

    lines.extend(
        [
            "## Next Tools",
            "",
            "- `get_cluster_profile` for one cluster's capped evidence rows.",
            "- `get_feature_signal` for selected feature columns across clusters.",
            "- `get_cluster_signal_matrix` for a compact cross-cluster matrix.",
            "- `export_html_report` for a human-facing full report outside agent context.",
        ]
    )
    return "\n".join(lines).strip()


def cluster_profile_payload_to_markdown(payload: dict[str, Any]) -> str:
    cluster = payload["cluster"]
    neighbor_text = (
        ", ".join(
            str(neighbor["cluster_id"]) for neighbor in cluster["topological_neighbors"]
        )
        or "None"
    )
    lines = [
        "# Cluster Profile",
        "",
        f"## Cluster {cluster['cluster_id']}: {cluster['semantic_name']}",
        f"- Size: {cluster['size']} ({cluster['size_pct']:.1f}%)",
        f"- Topological neighbors: {neighbor_text}",
        f"- Detail: {payload.get('detail', 'standard')}",
        f"- Feature rows returned: {cluster['features_returned']['total']}",
        f"- Feature rows omitted: {cluster['features_omitted']['total']}",
        "",
        "### Numeric signals",
    ]
    if not cluster["numeric_features"]:
        lines.append("- None in selected tiers.")
    for row in cluster["numeric_features"]:
        lines.append(f"- {_format_numeric_signal(row)}")

    lines.extend(["", "### Categorical signals"])
    if not cluster["categorical_features"]:
        lines.append("- None in selected tiers.")
    for row in cluster["categorical_features"]:
        lines.append(f"- {_format_categorical_signal(row)}")

    lines.extend(
        [
            "",
            "### Next Tools",
            "- Use `get_feature_signal` to compare a listed feature across clusters.",
            "- Use `compare_clusters` for pairwise statistical comparison.",
        ]
    )
    return "\n".join(lines).strip()


def feature_signal_payload_to_markdown(signals: dict[str, Any]) -> str:
    lines = [
        "# Feature Signal Summary",
        "",
        f"- Features: {', '.join(signals.get('feature_names', [])) or 'none'}",
        f"- Detail: {signals.get('detail', 'summary')}",
        f"- Clusters returned: {signals.get('clusters_returned', 0)}",
        f"- Clusters omitted: {signals.get('clusters_omitted', 0)}",
        "",
    ]
    if signals.get("clusters_omitted", 0):
        lines.extend(
            [
                "> More clusters had matching signals. Pass explicit `cluster_ids` "
                "or raise `max_clusters` only if you need the tail.",
                "",
            ]
        )
    for cluster in signals.get("clusters", []):
        lines.append(f"## Cluster {cluster['cluster_id']}: {cluster['semantic_name']}")
        lines.append(f"- Size: {cluster['cluster_size']}")
        numeric = cluster.get("numeric_features", [])
        categorical = cluster.get("categorical_features", [])
        if numeric:
            lines.append("- Numeric:")
            for row in numeric:
                lines.append(f"  - {_format_numeric_signal(row)}")
        if categorical:
            lines.append("- Categorical:")
            for row in categorical:
                lines.append(f"  - {_format_categorical_signal(row)}")
        if not numeric and not categorical:
            lines.append("- No selected feature signals.")
        lines.append("")
    return "\n".join(lines).strip()


def build_sweep_summary_payload(
    response: dict[str, Any],
    *,
    component_limit: int = 20,
    include_config_yaml: bool = False,
) -> dict[str, Any]:
    metrics = response.get("metrics", {}) or {}
    component_sizes = list(metrics.get("component_sizes", []) or [])
    component_preview = component_sizes[:component_limit]
    payload: dict[str, Any] = {
        "status": response.get("status", "ok"),
        "detail": "summary",
        "run_id": response.get("run_id"),
        "dataset_id": response.get("dataset_id"),
        "data_shape": response.get("data_shape"),
        "construction_threshold": response.get("construction_threshold"),
        "graph_health": response.get("graph_health"),
        "analysis_status": response.get("analysis_status"),
        "next_required_check": response.get("next_required_check"),
        "finalization_gate": response.get("finalization_gate"),
        "constructed_graph_connected": response.get("constructed_graph_connected"),
        "full_affinity_connected": response.get("full_affinity_connected"),
        "spectral_clustering_allowed": response.get("spectral_clustering_allowed"),
        "key_metrics": _sweep_key_metrics(metrics),
        "component_sizes_preview": component_preview,
        "component_sizes_omitted": max(
            len(component_sizes) - len(component_preview), 0
        ),
        "diff_summary": _sweep_diff_summary(response.get("diff", [])),
        "threshold_stability_summary": response.get("threshold_stability_summary"),
        "projection_method": response.get("projection_method"),
        "projection_cached": response.get("projection_cached"),
        "projection_cache_status": response.get("projection_cache_status"),
        "pca_cached": response.get("pca_cached"),
        "pca_cache_status": response.get("pca_cache_status"),
        "memory_usage_mb": response.get("memory_usage_mb"),
        "config_advisory": response.get("config_advisory"),
        "config_yaml_included": include_config_yaml,
        "config_yaml_available_via": "get_runtime_context",
        "next_tools": [
            "diagnose_cosmic_graph",
            "get_sweep_history",
            "get_threshold_stability_curve",
            "generate_cluster_dossier",
        ],
    }
    if include_config_yaml:
        payload["config_yaml_normalized"] = response.get("config_yaml_normalized")
    if response.get("saved_config_path"):
        payload["saved_config_path"] = response["saved_config_path"]
    return payload


def sweep_payload_to_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("key_metrics", {})
    gate = payload.get("finalization_gate", {}) or {}
    diff = payload.get("diff_summary", {})
    lines = [
        "# Sweep Run Complete",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Dataset ID: `{payload.get('dataset_id')}`",
        f"- Graph health status: {_format_graph_health_status(payload)}",
        f"- Constructed graph connected: {payload.get('constructed_graph_connected')}",
        f"- Construction threshold (edge-weight cut): "
        f"{_format_metric_value(payload.get('construction_threshold'))}",
        f"- Analysis status: {payload.get('analysis_status')}",
        f"- Next required check: `{payload.get('next_required_check')}`",
        f"- Finalization gate: {gate.get('status', 'unknown')}",
    ]
    if payload.get("config_advisory"):
        lines.append(f"- Config advisory: {payload['config_advisory']}")
    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for key, value in metrics.items():
        lines.append(f"| {key} | {_format_metric_value(value)} |")

    metric_changes = diff.get("metric_changes", {})
    if diff.get("parameter_changes") or metric_changes:
        lines.extend(["", "## Diff From Previous Sweep", ""])
        if diff.get("parameter_changes"):
            lines.append(
                "- Parameter changes: "
                + ", ".join(f"`{item}`" for item in diff["parameter_changes"])
            )
        if metric_changes:
            lines.extend(["", "| Metric | Previous | Current |", "|---|---:|---:|"])
            for key, change in metric_changes.items():
                lines.append(
                    f"| {key} | {_format_metric_value(change.get('previous'))} | "
                    f"{_format_metric_value(change.get('current'))} |"
                )

    stability = payload.get("threshold_stability_summary")
    if stability:
        lines.extend(
            [
                "",
                "## Threshold Stability",
                "",
                f"- Selected threshold: {_format_metric_value(stability.get('selected_threshold'))}",
            ]
        )
        if stability.get("warning"):
            lines.append(f"- Warning: {stability['warning']}")

    if gate.get("status") in {"blocked", "caution"}:
        lines.extend(
            [
                "",
                "## Gate Advisory",
                "",
                f"- Status: `{gate.get('status')}`",
                f"- Code: `{gate.get('code')}`",
                f"- Message: {gate.get('message')}",
            ]
        )
        action = gate.get("recommended_action") or {}
        if action:
            threshold = action.get("interpretation_edge_weight_threshold")
            command = (
                f"{action.get('tool', 'generate_cluster_dossier')}"
                f'(method="{action.get("method", "components")}"'
            )
            if threshold is not None:
                command += (
                    ", interpretation_edge_weight_threshold="
                    f"{_format_metric_value(threshold)}"
                )
            command += ")"
            lines.append(f"- Recommended action: `{command}`")
            reason = action.get("reason")
            if reason:
                lines.append(f"- Reason: {reason}")

    lines.extend(
        [
            "",
            "## Component Sizes",
            "",
            "- Preview: "
            + ", ".join(
                str(value) for value in payload.get("component_sizes_preview", [])
            ),
        ]
    )
    omitted = payload.get("component_sizes_omitted", 0)
    if omitted > 0:
        lines.append(f"- Omitted: {omitted}")
    lines.extend(
        [
            "",
            "## Next Tools",
            "",
        ]
    )
    lines.extend(f"- `{tool}`" for tool in payload.get("next_tools", []))
    return "\n".join(lines).strip()


def _format_graph_health_status(payload: dict[str, Any]) -> str:
    health = payload.get("graph_health")
    metrics = payload.get("key_metrics", {}) or {}
    giant = float(metrics.get("giant_fraction", 0.0) or 0.0)
    components = int(metrics.get("component_count", 0) or 0)
    singletons = float(metrics.get("singleton_fraction", 0.0) or 0.0)

    if health == "connected" and components > 1:
        if giant >= 0.85:
            return "usable_giant_component_with_tail"
        return "usable_multicomponent"
    if health == "fragmented" and singletons >= 0.25:
        return "fragmented_singleton_heavy"
    return str(health)


def _sweep_key_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "n_nodes",
        "n_edges",
        "density",
        "component_count",
        "giant_fraction",
        "singleton_fraction",
        "n_ball_maps",
        "grid_adequacy_status",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def _sweep_diff_summary(diff: list[dict[str, Any]]) -> dict[str, Any]:
    field_map = {
        "pca_dims": "sweep.projection.dimensions.values",
        "projection_dimensions": "sweep.projection.dimensions.values",
        "epsilon": "cosmic_graph.epsilon",
        "dataset_id": "dataset_id",
    }
    metric_fields = {"n_edges", "component_count", "giant_fraction"}
    parameter_changes = []
    metric_changes: dict[str, dict[str, Any]] = {}
    for row in diff:
        field = row.get("field")
        if field in metric_fields:
            metric_changes[field] = {
                "previous": row.get("previous"),
                "current": row.get("current"),
            }
        else:
            parameter_changes.append(field_map.get(field, field))
    return {
        "parameter_changes": parameter_changes,
        "metric_changes": metric_changes,
        "changes_returned": len(parameter_changes) + len(metric_changes),
    }


def _cluster_fragmentation_payload(sizes: pd.Series, total: int) -> dict[str, Any]:
    values = sorted((int(value) for value in sizes.tolist()), reverse=True)
    if total <= 0 or not values:
        return {
            "n_clusters": 0,
            "singleton_cluster_count": 0,
            "singleton_cluster_ratio": 0.0,
            "singleton_point_fraction": 0.0,
            "small_cluster_count": 0,
            "small_cluster_point_fraction": 0.0,
            "nontrivial_cluster_count": 0,
            "nontrivial_point_fraction": 0.0,
            "significant_cluster_count": 0,
            "largest_non_giant_cluster_pct": 0.0,
            "reference_nontrivial_min_size": 0,
            "reference_significant_min_size": 0,
        }

    nontrivial_min = max(10, int(np.ceil(total * 0.01)))
    significant_min = max(25, int(np.ceil(total * 0.03)))
    tail = values[1:]
    singleton_count = sum(1 for value in values if value == 1)
    small = [value for value in values if value < nontrivial_min]
    nontrivial = [value for value in values if value >= nontrivial_min]
    significant = [value for value in values if value >= significant_min]

    return {
        "n_clusters": len(values),
        "singleton_cluster_count": singleton_count,
        "singleton_cluster_ratio": round(singleton_count / len(values), 4),
        "singleton_point_fraction": round(singleton_count / total, 4),
        "small_cluster_count": len(small),
        "small_cluster_point_fraction": round(sum(small) / total, 4),
        "nontrivial_cluster_count": len(nontrivial),
        "nontrivial_point_fraction": round(sum(nontrivial) / total, 4),
        "significant_cluster_count": len(significant),
        "largest_non_giant_cluster_pct": round(max(tail, default=0) / total, 4),
        "reference_nontrivial_min_size": nontrivial_min,
        "reference_significant_min_size": significant_min,
    }


def _interpretation_readiness(fragmentation: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    if fragmentation["singleton_cluster_count"] > 0:
        reason_codes.append("SINGLETON_TAIL_PRESENT")
    if (
        fragmentation["singleton_cluster_count"]
        > fragmentation["nontrivial_cluster_count"]
        and fragmentation["singleton_point_fraction"]
        > fragmentation["largest_non_giant_cluster_pct"]
    ):
        reason_codes.append("SINGLETON_TAIL_DOMINATES_NON_GIANT_STRUCTURE")
    if (
        fragmentation["small_cluster_point_fraction"]
        > fragmentation["largest_non_giant_cluster_pct"]
        and fragmentation["small_cluster_count"] > 1
    ):
        reason_codes.append("TINY_CLUSTER_MASS_DOMINATES_TAIL")
    if (
        fragmentation["n_clusters"] > 1
        and fragmentation["nontrivial_cluster_count"] <= 1
    ):
        reason_codes.append("LOW_NONTRIVIAL_CLUSTER_COUNT")

    if "SINGLETON_TAIL_DOMINATES_NON_GIANT_STRUCTURE" in reason_codes:
        status = "review_required"
        message = (
            "Cluster labels are dominated by singleton/tiny tail structure. Treat "
            "the labels as exploratory until threshold stability or sweep history "
            "shows coherent non-giant mass."
        )
        avoid = ["cluster naming before reviewing stability", "full dossier first"]
    elif reason_codes:
        status = "caution"
        message = (
            "Cluster labels include a small-cluster tail. Interpret only after "
            "checking whether the tail is stable signal or ice-chipping."
        )
        avoid = ["cluster naming before checking small-cluster evidence"]
    else:
        status = "ready"
        message = "No singleton-heavy interpretation warning detected."
        avoid = []

    return {
        "status": status,
        "reason_codes": reason_codes,
        "basis": "advisory_from_relative_cluster_mass; not a hard quality threshold",
        "message": message,
        "allowed_next_steps": [
            "get_sweep_history",
            "get_threshold_stability_curve",
            "get_cluster_profile",
            "get_feature_signal",
        ],
        "avoid": avoid,
    }


def _tier_counts(tiers: dict[str, list]) -> dict[str, int]:
    return {tier: len(rows) for tier, rows in (tiers or {}).items()}


def _tier_counts_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        tier: sum(1 for row in rows if row.get("signal_tier") == tier)
        for tier in ("core", "supporting", "context", "noise")
    }


def _feature_preview_row(row: dict[str, Any], kind: str) -> dict[str, Any]:
    out = {
        "kind": kind,
        "column": row.get("column"),
        "signal_tier": row.get("signal_tier"),
        "aggregate_score": row.get("aggregate_score"),
    }
    if kind == "numeric":
        out["direction"] = row.get("direction")
        out["mean"] = row.get("mean")
        out["mean_rest"] = row.get("mean_rest")
    else:
        out["value"] = row.get("value")
        out["lift"] = row.get("lift")
        out["prevalence_cluster"] = row.get("prevalence_cluster")
    return out


def _feature_preview_from_profile(profile: Any, limit: int) -> dict[str, Any]:
    numeric_rows = [
        ("numeric", row)
        for row in profile.numeric_features
        if row.get("signal_tier") in {"core", "supporting"}
    ]
    categorical_rows = [
        ("categorical", row)
        for row in profile.categorical_features
        if row.get("signal_tier") in {"core", "supporting"}
    ]
    ranked = sorted(
        numeric_rows + categorical_rows,
        key=lambda item: -abs(float(item[1].get("aggregate_score", 0.0))),
    )
    total_available = sum(_tier_counts(profile.numeric_tiers).values()) + sum(
        _tier_counts(profile.categorical_tiers).values()
    )
    preview = [_feature_preview_row(row, kind) for kind, row in ranked[:limit]]
    return {
        "feature_preview": preview,
        "feature_preview_limit": limit,
        "features_previewed": len(preview),
        "features_omitted": max(total_available - len(preview), 0),
    }


def _feature_preview_from_bundle(bundle: dict[str, Any], limit: int) -> dict[str, Any]:
    numeric_rows = [
        ("numeric", row)
        for row in bundle.get("numeric", [])
        if row.get("signal_tier") in {"core", "supporting"}
    ]
    categorical_rows = [
        ("categorical", row)
        for row in bundle.get("categorical", [])
        if row.get("signal_tier") in {"core", "supporting"}
    ]
    ranked = sorted(
        numeric_rows + categorical_rows,
        key=lambda item: -abs(float(item[1].get("aggregate_score", 0.0))),
    )
    total_available = sum(
        1
        for row in [*bundle.get("numeric", []), *bundle.get("categorical", [])]
        if row.get("signal_tier") in {"core", "supporting", "context"}
    )
    preview = [_feature_preview_row(row, kind) for kind, row in ranked[:limit]]
    return {
        "feature_preview": preview,
        "feature_preview_limit": limit,
        "features_previewed": len(preview),
        "features_omitted": max(total_available - len(preview), 0),
    }


def _cluster_payload_from_bundle(
    bundle: dict[str, Any], feature_preview_limit: int
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "cluster_id": int(bundle["cluster_id"]),
        "size": int(bundle["size"]),
        "size_pct": float(bundle["size_pct"]),
        "semantic_name": str(bundle["semantic_name"]),
        "topological_neighbors": list(bundle["topological_neighbors"]),
        "numeric_tier_counts": _tier_counts_from_rows(bundle.get("numeric", [])),
        "categorical_tier_counts": _tier_counts_from_rows(
            bundle.get("categorical", [])
        ),
    }
    entry.update(_feature_preview_from_bundle(bundle, feature_preview_limit))
    return entry


def _format_tier_counts(counts: dict[str, int]) -> str:
    return ", ".join(
        f"{tier}={int(counts.get(tier, 0))}"
        for tier in ("core", "supporting", "context", "noise")
    )


def _format_feature_preview(row: dict[str, Any]) -> str:
    score = row.get("aggregate_score")
    score_text = f", score={float(score):.3f}" if score is not None else ""
    if row.get("kind") == "numeric":
        return (
            f"{row.get('column')}: {row.get('direction', 'mixed')} "
            f"({row.get('signal_tier', 'unknown')}{score_text})"
        )
    return (
        f"{row.get('column')}={row.get('value')}: "
        f"{row.get('signal_tier', 'unknown')} signal"
        f"{score_text}"
    )


def _format_numeric_signal(row: dict[str, Any]) -> str:
    z_score = row.get("z_score")
    z_text = f", z={float(z_score):.2f}" if z_score is not None else ""
    return (
        f"{row['column']}: {row.get('direction', 'mixed')} "
        f"({row.get('signal_tier', 'unknown')}{z_text}, "
        f"score={float(row.get('aggregate_score', 0.0)):.3f})"
    )


def _format_categorical_signal(row: dict[str, Any]) -> str:
    prevalence = row.get("prevalence_cluster", row.get("in_cluster_prevalence"))
    prevalence_text = (
        f", prevalence={float(prevalence):.1f}%" if prevalence is not None else ""
    )
    return (
        f"{row['column']}={row['value']}: {row.get('signal_tier', 'unknown')} "
        f"(lift={float(row.get('lift', 0.0)):.2f}{prevalence_text}, "
        f"score={float(row.get('aggregate_score', 0.0)):.3f})"
    )


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _evidence_metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    stats_failures = metadata.get("stats_failures", {}) if metadata else {}
    return {
        "stats_failure_counts": {
            key: len(value) if hasattr(value, "__len__") else 0
            for key, value in stats_failures.items()
        },
        "numeric_features_screened": metadata.get("numeric_features_screened", 0),
        "categorical_columns_screened": metadata.get("categorical_columns_screened", 0),
        "categorical_columns_gated": metadata.get("categorical_columns_gated", []),
    }


def _cluster_payload_from_dossier(
    dossier: Any, *, detail: str = "standard", feature_preview_limit: int = 5
) -> list[dict[str, Any]]:
    payloads = []
    for profile in dossier.clusters:
        entry: dict[str, Any] = {
            "cluster_id": profile.cluster_id,
            "size": profile.size,
            "size_pct": profile.size_pct,
            "semantic_name": profile.semantic_name,
            "topological_neighbors": profile.topological_neighbors,
        }
        if detail == "summary":
            entry["numeric_tier_counts"] = _tier_counts(profile.numeric_tiers)
            entry["categorical_tier_counts"] = _tier_counts(profile.categorical_tiers)
            entry.update(_feature_preview_from_profile(profile, feature_preview_limit))
        else:
            entry["numeric_features"] = profile.numeric_features
            entry["categorical_features"] = profile.categorical_features
            entry["numeric_tiers"] = profile.numeric_tiers
            entry["categorical_tiers"] = profile.categorical_tiers
            entry["central_rows"] = profile.central_rows
        payloads.append(entry)
    return payloads


def _threshold_surface_payload(
    *,
    construction_threshold: float,
    interpretation_threshold: float,
    threshold_inherited: bool,
    threshold_source: str,
) -> dict[str, Any]:
    import numpy as np

    if threshold_source == "spectral_default_full_affinity":
        status = "full_affinity_spectral"
        guidance = (
            "Spectral clustering used the full retained weighted affinity matrix. "
            "The persisted diagnostic graph was not rebuilt."
        )
    elif threshold_inherited:
        status = "matched"
        guidance = "Interpretation inherited the constructed graph surface."
    elif np.isclose(interpretation_threshold, construction_threshold):
        status = "matched_explicit"
        guidance = "Interpretation explicitly matches the constructed graph surface."
    elif interpretation_threshold < construction_threshold:
        status = "looser_than_construction"
        guidance = (
            "Interpretation uses a looser slice of the retained weighted matrix "
            "than the persisted diagnostic graph. Cluster labels may not match "
            "topological neighbor and centrality evidence from the persisted graph."
        )
    else:
        status = "stricter_than_construction"
        guidance = (
            "Interpretation uses a stricter slice of the retained weighted matrix "
            "than the persisted diagnostic graph. Cluster labels may not match "
            "topological neighbor and centrality evidence from the persisted graph."
        )
    return {
        "status": status,
        "construction_threshold": construction_threshold,
        "interpretation_edge_weight_threshold": interpretation_threshold,
        "threshold_inherited": threshold_inherited,
        "threshold_source": threshold_source,
        "guidance": guidance,
    }


def _prepend_threshold_markdown(markdown: str, surface: dict[str, Any]) -> str:
    header = "\n".join(
        [
            "# Threshold Surface",
            "",
            f"- Construction threshold: {surface['construction_threshold']:.4f}",
            "- Interpretation edge weight threshold: "
            f"{surface['interpretation_edge_weight_threshold']:.4f}",
            f"- Status: {surface['status']}",
            f"- Guidance: {surface['guidance']}",
            "",
        ]
    )
    return f"{header}\n{markdown}"
