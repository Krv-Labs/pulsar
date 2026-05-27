"""Compact MCP payload builders for cluster interpretation tools."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def singleton_count_at_threshold(adj: np.ndarray, threshold: float) -> int:
    row_max = adj.max(axis=1)
    return int((row_max <= threshold).sum())


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
        "cluster_result": cluster_meta,
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
            "summarize_sweep_history",
            "get_threshold_stability_curve",
            "get_cluster_profile",
            "get_feature_signal",
        ],
        "avoid": avoid,
    }


def _tier_counts(tiers: dict[str, list]) -> dict[str, int]:
    return {tier: len(rows) for tier, rows in (tiers or {}).items()}


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


def _evidence_metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    stats_failures = metadata.get("stats_failures", {}) if metadata else {}
    return {
        "stats_failure_counts": {
            key: len(value) if hasattr(value, "__len__") else 0
            for key, value in stats_failures.items()
        },
        "numeric_features_screened": metadata.get("numeric_features_screened", 0),
        "categorical_features_screened": metadata.get(
            "categorical_features_screened", 0
        ),
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
