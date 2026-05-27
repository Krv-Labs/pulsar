"""Compact MCP response shaping for dataset characterization."""

from __future__ import annotations

from typing import Any


_DEFAULT_PREVIEW_LIMIT = 20


def compact_characterization_payload(
    geo: dict[str, Any],
    *,
    preview_limit: int = _DEFAULT_PREVIEW_LIMIT,
) -> dict[str, Any]:
    """Return an agent-sized characterization summary.

    The analysis layer still produces complete ``column_profiles`` for internal
    calibration. The MCP map view should not dump O(width) schema by default.
    """
    column_profiles = list(geo.get("column_profiles", []))
    n_samples = int(geo.get("n_samples", 0) or 0)
    preview = _interesting_column_preview(column_profiles, n_samples, preview_limit)

    return {
        "status": "ok",
        "n_samples": geo.get("n_samples"),
        "n_features": geo.get("n_features"),
        "n_columns_total": geo.get("n_columns_total"),
        "missingness_pct": geo.get("missingness_pct"),
        "raw_numeric_geometry": {
            "knn_k5_mean": geo.get("knn_k5_mean"),
            "knn_k10_mean": geo.get("knn_k10_mean"),
            "knn_k20_mean": geo.get("knn_k20_mean"),
            "pca_cumulative_variance": geo.get("pca_cumulative_variance", []),
        },
        "schema_summary": _schema_summary(column_profiles, n_samples, len(preview)),
        "column_profile_preview": preview,
        "omitted_column_profiles": max(len(column_profiles) - len(preview), 0),
        "recommended_next_tool": "create_config",
        "agent_guidance": (
            "Use create_config(dataset_id) for processed-space calibration. "
            "Use probe_columns(dataset_id, columns) only for specific columns "
            "listed in column_profile_preview or named by the user."
        ),
    }


def _schema_summary(
    column_profiles: list[dict[str, Any]],
    n_samples: int,
    preview_count: int,
) -> dict[str, int]:
    numeric_count = sum(1 for cp in column_profiles if cp.get("is_numeric"))
    missing_count = sum(1 for cp in column_profiles if int(cp.get("n_missing", 0)) > 0)
    high_cardinality_count = sum(
        1 for cp in column_profiles if _is_high_cardinality(cp, n_samples)
    )
    all_missing_count = sum(
        1 for cp in column_profiles if float(cp.get("missing_pct", 0.0)) >= 100.0
    )

    return {
        "numeric_columns": numeric_count,
        "categorical_columns": len(column_profiles) - numeric_count,
        "columns_with_missing": missing_count,
        "high_cardinality_columns": high_cardinality_count,
        "all_missing_columns": all_missing_count,
        "previewed_column_profiles": preview_count,
    }


def _interesting_column_preview(
    column_profiles: list[dict[str, Any]],
    n_samples: int,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = [
        (_interest_score(cp, n_samples), cp)
        for cp in column_profiles
        if _interest_score(cp, n_samples) > 0
    ]
    ranked.sort(
        key=lambda item: (
            -item[0],
            -float(item[1].get("missing_pct", 0.0)),
            str(item[1].get("name", "")),
        )
    )
    return [_compact_column_profile(cp) for _, cp in ranked[:limit]]


def _interest_score(cp: dict[str, Any], n_samples: int) -> int:
    score = 0
    if float(cp.get("missing_pct", 0.0)) >= 100.0:
        score += 8
    elif int(cp.get("n_missing", 0)) > 0:
        score += 4
    if not cp.get("is_numeric"):
        score += 2
    if _is_high_cardinality(cp, n_samples):
        score += 3
    return score


def _is_high_cardinality(cp: dict[str, Any], n_samples: int) -> bool:
    if cp.get("is_numeric"):
        return False
    n_unique = int(cp.get("n_unique", 0))
    return n_unique > 50 or (n_samples > 0 and n_unique / n_samples > 0.9)


def _compact_column_profile(cp: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": cp.get("name"),
        "dtype": cp.get("dtype"),
        "is_numeric": cp.get("is_numeric"),
        "n_unique": cp.get("n_unique"),
        "n_missing": cp.get("n_missing"),
        "missing_pct": cp.get("missing_pct"),
    }
