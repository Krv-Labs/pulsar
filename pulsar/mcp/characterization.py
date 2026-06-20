"""Compact MCP response shaping for dataset characterization."""

from __future__ import annotations

from typing import Any

from pulsar.mcp.minhash_advisor import massive_dataset_advisory


_DEFAULT_PREVIEW_LIMIT = 20
_COLUMN_NAME_PREVIEW_LIMIT = 20
_VALUE_PREVIEW_LIMIT = 80


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
    column_name_preview = _column_name_preview(column_profiles)
    minhash_advisory = massive_dataset_advisory(n_samples)

    payload = {
        "status": "ok",
        "n_samples": geo.get("n_samples"),
        "n_features": geo.get("n_features"),
        "n_columns_total": geo.get("n_columns_total"),
        "missingness_pct": geo.get("missingness_pct"),
        "numeric_missingness_pct": geo.get("missingness_pct"),
        "overall_missingness_pct": _overall_missingness_pct(column_profiles, n_samples),
        "raw_numeric_geometry": {
            "knn_k5_mean": geo.get("knn_k5_mean"),
            "knn_k10_mean": geo.get("knn_k10_mean"),
            "knn_k20_mean": geo.get("knn_k20_mean"),
            "pca_cumulative_variance": geo.get("pca_cumulative_variance", []),
        },
        "schema_summary": _schema_summary(column_profiles, n_samples, len(preview)),
        "column_name_preview": column_name_preview,
        "column_profile_preview": preview,
        "omitted_column_profiles": max(len(column_profiles) - len(preview), 0),
        "recommended_next_tool": "create_config",
        "agent_guidance": (
            "Use create_config(dataset_id) for processed-space calibration. "
            "Use probe_columns(dataset_id, columns) only for specific columns "
            "listed in column_profile_preview or named by the user."
        ),
    }
    # Only present on massive datasets, so omit the key entirely otherwise.
    if minhash_advisory:
        payload["minhash_advisory"] = minhash_advisory
    return payload


def characterization_payload_to_markdown(payload: dict[str, Any]) -> str:
    schema = payload.get("schema_summary", {})
    geometry = payload.get("raw_numeric_geometry", {})
    name_preview = payload.get("column_name_preview", {})
    lines = [
        "# Dataset Characterization",
        "",
        f"- Rows: {payload.get('n_samples')}",
        f"- Numeric features: {payload.get('n_features')}",
        f"- Total columns: {payload.get('n_columns_total')}",
        f"- Numeric missingness: {_fmt_pct(payload.get('numeric_missingness_pct'))}",
        f"- Overall missingness: {_fmt_pct(payload.get('overall_missingness_pct'))}",
        "",
        "## Schema",
        "",
        f"- Numeric columns: {schema.get('numeric_columns', 0)}",
        f"- Categorical columns: {schema.get('categorical_columns', 0)}",
        f"- Columns with missing values: {schema.get('columns_with_missing', 0)}",
        f"- High-cardinality columns: {schema.get('high_cardinality_columns', 0)}",
        f"- All-missing columns: {schema.get('all_missing_columns', 0)}",
        "",
        "## Geometry",
        "",
        f"- kNN mean k=5: {_fmt_float(geometry.get('knn_k5_mean'))}",
        f"- kNN mean k=10: {_fmt_float(geometry.get('knn_k10_mean'))}",
        f"- kNN mean k=20: {_fmt_float(geometry.get('knn_k20_mean'))}",
    ]
    variance = geometry.get("pca_cumulative_variance", [])[:6]
    if variance:
        lines.append(
            "- PCA cumulative variance: "
            + ", ".join(f"{dim}d={float(frac):.3f}" for dim, frac in variance)
        )

    lines.extend(["", "## Column Name Preview", ""])
    for kind in ("numeric", "categorical"):
        block = name_preview.get(kind, {})
        names = block.get("columns", [])
        omitted = block.get("omitted", 0)
        label = "Numeric" if kind == "numeric" else "Categorical"
        if names:
            lines.append(f"- {label}: {', '.join(f'`{name}`' for name in names)}")
        else:
            lines.append(f"- {label}: none")
        if omitted:
            lines.append(f"  - {omitted} more {kind} columns omitted.")

    lines.extend(["", "## Columns To Inspect", ""])
    preview = payload.get("column_profile_preview", [])
    if preview:
        for cp in preview:
            lines.append(
                f"- `{cp.get('name')}`: {cp.get('dtype')}, "
                f"unique={cp.get('n_unique')}, "
                f"missing={_fmt_pct(cp.get('missing_pct'))}"
            )
    else:
        lines.append(
            "- No high-missingness, categorical, or high-cardinality columns flagged."
        )
    omitted_profiles = payload.get("omitted_column_profiles", 0)
    if omitted_profiles:
        lines.append(f"- Column profiles omitted from preview: {omitted_profiles}")

    advisory = payload.get("minhash_advisory")
    if advisory:
        cur = advisory.get("current_profile", {})
        sug = advisory.get("suggested_profile", {})
        lines.extend(
            [
                "",
                "## Cosmic Graph Construction (MinHash)",
                "",
                f"- {advisory.get('message')}",
                f"- Current `minhash_d={advisory.get('current')}`: "
                f"95% CI ±{cur.get('ci95_half_width_worst')}, "
                f"signature {cur.get('signature_memory_human')}.",
                f"- Suggested `minhash_d={advisory.get('suggested')}`: "
                f"95% CI ±{sug.get('ci95_half_width_worst')}, "
                f"signature {sug.get('signature_memory_human')}.",
                "- Set via `refine_config(..., cosmic_graph.minhash_d=<value>)`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Next Tools",
            "",
            f"- `{payload.get('recommended_next_tool', 'create_config')}` for processed-space calibration.",
            "- `probe_columns` for specific columns from the previews above.",
        ]
    )
    return "\n".join(lines).strip()


def probe_columns_payload(
    profiles: list[dict[str, Any]],
    requested_columns: list[str],
    missing_columns: list[str],
) -> dict[str, Any]:
    compact_profiles = [
        _compact_detailed_column_profile(profile) for profile in profiles
    ]
    return {
        "status": "ok",
        "columns_requested": len(requested_columns),
        "columns_returned": len(compact_profiles),
        "missing_columns": missing_columns,
        "column_profiles": compact_profiles,
    }


def probe_columns_payload_to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Column Probe",
        "",
        f"- Columns requested: {payload.get('columns_requested', 0)}",
        f"- Columns returned: {payload.get('columns_returned', 0)}",
    ]
    missing_columns = payload.get("missing_columns", [])
    if missing_columns:
        lines.append(
            "- Missing columns: " + ", ".join(f"`{name}`" for name in missing_columns)
        )

    for profile in payload.get("column_profiles", []):
        lines.extend(
            [
                "",
                f"## `{profile.get('name')}`",
                "",
                f"- Type: {profile.get('dtype')}",
                f"- Numeric: {profile.get('is_numeric')}",
                f"- Unique values: {profile.get('n_unique')}",
                f"- Missing: {profile.get('n_missing')} ({_fmt_pct(profile.get('missing_pct'))})",
            ]
        )
        if profile.get("is_numeric"):
            lines.extend(
                [
                    f"- Mean: {_fmt_float(profile.get('mean'))}",
                    f"- Std: {_fmt_float(profile.get('std'))}",
                    f"- Min / max: {_fmt_float(profile.get('min_val'))} / {_fmt_float(profile.get('max_val'))}",
                ]
            )
        top_values = profile.get("top_values") or []
        if top_values:
            lines.append("- Top values:")
            for value, count in top_values:
                lines.append(f"  - `{value}`: {count}")
        samples = profile.get("sample_values") or []
        if samples:
            lines.append("- Sample values: " + ", ".join(f"`{v}`" for v in samples))
        truncation = profile.get("truncation", {})
        if any(truncation.values()):
            lines.append(
                "- Truncation: "
                + ", ".join(f"{key}={value}" for key, value in truncation.items())
            )

    return "\n".join(lines).strip()


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


def _overall_missingness_pct(
    column_profiles: list[dict[str, Any]], n_samples: int
) -> float:
    if not column_profiles or n_samples <= 0:
        return 0.0
    missing = sum(int(cp.get("n_missing", 0) or 0) for cp in column_profiles)
    total = n_samples * len(column_profiles)
    return round(float(missing / total * 100.0), 2)


def _column_name_preview(
    column_profiles: list[dict[str, Any]],
    *,
    limit: int = _COLUMN_NAME_PREVIEW_LIMIT,
) -> dict[str, dict[str, Any]]:
    numeric = [str(cp.get("name")) for cp in column_profiles if cp.get("is_numeric")]
    categorical = [
        str(cp.get("name")) for cp in column_profiles if not cp.get("is_numeric")
    ]
    return {
        "numeric": {
            "columns": numeric[:limit],
            "omitted": max(len(numeric) - limit, 0),
        },
        "categorical": {
            "columns": categorical[:limit],
            "omitted": max(len(categorical) - limit, 0),
        },
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


def _compact_detailed_column_profile(profile: dict[str, Any]) -> dict[str, Any]:
    sample_values, truncated_samples = _truncate_values(
        profile.get("sample_values") or []
    )
    top_values = profile.get("top_values")
    truncated_top_values = 0
    if top_values:
        compact_top_values = []
        for raw_value, count in top_values:
            value, omitted = _truncate_value(raw_value)
            truncated_top_values += int(omitted)
            compact_top_values.append((value, count))
        top_values = compact_top_values
    return {
        "name": profile.get("name"),
        "dtype": profile.get("dtype"),
        "is_numeric": profile.get("is_numeric"),
        "n_unique": profile.get("n_unique"),
        "n_missing": profile.get("n_missing"),
        "missing_pct": profile.get("missing_pct"),
        "sample_values": sample_values,
        "mean": profile.get("mean"),
        "std": profile.get("std"),
        "min_val": profile.get("min_val"),
        "max_val": profile.get("max_val"),
        "top_values": top_values,
        "truncation": {
            "sample_values_truncated": truncated_samples,
            "top_values_truncated": truncated_top_values,
        },
    }


def _truncate_values(values: list[Any]) -> tuple[list[str], int]:
    out = []
    truncated = 0
    for value in values:
        text, was_truncated = _truncate_value(value)
        out.append(text)
        truncated += int(was_truncated)
    return out, truncated


def _truncate_value(value: Any) -> tuple[str, bool]:
    text = str(value)
    if len(text) <= _VALUE_PREVIEW_LIMIT:
        return text, False
    return f"{text[:_VALUE_PREVIEW_LIMIT]}...", True


def _fmt_pct(value: Any) -> str:
    return f"{float(value or 0.0):.2f}%"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
