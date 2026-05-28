"""
Preprocessing recommendation and repair helpers for Pulsar MCP tools.

Extracted from server.py so that preprocessing domain logic is independently
importable and testable without instantiating the FastMCP server.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import yaml

from pulsar.mcp.errors import mcp_error

logger = logging.getLogger(__name__)

_DIRTY_NUMERIC_SAMPLE_SIZE = 10
_PREPROCESSING_EXPANSION_WARNING_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Column profile normalisation
# ---------------------------------------------------------------------------


def _normalize_profile(cp: Any) -> dict[str, Any]:
    """Return cp as a plain dict regardless of whether it is a dict or dataclass."""
    if isinstance(cp, dict):
        out = dict(cp)
        out.setdefault("sample_values", [])
        return out
    return {
        "name": cp.name,
        "is_numeric": cp.is_numeric,
        "n_unique": cp.n_unique,
        "n_missing": cp.n_missing,
        "missing_pct": cp.missing_pct,
        "sample_values": cp.sample_values,
    }


def enrich_dirty_numeric_samples(
    column_profiles: list[Any],
    dataframe: Any | None,
    *,
    sample_size: int = _DIRTY_NUMERIC_SAMPLE_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add bounded object-column samples so dirty-numeric rules can run."""
    profiles = [_normalize_profile(cp) for cp in column_profiles]
    object_columns = [
        str(cp["name"])
        for cp in profiles
        if not cp.get("is_numeric") and not cp.get("sample_values")
    ]
    if dataframe is None:
        return profiles, {
            "status": "not_available" if object_columns else "enabled",
            "object_columns_sampled": 0,
            "sample_size_per_column": sample_size,
        }

    sampled = 0
    for cp in profiles:
        name = cp["name"]
        if cp.get("is_numeric") or cp.get("sample_values") or name not in dataframe:
            continue
        values = dataframe[name].dropna().head(sample_size * 1000).unique()[:sample_size]
        cp["sample_values"] = [str(value) for value in values]
        sampled += 1
    return profiles, {
        "status": "enabled",
        "object_columns_sampled": sampled,
        "sample_size_per_column": sample_size,
    }


# ---------------------------------------------------------------------------
# Dirty-numeric detection
# ---------------------------------------------------------------------------


def _try_float(s: str) -> bool:
    """Return True if s can be parsed as a float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _looks_like_dirty_numeric(sample_values: list[str]) -> bool:
    """Return True if majority of sample values parse as float.

    Used to detect columns where string sentinels (e.g. 'N/A') caused pandas
    to cast an otherwise numeric column to object dtype.
    """
    if not sample_values:
        return False
    parseable = sum(1 for v in sample_values if _try_float(v))
    return parseable / len(sample_values) > 0.5


# ---------------------------------------------------------------------------
# Preprocessing decision tree
# ---------------------------------------------------------------------------


def _recommend_preprocessing_block(
    column_profiles: list[Any],
    n_samples: int,
) -> tuple[list[str], dict[str, Any], dict[str, Any], list[tuple[str, str, str]]]:
    """Apply decision tree to column profiles and return preprocessing components.

    Returns:
        (drop_columns, impute_dict, encode_dict, rationale_rows)
        where rationale_rows is list of (column, decision_label, rationale).
    """
    drop: list[str] = []
    impute: dict[str, Any] = {}
    encode: dict[str, Any] = {}
    rationale: list[tuple[str, str, str]] = []

    for raw_cp in column_profiles:
        cp = _normalize_profile(raw_cp)
        name = cp["name"]
        is_numeric = cp["is_numeric"]
        n_unique = cp["n_unique"]
        n_missing = cp["n_missing"]
        missing_pct = cp["missing_pct"]
        sample_values = cp["sample_values"]

        # Rule 1: All-missing
        if missing_pct >= 100.0:
            drop.append(name)
            rationale.append((name, "drop", "All values are missing — cannot impute"))
            continue

        if not is_numeric:
            # Rule 2: Dirty numeric — object column where values are mostly parseable as float
            if _looks_like_dirty_numeric(sample_values):
                if missing_pct >= 30:
                    impute[name] = {"method": "sample_normal", "seed": 42}
                    rationale.append(
                        (
                            name,
                            "impute: sample_normal",
                            f"Dirty numeric ({missing_pct:.0f}% missing); string sentinels detected — coercion will rescue it",
                        )
                    )
                else:
                    impute[name] = {"method": "fill_mean", "seed": 42}
                    rationale.append(
                        (
                            name,
                            "impute: fill_mean",
                            f"Dirty numeric ({missing_pct:.0f}% missing); string sentinels detected — coercion will rescue it",
                        )
                    )
                continue

            # Rule 3: High-cardinality ID
            if n_samples > 0 and n_unique / n_samples > 0.9:
                drop.append(name)
                rationale.append(
                    (
                        name,
                        "drop",
                        f"ID-like column ({n_unique}/{n_samples} unique) — no topological signal",
                    )
                )
                continue

            # Rule 4: Too many categories to safely one-hot
            if n_unique > 50:
                drop.append(name)
                rationale.append(
                    (
                        name,
                        "drop",
                        f"{n_unique} categories would add {n_unique} dimensions — distorts Euclidean distance",
                    )
                )
                continue

            # Rule 5: Medium cardinality — cap at actual count so the
            # validation gate (n_cats > max_categories) passes on first try.
            if n_unique > 20:
                encode[name] = {"method": "one_hot", "max_categories": n_unique}
                if n_missing > 0:
                    impute[name] = {"method": "sample_categorical", "seed": 42}
                rationale.append(
                    (
                        name,
                        f"encode: one_hot (max {n_unique})",
                        f"{n_unique} categories — cap set to actual count to avoid validation failure",
                    )
                )
                continue

            # Rule 6: Binary — fill_mode is safe
            if n_unique == 2:
                encode[name] = {"method": "one_hot"}
                if n_missing > 0:
                    impute[name] = {"method": "fill_mode", "seed": 42}
                rationale.append(
                    (
                        name,
                        "encode: one_hot",
                        f"Binary column ({n_unique} values)"
                        + (
                            f"; impute: fill_mode ({n_missing} missing)"
                            if n_missing > 0
                            else ""
                        ),
                    )
                )
                continue

            # Rule 7: Low cardinality categorical
            encode[name] = {"method": "one_hot"}
            if n_missing > 0:
                impute[name] = {"method": "sample_categorical", "seed": 42}
            rationale.append(
                (
                    name,
                    "encode: one_hot",
                    f"{n_unique} unique values — safe cardinality"
                    + (
                        f"; impute: sample_categorical ({n_missing} missing)"
                        if n_missing > 0
                        else ""
                    ),
                )
            )
            continue

        # Numeric column
        if missing_pct >= 30:
            impute[name] = {"method": "sample_normal", "seed": 42}
            rationale.append(
                (
                    name,
                    "impute: sample_normal",
                    f"Numeric, {missing_pct:.0f}% missing — sample_normal preserves distribution shape",
                )
            )
        elif missing_pct >= 10:
            impute[name] = {"method": "sample_normal", "seed": 42}
            rationale.append(
                (
                    name,
                    "impute: sample_normal",
                    f"Numeric, {missing_pct:.0f}% missing — variance preservation required to prevent artificial topological gravity",
                )
            )
        elif missing_pct > 0:
            impute[name] = {"method": "fill_mean", "seed": 42}
            rationale.append(
                (
                    name,
                    "impute: fill_mean",
                    f"Numeric, {missing_pct:.0f}% missing — low missingness, mean fill is stable",
                )
            )
        else:
            rationale.append(
                (name, "no action", "Numeric, complete — no preprocessing needed")
            )

    # Safety net: catch columns with missing values that slipped through
    # (e.g. non-numeric columns routed to drop by rules 3-4, or edge cases
    # in the dirty-numeric detection).
    for raw_cp in column_profiles:
        cp = _normalize_profile(raw_cp)
        name = cp["name"]
        n_missing = cp["n_missing"]
        col_is_numeric = cp["is_numeric"]
        if n_missing > 0 and name not in impute and name not in drop:
            if col_is_numeric:
                impute[name] = {"method": "fill_mean", "seed": 42}
            else:
                impute[name] = {"method": "fill_mode", "seed": 42}
            rationale.append(
                (
                    name,
                    f"impute: {'fill_mean' if col_is_numeric else 'fill_mode'}",
                    f"Safety net — {n_missing} missing values not covered by primary rules",
                )
            )

    return drop, impute, encode, rationale


# ---------------------------------------------------------------------------
# YAML rendering helpers
# ---------------------------------------------------------------------------


def _preprocessing_block_to_yaml(
    drop: list[str],
    impute: dict[str, Any],
    encode: dict[str, Any],
) -> str:
    """Render drop/impute/encode dicts as a preprocessing: YAML block."""
    lines = ["preprocessing:"]
    lines.append(f"  drop_columns: {json.dumps(drop)}")
    if impute:
        lines.append("  impute:")
        for col, spec in impute.items():
            lines.append(
                f"    {col}: {{method: {spec['method']}, seed: {spec.get('seed', 42)}}}"
            )
    else:
        lines.append("  impute: {}")
    if encode:
        lines.append("  encode:")
        for col, spec in encode.items():
            parts = f"method: {spec['method']}"
            if "max_categories" in spec:
                parts += f", max_categories: {spec['max_categories']}"
            lines.append(f"    {col}: {{{parts}}}")
    else:
        lines.append("  encode: {}")
    return "\n".join(lines)


def build_preprocessing_recommendation_payload(
    *,
    drop: list[str],
    impute: dict[str, Any],
    encode: dict[str, Any],
    rationale: list[tuple[str, str, str]],
    column_profiles: list[Any],
    preprocessing_yaml: str,
    expansion_estimate: int,
    dirty_numeric_detection: dict[str, Any],
    detail: str = "summary",
    rationale_limit: int = 20,
) -> dict[str, Any]:
    rows = [
        {"column": col, "decision": decision, "reason": reason}
        for col, decision, reason in rationale
    ]
    counts = _decision_counts(rows)
    rationale_preview = rows[:rationale_limit]
    payload: dict[str, Any] = {
        "status": "ok",
        "detail": detail,
        "preprocessing_yaml": preprocessing_yaml,
        "decision_counts": counts,
        "expansion_estimate": expansion_estimate,
        "warnings": _preprocessing_warnings(
            drop=drop,
            rationale_rows=rows,
            expansion_estimate=expansion_estimate,
            dirty_numeric_detection=dirty_numeric_detection,
        ),
        "dirty_numeric_detection": dirty_numeric_detection,
        "actioned_columns_preview": {
            "drop": drop[:rationale_limit],
            "impute": list(impute)[:rationale_limit],
            "encode": list(encode)[:rationale_limit],
        },
        "rationale_preview": rationale_preview,
        "rationale_omitted": max(len(rows) - len(rationale_preview), 0),
        "recommended_next_tools": [
            "create_config",
            "validate_config",
            "probe_columns",
        ],
    }
    if detail == "full":
        payload["rationale"] = rows
        payload["column_decisions"] = _column_decisions(rows)
        payload["profile_source"] = {
            "columns_profiled": len(column_profiles),
            "dirty_numeric_detection": dirty_numeric_detection,
        }
    return payload


def preprocessing_recommendation_to_markdown(payload: dict[str, Any]) -> str:
    counts = payload.get("decision_counts", {})
    lines = [
        "# Preprocessing Recommendation",
        "",
        f"- Detail: {payload.get('detail', 'summary')}",
        f"- Drop: {counts.get('drop', 0)}",
        f"- Impute: {counts.get('impute', 0)}",
        f"- Encode: {counts.get('encode', 0)}",
        f"- No action: {counts.get('no_action', 0)}",
        f"- Expansion estimate: {payload.get('expansion_estimate', 0)}",
        "",
    ]
    warnings = payload.get("warnings", [])
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- `{warning['code']}`: {warning['message']}")
        lines.append("")
    dirty = payload.get("dirty_numeric_detection", {})
    lines.extend(
        [
            "## Dirty Numeric Detection",
            "",
            f"- Status: {dirty.get('status', 'unknown')}",
            f"- Object columns sampled: {dirty.get('object_columns_sampled', 0)}",
            f"- Sample size per column: {dirty.get('sample_size_per_column', 0)}",
            "",
            "## Recommended YAML",
            "",
            "```yaml",
            payload.get("preprocessing_yaml", ""),
            "```",
            "",
            "## Rationale Preview",
            "",
        ]
    )
    rows = payload.get("rationale_preview", [])
    if rows:
        lines.extend(["| Column | Decision | Rationale |", "|---|---|---|"])
        for row in rows:
            lines.append(
                f"| `{row['column']}` | {row['decision']} | {row['reason']} |"
            )
    else:
        lines.append("- No rationale rows.")
    omitted = payload.get("rationale_omitted", 0)
    if omitted:
        lines.append(f"\n{omitted} rationale rows omitted. Use `detail='full'` for audit.")
    lines.extend(
        [
            "",
            "## Next Tools",
            "",
            "- `create_config` to build a calibrated config with these preprocessing choices.",
            "- `validate_config` to check schema and preprocessing compatibility.",
            "- `probe_columns` to inspect suspicious columns before accepting drops/encodings.",
        ]
    )
    return "\n".join(lines).strip()


def _decision_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        "drop": sum(1 for row in rows if row["decision"] == "drop"),
        "impute": sum(1 for row in rows if row["decision"].startswith("impute")),
        "encode": sum(1 for row in rows if row["decision"].startswith("encode")),
        "no_action": sum(1 for row in rows if row["decision"] == "no action"),
    }


def _column_decisions(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        row["column"]: {"decision": row["decision"], "reason": row["reason"]}
        for row in rows
    }


def _preprocessing_warnings(
    *,
    drop: list[str],
    rationale_rows: list[dict[str, str]],
    expansion_estimate: int,
    dirty_numeric_detection: dict[str, Any],
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if expansion_estimate > _PREPROCESSING_EXPANSION_WARNING_THRESHOLD:
        warnings.append(
            {
                "code": "HIGH_DIMENSION_EXPANSION",
                "severity": "warning",
                "message": (
                    f"One-hot encoding may add {expansion_estimate} dimensions; "
                    "high expansion can distort Euclidean topology."
                ),
            }
        )
    high_cardinality_drops = [
        row["column"]
        for row in rationale_rows
        if row["column"] in drop
        and ("ID-like" in row["reason"] or "categories would add" in row["reason"])
    ]
    if high_cardinality_drops:
        warnings.append(
            {
                "code": "DROPPED_HIGH_CARDINALITY_COLUMNS",
                "severity": "info",
                "message": (
                    f"Dropped {len(high_cardinality_drops)} high-cardinality column(s): "
                    + ", ".join(high_cardinality_drops[:5])
                ),
            }
        )
    all_missing = [
        row["column"]
        for row in rationale_rows
        if row["column"] in drop and "All values are missing" in row["reason"]
    ]
    if all_missing:
        warnings.append(
            {
                "code": "ALL_MISSING_COLUMNS_DROPPED",
                "severity": "info",
                "message": f"Dropped {len(all_missing)} all-missing column(s).",
            }
        )
    dirty_numeric = [
        row["column"] for row in rationale_rows if "Dirty numeric" in row["reason"]
    ]
    if dirty_numeric:
        warnings.append(
            {
                "code": "DIRTY_NUMERIC_DETECTED",
                "severity": "warning",
                "message": (
                    f"Detected {len(dirty_numeric)} object column(s) with numeric-like values."
                ),
            }
        )
    if dirty_numeric_detection.get("status") == "not_available":
        warnings.append(
            {
                "code": "DIRTY_NUMERIC_SAMPLING_UNAVAILABLE",
                "severity": "warning",
                "message": "Object-column samples were unavailable; dirty numeric detection may be incomplete.",
            }
        )
    return warnings


# ---------------------------------------------------------------------------
# Preprocessing config repair
# ---------------------------------------------------------------------------


def repair_config(
    config_yaml: str,
    error_message: str,
    profiles_by_name: dict[str, Any],
) -> str:
    """Auto-repair a preprocessing config given a runtime error message.

    Args:
        config_yaml: The config_yaml that caused the error.
        error_message: The full error text from the failed sweep.
        profiles_by_name: Mapping of column name → column profile (dict or dataclass).

    Returns:
        Markdown with error classification, change log table, and patched config_yaml,
        or a JSON mcp_error envelope if the pattern is unrecognized.
    """
    config_dict = yaml.safe_load(config_yaml)
    if not isinstance(config_dict, dict):
        return mcp_error(
            "repair_preprocessing_config",
            "config_yaml must be a valid YAML mapping.",
        )

    pre = config_dict.setdefault("preprocessing", {})
    drop_list: list[str] = pre.setdefault("drop_columns", [])
    impute_dict: dict[str, Any] = pre.setdefault("impute", {})
    encode_dict: dict[str, Any] = pre.setdefault("encode", {})

    changes: list[tuple[str, str, str, str]] = []  # (col, old, new, rationale)

    def _get_profile_field(col: str, field: str, default: Any = None) -> Any:
        cp = profiles_by_name.get(col)
        if cp is None:
            return default
        return (
            cp.get(field, default)
            if isinstance(cp, dict)
            else getattr(cp, field, default)
        )

    # Pattern 1: Coercion failure — "configured for numeric imputation"
    m = re.search(
        r"Column '([^']+)' is configured for numeric imputation \(([^)]+)\)",
        error_message,
    )
    if m:
        col, old_method = m.group(1), m.group(2)
        n_unique = _get_profile_field(col, "n_unique", 999)
        new_method = "fill_mode" if n_unique <= 10 else "sample_categorical"
        impute_dict[col] = {"method": new_method, "seed": 42}
        changes.append(
            (
                col,
                f"impute: {old_method}",
                f"impute: {new_method}",
                "Column contains non-numeric values; switched to string imputation",
            )
        )

    # Pattern 2: NaN remaining
    elif "NaN values remain after imputation" in error_message:
        nan_cols = re.findall(r"'([^']+)' \(\d+ rows\)", error_message)
        for col in nan_cols:
            is_numeric = _get_profile_field(col, "is_numeric", True)
            n_missing = _get_profile_field(col, "n_missing", 0)
            new_method = "fill_mean" if is_numeric else "sample_categorical"
            impute_dict[col] = {"method": new_method, "seed": 42}
            changes.append(
                (
                    col,
                    "no impute rule",
                    f"impute: {new_method}",
                    f"{'Numeric' if is_numeric else 'Categorical'} column with {n_missing} missing rows",
                )
            )

    # Pattern 3: Non-numeric columns remaining
    elif "Non-numeric columns remain" in error_message:
        bad_cols = re.findall(r"'([^']+)' \(dtype=\w+\)", error_message)
        for col in bad_cols:
            n_unique = _get_profile_field(col, "n_unique", 999)
            if n_unique > 50:
                drop_list.append(col)
                changes.append(
                    (
                        col,
                        "no rule",
                        "drop_columns",
                        f"{n_unique} categories — one-hot would add {n_unique} dimensions",
                    )
                )
            else:
                encode_dict[col] = {"method": "one_hot"}
                changes.append(
                    (
                        col,
                        "no rule",
                        "encode: one_hot",
                        f"{n_unique} unique values — safe to one-hot",
                    )
                )

    # Pattern 4: All-missing
    elif "is all-missing" in error_message:
        m2 = re.search(r"Column '([^']+)' is all-missing", error_message)
        if m2:
            col = m2.group(1)
            drop_list.append(col)
            changes.append(
                (
                    col,
                    "impute/encode",
                    "drop_columns",
                    "Column is entirely null — cannot impute or encode",
                )
            )

    # Pattern 5: Cardinality exceeded
    elif "exceeding max_categories" in error_message:
        m3 = re.search(r"Column '([^']+)' has (\d+) categories", error_message)
        if m3:
            col, n_cats = m3.group(1), int(m3.group(2))
            if n_cats > 50:
                drop_list.append(col)
                if col in encode_dict:
                    del encode_dict[col]
                changes.append(
                    (
                        col,
                        "encode: one_hot",
                        "drop_columns",
                        f"{n_cats} categories is too many for one-hot encoding",
                    )
                )
            else:
                # Raise cap to match actual cardinality so the
                # validation gate (n_cats > max_categories) passes.
                encode_dict[col] = {"method": "one_hot", "max_categories": n_cats}
                changes.append(
                    (
                        col,
                        "encode: one_hot (max_categories too low)",
                        f"encode: one_hot (max_categories={n_cats})",
                        f"Raised cap to match actual {n_cats} categories",
                    )
                )

    else:
        return mcp_error(
            "repair_preprocessing_config",
            f"Unrecognized preprocessing error pattern. Raw error:\n{error_message}",
        )

    if not changes:
        return mcp_error(
            "repair_preprocessing_config",
            "Error was classified but no changes were needed. Review the error message manually.",
        )

    # Classify error for display
    if "NaN values remain" in error_message:
        classification = "`NaN values remain after imputation`"
    elif "Non-numeric columns" in error_message:
        classification = "`Non-numeric columns remain after preprocessing`"
    elif "configured for numeric imputation" in error_message:
        classification = "`Numeric imputation on non-numeric column`"
    elif "all-missing" in error_message:
        classification = "`All-missing column`"
    elif "exceeding max_categories" in error_message:
        classification = "`Cardinality limit exceeded`"
    else:
        classification = "`Preprocessing error`"

    result = f"## Error Classification\n{classification} — {len(changes)} change(s) made.\n\n"
    result += "## Changes Made\n\n"
    result += "| Column | Was | Now | Rationale |\n|---|---|---|---|\n"
    for col, was, now, reason in changes:
        result += f"| `{col}` | {was} | {now} | {reason} |\n"

    patched_yaml = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
    result += f"\n## Patched Config\n\n```yaml\n{patched_yaml}```\n"
    result += "\nCall `validate_preprocessing_config` with this config before re-running `run_topological_sweep`."
    return result


def _calibrate_processed_space(
    df: Any,
    column_profiles: list[Any],
    n_samples: int,
    data_path: str,
) -> Any:
    """Run recommended preprocessing + scaling, then profile the result.

    Returns ``None`` if preprocessing fails (e.g. too few numeric columns
    after encoding).  The caller should fall back to raw-space calibration.
    """
    drop, impute_dict, encode_dict, _ = _recommend_preprocessing_block(
        column_profiles, n_samples
    )

    # Build a minimal config just for preprocessing (sweep/cosmic values
    # are irrelevant — we only need preprocessing fields + data path).
    from pulsar.config import (
        BallMapperSpec,
        CosmicGraphSpec,
        EncodeSpec,
        ImputeSpec,
        PCASpec,
        PulsarConfig,
    )
    import numpy as np
    from sklearn.preprocessing import StandardScaler as SkScaler
    from pulsar.preprocessing import preprocess_dataframe
    from pulsar.analysis.characterization import profile_numeric_matrix

    impute_specs = {
        col: ImputeSpec(method=spec["method"], seed=spec.get("seed", 42))
        for col, spec in impute_dict.items()
    }
    encode_specs = {
        col: EncodeSpec(
            method=spec["method"], max_categories=spec.get("max_categories")
        )
        for col, spec in encode_dict.items()
    }
    temp_cfg = PulsarConfig(
        data=data_path,
        impute=impute_specs,
        encode=encode_specs,
        drop_columns=drop,
        pca=PCASpec(),
        ball_mapper=BallMapperSpec(),
        cosmic_graph=CosmicGraphSpec(),
    )

    try:
        df_processed, layout = preprocess_dataframe(df, temp_cfg)
    except (ValueError, TypeError):
        logger.info("Processed-space calibration failed; falling back to raw geometry")
        return None

    X = df_processed.to_numpy(dtype=np.float64)
    if X.shape[1] < 2:
        return None
    X_scaled = SkScaler().fit_transform(X)
    return profile_numeric_matrix(X_scaled)

