"""Processed feature-space dry run and target proxy overlap measurement."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from pulsar.config import load_config
from pulsar.mcp.preprocessing_plan.model import PreprocessingPlan
from pulsar.mcp.preprocessing_plan.render import plan_to_preprocessing_dict
from pulsar.preprocessing import preprocess_dataframe


def _target_non_zero_mask(series: pd.Series) -> np.ndarray:
    """Return boolean mask of rows where *series* is non-zero / True."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().any():
        return num.fillna(0.0).to_numpy(dtype=np.float64) != 0.0
    vals = series.dropna().astype(str).str.strip().str.lower()
    mask = np.zeros(len(series), dtype=bool)
    notna_idx = np.flatnonzero(series.notna().to_numpy())
    mask[notna_idx] = ~vals.isin(
        {"", "0", "0.0", "false", "no", "f", "n", "none", "nan"}
    ).to_numpy(dtype=bool)
    return mask


def _resolve_proxy_targets(df: pd.DataFrame, plan: PreprocessingPlan) -> list[str]:
    """Return explicit target columns, falling back to boolean mask-dependency partners."""
    target_cols = [
        col
        for col, entry in plan.entries.items()
        if entry.role == "target" and col in df.columns
    ]
    if not target_cols:
        ev_by_col = {c.name: c for c in plan.evidence.columns}
        for md in plan.evidence.mask_dependencies:
            p = md.partner
            if (
                p not in target_cols
                and p in df.columns
                and p in ev_by_col
                and ev_by_col[p].kind == "boolean"
            ):
                target_cols.append(p)
    return target_cols


def _candidate_proxy_columns(
    df: pd.DataFrame,
    df_proc: pd.DataFrame,
    output_names: Sequence[str],
    encode_prefixes: tuple[str, ...],
) -> list[str]:
    """Return processed marker and one-hot dummy columns eligible for target proxy checks."""
    input_cols = set(df.columns)
    candidate_cols: list[str] = []
    for c in output_names:
        c_str = str(c)
        is_marker = c_str.endswith("_was_missing")
        is_dummy = (
            not is_marker
            and c not in input_cols
            and bool(encode_prefixes)
            and c_str.startswith(encode_prefixes)
        )
        if (is_marker or is_dummy) and c in df_proc.columns:
            candidate_cols.append(c_str)
    return candidate_cols


def _measure_target_proxies(
    df: pd.DataFrame,
    df_proc: pd.DataFrame,
    output_names: Sequence[str],
    encode_prefixes: tuple[str, ...],
    plan: PreprocessingPlan,
) -> list[dict[str, Any]]:
    """Measure overlap between processed marker/dummy columns and target or boolean event-flag columns."""
    n_rows = len(df)
    if n_rows == 0 or df_proc.shape[0] != n_rows:
        return []

    target_cols = _resolve_proxy_targets(df, plan)
    if not target_cols:
        return []

    candidate_cols = _candidate_proxy_columns(
        df, df_proc, output_names, encode_prefixes
    )
    if not candidate_cols:
        return []

    proxies: list[dict[str, Any]] = []
    for tgt in target_cols:
        t_mask = _target_non_zero_mask(df[tgt])
        n_target = int(np.sum(t_mask))
        base_rate = round(float(n_target / n_rows), 4)
        for col in candidate_cols:
            arr = df_proc[col].to_numpy(dtype=np.float64)
            m1 = arr == 1.0
            m1_rows = int(np.sum(m1))
            n_both_1 = int(np.sum(m1 & t_mask))
            if col.endswith("_was_missing") and n_both_1 == 0:
                m0 = arr == 0.0
                m0_rows = int(np.sum(m0))
                n_both_0 = int(np.sum(m0 & t_mask))
                if m0_rows > 0 and n_both_0 == m0_rows:
                    proxies.append(
                        {
                            "column": col,
                            "target": tgt,
                            "marker_rows": m0_rows,
                            "share_with_target": 1.0,
                            "target_base_rate": base_rate,
                        }
                    )
                    continue
            if m1_rows <= 0:
                continue
            share = (
                1.0
                if n_both_1 == m1_rows
                else min(0.9999, round(float(n_both_1 / m1_rows), 4))
            )
            proxies.append(
                {
                    "column": col,
                    "target": tgt,
                    "marker_rows": m1_rows,
                    "share_with_target": share,
                    "target_base_rate": base_rate,
                }
            )

    proxies.sort(
        key=lambda item: (
            -item["share_with_target"],
            -item["marker_rows"],
            item["column"],
            item["target"],
        )
    )
    return proxies


def dry_run_processed_space(
    df: pd.DataFrame, plan: PreprocessingPlan
) -> tuple[dict[str, int] | None, dict[str, Any]]:
    """Run ``preprocess_dataframe`` on *df* using *plan* without raising on failure."""
    try:
        pre_dict = plan_to_preprocessing_dict(plan)
        cfg = load_config(
            {
                "run": {"name": "dry_run", "data": "in_memory"},
                "preprocessing": pre_dict,
            }
        )
        df_proc, layout = preprocess_dataframe(df, cfg)
        output_names = layout.feature_names
        marker_count = sum(1 for c in output_names if str(c).endswith("_was_missing"))
        encode_prefixes = tuple(f"{col}_" for col in cfg.encode.keys())
        input_cols = set(df.columns)
        dummy_count = (
            sum(
                1
                for c in output_names
                if not str(c).endswith("_was_missing")
                and c not in input_cols
                and str(c).startswith(encode_prefixes)
            )
            if encode_prefixes
            else 0
        )
        plan.target_proxies = _measure_target_proxies(
            df, df_proc, output_names, encode_prefixes, plan
        )
        return (
            {
                "feature_count": int(df_proc.shape[1]),
                "missing_marker_count": int(marker_count),
                "dummy_column_count": int(dummy_count),
            },
            {"valid": True, "error": None},
        )
    except Exception as exc:
        plan.target_proxies = []
        return None, {"valid": False, "error": str(exc)}
