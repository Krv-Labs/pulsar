"""
Parity harness for `pulsar.mcp.interpreter._compute_numeric_rows`.

This is the gate for any reimplementation of the numeric evidence computation.
The current implementation generates baseline JSON snapshots in
``tests/fixtures/feature_evidence_parity/``. Future implementations must
reproduce those snapshots within the per-field tolerances declared below, and
must produce identical `failure_reasons` sets per (cluster_id, column).

To regenerate baselines after an intentional change, set
``PULSAR_REGENERATE_PARITY_BASELINES=1`` and re-run this file. Do NOT
regenerate to silence a parity failure — investigate the divergence first.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from pulsar.mcp.interpreter import _compute_numeric_rows


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "feature_evidence_parity"
REGENERATE = os.environ.get("PULSAR_REGENERATE_PARITY_BASELINES") == "1"


# Per-field absolute tolerances. Stats that emerge from the same arithmetic in
# both implementations get a tight bound; rank-based tests that ride scipy
# normal-approximation paths get a slightly looser one to absorb float reorder.
_FIELD_TOLERANCES: dict[str, float] = {
    # Trivial arithmetic
    "mean": 1e-12,
    "mean_rest": 1e-12,
    "median": 1e-12,
    "median_rest": 1e-12,
    "std_cluster": 1e-12,
    "std_rest": 1e-12,
    "mad_cluster": 1e-12,
    "mad_rest": 1e-12,
    "global_mean": 1e-12,
    "global_mean_rest": 1e-12,
    "z_score": 1e-10,
    "homogeneity": 1e-10,
    "effect_mean_std": 1e-10,
    "effect_median_mad": 1e-10,
    # Closed-form rank/ECDF tests
    "ks_stat": 1e-12,
    "wasserstein_norm": 1e-12,
    "variance_ratio_log": 1e-10,
    "concentration_gain": 1e-10,
    # p-values from scipy / vectorized equivalents
    "one_vs_rest_p": 1e-10,
    "one_vs_rest_q": 1e-10,
    "global_p": 1e-10,
    "global_q": 1e-10,
    "neighbor_p": 1e-10,
    # Effect sizes (composite)
    "global_cluster_signal": 1e-10,
    "neighbor_effect": 1e-10,
    "specificity_score": 1e-10,
    "aggregate_score": 1e-10,
    "percentile_score": 1e-10,
}

# Fields compared by equality, not numerical tolerance.
_EXACT_FIELDS = frozenset(
    {
        "column",
        "cluster_id",
        "direction",
        "signal_tier",
        "sparkline",
        "neighbor_cluster_id",
    }
)


# ---------------------------------------------------------------------------
# Synthetic case builders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    name: str
    build: Callable[[], tuple[pd.DataFrame, list[str], pd.Series, dict[int, list[dict]]]]


def _adjacency_from_cluster_pairs(
    cluster_ids: list[int], pairs: list[tuple[int, int]]
) -> dict[int, list[dict[str, Any]]]:
    adj: dict[int, list[dict[str, Any]]] = {cid: [] for cid in cluster_ids}
    for a, b in pairs:
        adj[a].append(
            {
                "cluster_id": b,
                "bridge_weight": 1.0,
                "normalized_weight": 1.0,
                "bridge_strength": 1.0,
            }
        )
        adj[b].append(
            {
                "cluster_id": a,
                "bridge_weight": 1.0,
                "normalized_weight": 1.0,
                "bridge_strength": 1.0,
            }
        )
    return adj


def _full_adjacency(cluster_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    pairs = [
        (cluster_ids[i], cluster_ids[j])
        for i in range(len(cluster_ids))
        for j in range(i + 1, len(cluster_ids))
    ]
    return _adjacency_from_cluster_pairs(cluster_ids, pairs)


def _build_continuous() -> tuple[pd.DataFrame, list[str], pd.Series, dict]:
    rng = np.random.default_rng(0)
    n = 200
    data = pd.DataFrame(
        {
            "f_normal": rng.standard_normal(n),
            "f_shifted": rng.standard_normal(n) + 0.5,
            "f_scaled": rng.standard_normal(n) * 2.0,
            "f_skewed": rng.exponential(1.0, n),
            "f_uniform": rng.uniform(-1, 1, n),
        }
    )
    clusters = pd.Series(np.repeat([0, 1, 2], [70, 70, 60]))
    return data, list(data.columns), clusters, _full_adjacency([0, 1, 2])


def _build_tied_integer() -> tuple[pd.DataFrame, list[str], pd.Series, dict]:
    rng = np.random.default_rng(1)
    n = 200
    data = pd.DataFrame(
        {
            "i_binary": rng.integers(0, 2, n),
            "i_quartile": rng.integers(0, 4, n),
            "i_octile": rng.integers(0, 8, n),
            "i_capped": np.clip(rng.standard_normal(n) * 3, -2, 2).round().astype(int),
        }
    )
    clusters = pd.Series(np.repeat([0, 1, 2], [80, 60, 60]))
    return data, list(data.columns), clusters, _full_adjacency([0, 1, 2])


def _build_all_identical_column() -> tuple[pd.DataFrame, list[str], pd.Series, dict]:
    rng = np.random.default_rng(2)
    n = 150
    data = pd.DataFrame(
        {
            "varied": rng.standard_normal(n),
            "constant": np.full(n, 3.14),
            "binary_tied": np.concatenate([np.zeros(75), np.ones(75)]),
        }
    )
    clusters = pd.Series(np.repeat([0, 1, 2], 50))
    return data, list(data.columns), clusters, _full_adjacency([0, 1, 2])


def _build_constant_within_cluster() -> tuple[pd.DataFrame, list[str], pd.Series, dict]:
    rng = np.random.default_rng(3)
    n = 120
    clusters = pd.Series(np.repeat([0, 1, 2], 40))
    # `bucketed`: identical within each cluster, distinct across clusters.
    bucketed = np.where(clusters == 0, 1.0, np.where(clusters == 1, 5.0, 10.0))
    data = pd.DataFrame(
        {
            "bucketed": bucketed,
            "noise": rng.standard_normal(n),
        }
    )
    return data, list(data.columns), clusters, _full_adjacency([0, 1, 2])


def _build_singleton_clusters() -> tuple[pd.DataFrame, list[str], pd.Series, dict]:
    rng = np.random.default_rng(4)
    # 100 rows; cluster 0 has 95, clusters 1 and 2 each have a single point.
    cluster_labels = np.concatenate(
        [np.zeros(95, dtype=int), [1], np.full(4, 0, dtype=int)]
    )
    cluster_labels[96] = 2
    clusters = pd.Series(cluster_labels)
    data = pd.DataFrame(
        {
            "f1": rng.standard_normal(100),
            "f2": rng.standard_normal(100) * 2,
        }
    )
    return data, list(data.columns), clusters, _full_adjacency([0, 1, 2])


def _build_nan_rich() -> tuple[pd.DataFrame, list[str], pd.Series, dict]:
    rng = np.random.default_rng(5)
    n = 180
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    f3 = rng.standard_normal(n)
    # 30% NaN in f1, 60% NaN in f2 (heavy), 0% in f3.
    f1_mask = rng.random(n) < 0.30
    f2_mask = rng.random(n) < 0.60
    f1[f1_mask] = np.nan
    f2[f2_mask] = np.nan
    data = pd.DataFrame({"f1_30pct_nan": f1, "f2_60pct_nan": f2, "f3_clean": f3})
    clusters = pd.Series(np.repeat([0, 1, 2], 60))
    return data, list(data.columns), clusters, _full_adjacency([0, 1, 2])


def _build_imbalanced() -> tuple[pd.DataFrame, list[str], pd.Series, dict]:
    rng = np.random.default_rng(6)
    n = 200
    # Cluster 0: 2 rows (1% minority). Cluster 1: 198 rows.
    clusters = pd.Series(np.concatenate([np.zeros(2, dtype=int), np.ones(198, dtype=int)]))
    data = pd.DataFrame(
        {
            "f1": rng.standard_normal(n),
            "f2": rng.exponential(1.0, n),
        }
    )
    return data, list(data.columns), clusters, _full_adjacency([0, 1])


def _build_no_neighbor() -> tuple[pd.DataFrame, list[str], pd.Series, dict]:
    rng = np.random.default_rng(7)
    n = 100
    clusters = pd.Series(np.repeat([0, 1], 50))
    data = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
    # Empty adjacency — every cluster reports neighbor_id=None.
    adjacency = {0: [], 1: []}
    return data, list(data.columns), clusters, adjacency


def _build_kruskal_raises_case() -> tuple[pd.DataFrame, list[str], pd.Series, dict]:
    """All clusters share identical values on one column.

    scipy.stats.kruskal raises ValueError("All numbers are identical"). The
    contract is to surface that as a `kruskal: ...` entry in failure_reasons.
    """
    n = 60
    clusters = pd.Series(np.repeat([0, 1, 2], 20))
    rng = np.random.default_rng(8)
    data = pd.DataFrame(
        {
            "kruskal_degenerate": np.full(n, 7.0),
            "ok": rng.standard_normal(n),
        }
    )
    return data, list(data.columns), clusters, _full_adjacency([0, 1, 2])


CASES: list[Case] = [
    Case("continuous", _build_continuous),
    Case("tied_integer", _build_tied_integer),
    Case("all_identical_column", _build_all_identical_column),
    Case("constant_within_cluster", _build_constant_within_cluster),
    Case("singleton_clusters", _build_singleton_clusters),
    Case("nan_rich", _build_nan_rich),
    Case("imbalanced_1pct_99pct", _build_imbalanced),
    Case("no_neighbor", _build_no_neighbor),
    Case("kruskal_raises", _build_kruskal_raises_case),
]


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def _python_safe(value: Any) -> Any:
    """Convert numpy / pandas scalars to plain Python for JSON."""
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f):
            return {"__nan__": True}
        if math.isinf(f):
            return {"__inf__": f > 0}
        return f
    if isinstance(value, (np.integer, int, bool, np.bool_)):
        return int(value) if not isinstance(value, bool) else bool(value)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_python_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _python_safe(v) for k, v in value.items()}
    return value


def _python_restore(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__nan__"):
            return float("nan")
        if "__inf__" in value:
            return float("inf") if value["__inf__"] else float("-inf")
        return {k: _python_restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_python_restore(v) for v in value]
    return value


def _snapshot(rows: list[dict], global_stats: dict[str, dict]) -> dict[str, Any]:
    return {
        "rows": [_python_safe(row) for row in rows],
        "global_stats": _python_safe(global_stats),
    }


def _load_baseline(case_name: str) -> dict[str, Any]:
    path = FIXTURE_DIR / f"{case_name}.json"
    return _python_restore(json.loads(path.read_text()))


def _save_baseline(case_name: str, snapshot: dict[str, Any]) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / f"{case_name}.json"
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _format_row_key(row: dict) -> tuple[Any, Any]:
    return (row["cluster_id"], row["column"])


def _assert_rows_equal(actual: list[dict], expected: list[dict]) -> None:
    actual_by_key = {_format_row_key(r): r for r in actual}
    expected_by_key = {_format_row_key(r): r for r in expected}

    missing = expected_by_key.keys() - actual_by_key.keys()
    extra = actual_by_key.keys() - expected_by_key.keys()
    assert not missing, f"Missing rows in actual: {sorted(missing)}"
    assert not extra, f"Extra rows in actual: {sorted(extra)}"

    for key in expected_by_key:
        exp = expected_by_key[key]
        act = actual_by_key[key]
        # Failure strings: set equality (order-independent).
        exp_failures = set(exp.get("failure_reasons", []) or [])
        act_failures = set(act.get("failure_reasons", []) or [])
        assert exp_failures == act_failures, (
            f"{key}: failure_reasons mismatch.\n"
            f"  expected: {sorted(exp_failures)}\n"
            f"  actual:   {sorted(act_failures)}"
        )

        for field, value in exp.items():
            if field == "failure_reasons":
                continue
            if field in _EXACT_FIELDS:
                assert act[field] == value, (
                    f"{key} field {field!r}: expected {value!r}, got {act[field]!r}"
                )
                continue
            tol = _FIELD_TOLERANCES.get(field)
            if tol is None:
                # Unknown numeric field — require exact match to flag schema drift.
                assert act[field] == value, (
                    f"{key} field {field!r}: no tolerance declared; "
                    f"expected exact match. expected={value!r} actual={act[field]!r}"
                )
                continue
            exp_f = float(value) if not isinstance(value, dict) else float("nan")
            act_f = float(act[field]) if not isinstance(act[field], dict) else float("nan")
            if math.isnan(exp_f) and math.isnan(act_f):
                continue
            assert abs(exp_f - act_f) <= tol, (
                f"{key} field {field!r} drift exceeds tol={tol}.\n"
                f"  expected: {exp_f}\n"
                f"  actual:   {act_f}\n"
                f"  delta:    {abs(exp_f - act_f)}"
            )


def _assert_global_stats_equal(actual: dict, expected: dict) -> None:
    assert set(actual.keys()) == set(expected.keys()), (
        f"global_stats column keys mismatch.\n"
        f"  expected: {sorted(expected.keys())}\n"
        f"  actual:   {sorted(actual.keys())}"
    )
    for column, exp_entry in expected.items():
        act_entry = actual[column]
        exp_fail = set(exp_entry.get("failure_reasons", []) or [])
        act_fail = set(act_entry.get("failure_reasons", []) or [])
        assert exp_fail == act_fail, (
            f"global_stats[{column!r}] failure_reasons mismatch.\n"
            f"  expected: {sorted(exp_fail)}\n"
            f"  actual:   {sorted(act_fail)}"
        )
        for field, value in exp_entry.items():
            if field == "failure_reasons":
                continue
            tol = 1e-10
            exp_f = float(value)
            act_f = float(act_entry[field])
            if math.isnan(exp_f) and math.isnan(act_f):
                continue
            assert abs(exp_f - act_f) <= tol, (
                f"global_stats[{column!r}] field {field!r} drift exceeds tol={tol}.\n"
                f"  expected: {exp_f}\n"
                f"  actual:   {act_f}"
            )


# ---------------------------------------------------------------------------
# The parity test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_numeric_rows_parity(case: Case) -> None:
    data, numeric_cols, clusters, adjacency = case.build()
    rows, global_stats = _compute_numeric_rows(data, numeric_cols, clusters, adjacency)
    snapshot = _snapshot(rows, global_stats)

    if REGENERATE:
        _save_baseline(case.name, snapshot)
        pytest.skip(f"regenerated baseline for {case.name}")

    # Both sides go through the same safe→restore round-trip so NaN/Inf land
    # as real Python floats on both sides of the comparison.
    actual = _python_restore(snapshot)
    expected = _load_baseline(case.name)
    _assert_rows_equal(actual["rows"], expected["rows"])
    _assert_global_stats_equal(actual["global_stats"], expected["global_stats"])
