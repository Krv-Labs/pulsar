"""
Unit tests for ``_NumericPrecompute`` and ``_build_numeric_precompute``.

These are isolation tests — production code does not yet use the precompute.
They verify that the precomputed fields agree with the existing scalar helpers
on the same data, and that NaN / tie / degenerate inputs are handled.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from pulsar.mcp.interpreter import (
    _build_numeric_precompute,
    _column_tie_correction,
    _normalize_numeric,
    _safe_mad,
    _safe_mean,
    _safe_median,
    _safe_std,
    _safe_var,
)


def test_precompute_shapes_for_basic_data():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.standard_normal(50), "b": rng.standard_normal(50)})
    pre = _build_numeric_precompute(df, ["a", "b"])
    assert pre.X.shape == (50, 2)
    assert pre.valid.shape == (50, 2)
    assert pre.ranks.shape == (50, 2)
    assert pre.sort_idx.shape == (50, 2)
    assert pre.n_total_j.shape == (2,)
    assert pre.col_mean.shape == (2,)
    assert pre.col_std_pop.shape == (2,)
    assert pre.col_std_sample.shape == (2,)
    assert pre.col_median.shape == (2,)
    assert pre.col_mad.shape == (2,)
    assert pre.col_iqr.shape == (2,)
    assert pre.tie_correction.shape == (2,)


def test_precompute_matches_safe_helpers_on_clean_data():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "x": rng.standard_normal(100),
            "y": rng.exponential(1.0, 100),
        }
    )
    pre = _build_numeric_precompute(df, ["x", "y"])
    for j, col in enumerate(["x", "y"]):
        vals = _normalize_numeric(df[col])
        assert pre.n_total_j[j] == vals.size
        assert math.isclose(float(pre.col_mean[j]), _safe_mean(vals), abs_tol=1e-12)
        assert math.isclose(
            float(pre.col_std_sample[j]), _safe_std(vals), abs_tol=1e-12
        )
        assert math.isclose(
            float(pre.col_std_pop[j]),
            float(np.std(vals, ddof=0)),
            abs_tol=1e-12,
        )
        assert math.isclose(
            float(pre.col_var_sample[j]), _safe_var(vals), abs_tol=1e-12
        )
        assert math.isclose(float(pre.col_median[j]), _safe_median(vals), abs_tol=1e-12)
        assert math.isclose(float(pre.col_mad[j]), _safe_mad(vals), abs_tol=1e-12)
        assert math.isclose(
            float(pre.col_iqr[j]),
            float(stats.iqr(vals)),
            abs_tol=1e-12,
        )


def test_precompute_per_column_nan_counts():
    rng = np.random.default_rng(2)
    n = 200
    col_a = rng.standard_normal(n)
    col_b = rng.standard_normal(n)
    col_a[rng.choice(n, 50, replace=False)] = np.nan
    col_b[rng.choice(n, 120, replace=False)] = np.nan
    df = pd.DataFrame({"a": col_a, "b": col_b})

    pre = _build_numeric_precompute(df, ["a", "b"])
    assert pre.n_total_j[0] == int((~np.isnan(col_a)).sum())
    assert pre.n_total_j[1] == int((~np.isnan(col_b)).sum())

    # NaN rows stay marked invalid.
    assert (~pre.valid[:, 0]).sum() == 50
    assert (~pre.valid[:, 1]).sum() == 120

    # nanmean ignores NaN positions.
    assert math.isclose(
        float(pre.col_mean[0]),
        float(np.nanmean(col_a)),
        abs_tol=1e-12,
    )


def test_precompute_ranks_match_scipy_rankdata_omit():
    rng = np.random.default_rng(3)
    n = 80
    col = rng.standard_normal(n)
    col[5] = np.nan
    col[42] = np.nan
    df = pd.DataFrame({"x": col})
    pre = _build_numeric_precompute(df, ["x"])
    expected = stats.rankdata(col, nan_policy="omit")
    np.testing.assert_allclose(pre.ranks[:, 0], expected, atol=1e-12, equal_nan=True)


def test_precompute_sort_idx_puts_nan_at_end():
    col = np.array([3.0, np.nan, 1.0, 2.0, np.nan, 0.5])
    df = pd.DataFrame({"x": col})
    pre = _build_numeric_precompute(df, ["x"])
    order = pre.sort_idx[:, 0]
    # First n_valid positions point at non-NaN entries; remainder point at NaN.
    n_valid = int(pre.n_total_j[0])
    assert n_valid == 4
    finite_vals = col[order[:n_valid]]
    assert np.all(np.isfinite(finite_vals))
    assert np.all(finite_vals[:-1] <= finite_vals[1:]), "sorted prefix not ascending"
    nan_tail = col[order[n_valid:]]
    assert np.all(np.isnan(nan_tail))


def test_tie_correction_matches_scipy_kruskal_on_tied_input():
    """The precompute's tie correction must equal what scipy.kruskal applies.

    Verification: construct two groups whose Kruskal H differs from the
    tie-corrected H by exactly the factor ``tie_correction``.
    """
    # Heavily tied integer data, two groups.
    group_a = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4], dtype=float)
    group_b = np.array([2, 2, 3, 3, 4, 4, 4, 5, 5, 5], dtype=float)
    pooled = np.concatenate([group_a, group_b])
    pooled_sorted = np.sort(pooled)
    correction = _column_tie_correction(pooled_sorted, pooled.size)

    # Recover the uncorrected H from scipy's corrected output.
    h_corrected, _p = stats.kruskal(group_a, group_b)
    # scipy returns H already divided by tie correction. Multiplying back must
    # match the closed-form uncorrected H.
    h_uncorrected_from_scipy = h_corrected * correction

    ranks = stats.rankdata(pooled)
    n = pooled.size
    ranks_a = ranks[: group_a.size]
    ranks_b = ranks[group_a.size :]
    rank_sum_a = ranks_a.sum()
    rank_sum_b = ranks_b.sum()
    expected_h = (12.0 / (n * (n + 1))) * (
        rank_sum_a**2 / group_a.size + rank_sum_b**2 / group_b.size
    ) - 3.0 * (n + 1)

    assert math.isclose(h_uncorrected_from_scipy, expected_h, rel_tol=1e-9)


def test_tie_correction_returns_zero_for_all_identical():
    # Every value identical → tie correction collapses to 0 (scipy divides by it
    # and emits NaN; we must signal "undefined" rather than 1.0).
    sorted_col = np.array([7.0, 7.0, 7.0, 7.0, 7.0])
    assert _column_tie_correction(sorted_col, 5) == 0.0


def test_tie_correction_is_one_when_no_ties():
    sorted_col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _column_tie_correction(sorted_col, 5) == 1.0


def test_precompute_empty_columns_returns_zero_width():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    pre = _build_numeric_precompute(df, [])
    assert pre.X.shape == (3, 0)
    assert pre.column_names == []
    assert pre.col_mean.size == 0


def test_precompute_handles_string_coerced_to_nan():
    # _normalize_numeric drops non-numeric via to_numeric(errors='coerce');
    # the precompute must produce the same effective valid mask.
    df = pd.DataFrame({"mixed": ["1.0", "bad", "3.0", None, "5"]})
    pre = _build_numeric_precompute(df, ["mixed"])
    assert pre.n_total_j[0] == 3
    np.testing.assert_array_equal(
        pre.valid[:, 0], np.array([True, False, True, False, True])
    )
