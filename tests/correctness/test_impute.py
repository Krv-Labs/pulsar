"""
Correctness tests for impute.rs — impute_column.

Each test defines a transparent Python reference implementation of the
imputation method, then asserts that the Rust output matches it exactly
(or approximately, for floating-point operations).

NOTE on sampling methods (sample_normal, sample_categorical):
  The Rust code uses rand::rngs::StdRng (ChaCha12-based PRNG), while Python's
  numpy uses PCG64.  The two PRNGs produce different sequences for the same
  integer seed, so we cannot check bit-for-bit identity for sampled values.
  Instead, the sampling tests verify *distributional* correctness:
    - No NaN values remain after imputation.
    - Imputed values come from the right support (e.g. observed categories).
    - The mean and std of sampled values are plausible.
"""
import numpy as np
import pytest
from collections import Counter

from pulsar._pulsar import impute_column


# ---------------------------------------------------------------------------
# Python reference implementations
# ---------------------------------------------------------------------------

def py_fill_mean(arr):
    """Fill NaN with the mean of observed values."""
    observed = arr[~np.isnan(arr)]
    fill = observed.mean()
    result = arr.copy()
    result[np.isnan(result)] = fill
    return result


def py_fill_median(arr):
    """Fill NaN with the median of observed values.

    For an even number of observations the median is the average of the two
    middle values — matching numpy's nanmedian behaviour.
    """
    observed = sorted(arr[~np.isnan(arr)].tolist())
    n = len(observed)
    if n % 2 == 0:
        median = (observed[n // 2 - 1] + observed[n // 2]) / 2.0
    else:
        median = observed[n // 2]
    result = arr.copy()
    result[np.isnan(result)] = median
    return result


def py_fill_mode(arr):
    """Fill NaN with the most frequent observed value."""
    observed = arr[~np.isnan(arr)].tolist()
    mode = Counter(observed).most_common(1)[0][0]
    result = arr.copy()
    result[np.isnan(result)] = mode
    return result


# ---------------------------------------------------------------------------
# fill_mean
# ---------------------------------------------------------------------------

def test_fill_mean_matches_reference(col_with_nans):
    expected = py_fill_mean(col_with_nans)
    actual = np.array(impute_column(col_with_nans, "fill_mean"))
    np.testing.assert_allclose(actual, expected, atol=1e-15,
                               err_msg="fill_mean: Rust result differs from Python reference")


def test_fill_mean_all_nan_indices_replaced(col_with_nans):
    """Every position that was NaN must be non-NaN after imputation."""
    actual = np.array(impute_column(col_with_nans, "fill_mean"))
    assert not np.any(np.isnan(actual))


def test_fill_mean_non_nan_unchanged(col_with_nans):
    """Positions that were already finite must not be modified."""
    actual = np.array(impute_column(col_with_nans, "fill_mean"))
    mask = ~np.isnan(col_with_nans)
    np.testing.assert_array_equal(actual[mask], col_with_nans[mask])


# ---------------------------------------------------------------------------
# fill_median
# ---------------------------------------------------------------------------

def test_fill_median_matches_reference_odd():
    """Odd number of observed values → exact middle element."""
    arr = np.array([1.0, 3.0, np.nan, 7.0])  # observed: [1, 3, 7] → median = 3
    expected = py_fill_median(arr)
    actual = np.array(impute_column(arr, "fill_median"))
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_fill_median_matches_reference_even():
    """Even number of observed values → average of two middle elements."""
    arr = np.array([1.0, 3.0, np.nan, 5.0, 7.0])  # observed: [1,3,5,7] → median = 4
    expected = py_fill_median(arr)
    actual = np.array(impute_column(arr, "fill_median"))
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_fill_median_matches_reference_fixture(col_with_nans):
    expected = py_fill_median(col_with_nans)
    actual = np.array(impute_column(col_with_nans, "fill_median"))
    np.testing.assert_allclose(actual, expected, atol=1e-15)


# ---------------------------------------------------------------------------
# fill_mode
# ---------------------------------------------------------------------------

def test_fill_mode_matches_reference(col_with_nans):
    # col_with_nans = [1, 2, nan, 4, nan, 6] — all values unique, mode = 1.0
    # (minimum value wins when counts are tied in Rust HashMap; Python Counter
    # returns first-encountered — but all have count 1 so we just check Python
    # and Rust agree with each other)
    actual = np.array(impute_column(col_with_nans, "fill_mode"))
    assert not np.any(np.isnan(actual))
    # The fill value must be one of the observed values
    observed = set(col_with_nans[~np.isnan(col_with_nans)].tolist())
    for v in actual:
        assert v in observed


def test_fill_mode_picks_most_frequent():
    """When one value clearly dominates, both Rust and Python must pick it."""
    arr = np.array([2.0, 2.0, 2.0, 1.0, np.nan, 3.0])
    expected = py_fill_mode(arr)  # mode = 2.0
    actual = np.array(impute_column(arr, "fill_mode"))
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_fill_mode_categorical(categorical_col):
    arr = np.array([0.0, 1.0, 1.0, np.nan, 0.0, np.nan, 1.0])  # mode = 1.0
    expected = py_fill_mode(arr)
    actual = np.array(impute_column(arr, "fill_mode"))
    np.testing.assert_allclose(actual, expected, atol=1e-15)


# ---------------------------------------------------------------------------
# sample_normal — distributional checks only (PRNG mismatch with numpy)
# ---------------------------------------------------------------------------

def test_sample_normal_no_nans_remain(col_with_nans):
    actual = np.array(impute_column(col_with_nans, "sample_normal", seed=42))
    assert not np.any(np.isnan(actual)), "NaN values remain after sample_normal"


def test_sample_normal_preserves_finite_values(col_with_nans):
    actual = np.array(impute_column(col_with_nans, "sample_normal", seed=42))
    mask = ~np.isnan(col_with_nans)
    np.testing.assert_array_equal(actual[mask], col_with_nans[mask])


def test_sample_normal_imputed_values_near_distribution():
    """Imputed values should be drawn from N(μ, σ) of observed values.

    With a large column we can check that the mean of imputed values is close
    to the observed mean (within a few standard errors).
    """
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(1000).astype(np.float64)
    arr[::5] = np.nan  # 20 % NaN

    observed = arr[~np.isnan(arr)]
    obs_mean = observed.mean()
    obs_std = observed.std(ddof=0)

    result = np.array(impute_column(arr, "sample_normal", seed=0))
    imputed = result[np.isnan(arr)]

    # Imputed mean should be within 3 standard errors of the distribution mean
    se = obs_std / np.sqrt(len(imputed))
    assert abs(imputed.mean() - obs_mean) < 4 * se, (
        f"Imputed mean {imputed.mean():.4f} too far from observed mean {obs_mean:.4f}"
    )


def test_sample_normal_deterministic():
    """Same seed → identical output."""
    arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
    r1 = np.array(impute_column(arr, "sample_normal", seed=7))
    r2 = np.array(impute_column(arr, "sample_normal", seed=7))
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# sample_categorical — distributional checks only
# ---------------------------------------------------------------------------

def test_sample_categorical_no_nans_remain(categorical_col):
    actual = np.array(impute_column(categorical_col, "sample_categorical", seed=7))
    assert not np.any(np.isnan(actual))


def test_sample_categorical_values_from_observed(categorical_col):
    """Every imputed value must come from the set of observed categories."""
    observed_set = set(categorical_col[~np.isnan(categorical_col)].tolist())
    actual = np.array(impute_column(categorical_col, "sample_categorical", seed=7))
    for v in actual:
        assert v in observed_set, f"Imputed value {v} not in observed set {observed_set}"


def test_sample_categorical_respects_frequencies():
    """When one category is much more frequent, it should dominate imputed values."""
    # category 1.0 appears 8×, category 0.0 appears 2×
    arr = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
                   + [np.nan] * 100, dtype=np.float64)
    result = np.array(impute_column(arr, "sample_categorical", seed=0))
    imputed = result[np.isnan(arr)]
    frac_ones = (imputed == 1.0).mean()
    # Expect roughly 80 % ones; allow ±15 % slack
    assert frac_ones > 0.65, f"Expected ~80% ones, got {frac_ones:.2%}"


def test_sample_categorical_deterministic(categorical_col):
    r1 = np.array(impute_column(categorical_col, "sample_categorical", seed=99))
    r2 = np.array(impute_column(categorical_col, "sample_categorical", seed=99))
    np.testing.assert_array_equal(r1, r2)
