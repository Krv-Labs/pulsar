import numpy as np
import pytest
from pulsar._pulsar import impute_column


def test_no_nans_passthrough():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    result = impute_column(arr, "fill_mean")
    np.testing.assert_array_equal(result, arr)


def test_fill_mean_correct():
    arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
    result = impute_column(arr, "fill_mean")
    expected_mean = np.mean([1.0, 2.0, 4.0])
    assert not np.any(np.isnan(result))
    np.testing.assert_allclose(result[2], expected_mean)
    np.testing.assert_allclose(result[4], expected_mean)


def test_fill_median_odd():
    arr = np.array([1.0, 3.0, np.nan, 5.0])
    result = impute_column(arr, "fill_median")
    assert not np.any(np.isnan(result))
    np.testing.assert_allclose(result[2], 3.0)


def test_fill_median_even():
    arr = np.array([1.0, 3.0, np.nan, 5.0, 7.0])
    result = impute_column(arr, "fill_median")
    # median of [1, 3, 5, 7] = (3 + 5) / 2 = 4.0
    assert not np.any(np.isnan(result))
    np.testing.assert_allclose(result[2], 4.0)


def test_fill_mode():
    arr = np.array([1.0, 2.0, 2.0, np.nan, 3.0])
    result = impute_column(arr, "fill_mode")
    assert not np.any(np.isnan(result))
    np.testing.assert_allclose(result[3], 2.0)


def test_sample_normal_no_nans_after():
    rng = np.random.default_rng(42)
    arr = rng.standard_normal(20).astype(np.float64)
    arr[[3, 7, 15]] = np.nan
    result = impute_column(arr, "sample_normal", seed=42)
    assert not np.any(np.isnan(result))
    assert result.shape == arr.shape


def test_sample_normal_deterministic():
    arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
    r1 = impute_column(arr, "sample_normal", seed=99)
    r2 = impute_column(arr, "sample_normal", seed=99)
    np.testing.assert_array_equal(r1, r2)


def test_sample_normal_different_seeds():
    arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
    r1 = impute_column(arr, "sample_normal", seed=1)
    r2 = impute_column(arr, "sample_normal", seed=2)
    # NaN positions should differ (extremely unlikely to be equal)
    assert not np.allclose(r1[2], r2[2]) or not np.allclose(r1[4], r2[4])


def test_sample_categorical_deterministic():
    arr = np.array([0.0, 1.0, np.nan, 0.0, np.nan, 1.0])
    r1 = impute_column(arr, "sample_categorical", seed=7)
    r2 = impute_column(arr, "sample_categorical", seed=7)
    np.testing.assert_array_equal(r1, r2)


def test_sample_categorical_values_from_observed():
    arr = np.array([0.0, 1.0, np.nan, 0.0, np.nan, 1.0])
    result = impute_column(arr, "sample_categorical", seed=7)
    assert not np.any(np.isnan(result))
    # Imputed values must be from {0.0, 1.0}
    for v in result:
        assert v in {0.0, 1.0}


def test_unknown_method_raises():
    arr = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="unknown imputation method.*fill_value"):
        impute_column(arr, "bogus_method")


def test_fill_value_replaces_nans():
    arr = np.array([1.0, np.nan, 3.0, np.nan])
    result = impute_column(arr, "fill_value", value=0.0)
    np.testing.assert_array_equal(result, np.array([1.0, 0.0, 3.0, 0.0]))
    # Input array is not mutated
    assert np.isnan(arr[1]) and np.isnan(arr[3])


def test_fill_value_all_nan():
    arr = np.array([np.nan, np.nan, np.nan])
    result = impute_column(arr, "fill_value", value=-999.0)
    np.testing.assert_array_equal(result, np.array([-999.0, -999.0, -999.0]))


def test_fill_value_no_nans_passthrough():
    arr = np.array([1.0, 2.0, 3.0])
    result = impute_column(arr, "fill_value", value=0.0)
    np.testing.assert_array_equal(result, arr)


def test_fill_value_missing_value_raises():
    arr = np.array([1.0, np.nan])
    with pytest.raises(ValueError, match="fill_value requires a 'value'"):
        impute_column(arr, "fill_value")


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_fill_value_non_finite_raises(bad_val):
    arr = np.array([1.0, np.nan])
    with pytest.raises(ValueError, match="fill_value requires a finite value"):
        impute_column(arr, "fill_value", value=bad_val)


def test_value_with_other_method_raises():
    arr = np.array([1.0, np.nan, 3.0])
    with pytest.raises(
        ValueError, match="'value' is only valid with method 'fill_value'"
    ):
        impute_column(arr, "fill_mean", value=0.0)
