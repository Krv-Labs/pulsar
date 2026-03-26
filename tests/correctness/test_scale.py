"""
Correctness tests for scale.rs — StandardScaler.

The Python reference implements standard scaling explicitly using numpy
arithmetic (not sklearn).  This makes the algorithm fully visible:
  1. Compute column mean and population std (ddof=0).
  2. Transform: (x - mean) / std
  3. Inverse: x_scaled * std + mean

We verify that the Rust output matches the Python reference to machine
precision (atol=1e-12).
"""
import numpy as np
import pytest

from pulsar._pulsar import StandardScaler


# ---------------------------------------------------------------------------
# Python reference implementation
# ---------------------------------------------------------------------------

def py_fit_transform(X):
    """Standard-scale X column-wise.

    Uses population std (ddof=0) to match Rust and sklearn's default.
    Constant columns (std < 1e-10) are left as-is (fill std with 1.0).

    Returns (X_scaled, means, stds).
    """
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    stds = np.where(stds < 1e-10, 1.0, stds)
    X_scaled = (X - means) / stds
    return X_scaled, means, stds


def py_transform(X, means, stds):
    """Apply stored statistics to new data."""
    return (X - means) / stds


def py_inverse_transform(X_scaled, means, stds):
    """Undo standard scaling."""
    return X_scaled * stds + means


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fit_transform_matches_reference(rng_data):
    expected, _, _ = py_fit_transform(rng_data)
    scaler = StandardScaler()
    actual = np.array(scaler.fit_transform(rng_data))
    np.testing.assert_allclose(actual, expected, atol=1e-12,
                               err_msg="fit_transform: Rust differs from Python reference")


def test_transform_matches_reference(rng_data, small_data):
    _, means, stds = py_fit_transform(rng_data)
    expected = py_transform(small_data, means[:small_data.shape[1]],
                            stds[:small_data.shape[1]])

    # Use a dataset with the same number of columns (small_data has 3, rng_data 4)
    # Re-fit on a 50×3 slice instead
    data3 = rng_data[:, :3]
    _, means3, stds3 = py_fit_transform(data3)
    expected3 = py_transform(small_data, means3, stds3)

    scaler = StandardScaler()
    scaler.fit_transform(data3)
    actual = np.array(scaler.transform(small_data))

    np.testing.assert_allclose(actual, expected3, atol=1e-12,
                               err_msg="transform: Rust differs from Python reference")


def test_inverse_transform_matches_reference(rng_data):
    X_expected, means, stds = py_fit_transform(rng_data)

    scaler = StandardScaler()
    X_scaled = np.array(scaler.fit_transform(rng_data))

    expected_recovered = py_inverse_transform(X_scaled, means, stds)
    actual_recovered = np.array(scaler.inverse_transform(X_scaled))

    np.testing.assert_allclose(actual_recovered, expected_recovered, atol=1e-12,
                               err_msg="inverse_transform: Rust differs from Python reference")


def test_inverse_is_left_inverse_of_transform(rng_data):
    """Applying inverse_transform after fit_transform recovers the original data."""
    scaler = StandardScaler()
    X_scaled = np.array(scaler.fit_transform(rng_data))
    X_recovered = np.array(scaler.inverse_transform(X_scaled))
    np.testing.assert_allclose(X_recovered, rng_data, atol=1e-10,
                               err_msg="inverse_transform did not recover original data")


def test_scaled_mean_is_zero(rng_data):
    scaler = StandardScaler()
    X_scaled = np.array(scaler.fit_transform(rng_data))
    np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10,
                               err_msg="Column means of scaled data should be ~0")


def test_scaled_std_is_one(rng_data):
    scaler = StandardScaler()
    X_scaled = np.array(scaler.fit_transform(rng_data))
    np.testing.assert_allclose(X_scaled.std(axis=0, ddof=0), 1.0, atol=1e-10,
                               err_msg="Column stds of scaled data should be ~1")


def test_constant_column_does_not_raise():
    """A column with zero variance should produce all-zero output (std clamped to 1)."""
    X = np.ones((5, 2), dtype=np.float64)
    scaler = StandardScaler()
    result = np.array(scaler.fit_transform(X))
    np.testing.assert_array_equal(result, np.zeros_like(X))
