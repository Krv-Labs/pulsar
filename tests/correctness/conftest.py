"""
Shared fixtures for correctness tests.

These tests compare a readable Python reference implementation of each
algorithm to the compiled Rust output.  The Python side is intentionally
written in a "close to the math" style so you can audit it line by line.
"""
import numpy as np
import pytest


@pytest.fixture
def rng_data():
    """50×4 float64 array drawn from a standard normal distribution."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((50, 4)).astype(np.float64)


@pytest.fixture
def small_data():
    """10×3 float64 array — small enough to inspect by eye."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 3)).astype(np.float64)


@pytest.fixture
def col_with_nans():
    """1-D column with two NaN values at known positions (indices 2 and 4)."""
    return np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])


@pytest.fixture
def categorical_col():
    """Float-encoded categorical column: categories {0.0, 1.0}, two NaNs."""
    return np.array([0.0, 1.0, np.nan, 0.0, np.nan, 1.0])
