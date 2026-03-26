import numpy as np
import pytest


@pytest.fixture
def small_array():
    rng = np.random.default_rng(0)
    return rng.standard_normal((50, 4)).astype(np.float64)


@pytest.fixture
def array_with_nans():
    arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
    return arr


@pytest.fixture
def categorical_array():
    # Encoded as float: 0.0, 1.0, 2.0
    return np.array([0.0, 1.0, np.nan, 0.0, np.nan, 1.0])
