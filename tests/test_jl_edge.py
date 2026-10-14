import numpy as np
import pytest
from pulsar._pulsar import JLProjection, jl_grid


def test_jl_grid_zero_rows_raises():
    with pytest.raises(ValueError, match="0 rows"):
        jl_grid(np.zeros((0, 4)), [2], [0])


def test_jl_grid_zero_columns_raises():
    with pytest.raises(ValueError, match="0 features"):
        jl_grid(np.zeros((5, 0)), [2], [0])


def test_jl_fit_transform_zero_rows_raises():
    with pytest.raises(ValueError, match="0 rows"):
        JLProjection(2, 0).fit_transform(np.zeros((0, 4)))
