import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler as SklearnScaler

from pulsar._pulsar import StandardScaler


def test_fit_transform_shape(small_array):
    scaler = StandardScaler()
    result = np.array(scaler.fit_transform(small_array))
    assert result.shape == small_array.shape


def test_fit_transform_mean_zero(small_array):
    scaler = StandardScaler()
    result = np.array(scaler.fit_transform(small_array))
    np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-10)


def test_fit_transform_std_one(small_array):
    scaler = StandardScaler()
    result = np.array(scaler.fit_transform(small_array))
    np.testing.assert_allclose(result.std(axis=0), 1.0, atol=1e-10)


def test_inverse_transform_roundtrip(small_array):
    scaler = StandardScaler()
    scaled = np.array(scaler.fit_transform(small_array))
    recovered = np.array(scaler.inverse_transform(scaled))
    np.testing.assert_allclose(recovered, small_array, atol=1e-10)


def test_transform_after_fit(small_array):
    scaler = StandardScaler()
    scaled1 = np.array(scaler.fit_transform(small_array))
    scaled2 = np.array(scaler.transform(small_array))
    np.testing.assert_allclose(scaled1, scaled2, atol=1e-10)


def test_matches_sklearn(small_array):
    pulsar_scaler = StandardScaler()
    pulsar_result = np.array(pulsar_scaler.fit_transform(small_array))

    sklearn_scaler = SklearnScaler()
    sklearn_result = sklearn_scaler.fit_transform(small_array)

    np.testing.assert_allclose(pulsar_result, sklearn_result, atol=1e-10)


def test_transform_before_fit_raises():
    scaler = StandardScaler()
    arr = np.ones((3, 2))
    with pytest.raises(ValueError, match="fit_transform"):
        scaler.transform(arr)


def test_inverse_transform_before_fit_raises():
    scaler = StandardScaler()
    arr = np.ones((3, 2))
    with pytest.raises(ValueError, match="fit_transform"):
        scaler.inverse_transform(arr)
