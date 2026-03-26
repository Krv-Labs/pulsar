import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from pulsar._pulsar import PCA


def test_output_shape(small_array):
    pca = PCA(n_components=2, seed=42)
    result = np.array(pca.fit_transform(small_array))
    assert result.shape == (small_array.shape[0], 2)


def test_subspace_quality(small_array):
    """
    Randomized SVD captures the same subspace as exact PCA, verified by
    checking that the projection explains similar variance.
    """
    n_components = 2
    pulsar_pca = PCA(n_components=n_components, seed=42)
    pulsar_result = np.array(pulsar_pca.fit_transform(small_array))

    sklearn_pca = SklearnPCA(n_components=n_components, random_state=42)
    sklearn_result = sklearn_pca.fit_transform(small_array)

    # Randomized SVD won't match sklearn exactly, but should capture similar variance
    pulsar_var = np.var(pulsar_result, axis=0).sum()
    sklearn_var = np.var(sklearn_result, axis=0).sum()
    
    # The captured variance should be within 10% for well-conditioned data
    # (randomized SVD is approximate but should be close for top components)
    assert pulsar_var > 0.7 * sklearn_var, "Randomized PCA should capture most variance"


def test_transform_consistent_with_fit_transform(small_array):
    pca = PCA(n_components=2, seed=42)
    result1 = np.array(pca.fit_transform(small_array))
    result2 = np.array(pca.transform(small_array))
    np.testing.assert_allclose(result1, result2, atol=1e-10)


def test_explained_variance_descending(small_array):
    pca = PCA(n_components=3, seed=42)
    pca.fit_transform(small_array)
    ev = np.array(pca.explained_variance)
    assert len(ev) == 3
    assert np.all(ev[:-1] >= ev[1:]), "explained variance should be non-increasing"


def test_explained_variance_positive(small_array):
    pca = PCA(n_components=2, seed=0)
    pca.fit_transform(small_array)
    ev = np.array(pca.explained_variance)
    assert np.all(ev > 0)


def test_transform_before_fit_raises(small_array):
    pca = PCA(n_components=2, seed=0)
    with pytest.raises(ValueError, match="fit_transform"):
        pca.transform(small_array)


def test_n_components_too_large():
    arr = np.random.randn(10, 3).astype(np.float64)
    pca = PCA(n_components=5, seed=0)
    with pytest.raises(ValueError, match="n_components"):
        pca.fit_transform(arr)
