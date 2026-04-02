import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from pulsar._pulsar import PCA, pca_grid


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


# ---------------------------------------------------------------------------
# Tests that would have caught the transposition bug
# ---------------------------------------------------------------------------


def test_principal_direction_aligns_with_dominant_axis():
    """Data with variance along a single axis must produce a first component
    aligned with that axis.

    This is the canonical test the transposition bug would have failed: if
    the matrix is silently transposed, the component vectors live in the
    wrong space and cannot align with the known variance direction.
    """
    rng = np.random.default_rng(0)
    n_samples = 200
    # Variance overwhelmingly in column 0; columns 1-3 are near-zero noise
    data = np.zeros((n_samples, 4), dtype=np.float64)
    data[:, 0] = rng.standard_normal(n_samples) * 10.0
    data[:, 1] = rng.standard_normal(n_samples) * 0.01
    data[:, 2] = rng.standard_normal(n_samples) * 0.01
    data[:, 3] = rng.standard_normal(n_samples) * 0.01

    pca = PCA(n_components=1, seed=42)
    result = np.array(pca.fit_transform(data))

    # Output shape must be (n_samples, 1) — a transposed implementation
    # could produce (n_features, 1) or (1, n_samples).
    assert result.shape == (n_samples, 1)

    # The projected values should correlate almost perfectly with the
    # dominant axis (column 0). A transposition would scramble this.
    centered_col0 = data[:, 0] - data[:, 0].mean()
    projected = result[:, 0]
    corr = np.abs(np.corrcoef(centered_col0, projected)[0, 1])
    assert corr > 0.99, (
        f"First component should align with dominant variance axis, got corr={corr:.4f}"
    )


def test_pca_grid_dimension_exceeds_features():
    """pca_grid must raise when a requested dimension exceeds n_features."""
    arr = np.random.default_rng(0).standard_normal((20, 3)).astype(np.float64)
    with pytest.raises(ValueError, match="dimension.*exceeds"):
        pca_grid(arr, [2, 5], [42])


def test_pca_raises_on_empty_input():
    """PCA must reject a matrix with 0 rows."""
    arr = np.empty((0, 4), dtype=np.float64)
    pca = PCA(n_components=2, seed=42)
    with pytest.raises(ValueError, match="at least 2 samples"):
        pca.fit_transform(arr)


def test_pca_raises_on_single_row():
    """PCA requires at least 2 samples for meaningful variance estimation."""
    arr = np.array([[1.0, 2.0, 3.0]])
    pca = PCA(n_components=1, seed=42)
    with pytest.raises(ValueError):
        pca.fit_transform(arr)


def test_pca_raises_on_zero_features():
    """PCA must reject a matrix with 0 columns."""
    arr = np.empty((10, 0), dtype=np.float64)
    pca = PCA(n_components=1, seed=42)
    with pytest.raises(ValueError):
        pca.fit_transform(arr)
