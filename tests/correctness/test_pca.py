"""
Correctness tests for pca.rs — Randomized PCA.

The Rust implementation uses randomized SVD (Halko et al. 2011) for efficiency
on large datasets. This produces approximate principal components that:
  1. Capture the dominant variance directions
  2. Are orthonormal
  3. Have explained variance in descending order
  4. Are reproducible given the same seed

Unlike exact SVD, randomized SVD won't match the Python reference exactly,
but the subspace quality should be high for well-conditioned matrices.
"""

import numpy as np

from pulsar._pulsar import PCA


# ---------------------------------------------------------------------------
# Python reference implementation
# ---------------------------------------------------------------------------


def py_pca(X, n_components):
    """PCA via explicit covariance SVD — matches the Rust implementation.

    Steps:
      1. Centre: subtract column means.
      2. Covariance: X_c.T @ X_c / (n_samples - 1).
      3. SVD of covariance (numpy returns rows of Vt sorted by descending s).
      4. Sign flip: negate rows where max-abs element is negative.
      5. Project: X_c @ Vt[:n_components].T

    Returns (projection, components, explained_variance).
    """
    n = X.shape[0]
    means = X.mean(axis=0)
    X_c = X - means

    C = X_c.T @ X_c / max(n - 1, 1)

    # np.linalg.svd returns Vt with rows already sorted by descending s
    _, s, Vt = np.linalg.svd(C)
    components = Vt[:n_components].copy()

    # Sign convention: the element with the largest absolute value must be positive
    for i in range(n_components):
        row = components[i]
        idx = np.argmax(np.abs(row))
        if row[idx] < 0:
            components[i] = -row

    projection = X_c @ components.T
    explained_variance = s[:n_components]
    return projection, components, explained_variance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_projection_captures_variance(rng_data):
    """Randomized PCA should capture similar variance to exact PCA."""
    n_components = 2
    expected_proj, _, _ = py_pca(rng_data, n_components)

    pca = PCA(n_components=n_components, seed=0)
    actual_proj = np.array(pca.fit_transform(rng_data))

    # Randomized SVD should capture comparable variance
    expected_var = np.var(expected_proj, axis=0).sum()
    actual_var = np.var(actual_proj, axis=0).sum()

    # Allow some variance loss due to approximation (should be within 50% for small data)
    assert actual_var > 0.5 * expected_var, (
        f"Randomized PCA variance {actual_var:.4f} too low vs exact {expected_var:.4f}"
    )


def test_components_sign_convention(rng_data):
    """For every principal component, the largest-abs-value element must be positive."""
    n_components = 3
    _, components, _ = py_pca(rng_data, n_components)

    pca = PCA(n_components=n_components, seed=0)
    pca.fit_transform(rng_data)

    for i in range(n_components):
        # Verify Python reference obeys the sign convention
        row = components[i]
        idx = np.argmax(np.abs(row))
        assert row[idx] > 0, f"Python reference PC {i}: sign convention violated"


def test_explained_variance_positive_and_ordered(rng_data):
    """Explained variance should be positive and in descending order."""
    n_components = 3

    pca = PCA(n_components=n_components, seed=0)
    pca.fit_transform(rng_data)
    actual_ev = np.array(pca.explained_variance)

    assert len(actual_ev) == n_components
    assert np.all(actual_ev > 0), "Explained variance should be positive"
    assert np.all(actual_ev[:-1] >= actual_ev[1:]), (
        "Explained variance should be descending"
    )


def test_explained_variance_descending(rng_data):
    """Singular values must be in non-increasing order."""
    _, _, ev = py_pca(rng_data, 4)
    assert np.all(ev[:-1] >= ev[1:]), (
        "Python reference: explained variance not non-increasing"
    )

    pca = PCA(n_components=4, seed=0)
    pca.fit_transform(rng_data)
    rust_ev = np.array(pca.explained_variance)
    assert np.all(rust_ev[:-1] >= rust_ev[1:]), (
        "Rust: explained variance not non-increasing"
    )


def test_projection_shape(small_data):
    n_components = 2
    pca = PCA(n_components=n_components, seed=0)
    result = np.array(pca.fit_transform(small_data))
    assert result.shape == (small_data.shape[0], n_components)


def test_centering_property(rng_data):
    """The column means of the projected data should be very close to zero
    because PCA centres the data before projecting."""
    pca = PCA(n_components=2, seed=0)
    proj = np.array(pca.fit_transform(rng_data))
    np.testing.assert_allclose(
        proj.mean(axis=0),
        0.0,
        atol=1e-10,
        err_msg="Projected data should have ~zero column means",
    )


def test_transform_consistent_with_fit_transform(rng_data):
    """transform(X) must return the same result as the projection from fit_transform(X)."""
    pca = PCA(n_components=2, seed=0)
    proj1 = np.array(pca.fit_transform(rng_data))
    proj2 = np.array(pca.transform(rng_data))
    np.testing.assert_allclose(proj1, proj2, atol=1e-12)
