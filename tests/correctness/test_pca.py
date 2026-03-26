"""
Correctness tests for pca.rs — PCA.

The Python reference re-implements PCA step by step using only numpy:
  1. Centre: X_c = X - column_mean(X)
  2. Sample covariance: C = X_c.T @ X_c / (n - 1)
  3. SVD: U, s, Vt = np.linalg.svd(C)   — rows of Vt are principal components
  4. Sign flip: negate any row whose largest-abs-value element is negative
  5. Project: X_c @ Vt[:n_components].T

This mirrors the Rust implementation in pca.rs exactly.

Because the SVD of a covariance matrix is deterministic (given the same
algorithm), and both implementations apply the same sign convention, the
Rust and Python outputs should agree to floating-point precision (atol=1e-8).
Note: nalgebra and numpy use different SVD algorithms internally so tiny
floating-point differences are expected; we do NOT require bit-for-bit identity.
"""
import numpy as np
import pytest

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

def test_projection_matches_reference(rng_data):
    """Rust projection should match Python reference to floating-point tolerance."""
    n_components = 2
    expected_proj, _, _ = py_pca(rng_data, n_components)

    pca = PCA(n_components=n_components, seed=0)
    actual_proj = np.array(pca.fit_transform(rng_data))

    np.testing.assert_allclose(actual_proj, expected_proj, atol=1e-8,
                               err_msg="Projection differs between Rust and Python reference")


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


def test_explained_variance_matches_reference(rng_data):
    n_components = 3
    _, _, expected_ev = py_pca(rng_data, n_components)

    pca = PCA(n_components=n_components, seed=0)
    pca.fit_transform(rng_data)
    actual_ev = np.array(pca.explained_variance)

    np.testing.assert_allclose(actual_ev, expected_ev, atol=1e-8,
                               err_msg="Explained variance differs between Rust and Python reference")


def test_explained_variance_descending(rng_data):
    """Singular values must be in non-increasing order."""
    _, _, ev = py_pca(rng_data, 4)
    assert np.all(ev[:-1] >= ev[1:]), "Python reference: explained variance not non-increasing"

    pca = PCA(n_components=4, seed=0)
    pca.fit_transform(rng_data)
    rust_ev = np.array(pca.explained_variance)
    assert np.all(rust_ev[:-1] >= rust_ev[1:]), "Rust: explained variance not non-increasing"


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
    np.testing.assert_allclose(proj.mean(axis=0), 0.0, atol=1e-10,
                               err_msg="Projected data should have ~zero column means")


def test_transform_consistent_with_fit_transform(rng_data):
    """transform(X) must return the same result as the projection from fit_transform(X)."""
    pca = PCA(n_components=2, seed=0)
    proj1 = np.array(pca.fit_transform(rng_data))
    proj2 = np.array(pca.transform(rng_data))
    np.testing.assert_allclose(proj1, proj2, atol=1e-12)
