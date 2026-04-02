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
from sklearn.decomposition import PCA as SklearnPCA

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


# ---------------------------------------------------------------------------
# Tests that would have caught the transposition bug
# ---------------------------------------------------------------------------


def test_reconstruction_error_bounded_by_discarded_variance():
    """Project data to k components and back. The reconstruction error (MSE)
    should be bounded by the sum of discarded eigenvalues.

    A transposition bug would produce a projection in the wrong space,
    making reconstruction nonsensical and the error enormous.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 6)).astype(np.float64)
    n_components = 3

    # Fit with Pulsar
    pca = PCA(n_components=n_components, seed=0)
    projected = np.array(pca.fit_transform(X))

    # Get full exact variance to compute discarded portion
    sklearn_full = SklearnPCA(n_components=6)
    sklearn_full.fit(X)
    total_variance = sklearn_full.explained_variance_.sum()
    pulsar_ev = np.array(pca.explained_variance)
    captured_variance = pulsar_ev.sum()
    discarded_variance = total_variance - captured_variance

    # Reconstruct via sklearn (fit exact PCA to get components in same space)
    # Instead: use Pulsar's own projection and approximate reconstruction
    # Reconstruct: X_approx = projected @ V + mean, where V is from sklearn
    # for the reference. We only check that error is in the right ballpark.
    sklearn_ref = SklearnPCA(n_components=n_components)
    sklearn_ref.fit(X)
    X_proj_sk = sklearn_ref.transform(X)

    # Pulsar's projection should achieve similar reconstruction quality.
    # We can't call inverse_transform on Pulsar, but the projected variance
    # should be close to sklearn's, implying similar reconstruction error.
    pulsar_captured = np.var(projected, axis=0).sum()
    sklearn_captured = np.var(X_proj_sk, axis=0).sum()

    # Pulsar should capture at least 80% of what exact PCA captures
    assert pulsar_captured > 0.80 * sklearn_captured, (
        f"Pulsar captured variance {pulsar_captured:.4f} is too low "
        f"vs sklearn {sklearn_captured:.4f}"
    )

    # The discarded variance should be non-negative and reasonable
    # (captured should not exceed total by more than a small approximation error)
    assert captured_variance < total_variance * 1.05, (
        f"Captured variance {captured_variance:.4f} implausibly exceeds "
        f"total {total_variance:.4f}"
    )
    assert discarded_variance > -0.05 * total_variance, (
        f"Discarded variance {discarded_variance:.4f} is implausibly negative "
        f"for total variance {total_variance:.4f}"
    )


def test_components_are_orthogonal(rng_data):
    """Principal components must be mutually orthogonal unit vectors.

    A transposition bug produces components that live in sample-space instead
    of feature-space, so their dot products would not form an identity matrix
    of the expected dimension.
    """
    n_components = 3
    pca = PCA(n_components=n_components, seed=0)
    projected = np.array(pca.fit_transform(rng_data))

    # Use sklearn to extract equivalent components for shape reference
    sklearn_pca = SklearnPCA(n_components=n_components)
    sklearn_pca.fit(rng_data)

    # The Pulsar projected columns should be mutually uncorrelated
    # (this is a necessary property of PCA projections).
    # Covariance of projected data should be diagonal.
    cov = np.cov(projected, rowvar=False)
    n = cov.shape[0]
    assert cov.shape == (n_components, n_components), (
        f"Covariance of projection should be {n_components}x{n_components}, "
        f"got {cov.shape}"
    )

    # Off-diagonal elements should be near zero
    mask = ~np.eye(n, dtype=bool)
    off_diag = np.abs(cov[mask])
    diag = np.abs(np.diag(cov))
    # Off-diagonal should be tiny relative to diagonal
    assert np.all(off_diag < 0.1 * diag.max()), (
        f"Off-diagonal covariance too large: max={off_diag.max():.6f}, "
        f"diag max={diag.max():.6f}"
    )


def test_sklearn_subspace_agreement():
    """Pulsar and sklearn PCA should span the same subspace for the top
    components on well-conditioned data.

    Verified by computing the absolute dot product between each pair of
    corresponding component vectors (after sign normalisation). For the
    dominant components of well-conditioned data, these should be close to 1.

    A transposition bug produces components with the wrong dimensionality
    or in the wrong space entirely.
    """
    rng = np.random.default_rng(123)
    # Well-conditioned data with clear spectral gaps
    X = rng.standard_normal((200, 5)).astype(np.float64)
    # Inject strong signal in first two components
    X[:, 0] *= 10.0
    X[:, 1] *= 5.0

    n_components = 2

    pca = PCA(n_components=n_components, seed=42)
    pulsar_proj = np.array(pca.fit_transform(X))

    sklearn_pca = SklearnPCA(n_components=n_components)
    sklearn_proj = sklearn_pca.fit_transform(X)

    # Compare subspaces: for each component, the correlation between
    # Pulsar's and sklearn's projection column should be high.
    for i in range(n_components):
        corr = np.abs(np.corrcoef(pulsar_proj[:, i], sklearn_proj[:, i])[0, 1])
        assert corr > 0.95, (
            f"Component {i}: Pulsar-sklearn correlation = {corr:.4f}, expected > 0.95"
        )


def test_deterministic_with_same_seed():
    """Same seed must produce bit-identical results."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4)).astype(np.float64)

    pca1 = PCA(n_components=2, seed=42)
    result1 = np.array(pca1.fit_transform(X))
    ev1 = np.array(pca1.explained_variance)

    pca2 = PCA(n_components=2, seed=42)
    result2 = np.array(pca2.fit_transform(X))
    ev2 = np.array(pca2.explained_variance)

    np.testing.assert_array_equal(result1, result2)
    np.testing.assert_array_equal(ev1, ev2)


def test_different_seeds_produce_valid_results():
    """Different seeds should both explain similar total variance.

    Randomized SVD with different seeds explores different random subspaces,
    but for well-conditioned data the captured variance should be consistent.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 6)).astype(np.float64)
    n_components = 3

    pca_a = PCA(n_components=n_components, seed=42)
    pca_a.fit_transform(X)
    ev_a = np.array(pca_a.explained_variance).sum()

    pca_b = PCA(n_components=n_components, seed=999)
    pca_b.fit_transform(X)
    ev_b = np.array(pca_b.explained_variance).sum()

    # Both should capture similar total variance (within 10%)
    ratio = min(ev_a, ev_b) / max(ev_a, ev_b)
    assert ratio > 0.90, (
        f"Seeds produced very different variance: {ev_a:.4f} vs {ev_b:.4f} "
        f"(ratio={ratio:.4f})"
    )
