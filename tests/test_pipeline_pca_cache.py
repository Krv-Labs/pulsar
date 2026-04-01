"""
Tests for PCA embedding caching in ThemaRS.fit() and MCP session.

Tests that _precomputed_embeddings parameter works correctly,
and that the MCP session properly invalidates/reuses cache.
"""

import hashlib
import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from pulsar.config import load_config
from pulsar.pipeline import ThemaRS
from pulsar.mcp.server import _pca_fingerprint


@pytest.fixture
def basic_config():
    """Minimal config for quick fitting."""
    return {
        "run": {"name": "test"},
        "preprocessing": {"drop_columns": [], "impute": {}},
        "sweep": {
            "pca": {
                "dimensions": {"values": [2, 3]},
                "seed": {"values": [42]},
            },
            "ball_mapper": {
                "epsilon": {"values": [0.5, 1.0]},
            },
        },
        "cosmic_graph": {"threshold": "0.0"},
        "output": {"n_reps": 1},
    }


@pytest.fixture
def test_data():
    """Simple test dataset."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((50, 3)),
        columns=["x", "y", "z"],
    )


def test_precomputed_embeddings_bypasses_pca_grid(basic_config, test_data):
    """Assert that passing _precomputed_embeddings skips pca_grid call."""
    cfg = load_config(basic_config)

    # First fit: compute embeddings normally
    model1 = ThemaRS(cfg)
    model1.fit(data=test_data)
    embeddings1 = model1._embeddings

    # Monkeypatch pca_grid to track if it's called
    with mock.patch("pulsar.pipeline.pca_grid") as mock_pca_grid:
        mock_pca_grid.side_effect = Exception("pca_grid should not be called!")

        # Second fit: pass precomputed embeddings
        model2 = ThemaRS(cfg)
        model2.fit(data=test_data, _precomputed_embeddings=embeddings1)

        # Verify pca_grid was NOT called
        assert not mock_pca_grid.called, "pca_grid was called but should have been skipped"


def test_output_equivalence_with_cache(basic_config, test_data):
    """Assert that output is identical with and without cache."""
    cfg = load_config(basic_config)

    # First fit: normal path
    model1 = ThemaRS(cfg)
    model1.fit(data=test_data)
    embeddings = model1._embeddings
    wadj1 = np.array(model1.weighted_adjacency)

    # Second fit: with cached embeddings (must use same epsilon range)
    model2 = ThemaRS(cfg)
    model2.fit(data=test_data, _precomputed_embeddings=embeddings)
    wadj2 = np.array(model2.weighted_adjacency)

    # Weighted adjacency should be numerically identical
    # (ball mapper is deterministic given same embeddings and epsilons)
    np.testing.assert_array_almost_equal(wadj1, wadj2, decimal=10,
        err_msg="Weighted adjacency differs with cached embeddings"
    )


def test_shape_assertion_on_mismatch(basic_config, test_data):
    """Assert that shape mismatch in embeddings raises a clear error."""
    cfg = load_config(basic_config)

    # Create embeddings with wrong row count (different data size)
    wrong_shape_embeddings = [np.random.randn(25, 2), np.random.randn(25, 3)]

    model = ThemaRS(cfg)
    with pytest.raises(AssertionError, match="Precomputed embedding row count"):
        model.fit(data=test_data, _precomputed_embeddings=wrong_shape_embeddings)


def test_embeddings_stored_after_fit(basic_config, test_data):
    """Assert that self._embeddings is set after fit()."""
    cfg = load_config(basic_config)
    model = ThemaRS(cfg)

    # Before fit
    assert not hasattr(model, "_embeddings") or model._embeddings is None

    # After fit
    model.fit(data=test_data)
    assert hasattr(model, "_embeddings"), "model._embeddings not set after fit"
    assert model._embeddings is not None, "model._embeddings is None after fit"
    assert isinstance(model._embeddings, list), "model._embeddings is not a list"
    assert len(model._embeddings) > 0, "model._embeddings is empty"
    assert all(isinstance(e, np.ndarray) for e in model._embeddings), \
        "Not all embeddings are numpy arrays"


def test_pca_fingerprint_changes_on_dimension_change():
    """Assert fingerprint changes when PCA dimensions change."""
    cfg1 = type("MockConfig", (), {
        "data": "/path/to/data.csv",
        "pca": type("MockPCA", (), {
            "dimensions": [2, 3],
            "seeds": [42],
        })()
    })()

    cfg2 = type("MockConfig", (), {
        "data": "/path/to/data.csv",
        "pca": type("MockPCA", (), {
            "dimensions": [2, 4],  # Different
            "seeds": [42],
        })()
    })()

    fp1 = _pca_fingerprint(cfg1, 100)
    fp2 = _pca_fingerprint(cfg2, 100)

    assert fp1 != fp2, "Fingerprint should change when dimensions change"


def test_pca_fingerprint_changes_on_seed_change():
    """Assert fingerprint changes when PCA seeds change."""
    cfg1 = type("MockConfig", (), {
        "data": "/path/to/data.csv",
        "pca": type("MockPCA", (), {
            "dimensions": [2, 3],
            "seeds": [42],
        })()
    })()

    cfg2 = type("MockConfig", (), {
        "data": "/path/to/data.csv",
        "pca": type("MockPCA", (), {
            "dimensions": [2, 3],
            "seeds": [99],  # Different
        })()
    })()

    fp1 = _pca_fingerprint(cfg1, 100)
    fp2 = _pca_fingerprint(cfg2, 100)

    assert fp1 != fp2, "Fingerprint should change when seeds change"


def test_pca_fingerprint_changes_on_shape_change():
    """Assert fingerprint changes when data row count changes."""
    cfg = type("MockConfig", (), {
        "data": "/path/to/data.csv",
        "pca": type("MockPCA", (), {
            "dimensions": [2, 3],
            "seeds": [42],
        })()
    })()

    fp1 = _pca_fingerprint(cfg, 100)
    fp2 = _pca_fingerprint(cfg, 150)  # Different row count

    assert fp1 != fp2, "Fingerprint should change when row count changes"


def test_pca_fingerprint_changes_on_path_change():
    """Assert fingerprint changes when data path changes."""
    cfg1 = type("MockConfig", (), {
        "data": "/path/to/data1.csv",
        "pca": type("MockPCA", (), {
            "dimensions": [2, 3],
            "seeds": [42],
        })()
    })()

    cfg2 = type("MockConfig", (), {
        "data": "/path/to/data2.csv",  # Different
        "pca": type("MockPCA", (), {
            "dimensions": [2, 3],
            "seeds": [42],
        })()
    })()

    fp1 = _pca_fingerprint(cfg1, 100)
    fp2 = _pca_fingerprint(cfg2, 100)

    assert fp1 != fp2, "Fingerprint should change when data path changes"


def test_pca_fingerprint_stable_on_epsilon_change():
    """Assert fingerprint is STABLE when only epsilon changes (key correctness test)."""
    cfg_base = {
        "data": "/path/to/data.csv",
        "pca": type("MockPCA", (), {
            "dimensions": [2, 3],
            "seeds": [42],
        })()
    }

    # Create two configs that differ only in epsilon
    cfg1 = type("MockConfig1", (), cfg_base)()
    cfg1.epsilon = 0.5

    cfg2 = type("MockConfig2", (), cfg_base)()
    cfg2.epsilon = 1.5

    fp1 = _pca_fingerprint(cfg1, 100)
    fp2 = _pca_fingerprint(cfg2, 100)

    assert fp1 == fp2, (
        "Fingerprint should be STABLE when only epsilon changes. "
        "This is the key correctness test for cache hits on Auto-Focus retries."
    )


def test_pca_fingerprint_format_is_hash():
    """Assert fingerprint is a valid SHA256 hex string."""
    cfg = type("MockConfig", (), {
        "data": "/path/to/data.csv",
        "pca": type("MockPCA", (), {
            "dimensions": [2, 3],
            "seeds": [42],
        })()
    })()

    fp = _pca_fingerprint(cfg, 100)

    # Check it's a hex string of length 64 (SHA256)
    assert isinstance(fp, str), "Fingerprint should be a string"
    assert len(fp) == 64, "SHA256 hex should be 64 characters"
    assert all(c in "0123456789abcdef" for c in fp), "Should be valid hex"
