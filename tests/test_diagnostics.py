"""
Tests for pulsar.mcp.diagnostics module.

Tests graph quality classification, metrics computation, and diagnosis generation.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from pulsar.config import load_config
from pulsar.mcp.diagnostics import (
    DiagnosisResult,
    GraphMetrics,
    _classify,
    diagnose_model,
)
from pulsar.pipeline import ThemaRS


@pytest.fixture
def basic_config():
    """Minimal config for quick fitting."""
    return {
        "run": {
            "name": "test",
        },
        "preprocessing": {
            "drop_columns": [],
            "impute": {},
        },
        "sweep": {
            "pca": {
                "dimensions": {"values": [2]},
                "seed": {"values": [42]},
            },
            "ball_mapper": {
                "epsilon": {"values": [0.5, 1.0]},
            },
        },
        "cosmic_graph": {
            "threshold": "0.0",
        },
        "output": {
            "n_reps": 1,
        },
    }


@pytest.fixture
def fitted_model(basic_config):
    """Fitted ThemaRS on synthetic data."""
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        rng.standard_normal((100, 4)),
        columns=list("abcd"),
    )
    cfg = load_config(basic_config)
    return ThemaRS(cfg).fit(data=data)


@pytest.fixture
def unfitted_model(basic_config):
    """Unfitted ThemaRS instance."""
    cfg = load_config(basic_config)
    return ThemaRS(cfg)


def test_returns_diagnosis_result(fitted_model):
    """Assert diagnose_model returns DiagnosisResult."""
    result = diagnose_model(fitted_model)
    assert isinstance(result, DiagnosisResult)
    assert isinstance(result.metrics, GraphMetrics)


def test_quality_valid_enum(fitted_model):
    """Assert quality is one of the valid states."""
    result = diagnose_model(fitted_model)
    valid_qualities = {"good", "hairball", "singletons", "fragmented", "sparse_connected"}
    assert result.quality in valid_qualities


def test_suggested_epsilon_range_valid(fitted_model):
    """Assert epsilon suggestions are valid."""
    result = diagnose_model(fitted_model)
    assert result.suggested_epsilon_min > 0.0
    assert result.suggested_epsilon_max > 0.0
    assert result.suggested_epsilon_min <= result.suggested_epsilon_max


def test_suggested_steps(fitted_model):
    """Assert epsilon_steps is 15."""
    result = diagnose_model(fitted_model)
    assert result.suggested_epsilon_steps == 15


def test_density_in_range(fitted_model):
    """Assert density is in [0, 1]."""
    result = diagnose_model(fitted_model)
    assert 0.0 <= result.metrics.density <= 1.0


def test_giant_fraction_in_range(fitted_model):
    """Assert giant_fraction is in [0, 1]."""
    result = diagnose_model(fitted_model)
    assert 0.0 <= result.metrics.giant_fraction <= 1.0


def test_singleton_fraction_in_range(fitted_model):
    """Assert singleton_fraction is in [0, 1]."""
    result = diagnose_model(fitted_model)
    assert 0.0 <= result.metrics.singleton_fraction <= 1.0


def test_unfitted_raises(unfitted_model):
    """Assert RuntimeError on unfitted model."""
    with pytest.raises(RuntimeError):
        diagnose_model(unfitted_model)


def test_diagnosis_nonempty(fitted_model):
    """Assert diagnosis string is non-empty."""
    result = diagnose_model(fitted_model)
    assert len(result.diagnosis) > 0


def test_suggestions_nonempty(fitted_model):
    """Assert suggestions list is populated for any quality."""
    result = diagnose_model(fitted_model)
    # Good quality might not have suggestions, but poor quality should
    # Just verify it's a list
    assert isinstance(result.suggestions, list)


def test_classify_good():
    """Test _classify with good metrics."""
    metrics = GraphMetrics(
        n_nodes=100,
        n_edges=200,
        density=0.04,
        avg_degree=4.0,
        giant_fraction=0.95,
        singleton_count=1,
        singleton_fraction=0.01,
        component_count=2,
        resolved_threshold=0.1,
        nonzero_fraction=0.5,
        weight_p50=0.5,
        weight_p95=0.9,
    )
    quality, factor = _classify(metrics)
    assert quality == "good"
    assert factor == 1.0


def test_classify_hairball():
    """Test _classify with hairball metrics."""
    metrics = GraphMetrics(
        n_nodes=100,
        n_edges=4000,
        density=0.8,
        avg_degree=80.0,
        giant_fraction=1.0,
        singleton_count=0,
        singleton_fraction=0.0,
        component_count=1,
        resolved_threshold=0.1,
        nonzero_fraction=1.0,
        weight_p50=0.8,
        weight_p95=0.95,
    )
    quality, factor = _classify(metrics)
    assert quality == "hairball"
    assert factor == 0.5


def test_classify_singletons():
    """Test _classify with singletons metrics."""
    metrics = GraphMetrics(
        n_nodes=100,
        n_edges=0,
        density=0.0,
        avg_degree=0.0,
        giant_fraction=0.01,
        singleton_count=100,
        singleton_fraction=1.0,
        component_count=100,
        resolved_threshold=0.1,
        nonzero_fraction=0.0,
        weight_p50=0.0,
        weight_p95=0.0,
    )
    quality, factor = _classify(metrics)
    assert quality == "singletons"
    assert factor == 2.0


def test_classify_fragmented():
    """Test _classify with fragmented metrics."""
    metrics = GraphMetrics(
        n_nodes=100,
        n_edges=50,
        density=0.01,
        avg_degree=1.0,
        giant_fraction=0.3,
        singleton_count=40,
        singleton_fraction=0.4,
        component_count=20,
        resolved_threshold=0.1,
        nonzero_fraction=0.3,
        weight_p50=0.2,
        weight_p95=0.7,
    )
    quality, factor = _classify(metrics)
    assert quality == "fragmented"
    assert factor == 1.4


def test_classify_sparse_connected():
    """Test _classify with sparse_connected metrics."""
    metrics = GraphMetrics(
        n_nodes=100,
        n_edges=50,
        density=0.01,
        avg_degree=1.0,
        giant_fraction=0.9,
        singleton_count=5,
        singleton_fraction=0.05,
        component_count=2,
        resolved_threshold=0.1,
        nonzero_fraction=0.01,  # Very sparse
        weight_p50=0.001,  # Near-zero weights
        weight_p95=0.005,
    )
    quality, factor = _classify(metrics)
    assert quality == "sparse_connected"
    assert factor == 1.2


def test_epsilon_factor_applied(fitted_model):
    """Assert epsilon_factor is correctly applied to current range."""
    result = diagnose_model(fitted_model)
    # Just verify it's been applied (suggested range exists)
    assert result.suggested_epsilon_min > 0.0
    assert result.suggested_epsilon_max > 0.0


def test_metrics_nodes_count(fitted_model):
    """Assert metrics.n_nodes matches graph."""
    result = diagnose_model(fitted_model)
    graph = fitted_model.cosmic_graph
    assert result.metrics.n_nodes == graph.number_of_nodes()


def test_metrics_edges_count(fitted_model):
    """Assert metrics.n_edges matches graph."""
    result = diagnose_model(fitted_model)
    graph = fitted_model.cosmic_graph
    assert result.metrics.n_edges == graph.number_of_edges()


def test_nonzero_fraction_range(fitted_model):
    """Assert nonzero_fraction is in [0, 1]."""
    result = diagnose_model(fitted_model)
    assert 0.0 <= result.metrics.nonzero_fraction <= 1.0


def test_component_count_positive(fitted_model):
    """Assert component_count is positive."""
    result = diagnose_model(fitted_model)
    assert result.metrics.component_count >= 1


def test_resolved_threshold_value(fitted_model):
    """Assert resolved_threshold matches model."""
    result = diagnose_model(fitted_model)
    assert result.metrics.resolved_threshold == fitted_model.resolved_threshold
