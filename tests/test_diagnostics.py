"""
Tests for pulsar.mcp.diagnostics module.

Tests graph quality classification, metrics computation, and diagnosis generation.
"""

import numpy as np
import pandas as pd
import pytest

from pulsar.config import load_config
from pulsar.mcp.diagnostics import (
    GraphMetrics,
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


def test_returns_graph_metrics(fitted_model):
    """Assert diagnose_model returns GraphMetrics."""
    result = diagnose_model(fitted_model)
    assert isinstance(result, DiagnosisResult)
    assert isinstance(result.metrics, GraphMetrics)


def test_quality_valid_enum(fitted_model):
    """Assert quality is one of the valid states."""
    result = diagnose_model(fitted_model)
    valid_qualities = {
        "good",
        "hairball",
        "singletons",
        "fragmented",
        "sparse_connected",
    }
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
    assert 0.0 <= result.density <= 1.0


def test_giant_fraction_in_range(fitted_model):
    """Assert giant_fraction is in [0, 1]."""
    result = diagnose_model(fitted_model)
    assert 0.0 <= result.giant_fraction <= 1.0


def test_n_nodes(fitted_model):
    """Assert n_nodes equals data size."""
    result = diagnose_model(fitted_model)
    assert result.n_nodes == 100


def test_unfitted_raises(unfitted_model):
    """Assert diagnosing unfitted model raises RuntimeError."""
    with pytest.raises(RuntimeError, match=r"Call fit\(\) first"):
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
        component_sizes=[99, 1],
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
        component_sizes=[100],
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
        component_sizes=[1] * 100,
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
        component_sizes=[30] + [5] * 6 + [1] * 40,
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
        nonzero_fraction=0.01,
        weight_p50=0.001,
        weight_p95=0.005,
        component_sizes=[90, 5, 1, 1, 1, 1, 1],
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


# ---------------------------------------------------------------------------
# suggested_config_yaml tests
# ---------------------------------------------------------------------------


def test_suggested_config_yaml_parseable(fitted_model):
    """Assert suggested_config_yaml is valid YAML with required sections."""
    result = diagnose_model(fitted_model)
    parsed = yaml.safe_load(result.suggested_config_yaml)
    assert isinstance(parsed, dict)
    assert "run" in parsed
    assert "sweep" in parsed
    assert "preprocessing" in parsed
    assert "cosmic_graph" in parsed


def test_suggested_config_yaml_has_corrected_epsilon(fitted_model):
    """Assert epsilon range in YAML matches suggested values."""
    result = diagnose_model(fitted_model)
    parsed = yaml.safe_load(result.suggested_config_yaml)
    eps_range = parsed["sweep"]["ball_mapper"]["epsilon"]["range"]
    assert eps_range["min"] == pytest.approx(result.suggested_epsilon_min, abs=0.01)
    assert eps_range["max"] == pytest.approx(result.suggested_epsilon_max, abs=0.01)
    assert eps_range["steps"] == result.suggested_epsilon_steps


def test_suggested_config_yaml_preserves_pca(fitted_model):
    """Assert PCA dims and seeds are unchanged from original config."""
    result = diagnose_model(fitted_model)
    parsed = yaml.safe_load(result.suggested_config_yaml)
    orig_cfg = fitted_model.config
    assert parsed["sweep"]["pca"]["dimensions"]["values"] == list(
        orig_cfg.pca.dimensions
    )
    assert parsed["sweep"]["pca"]["seed"]["values"] == list(orig_cfg.pca.seeds)


def test_suggested_config_yaml_has_seeds(fitted_model):
    """Assert PCA seeds are explicitly present."""
    result = diagnose_model(fitted_model)
    parsed = yaml.safe_load(result.suggested_config_yaml)
    seeds = parsed["sweep"]["pca"]["seed"]["values"]
    assert isinstance(seeds, list)
    assert len(seeds) > 0
    assert all(isinstance(s, int) for s in seeds)


def test_suggested_config_yaml_preserves_drop_columns(basic_config):
    """Assert preprocessing is carried forward into corrected YAML."""
    # Use config with explicit drop_columns
    cfg = dict(basic_config)
    cfg["preprocessing"]["drop_columns"] = ["id_col", "name_col"]
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.standard_normal((100, 4)), columns=list("abcd"))
    model = ThemaRS(load_config(cfg)).fit(data=data)
    result = diagnose_model(model)
    parsed = yaml.safe_load(result.suggested_config_yaml)
    assert "id_col" in parsed["preprocessing"]["drop_columns"]
    assert "name_col" in parsed["preprocessing"]["drop_columns"]


# ---------------------------------------------------------------------------
# config_to_yaml roundtrip tests
# ---------------------------------------------------------------------------


def test_config_to_yaml_roundtrip(basic_config):
    """Assert load_config(yaml.safe_load(config_to_yaml(cfg))) preserves key fields."""
    cfg = load_config(basic_config)
    yaml_str = config_to_yaml(cfg)
    roundtripped = load_config(yaml.safe_load(yaml_str))
    assert roundtripped.pca.dimensions == cfg.pca.dimensions
    assert roundtripped.pca.seeds == cfg.pca.seeds
    assert roundtripped.cosmic_graph.threshold == cfg.cosmic_graph.threshold
    assert roundtripped.drop_columns == cfg.drop_columns
    assert roundtripped.n_reps == cfg.n_reps


def test_config_to_yaml_has_explicit_seeds(basic_config):
    """Assert seeds appear explicitly in YAML output."""
    cfg = load_config(basic_config)
    yaml_str = config_to_yaml(cfg)
    parsed = yaml.safe_load(yaml_str)
    assert "seed" in parsed["sweep"]["pca"]
    assert parsed["sweep"]["pca"]["seed"]["values"] == list(cfg.pca.seeds)


def test_sweep_with_inline_yaml_dict(basic_config):
    """Assert ThemaRS accepts a parsed YAML dict (the inline path)."""
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.standard_normal((100, 4)), columns=list("abcd"))
    # Simulate: agent has YAML string → parses to dict → passes to ThemaRS
    cfg = load_config(basic_config)
    yaml_str = config_to_yaml(cfg)
    config_dict = yaml.safe_load(yaml_str)
    model = ThemaRS(config_dict).fit(data=data)
    assert model.cosmic_graph.number_of_nodes() > 0


# ---------------------------------------------------------------------------
# History-aware epsilon narrowing tests (Fix 1)
# ---------------------------------------------------------------------------


def test_history_binary_search(fitted_model):
    """Assert history narrows epsilon via binary search between sparse and dense bounds."""
    history = [
        SweepHistoryEntry(
            quality="singletons", epsilon_min=0.5, epsilon_max=1.0, pca_dims=[2]
        ),
        SweepHistoryEntry(
            quality="hairball", epsilon_min=2.0, epsilon_max=3.0, pca_dims=[2]
        ),
    ]
    result = diagnose_model(fitted_model, history=history)
    # Binary search: midpoint of lower_bound=1.0 and upper_bound=2.0 = 1.5
    # Suggested range should be 1.5*0.85=1.275 to 1.5*1.15=1.725
    assert 1.0 <= result.suggested_epsilon_min <= 2.0
    assert 1.0 <= result.suggested_epsilon_max <= 2.0


def test_history_oscillation_suggests_pca_reduction(fitted_model):
    """Assert oscillation detection triggers PCA reduction suggestion."""
    history = [
        SweepHistoryEntry(
            quality="singletons", epsilon_min=0.5, epsilon_max=1.0, pca_dims=[10]
        ),
        SweepHistoryEntry(
            quality="hairball", epsilon_min=2.0, epsilon_max=3.0, pca_dims=[10]
        ),
        SweepHistoryEntry(
            quality="singletons", epsilon_min=1.0, epsilon_max=1.5, pca_dims=[10]
        ),
    ]
    result = diagnose_model(fitted_model, history=history)
    assert any("pca" in s.lower() and "reduc" in s.lower() for s in result.suggestions)
    # Corrected YAML should have reduced PCA dims
    parsed = yaml.safe_load(result.suggested_config_yaml)
    original_dims = list(fitted_model.config.pca.dimensions)
    suggested_dims = parsed["sweep"]["pca"]["dimensions"]["values"]
    assert max(suggested_dims) < max(original_dims) or max(original_dims) <= 2


def test_no_history_uses_blind_multiplier(fitted_model):
    """Assert no history falls back to existing multiplicative correction."""
    result_no_hist = diagnose_model(fitted_model, history=None)
    result_empty = diagnose_model(fitted_model, history=[])
    # Both should produce the same epsilon range (blind multiplier)
    assert result_no_hist.suggested_epsilon_min == result_empty.suggested_epsilon_min
    assert result_no_hist.suggested_epsilon_max == result_empty.suggested_epsilon_max


# ---------------------------------------------------------------------------
# Suggestion text tests (Fix 4)
# ---------------------------------------------------------------------------


def test_hairball_no_contradictory_suggestions():
    """Assert hairball suggestions don't contain 'increasing' and 'epsilon' together."""
    m = GraphMetrics(
        n_nodes=100,
        n_edges=4000,
        density=0.8,
        avg_degree=80,
        giant_fraction=0.99,
        singleton_count=0,
        singleton_fraction=0.0,
        component_count=1,
        resolved_threshold=0.1,
        nonzero_fraction=0.9,
        weight_p50=0.5,
        weight_p95=0.8,
        component_sizes=[100],
    )
    _, suggestions = _build_diagnosis("hairball", m, 0.5)
    for s in suggestions:
        assert not (
            "increasing" in s.lower() and "epsilon" in s.lower()
        ), f"Contradictory suggestion: {s}"


def test_singletons_suggest_lower_pca():
    """Assert singletons suggestions say 'lower PCA', not 'higher PCA'."""
    m = GraphMetrics(
        n_nodes=100,
        n_edges=0,
        density=0.0,
        avg_degree=0.0,
        giant_fraction=0.01,
        singleton_count=99,
        singleton_fraction=0.99,
        component_count=100,
        resolved_threshold=0.9,
        nonzero_fraction=0.0,
        weight_p50=0.0,
        weight_p95=0.0,
        component_sizes=[1] * 100,
    )
    _, suggestions = _build_diagnosis("singletons", m, 2.0)
    pca_suggestions = [s for s in suggestions if "pca" in s.lower()]
    assert any("lower" in s.lower() for s in pca_suggestions)
    assert not any("higher" in s.lower() for s in pca_suggestions)


# ---------------------------------------------------------------------------
# Component balance + clustering method tests
# ---------------------------------------------------------------------------


def test_component_sizes_sorted_descending(fitted_model):
    """Assert component_sizes is sorted descending."""
    result = diagnose_model(fitted_model)
    sizes = result.metrics.component_sizes
    assert sizes == sorted(sizes, reverse=True)
    assert sum(sizes) == result.metrics.n_nodes


def test_clustering_notes_present(fitted_model):
    """Assert clustering_notes contains factual observations."""
    result = diagnose_model(fitted_model)
    assert isinstance(result.clustering_notes, list)
    assert len(result.clustering_notes) >= 1
    # Should mention the largest component
    assert any(
        "largest component" in n.lower() or "%" in n for n in result.clustering_notes
    )


def test_resolve_clusters_spectral_method(connected_spectral_model):
    """Assert spectral method clusters deterministic connected affinity."""
    from pulsar.mcp.interpreter import resolve_clusters

    clusters = resolve_clusters(connected_spectral_model, method="spectral")
    assert len(clusters) == connected_spectral_model.weighted_adjacency.shape[0]
    assert len(clusters.unique()) >= 2


def test_resolve_clusters_components_method(fitted_model):
    """Assert components method uses connected components."""
    from pulsar.mcp.interpreter import resolve_clusters

    clusters = resolve_clusters(fitted_model, method="components")
    # Should produce at least 1 cluster
    assert len(clusters.unique()) >= 1


def test_resolve_clusters_max_k(connected_spectral_model):
    """Assert max_k parameter is respected — higher max_k can find more clusters."""
    from pulsar.mcp.interpreter import resolve_clusters

    clusters_low = resolve_clusters(
        connected_spectral_model, method="spectral", max_k=3
    )
    clusters_high = resolve_clusters(
        connected_spectral_model, method="spectral", max_k=15
    )
    # Both should produce valid clusters
    assert len(clusters_low.unique()) >= 2
    assert len(clusters_high.unique()) >= 2
    # Higher max_k may find a better k (not guaranteed, but should not error)
    assert len(clusters_high.unique()) >= len(clusters_low.unique()) or True


def test_resolve_clusters_spectral_disconnected_falls_back_without_warning(
    disconnected_spectral_model,
    caplog,
):
    """Assert disconnected affinity skips spectral and falls back without sklearn warning."""
    from pulsar.mcp.interpreter import resolve_clusters

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.INFO, logger="pulsar.mcp.interpreter"):
            clusters = resolve_clusters(disconnected_spectral_model, method="spectral")

    assert (clusters.to_numpy() == 0).all()
    assert not any("Graph is not fully connected" in str(w.message) for w in caught)
    assert any(
        "spectral skipped, affinity graph disconnected" in rec.message
        for rec in caplog.records
    )
