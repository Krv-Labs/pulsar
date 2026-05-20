"""
Tests for pulsar.mcp.diagnostics module.

Tests graph quality metrics computation and diagnosis generation.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from pulsar.config import load_config, config_to_yaml
from pulsar.mcp.diagnostics import (
    GraphMetrics,
    _graph_advisories,
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
            "construction_threshold": "0.0",
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


@pytest.fixture
def connected_spectral_model():
    """Fitted model guaranteed to produce a dense connected graph (low epsilon)."""
    cfg = load_config(
        {
            "run": {"name": "spectral_test"},
            "preprocessing": {"drop_columns": [], "impute": {}},
            "sweep": {
                "pca": {"dimensions": {"values": [2]}, "seed": {"values": [42]}},
                "ball_mapper": {"epsilon": {"values": [2.0, 3.0]}},
            },
            "cosmic_graph": {"construction_threshold": "0.0"},
            "output": {"n_reps": 1},
        }
    )
    rng = np.random.default_rng(7)
    # Tightly clustered data → dense graph
    data = pd.DataFrame(
        rng.standard_normal((60, 3)) * 0.3,
        columns=list("xyz"),
    )
    return ThemaRS(cfg).fit(data=data)


@pytest.fixture
def disconnected_spectral_model():
    """Fitted model that produces a disconnected graph (high threshold)."""
    cfg = load_config(
        {
            "run": {"name": "disconnected_test"},
            "preprocessing": {"drop_columns": [], "impute": {}},
            "sweep": {
                "pca": {"dimensions": {"values": [2]}, "seed": {"values": [42]}},
                "ball_mapper": {"epsilon": {"values": [0.1]}},
            },
            "cosmic_graph": {"construction_threshold": "0.0"},
            "output": {"n_reps": 1},
        }
    )
    rng = np.random.default_rng(99)
    # Spread data with small epsilon → sparse/disconnected graph
    data = pd.DataFrame(
        rng.standard_normal((50, 3)) * 5.0,
        columns=list("xyz"),
    )
    return ThemaRS(cfg).fit(data=data)


# ---------------------------------------------------------------------------
# Core GraphMetrics tests
# ---------------------------------------------------------------------------


def test_returns_graph_metrics(fitted_model):
    """Assert diagnose_model returns GraphMetrics."""
    result = diagnose_model(fitted_model)
    assert isinstance(result, GraphMetrics)


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
    with pytest.raises(RuntimeError):
        diagnose_model(unfitted_model)


def test_nodes_count(fitted_model):
    """Assert n_nodes matches graph."""
    result = diagnose_model(fitted_model)
    graph = fitted_model.cosmic_graph
    assert result.n_nodes == graph.number_of_nodes()


def test_edges_count(fitted_model):
    """Assert n_edges matches graph."""
    result = diagnose_model(fitted_model)
    graph = fitted_model.cosmic_graph
    assert result.n_edges == graph.number_of_edges()


def test_nonzero_fraction_range(fitted_model):
    """Assert nonzero_fraction is in [0, 1]."""
    result = diagnose_model(fitted_model)
    assert 0.0 <= result.nonzero_fraction <= 1.0


def test_component_count_positive(fitted_model):
    """Assert component_count is positive."""
    result = diagnose_model(fitted_model)
    assert result.component_count >= 1


def test_resolved_threshold_value(fitted_model):
    """Assert resolved_construction_threshold matches model."""
    result = diagnose_model(fitted_model)
    assert (
        result.resolved_construction_threshold
        == fitted_model.resolved_construction_threshold
    )


def test_component_sizes_sorted_descending(fitted_model):
    """Assert component_sizes is sorted descending."""
    result = diagnose_model(fitted_model)
    sizes = result.component_sizes
    assert sizes == sorted(sizes, reverse=True)
    assert sum(sizes) == result.n_nodes


def test_grid_adequacy_under_sampled(basic_config):
    """Assert small representation grids are flagged as under-sampled."""
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.standard_normal((40, 4)), columns=list("abcd"))
    cfg = load_config(basic_config)
    model = ThemaRS(cfg).fit(data=data)

    result = diagnose_model(model)

    assert result.n_ball_maps == 2
    assert result.grid_adequacy_status == "under_sampled"
    assert "Add PCA dimensions" in result.grid_adequacy_note


def test_grid_adequacy_sample_count_ok():
    """Assert representation-rich grids are flagged as adequate by count."""
    cfg = load_config(
        {
            "run": {"name": "grid_count_ok"},
            "preprocessing": {"drop_columns": [], "impute": {}},
            "sweep": {
                "pca": {
                    "dimensions": {"values": [2, 3, 4]},
                    "seed": {"values": [42, 7]},
                },
                "ball_mapper": {
                    "epsilon": {"range": {"min": 0.5, "max": 1.5, "steps": 8}},
                },
            },
            "cosmic_graph": {"construction_threshold": "0.0"},
            "output": {"n_reps": 1},
        }
    )
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.standard_normal((60, 5)), columns=list("abcde"))
    model = ThemaRS(cfg).fit(data=data)

    result = diagnose_model(model)

    assert result.n_ball_maps == 48
    assert result.grid_adequacy_status == "sample_count_ok"
    assert "adequate for a baseline" in result.grid_adequacy_note


# ---------------------------------------------------------------------------
# Config round-trip / pipeline integration tests
# ---------------------------------------------------------------------------


def test_sweep_with_inline_yaml_dict(basic_config):
    """Assert ThemaRS accepts a parsed YAML dict (the inline path)."""
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.standard_normal((100, 4)), columns=list("abcd"))
    cfg = load_config(basic_config)
    yaml_str = config_to_yaml(cfg)
    config_dict = yaml.safe_load(yaml_str)
    model = ThemaRS(config_dict).fit(data=data)
    assert model.cosmic_graph.number_of_nodes() > 0


def test_config_to_yaml_roundtrip(basic_config):
    """Assert load_config(yaml.safe_load(config_to_yaml(cfg))) preserves key fields."""
    cfg = load_config(basic_config)
    yaml_str = config_to_yaml(cfg)
    roundtripped = load_config(yaml.safe_load(yaml_str))
    assert roundtripped.pca.dimensions == cfg.pca.dimensions
    assert roundtripped.pca.seeds == cfg.pca.seeds
    assert (
        roundtripped.cosmic_graph.construction_threshold
        == cfg.cosmic_graph.construction_threshold
    )
    assert roundtripped.drop_columns == cfg.drop_columns
    assert roundtripped.n_reps == cfg.n_reps


def test_config_to_yaml_has_explicit_seeds(basic_config):
    """Assert seeds appear explicitly in YAML output."""
    cfg = load_config(basic_config)
    yaml_str = config_to_yaml(cfg)
    parsed = yaml.safe_load(yaml_str)
    assert "seed" in parsed["sweep"]["pca"]
    assert parsed["sweep"]["pca"]["seed"]["values"] == list(cfg.pca.seeds)


# ---------------------------------------------------------------------------
# resolve_clusters tests
# ---------------------------------------------------------------------------


def test_resolve_clusters_components_method(fitted_model):
    """Assert components method uses connected components."""
    from pulsar.mcp.interpreter import resolve_clusters

    result = resolve_clusters(fitted_model, method="components")
    assert result.method_used == "components"
    assert len(result.labels.unique()) >= 1


def test_resolve_clusters_spectral_method(connected_spectral_model):
    """Assert spectral method clusters deterministic connected affinity."""
    from pulsar.mcp.interpreter import resolve_clusters

    result = resolve_clusters(connected_spectral_model, method="spectral")
    assert result.method_used == "spectral"
    assert len(result.labels) == connected_spectral_model.weighted_adjacency.shape[0]
    assert len(result.labels.unique()) >= 2


def test_resolve_clusters_max_k(connected_spectral_model):
    """Assert max_k parameter is respected — higher max_k can find more clusters."""
    from pulsar.mcp.interpreter import resolve_clusters

    result_low = resolve_clusters(connected_spectral_model, method="spectral", max_k=3)
    result_high = resolve_clusters(
        connected_spectral_model, method="spectral", max_k=15
    )
    assert len(result_low.labels.unique()) >= 2
    assert len(result_high.labels.unique()) >= 2


def test_resolve_clusters_spectral_disconnected_raises(
    disconnected_spectral_model,
):
    """Assert spectral method raises ValueError on a disconnected affinity graph."""
    from pulsar.mcp.interpreter import resolve_clusters

    with pytest.raises(ValueError, match="disconnected"):
        resolve_clusters(disconnected_spectral_model, method="spectral")


# ---------------------------------------------------------------------------
# _graph_advisories unit tests
# ---------------------------------------------------------------------------


def test_graph_advisories_empty():
    """n_edges == 0 yields an EMPTY_GRAPH error advisory."""
    advisories = _graph_advisories(
        n_edges=0, singleton_fraction=0.0, giant_fraction=0.0, density=0.0
    )
    codes = {a["code"] for a in advisories}
    assert "EMPTY_GRAPH" in codes
    empty = next(a for a in advisories if a["code"] == "EMPTY_GRAPH")
    assert empty["severity"] == "error"
    assert "message" in empty
    assert "agent_action" in empty


def test_graph_advisories_high_singletons():
    """singleton_fraction > 0.8 yields a HIGH_SINGLETONS warning."""
    advisories = _graph_advisories(
        n_edges=10, singleton_fraction=0.9, giant_fraction=0.1, density=0.05
    )
    codes = {a["code"] for a in advisories}
    assert "HIGH_SINGLETONS" in codes
    high = next(a for a in advisories if a["code"] == "HIGH_SINGLETONS")
    assert high["severity"] == "warning"
    assert high["agent_action"]


def test_graph_advisories_dominant_component():
    """giant_fraction > 0.95 yields a DOMINANT_COMPONENT info advisory."""
    advisories = _graph_advisories(
        n_edges=100, singleton_fraction=0.0, giant_fraction=0.99, density=0.5
    )
    codes = {a["code"] for a in advisories}
    assert "DOMINANT_COMPONENT" in codes
    dom = next(a for a in advisories if a["code"] == "DOMINANT_COMPONENT")
    assert dom["severity"] == "info"
    assert dom["agent_action"]


def test_graph_advisories_hairball_density():
    """density > 0.8 with edges yields a HAIRBALL_DENSITY warning."""
    advisories = _graph_advisories(
        n_edges=500, singleton_fraction=0.0, giant_fraction=1.0, density=0.95
    )
    codes = {a["code"] for a in advisories}
    assert "HAIRBALL_DENSITY" in codes
    hair = next(a for a in advisories if a["code"] == "HAIRBALL_DENSITY")
    assert hair["severity"] == "warning"
    assert hair["agent_action"]


def test_graph_advisories_clean():
    """A well-formed graph yields no advisories."""
    advisories = _graph_advisories(
        n_edges=50, singleton_fraction=0.1, giant_fraction=0.5, density=0.3
    )
    assert advisories == []
