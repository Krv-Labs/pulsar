"""
Tests for pulsar.mcp.diagnostics module.

Tests graph quality metrics computation and diagnosis generation.
"""

import json

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
                "projection": {
                    "method": "pca",
                    "dimensions": {"values": [2]},
                    "seed": {"values": [42]},
                },
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


def test_weight_percentiles_are_ordered(fitted_model):
    """Assert diagnostics expose lower, middle, and upper weight percentiles."""
    result = diagnose_model(fitted_model)
    assert result.weight_p25 <= result.weight_p50 <= result.weight_p95


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
    assert "Add projection dimensions" in result.grid_adequacy_note


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


def test_load_config_defaults_construction_threshold_to_auto(basic_config):
    """Missing construction_threshold should remain an auto-selected graph cut."""
    basic_config.pop("cosmic_graph")

    cfg = load_config(basic_config)

    assert cfg.cosmic_graph.construction_threshold == "auto"


def test_load_config_preserves_explicit_zero_construction_threshold(basic_config):
    """A no-cutoff graph is only selected when explicitly requested."""
    basic_config["cosmic_graph"] = {"construction_threshold": 0.0}

    cfg = load_config(basic_config)

    assert cfg.cosmic_graph.construction_threshold == 0.0


def test_load_config_rejects_out_of_range_construction_threshold(basic_config):
    basic_config["cosmic_graph"] = {"construction_threshold": 1.5}

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        load_config(basic_config)


def test_load_config_rejects_legacy_cosmic_graph_threshold(basic_config):
    """Legacy threshold key must fail loudly instead of falling back to auto."""
    basic_config["cosmic_graph"] = {"threshold": 0.0}

    with pytest.raises(ValueError, match="cosmic_graph.construction_threshold"):
        load_config(basic_config)


def test_load_config_rejects_unknown_cosmic_graph_key(basic_config):
    """Unknown cosmic_graph fields must not be silently ignored."""
    basic_config["cosmic_graph"]["stale_field"] = 0.0

    with pytest.raises(ValueError, match="Unsupported cosmic_graph key"):
        load_config(basic_config)


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


def test_resolve_clusters_spectral_honors_explicit_method_with_threshold():
    """A positive interpretation threshold must not force component clustering."""
    from types import SimpleNamespace

    from pulsar.mcp.interpreter import resolve_clusters

    weighted = np.full((30, 30), 0.4, dtype=float)
    np.fill_diagonal(weighted, 0.0)
    weighted[:15, :15] = 0.9
    weighted[15:, 15:] = 0.9
    np.fill_diagonal(weighted, 0.0)
    model = SimpleNamespace(weighted_adjacency=weighted)

    result = resolve_clusters(
        model,
        method="spectral",
        interpretation_edge_weight_threshold=0.2,
    )

    assert result.method_used == "spectral"
    assert result.interpretation_edge_weight_threshold_applied == 0.2


def test_resolve_clusters_max_k(connected_spectral_model):
    """Assert max_k parameter is respected — higher max_k can find more clusters."""
    from pulsar.mcp.interpreter import resolve_clusters

    result_low = resolve_clusters(connected_spectral_model, method="spectral", max_k=3)
    result_high = resolve_clusters(
        connected_spectral_model, method="spectral", max_k=15
    )
    assert len(result_low.labels.unique()) >= 2
    assert len(result_high.labels.unique()) >= 2


def test_resolve_clusters_spectral_disconnected_degrades_gracefully(
    disconnected_spectral_model,
):
    """Spectral clustering no longer hard-fails on a disconnected affinity graph;
    it labels every node (giant component clustered, residual isolated). The
    exhaustive disconnected-path assertions live in test_spectral_robustness.py."""
    from pulsar.mcp.interpreter import resolve_clusters

    result = resolve_clusters(disconnected_spectral_model, method="spectral")
    n = disconnected_spectral_model.weighted_adjacency.shape[0]
    assert len(result.labels) == n


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
    assert "diagnostic_interpretation" in empty
    assert "agent_action" not in empty


def test_graph_advisories_high_singletons():
    """singleton_fraction > 0.8 yields a HIGH_SINGLETONS warning."""
    advisories = _graph_advisories(
        n_edges=10, singleton_fraction=0.9, giant_fraction=0.1, density=0.05
    )
    codes = {a["code"] for a in advisories}
    assert "HIGH_SINGLETONS" in codes
    high = next(a for a in advisories if a["code"] == "HIGH_SINGLETONS")
    assert high["severity"] == "warning"
    assert high["diagnostic_interpretation"]
    assert "agent_action" not in high


def test_graph_advisories_dominant_component():
    """giant_fraction > 0.95 yields a DOMINANT_COMPONENT warning advisory."""
    advisories = _graph_advisories(
        n_edges=100, singleton_fraction=0.0, giant_fraction=0.99, density=0.5
    )
    codes = {a["code"] for a in advisories}
    assert "DOMINANT_COMPONENT" in codes
    dom = next(a for a in advisories if a["code"] == "DOMINANT_COMPONENT")
    assert dom["severity"] == "warning"
    assert dom["diagnostic_interpretation"]
    assert "agent_action" not in dom


def test_graph_advisories_hairball_density():
    """density > 0.8 with edges yields a HAIRBALL_DENSITY warning."""
    advisories = _graph_advisories(
        n_edges=500, singleton_fraction=0.0, giant_fraction=1.0, density=0.95
    )
    codes = {a["code"] for a in advisories}
    assert "HAIRBALL_DENSITY" in codes
    hair = next(a for a in advisories if a["code"] == "HAIRBALL_DENSITY")
    assert hair["severity"] == "warning"
    assert hair["diagnostic_interpretation"]
    assert "agent_action" not in hair


def test_graph_advisories_clean():
    """A well-formed graph yields no advisories."""
    advisories = _graph_advisories(
        n_edges=50, singleton_fraction=0.1, giant_fraction=0.5, density=0.3
    )
    assert advisories == []


# ---------------------------------------------------------------------------
# B1: session run-state consistency for cached cluster state
# ---------------------------------------------------------------------------


def _two_block_spectral_session():
    """Session with a connected two-block weighted adjacency suitable for the
    spectral clustering path (mirrors test_mcp_workflow direct-session setup)."""
    import asyncio
    from types import SimpleNamespace

    import networkx as nx

    from pulsar.mcp.session import _sessions, _get_session

    _sessions.clear()
    session = _get_session(None)
    weighted = np.full((30, 30), 0.4, dtype=float)
    np.fill_diagonal(weighted, 0.0)
    weighted[:15, :15] = 0.9
    weighted[15:, 15:] = 0.9
    np.fill_diagonal(weighted, 0.0)
    session.model = SimpleNamespace(
        weighted_adjacency=weighted,
        cosmic_graph=nx.from_numpy_array((weighted > 0.2).astype(int)),
        resolved_construction_threshold=0.2,
    )
    session.data = pd.DataFrame(
        {
            "x": np.r_[np.ones(15), np.zeros(15)],
            "y": np.r_[np.zeros(15), np.ones(15)],
        }
    )
    return asyncio, session


def test_cluster_profile_stamps_run_and_serves_when_consistent():
    """A dossier stamps clusters with the active run; a matching profile read
    succeeds (no false-positive stale error)."""
    from pulsar.mcp.tools.clustering import (
        generate_cluster_dossier,
        get_cluster_profile,
    )

    asyncio, session = _two_block_spectral_session()
    session.latest_run_id = "run_first"

    dossier = json.loads(
        asyncio.run(generate_cluster_dossier(method="spectral", response_format="json"))
    )
    assert dossier["status"] == "ok"
    # The cache is stamped with the run it was computed from.
    assert session.clusters_run_id == "run_first"

    cluster_id = dossier["clusters"][0]["cluster_id"]
    profile = json.loads(
        asyncio.run(get_cluster_profile(cluster_id=cluster_id, response_format="json"))
    )
    assert profile["status"] == "ok"


def test_cluster_profile_rejects_stale_cache_after_new_sweep():
    """If a newer sweep advances latest_run_id without recomputing clusters,
    get_cluster_profile must refuse to serve the stale cache and instruct a
    re-run rather than returning mismatched data."""
    from pulsar.mcp.tools.clustering import (
        generate_cluster_dossier,
        get_cluster_profile,
    )

    asyncio, session = _two_block_spectral_session()
    session.latest_run_id = "run_first"

    dossier = json.loads(
        asyncio.run(generate_cluster_dossier(method="spectral", response_format="json"))
    )
    assert dossier["status"] == "ok"
    cluster_id = dossier["clusters"][0]["cluster_id"]

    # Simulate a second sweep: latest_run_id advances and the fitted model is
    # replaced, but clusters/feature_evidence are NOT recomputed (stale cache).
    session.latest_run_id = "run_second"

    profile = json.loads(
        asyncio.run(get_cluster_profile(cluster_id=cluster_id, response_format="json"))
    )

    assert profile["status"] == "error"
    assert profile["tool"] == "get_cluster_profile"
    assert profile["error_code"] == "CLUSTER_CACHE_STALE"
    assert "generate_cluster_dossier" in profile["agent_action"]
    assert profile["details"]["cached_run_id"] == "run_first"
    assert profile["details"]["current_run_id"] == "run_second"


def test_cluster_signal_matrix_rejects_stale_cache_after_new_sweep():
    """Run-state consistency also guards the cross-cluster signal matrix read."""
    from pulsar.mcp.tools.clustering import (
        generate_cluster_dossier,
        get_cluster_signal_matrix,
    )

    asyncio, session = _two_block_spectral_session()
    session.latest_run_id = "run_first"

    asyncio.run(generate_cluster_dossier(method="spectral", response_format="json"))
    session.latest_run_id = "run_second"

    result = json.loads(asyncio.run(get_cluster_signal_matrix(return_markdown=False)))
    assert result["status"] == "error"
    assert result["error_code"] == "CLUSTER_CACHE_STALE"


def test_generate_cluster_dossier_refreshes_stale_stamp():
    """Re-running the dossier after a new sweep refreshes the stamp, restoring
    consistent reads — the documented recovery path."""
    from pulsar.mcp.tools.clustering import (
        generate_cluster_dossier,
        get_cluster_profile,
    )

    asyncio, session = _two_block_spectral_session()
    session.latest_run_id = "run_first"
    asyncio.run(generate_cluster_dossier(method="spectral", response_format="json"))

    session.latest_run_id = "run_second"
    dossier = json.loads(
        asyncio.run(generate_cluster_dossier(method="spectral", response_format="json"))
    )
    assert dossier["status"] == "ok"
    assert session.clusters_run_id == "run_second"

    cluster_id = dossier["clusters"][0]["cluster_id"]
    profile = json.loads(
        asyncio.run(get_cluster_profile(cluster_id=cluster_id, response_format="json"))
    )
    assert profile["status"] == "ok"
