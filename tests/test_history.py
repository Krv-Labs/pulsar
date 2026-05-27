"""Tests for pulsar.mcp.history.summarize_history."""

from __future__ import annotations

from dataclasses import dataclass

from pulsar.mcp.history import summarize_history


@dataclass
class _FakeRecord:
    config_yaml: str
    metrics: dict


def test_summarize_history_empty():
    """Empty history returns n_runs=0 and a canned observation."""
    result = summarize_history([])
    assert result["n_runs"] == 0
    assert isinstance(result["observations"], list)
    assert len(result["observations"]) >= 1
    assert result["rationale"] == ""


def test_summarize_history_detects_hairball_and_singletons():
    """observations flag the offending PCA dim and threshold; rationale is non-empty."""
    hairball_yaml = """run:
  name: hairball
sweep:
  pca:
    dimensions:
      values: [2]
    seed:
      values: [42]
  ball_mapper:
    epsilon:
      values: [0.5]
cosmic_graph:
  construction_threshold: "0.0"
"""
    fragmented_yaml = """run:
  name: fragmented
sweep:
  pca:
    dimensions:
      values: [10]
    seed:
      values: [42]
  ball_mapper:
    epsilon:
      values: [0.1]
cosmic_graph:
  construction_threshold: 0.5
"""
    history = [
        _FakeRecord(
            config_yaml=hairball_yaml,
            metrics={
                "n_edges": 1000,
                "density": 0.95,
                "singleton_fraction": 0.0,
                "giant_fraction": 1.0,
            },
        ),
        _FakeRecord(
            config_yaml=fragmented_yaml,
            metrics={
                "n_edges": 5,
                "density": 0.01,
                "singleton_fraction": 0.7,
                "giant_fraction": 0.05,
            },
        ),
    ]

    result = summarize_history(history)

    assert result["n_runs"] == 2
    joined = " ".join(result["observations"])
    # PCA dim 2 appeared in a hairball run
    assert "2" in joined
    # PCA dim 10 appeared in a fragmented run
    assert "10" in joined
    # Threshold 0.5 coincided with elevated singletons
    assert "0.5" in joined
    assert "elevated singleton" in joined
    assert result["rationale"] != ""
    assert "fragmentation_trend" in result
