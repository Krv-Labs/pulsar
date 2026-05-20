from types import SimpleNamespace

from pulsar.mcp.history import summarize_history


def _record(component_sizes, *, singleton_fraction=0.0, config_yaml="sweep: {}"):
    total = sum(component_sizes)
    return SimpleNamespace(
        config_yaml=config_yaml,
        metrics={
            "n_nodes": total,
            "n_edges": 1,
            "density": 0.2,
            "component_count": len(component_sizes),
            "component_sizes": component_sizes,
            "giant_fraction": component_sizes[0] / total,
            "singleton_fraction": singleton_fraction,
        },
    )


def test_summarize_history_detects_ice_chipping():
    summary = summarize_history(
        [
            _record([980, 20]),
            _record([970, 10, 10, 5, 5]),
        ]
    )

    trend = summary["fragmentation_trend"]
    assert trend["status"] == "ice_chipping"
    assert trend["component_count_delta"] == 3
    assert trend["nontrivial_component_mass_delta"] == 0.0
    assert "higher component count" in trend["agent_action"]


def test_summarize_history_detects_meaningful_resolution():
    summary = summarize_history(
        [
            _record([980, 20]),
            _record([900, 70, 30]),
        ]
    )

    trend = summary["fragmentation_trend"]
    assert trend["status"] == "meaningful_resolution"
    assert trend["largest_non_giant_component_pct"] == 0.07
    assert trend["nontrivial_component_mass_delta"] == 0.08


def test_summarize_history_detects_over_fragmentation():
    summary = summarize_history(
        [
            _record([980, 20], singleton_fraction=0.0),
            _record([900, 40, *([1] * 60)], singleton_fraction=0.06),
        ]
    )

    trend = summary["fragmentation_trend"]
    assert trend["status"] == "over_fragmentation"
    assert trend["singleton_fraction_delta"] == 0.06
