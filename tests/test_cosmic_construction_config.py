"""cosmic_graph.construction toggle: default, accepted values, and validation."""

import pytest

from pulsar.config import load_config


def _cfg(**cosmic):
    return load_config(
        {
            "preprocessing": {},
            "sweep": {"ball_mapper": {"epsilon": {"values": [0.5]}}},
            "cosmic_graph": cosmic,
        }
    )


def test_construction_defaults_to_minhash():
    assert _cfg().cosmic_graph.construction == "minhash"


def test_construction_exact_accepted():
    assert _cfg(construction="exact").cosmic_graph.construction == "exact"


def test_construction_invalid_value_raises():
    with pytest.raises(ValueError, match="cosmic_graph.construction must be one of"):
        _cfg(construction="approximate")
