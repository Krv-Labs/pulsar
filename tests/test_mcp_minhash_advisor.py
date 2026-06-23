"""Tests for the MinHash signature-depth advisor (pure error math)."""

import math

import pytest

from pulsar.mcp.minhash_advisor import (
    DEFAULT_D,
    MASSIVE_N_THRESHOLD,
    ci_half_width,
    depth_for_epsilon,
    depth_profile,
    epsilon_at_confidence,
    massive_dataset_advisory,
    recommend_d,
    signature_memory_bytes,
    standard_error,
)


def test_hoeffding_inverse_consistency():
    """epsilon_at_confidence and depth_for_epsilon must be inverses."""
    for delta in (0.05, 0.1):
        for d in (64, 128, 256, 512, 738):
            eps = epsilon_at_confidence(d, delta)
            # depth_for_epsilon rounds up, so it returns <= d (d is sufficient).
            assert depth_for_epsilon(eps, delta) <= d + 1


def test_known_hoeffding_depths():
    """Textbook values: d≈738 for ε=0.05, d≈185 for ε=0.10 at 95%."""
    assert depth_for_epsilon(0.05, 0.05) == 738
    assert depth_for_epsilon(0.10, 0.05) == 185


def test_standard_error_and_ci():
    # Worst-case SE at W=0.5 is 1/(2*sqrt(d)).
    assert math.isclose(standard_error(0.5, 256), 1.0 / (2 * math.sqrt(256)))
    # CI half-width default is 1.96 * worst-case SE.
    assert math.isclose(ci_half_width(256), 1.96 / (2 * math.sqrt(256)))
    # Error shrinks as d grows.
    assert standard_error(0.5, 512) < standard_error(0.5, 256)


def test_signature_memory():
    # 256 * 1e6 * 4 bytes ≈ 1.02 GB.
    assert signature_memory_bytes(256, 1_000_000) == 256 * 1_000_000 * 4
    assert signature_memory_bytes(128, 1_000_000) < signature_memory_bytes(
        256, 1_000_000
    )


def test_depth_profile_fields():
    prof = depth_profile(256, 1_000_000)
    assert prof["d"] == 256
    assert prof["confidence"] == 0.95
    assert 0.05 < prof["ci95_half_width_worst"] < 0.07
    assert prof["signature_memory_human"].endswith("GB")


def test_recommend_d_with_memory_budget():
    """Largest standard depth fitting the budget is chosen."""
    n = 1_000_000
    budget = signature_memory_bytes(128, n)  # exactly fits d=128, not 256
    rec = recommend_d(n, memory_budget_bytes=budget)
    assert rec["recommended_d"] == 128
    assert rec["profile"]["signature_memory_bytes"] <= budget


def test_recommend_d_with_target_epsilon():
    rec = recommend_d(1000, target_eps=0.05)
    # Must meet the Hoeffding requirement (d >= 738).
    assert rec["recommended_d"] >= depth_for_epsilon(0.05)


def test_recommend_d_default_small_n():
    rec = recommend_d(1000)
    assert rec["recommended_d"] == DEFAULT_D
    assert "appropriate" in rec["note"]


def test_recommend_d_massive_n_lowers_depth():
    rec = recommend_d(MASSIVE_N_THRESHOLD + 1)
    assert rec["recommended_d"] < DEFAULT_D


def test_massive_advisory_gating():
    """Advisory only fires for massive n."""
    assert massive_dataset_advisory(1000) is None
    adv = massive_dataset_advisory(MASSIVE_N_THRESHOLD + 1)
    assert adv is not None
    assert adv["parameter"] == "cosmic_graph.minhash_d"
    assert adv["suggested"] < adv["current"]
    # Suggested depth uses less memory than current.
    assert (
        adv["suggested_profile"]["signature_memory_bytes"]
        < adv["current_profile"]["signature_memory_bytes"]
    )


def test_invalid_arguments():
    with pytest.raises(ValueError):
        epsilon_at_confidence(0)
    with pytest.raises(ValueError):
        depth_for_epsilon(0.0)
    with pytest.raises(ValueError):
        standard_error(0.5, 0)
