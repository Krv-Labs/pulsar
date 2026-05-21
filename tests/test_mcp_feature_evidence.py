import asyncio
import json
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd

from pulsar.mcp import server
from pulsar.mcp.interpreter import build_feature_evidence_index


def _make_disconnected_model(size_per_cluster: int = 3) -> SimpleNamespace:
    total = size_per_cluster * 2
    weighted = np.zeros((total, total), dtype=float)
    for start in (0, size_per_cluster):
        end = start + size_per_cluster
        for i in range(start, end):
            for j in range(i + 1, end):
                weighted[i, j] = weighted[j, i] = 0.8 + (0.05 * ((i + j) % 2))
    graph = nx.from_numpy_array((weighted > 0).astype(int))
    return SimpleNamespace(
        cosmic_graph=graph,
        weighted_adjacency=weighted,
        resolved_construction_threshold=0.0,
    )


def _make_singleton_tail_model() -> SimpleNamespace:
    total = 8
    weighted = np.zeros((total, total), dtype=float)
    for i in range(4):
        for j in range(i + 1, 4):
            weighted[i, j] = weighted[j, i] = 0.8
    graph = nx.from_numpy_array((weighted > 0).astype(int))
    return SimpleNamespace(
        cosmic_graph=graph,
        weighted_adjacency=weighted,
        resolved_construction_threshold=0.0,
    )


def _make_multifactor_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    cluster_a = 48
    cluster_b = 48
    data = pd.DataFrame(
        {
            "f1": np.concatenate(
                [rng.normal(3.2, 0.35, cluster_a), rng.normal(0.0, 0.35, cluster_b)]
            ),
            "f2": np.concatenate(
                [rng.normal(-2.4, 0.45, cluster_a), rng.normal(0.1, 0.45, cluster_b)]
            ),
            "f3": np.concatenate(
                [rng.normal(0.0, 0.15, cluster_a), rng.normal(0.0, 1.2, cluster_b)]
            ),
            "noise": rng.normal(0.0, 1.0, cluster_a + cluster_b),
            "segment": (["alpha"] * cluster_a) + (["beta"] * cluster_b),
        }
    )
    clusters = pd.Series(([0] * cluster_a) + ([1] * cluster_b), name="cluster")
    return data, clusters


def _make_small_server_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [3.1, 3.4, 2.9, 0.1, -0.2, 0.0],
            "f2": [-2.5, -2.1, -2.7, 0.2, 0.0, 0.4],
            "f3": [0.0, 0.1, -0.1, 1.8, -1.6, 1.4],
            "noise": [0.3, -0.1, 0.2, -0.4, 0.5, -0.2],
            "segment": ["alpha", "alpha", "alpha", "beta", "beta", "beta"],
        }
    )


def _make_singleton_tail_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [3.1, 3.4, 2.9, 3.2, -3.0, -2.0, 0.0, 2.0],
            "f2": [-2.5, -2.1, -2.7, -2.3, 1.0, 2.0, 3.0, 4.0],
            "segment": [
                "major",
                "major",
                "major",
                "major",
                "rare_a",
                "rare_b",
                "rare_c",
                "rare_d",
            ],
        }
    )


def _make_wide_server_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    cluster_a = 32
    cluster_b = 32
    columns = {
        f"signal_{idx:02d}": np.concatenate(
            [
                rng.normal(3.0 + (idx * 0.03), 0.25, cluster_a),
                rng.normal(0.0, 0.25, cluster_b),
            ]
        )
        for idx in range(36)
    }
    columns["segment"] = (["alpha"] * cluster_a) + (["beta"] * cluster_b)
    return pd.DataFrame(columns)


def test_build_feature_evidence_index_preserves_multifactor_signal():
    data, clusters = _make_multifactor_dataset()
    model = _make_disconnected_model(size_per_cluster=len(data) // 2)

    evidence = build_feature_evidence_index(model, data, clusters)
    cluster_zero = evidence.cluster_bundles[0]

    tiered_numeric = {
        row["column"]: row["signal_tier"]
        for row in cluster_zero["numeric"]
        if row["column"] in {"f1", "f2", "f3"}
    }
    top_numeric = [row["column"] for row in cluster_zero["numeric"][:4]]

    assert evidence.metadata["neighbor_contrast_enabled"] is True
    assert set(top_numeric) >= {"f1", "f2", "f3"}
    assert set(tiered_numeric) == {"f1", "f2", "f3"}
    assert all(tier != "noise" for tier in tiered_numeric.values())


def test_generate_cluster_dossier_supports_tiered_retrieval_without_payload_duplication():
    server._sessions.clear()
    session = server._get_session(None)
    session.model = _make_disconnected_model()
    session.data = _make_small_server_dataset()
    session.latest_run_id = "run_synthetic"

    summary_text = asyncio.run(
        server.generate_cluster_dossier(method="auto", detail="summary")
    )
    summary = json.loads(summary_text)
    full_text = asyncio.run(
        server.generate_cluster_dossier(method="auto", detail="full")
    )
    full = json.loads(full_text)
    cluster_profile = json.loads(asyncio.run(server.get_cluster_profile(0)))
    feature_payload = json.loads(
        asyncio.run(server.get_feature_signal(["f1", "segment=alpha"]))
    )
    matrix_payload = json.loads(asyncio.run(server.get_cluster_signal_matrix()))

    assert summary["status"] == "ok"
    assert summary["cluster_result"]["method_used"] in {
        "threshold_stability",
        "components",
    }
    assert summary["cluster_result"]["method_used"] != "spectral"
    assert summary["detail"] == "summary"
    assert "markdown_summary" not in summary
    assert len(summary_text) < len(full_text)
    assert full["detail"] == "full"
    # Summary mode must NOT carry the full tier dumps or signal_matrix
    assert "signal_matrix" not in summary, (
        "summary should expose only signal_matrix_summary"
    )
    assert "signal_matrix_summary" in summary
    assert "clusters_returned" in summary
    assert "numeric_global_ranking" not in summary
    assert "categorical_global_ranking" not in summary
    assert "numeric_global_ranking_preview" in summary
    assert "categorical_global_ranking_preview" in summary
    for cluster in summary["clusters"]:
        assert "numeric_features" not in cluster
        assert "categorical_features" not in cluster
        assert "numeric_tiers" not in cluster
        assert "categorical_tiers" not in cluster
        assert "central_rows" not in cluster
        assert "numeric_tier_counts" in cluster
        assert "feature_preview" in cluster
        assert cluster["features_previewed"] <= cluster["feature_preview_limit"]
    # Full mode still ships everything
    assert "signal_matrix" in full
    full_cluster = full["clusters"][0]
    assert "numeric_features" in full_cluster
    assert "categorical_features" in full_cluster
    assert "numeric_tiers" in full_cluster
    assert "central_rows" in full_cluster
    assert summary["recommended_next_tools"][0] == "get_cluster_profile"
    assert "cluster map" in summary["payload_guidance"]
    assert cluster_profile["cluster"]["cluster_id"] == 0
    assert cluster_profile["max_features"] == 16
    assert cluster_profile["cluster"]["feature_limit"] == 16
    assert cluster_profile["cluster"]["numeric_features"]
    assert feature_payload["signals"]["feature_names"] == ["f1", "segment=alpha"]
    assert feature_payload["signals"]["clusters"]
    assert feature_payload["signals"]["detail"] == "summary"
    assert "clusters_returned" in feature_payload["signals"]
    assert "clusters_omitted" in feature_payload["signals"]
    # Summary projection drops the heavyweight metrics
    sample_numeric = next(
        (
            row
            for cluster in feature_payload["signals"]["clusters"]
            for row in cluster["numeric_features"]
        ),
        None,
    )
    if sample_numeric is not None:
        assert "wasserstein_norm" not in sample_numeric
        assert "evidence_vector" not in sample_numeric
        assert "signal_tier" in sample_numeric
    full_feature_payload = json.loads(
        asyncio.run(
            server.get_feature_signal(
                ["f1", "segment=alpha"], detail="full", max_clusters=50
            )
        )
    )
    full_sample = next(
        (
            row
            for cluster in full_feature_payload["signals"]["clusters"]
            for row in cluster["numeric_features"]
        ),
        None,
    )
    if full_sample is not None:
        assert "wasserstein_norm" in full_sample
    assert matrix_payload["signal_matrix"]["numeric_rows"]
    assert "clusters_returned" in matrix_payload["signal_matrix"]


def test_get_cluster_profile_caps_wide_dataset_feature_output():
    server._sessions.clear()
    session = server._get_session(None)
    data = _make_wide_server_dataset()
    session.model = _make_disconnected_model(size_per_cluster=len(data) // 2)
    session.data = data
    session.latest_run_id = "run_wide_synthetic"

    asyncio.run(server.generate_cluster_dossier(method="auto", detail="summary"))
    default_capped = json.loads(asyncio.run(server.get_cluster_profile(0)))
    capped = json.loads(asyncio.run(server.get_cluster_profile(0, max_features=7)))
    uncapped = json.loads(asyncio.run(server.get_cluster_profile(0, max_features=200)))

    cluster = capped["cluster"]
    returned = cluster["numeric_features"] + cluster["categorical_features"]
    all_rows = (
        uncapped["cluster"]["numeric_features"]
        + uncapped["cluster"]["categorical_features"]
    )
    returned_keys = {(row.get("column"), row.get("value")) for row in returned}
    expected_keys = {
        (row.get("column"), row.get("value"))
        for row in sorted(
            all_rows,
            key=lambda row: -abs(float(row.get("aggregate_score", 0.0))),
        )[:7]
    }

    assert capped["max_features"] == 7
    assert default_capped["max_features"] == 16
    assert default_capped["cluster"]["features_returned"]["total"] <= 16
    assert cluster["feature_limit"] == 7
    assert cluster["features_returned"]["total"] == 7
    assert cluster["features_omitted"]["total"] > 0
    assert returned_keys == expected_keys


def test_generate_cluster_dossier_summary_caps_feature_preview():
    server._sessions.clear()
    session = server._get_session(None)
    data = _make_wide_server_dataset()
    session.model = _make_disconnected_model(size_per_cluster=len(data) // 2)
    session.data = data
    session.latest_run_id = "run_preview_cap"

    summary = json.loads(
        asyncio.run(
            server.generate_cluster_dossier(
                method="auto",
                detail="summary",
                feature_preview_limit=3,
            )
        )
    )

    assert summary["status"] == "ok"
    for cluster in summary["clusters"]:
        assert len(cluster["feature_preview"]) <= 3
        assert cluster["feature_preview_limit"] == 3
        assert "numeric_features" not in cluster
        assert "categorical_features" not in cluster


def test_generate_cluster_dossier_surfaces_singleton_heavy_readiness():
    server._sessions.clear()
    session = server._get_session(None)
    session.model = _make_singleton_tail_model()
    session.data = _make_singleton_tail_dataset()
    session.latest_run_id = "run_singleton_tail"

    summary = json.loads(
        asyncio.run(server.generate_cluster_dossier(method="components"))
    )

    assert summary["status"] == "ok"
    readiness = summary["cluster_result"]["interpretation_readiness"]
    fragmentation = summary["cluster_result"]["cluster_fragmentation"]
    assert readiness["status"] == "review_required"
    assert "SINGLETON_TAIL_DOMINATES_NON_GIANT_STRUCTURE" in readiness["reason_codes"]
    assert readiness["basis"].startswith("advisory")
    assert fragmentation["singleton_cluster_count"] == 4
    assert fragmentation["singleton_cluster_ratio"] == 0.8
    assert fragmentation["singleton_point_fraction"] == 0.5
    assert summary["clusters"]


def test_evidence_index_surfaces_stats_failures_metadata():
    """stats_failures metadata must always be present and structured."""
    data, clusters = _make_multifactor_dataset()
    model = _make_disconnected_model(size_per_cluster=len(data) // 2)
    evidence = build_feature_evidence_index(model, data, clusters)

    assert "stats_failures" in evidence.metadata
    stats_failures = evidence.metadata["stats_failures"]
    assert set(stats_failures.keys()) == {
        "numeric_column_level",
        "categorical_column_level",
        "numeric_row_level",
        "categorical_row_level",
    }
    assert stats_failures["numeric_column_level"] == {}
    assert stats_failures["categorical_column_level"] == {}
    assert stats_failures["numeric_row_level"] == []
    assert stats_failures["categorical_row_level"] == []


def test_evidence_index_captures_ks_failure_when_stats_call_raises(monkeypatch):
    """When scipy.stats raises ValueError, the failure must be recorded on the row."""
    from pulsar.mcp import interpreter

    call_count = {"n": 0}
    original_ks = interpreter.stats.ks_2samp

    def flaky_ks_2samp(a, b, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("synthetic failure for test")
        return original_ks(a, b, *args, **kwargs)

    monkeypatch.setattr(interpreter.stats, "ks_2samp", flaky_ks_2samp)

    data, clusters = _make_multifactor_dataset()
    model = _make_disconnected_model(size_per_cluster=len(data) // 2)
    evidence = build_feature_evidence_index(model, data, clusters)

    row_failures = evidence.metadata["stats_failures"]["numeric_row_level"]
    assert len(row_failures) == 1
    assert row_failures[0]["reasons"] == ["ks_2samp: synthetic failure for test"]
