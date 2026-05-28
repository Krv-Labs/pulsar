import asyncio
import json
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd

from pulsar.mcp.session import _sessions, _get_session
from pulsar.mcp.tools.clustering import (
    generate_cluster_dossier,
    get_cluster_profile,
    get_feature_signal,
    get_cluster_signal_matrix,
)
from pulsar.mcp.interpreter import FeatureEvidenceIndex, build_feature_evidence_index
from pulsar.mcp.interpreter import signal_matrix_payload


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
    _sessions.clear()
    session = _get_session(None)
    session.model = _make_disconnected_model()
    session.data = _make_small_server_dataset()
    session.latest_run_id = "run_synthetic"

    summary_text = asyncio.run(
        generate_cluster_dossier(
            method="auto", detail="summary", response_format="json"
        )
    )
    summary = json.loads(summary_text)
    full_text = asyncio.run(
        generate_cluster_dossier(method="auto", detail="full", response_format="json")
    )
    full = json.loads(full_text)
    cluster_profile = json.loads(
        asyncio.run(get_cluster_profile(0, response_format="json"))
    )
    feature_payload = json.loads(
        asyncio.run(get_feature_signal(["f1", "segment=alpha"], response_format="json"))
    )
    matrix_payload = json.loads(
        asyncio.run(get_cluster_signal_matrix(return_markdown=False))
    )

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
            get_feature_signal(
                ["f1", "segment=alpha"],
                detail="full",
                max_clusters=50,
                response_format="json",
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


def test_generate_cluster_dossier_defaults_to_compact_markdown_summary():
    _sessions.clear()
    session = _get_session(None)
    session.model = _make_disconnected_model()
    session.data = _make_small_server_dataset()
    session.latest_run_id = "run_markdown_default"

    markdown = asyncio.run(generate_cluster_dossier(method="auto"))

    assert "# Threshold Surface" in markdown
    assert "# Cluster Dossier Summary" in markdown
    assert "## Next Tools" in markdown
    assert "Central Representative Rows" not in markdown
    assert "Top Defining Numeric Features" not in markdown


def test_cluster_signal_matrix_tool_defaults_to_markdown():
    _sessions.clear()
    session = _get_session(None)
    session.model = _make_disconnected_model()
    session.data = _make_small_server_dataset()
    session.latest_run_id = "run_matrix_markdown_default"

    asyncio.run(
        generate_cluster_dossier(
            method="auto", detail="summary", response_format="json"
        )
    )
    markdown = asyncio.run(get_cluster_signal_matrix())

    assert "## Topological Signal Matrix" in markdown
    assert "### Numeric Feature Matrix" in markdown
    assert not markdown.lstrip().startswith("{")


def test_profile_and_feature_signal_tools_default_to_markdown():
    _sessions.clear()
    session = _get_session(None)
    session.model = _make_disconnected_model()
    session.data = _make_small_server_dataset()
    session.latest_run_id = "run_followup_markdown_defaults"

    asyncio.run(
        generate_cluster_dossier(
            method="auto", detail="summary", response_format="json"
        )
    )

    profile_markdown = asyncio.run(get_cluster_profile(0))
    feature_markdown = asyncio.run(get_feature_signal(["f1"]))

    assert "# Cluster Profile" in profile_markdown
    assert "### Next Tools" in profile_markdown
    assert not profile_markdown.lstrip().startswith("{")
    assert "# Feature Signal Summary" in feature_markdown
    assert "## Cluster" in feature_markdown
    assert not feature_markdown.lstrip().startswith("{")


def test_signal_matrix_caps_wide_feature_columns():
    _sessions.clear()
    session = _get_session(None)
    data = _make_wide_server_dataset()
    session.model = _make_disconnected_model(size_per_cluster=len(data) // 2)
    session.data = data
    session.latest_run_id = "run_matrix_feature_cap"

    asyncio.run(
        generate_cluster_dossier(
            method="auto", detail="summary", response_format="json"
        )
    )
    payload = json.loads(asyncio.run(get_cluster_signal_matrix(return_markdown=False)))
    matrix = payload["signal_matrix"]

    assert len(matrix["numeric_columns"]) <= 10
    assert matrix["numeric_features_omitted"] > 0


def test_get_cluster_profile_caps_wide_dataset_feature_output():
    _sessions.clear()
    session = _get_session(None)
    data = _make_wide_server_dataset()
    session.model = _make_disconnected_model(size_per_cluster=len(data) // 2)
    session.data = data
    session.latest_run_id = "run_wide_synthetic"

    asyncio.run(
        generate_cluster_dossier(
            method="auto", detail="summary", response_format="json"
        )
    )
    default_capped = json.loads(
        asyncio.run(get_cluster_profile(0, response_format="json"))
    )
    capped = json.loads(
        asyncio.run(get_cluster_profile(0, max_features=7, response_format="json"))
    )
    uncapped = json.loads(
        asyncio.run(get_cluster_profile(0, max_features=200, response_format="json"))
    )

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


def test_signal_matrix_payload_uses_real_assembled_cell_values():
    data, clusters = _make_multifactor_dataset()
    model = _make_disconnected_model(size_per_cluster=len(data) // 2)
    evidence = build_feature_evidence_index(model, data, clusters)

    payload = signal_matrix_payload(evidence, max_clusters=2, return_markdown=False)

    numeric_values = [
        value for row in payload["numeric_rows"] for value in row["values"].values()
    ]
    categorical_values = [
        value for row in payload["categorical_rows"] for value in row["values"].values()
    ]

    assert any(abs(value) > 0 for value in numeric_values)
    assert categorical_values
    assert all(value is not None for value in categorical_values)


def test_signal_matrix_cluster_cap_ranks_categorical_signal():
    evidence = FeatureEvidenceIndex(
        cluster_bundles={
            0: {
                "numeric": [{"column": "weak_numeric", "aggregate_score": 0.1}],
                "categorical": [],
            },
            1: {
                "numeric": [],
                "categorical": [
                    {
                        "column": "segment",
                        "value": "high_signal",
                        "aggregate_score": 10.0,
                    }
                ],
            },
        },
        numeric_global_ranking=[],
        categorical_global_ranking=[],
        signal_matrix={
            "numeric_columns": ["weak_numeric"],
            "categorical_values": [
                {"column": "segment", "value": "high_signal"},
            ],
            "numeric_rows": [
                {
                    "cluster_id": 0,
                    "size": 10,
                    "values": {
                        "weak_numeric": {
                            "aggregate_score": 0.1,
                            "signal_tier": "core",
                        }
                    },
                }
            ],
            "categorical_rows": [
                {
                    "cluster_id": 1,
                    "size": 10,
                    "values": {
                        "segment=high_signal": {
                            "aggregate_score": 10.0,
                            "signal_tier": "core",
                        }
                    },
                }
            ],
        },
        metadata={},
    )

    payload = signal_matrix_payload(evidence, max_clusters=1, return_markdown=False)

    assert payload["clusters_returned"] == 1
    assert payload["clusters_omitted"] == 1
    assert [row["cluster_id"] for row in payload["categorical_rows"]] == [1]
    assert payload["numeric_rows"] == []


def test_generate_cluster_dossier_summary_caps_feature_preview():
    _sessions.clear()
    session = _get_session(None)
    data = _make_wide_server_dataset()
    session.model = _make_disconnected_model(size_per_cluster=len(data) // 2)
    session.data = data
    session.latest_run_id = "run_preview_cap"

    summary = json.loads(
        asyncio.run(
            generate_cluster_dossier(
                method="auto",
                detail="summary",
                feature_preview_limit=3,
                response_format="json",
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
    _sessions.clear()
    session = _get_session(None)
    session.model = _make_singleton_tail_model()
    session.data = _make_singleton_tail_dataset()
    session.latest_run_id = "run_singleton_tail"

    summary = json.loads(
        asyncio.run(
            generate_cluster_dossier(method="components", response_format="json")
        )
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
    """When the KS implementation raises ValueError, the failure must be recorded.

    Monkeypatches ``interpreter._ks_two_sample_stat`` — the wrapped seam that
    replaced the direct ``scipy.stats.ks_2samp`` call when KS was vectorized.
    The emitted failure label (``"ks_2samp: ..."``) is preserved so the
    downstream ``stats_failures`` contract is unchanged.
    """
    from pulsar.mcp import interpreter

    call_count = {"n": 0}
    original_ks = interpreter._ks_two_sample_stat

    def flaky_ks(a, b, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("synthetic failure for test")
        return original_ks(a, b, *args, **kwargs)

    monkeypatch.setattr(interpreter, "_ks_two_sample_stat", flaky_ks)

    data, clusters = _make_multifactor_dataset()
    model = _make_disconnected_model(size_per_cluster=len(data) // 2)
    evidence = build_feature_evidence_index(model, data, clusters)

    row_failures = evidence.metadata["stats_failures"]["numeric_row_level"]
    assert len(row_failures) == 1
    assert row_failures[0]["reasons"] == ["ks_2samp: synthetic failure for test"]


def test_signal_matrix_software3_markdown_and_telemetry():
    """Verify that signal_matrix_payload produces beautiful Markdown and rich omitted telemetry."""
    evidence = FeatureEvidenceIndex(
        cluster_bundles={
            0: {
                "numeric": [{"column": "f1", "aggregate_score": 1.2}],
                "categorical": [],
            },
            1: {
                "numeric": [],
                "categorical": [
                    {"column": "cat1", "value": "v1", "aggregate_score": 3.4}
                ],
            },
            2: {
                "numeric": [{"column": "f1", "aggregate_score": 0.5}],
                "categorical": [],
            },
        },
        numeric_global_ranking=[],
        categorical_global_ranking=[],
        signal_matrix={
            "numeric_columns": ["f1"],
            "categorical_values": [{"column": "cat1", "value": "v1"}],
            "numeric_rows": [
                {
                    "cluster_id": 0,
                    "values": {"f1": {"value": 1.2, "signal_tier": "core"}},
                },
                {
                    "cluster_id": 2,
                    "values": {"f1": {"value": 0.5, "signal_tier": "core"}},
                },
            ],
            "categorical_rows": [
                {
                    "cluster_id": 1,
                    "values": {"cat1=v1": {"value": "v1", "signal_tier": "core"}},
                }
            ],
        },
        metadata={},
    )

    payload = signal_matrix_payload(evidence, max_clusters=2, return_markdown=True)

    assert payload["clusters_returned"] == 2
    assert payload["clusters_omitted"] == 1

    # Assert omitted telemetry has maximum aggregate signal scores
    omitted = payload["omitted_clusters"]
    assert len(omitted) == 1
    assert omitted[0]["cluster_id"] == 2
    assert omitted[0]["max_signal"] == 0.5

    # Assert Markdown report starts and includes expected markdown markers and tables
    report = payload["markdown_report"]
    assert "## Topological Signal Matrix" in report
    assert "### Telemetry" in report
    assert "### Numeric Feature Matrix" in report
    assert "### Categorical Feature Matrix" in report
    assert "| Cluster ID | f1 |" in report
    assert "| Cluster ID | cat1=v1 |" in report
    assert "Cluster 2" in report


def test_signal_matrix_defensive_safe_casting():
    """Verify that signal_matrix_payload handles malformed or missing aggregate scores without crashing."""
    evidence = FeatureEvidenceIndex(
        cluster_bundles={
            0: {
                "numeric": [
                    {"column": "f1", "aggregate_score": "N/A"},  # invalid string float
                    {"column": "f2", "aggregate_score": None},  # None float
                ],
                "categorical": [],
            }
        },
        numeric_global_ranking=[],
        categorical_global_ranking=[],
        signal_matrix={
            "numeric_columns": ["f1", "f2"],
            "categorical_values": [],
            "numeric_rows": [
                {
                    "cluster_id": 0,
                    "values": {
                        "f1": {"value": 1.0, "signal_tier": "core"},
                        "f2": {"value": 2.0, "signal_tier": "core"},
                    },
                }
            ],
            "categorical_rows": [],
        },
        metadata={},
    )

    # Should run successfully and default invalid scores to 0.0 without throwing ValueError
    payload = signal_matrix_payload(evidence, max_clusters=1, return_markdown=False)
    assert payload["clusters_returned"] == 1
    assert payload["clusters_omitted"] == 0
