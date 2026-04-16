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
        resolved_threshold=0.0,
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
    assert cluster_profile["cluster"]["cluster_id"] == 0
    assert cluster_profile["cluster"]["numeric_features"]
    assert feature_payload["signals"]["feature_names"] == ["f1", "segment=alpha"]
    assert feature_payload["signals"]["clusters"]
    assert matrix_payload["signal_matrix"]["numeric_rows"]
