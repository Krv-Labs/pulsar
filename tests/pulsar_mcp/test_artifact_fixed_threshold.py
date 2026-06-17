"""Regression coverage for fixed construction-threshold artifact dumps."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


class MemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self.get_keys: list[str] = []

    def put(self, key: str, data: bytes) -> None:
        self._data[key] = data

    def get(self, key: str) -> bytes:
        self.get_keys.append(key)
        return self._data[key]


class FixedThresholdModel:
    def __init__(self) -> None:
        self._weighted_adjacency = np.array(
            [
                [0.0, 0.75, 0.0],
                [0.75, 0.0, 0.25],
                [0.0, 0.25, 0.0],
            ],
            dtype=np.float64,
        )
        import networkx as nx

        self._cosmic_graph = nx.Graph()
        self._cosmic_graph.add_nodes_from([0, 1, 2])
        self._cosmic_graph.add_edge(0, 1, weight=0.75)
        self._cosmic_graph.add_edge(1, 2, weight=0.25)
        self._embeddings = [np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]], dtype=np.float32)]
        self._preprocessed_data = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        self._data = self._preprocessed_data.copy()
        self._stability_result = None
        self._resolved_construction_threshold = 0.5
        self._ball_maps = []

    @property
    def resolved_construction_threshold(self) -> float:
        return self._resolved_construction_threshold


def test_dump_artifact_handles_fixed_construction_threshold_without_stability_result():
    from pulsar.artifacts import dump_artifact, load_artifact

    store = MemoryStore()
    artifact = json.loads(
        json.dumps(
            dump_artifact(
                FixedThresholdModel(),
                dataset_id="ds-fixed",
                config_hash="cfg-fixed",
                prefix="u/ds-fixed/cfg-fixed",
                store=store,
            )
        )
    )

    assert artifact["stabilityCurve"] is None
    assert artifact["resolvedConstructionThreshold"] == 0.5
    view = load_artifact(artifact, store)
    assert view.resolved_construction_threshold == 0.5


def test_load_artifact_defers_blob_reads_until_properties_are_accessed():
    from pulsar.artifacts import dump_artifact, load_artifact

    store = MemoryStore()
    artifact = json.loads(
        json.dumps(
            dump_artifact(
                FixedThresholdModel(),
                dataset_id="ds-fixed",
                config_hash="cfg-fixed",
                prefix="u/ds-fixed/cfg-fixed",
                store=store,
            )
        )
    )

    view = load_artifact(artifact, store)
    assert store.get_keys == []
    assert view.dataset_id == "ds-fixed"
    assert view.config_hash == "cfg-fixed"
    assert view.resolved_construction_threshold == 0.5
    assert store.get_keys == []

    assert view.cosmic_graph.number_of_edges() == 2
    assert np.allclose(view.weighted_adjacency, FixedThresholdModel()._weighted_adjacency)
    assert store.get_keys == []

    assert list(view.data.columns) == ["x"]
    assert store.get_keys == ["u/ds-fixed/cfg-fixed/data.parquet"]

    assert list(view.preprocessed_data.columns) == ["x"]
    assert store.get_keys == [
        "u/ds-fixed/cfg-fixed/data.parquet",
        "u/ds-fixed/cfg-fixed/preprocessed.parquet",
    ]

    assert len(view._embeddings) == 1
    assert store.get_keys == [
        "u/ds-fixed/cfg-fixed/data.parquet",
        "u/ds-fixed/cfg-fixed/preprocessed.parquet",
        "u/ds-fixed/cfg-fixed/embeddings/emb_0.npy",
    ]
