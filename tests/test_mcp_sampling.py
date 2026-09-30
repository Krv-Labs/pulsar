"""Focused checks for deterministic, dtype-preserving entity sampling."""

import asyncio
import json

import pandas as pd
import pytest

from pulsar.mcp import registry as registry_module
from pulsar.mcp.session import _get_session, _resolve_dataset_path, _sessions
from pulsar.mcp.tools.ingestion import (
    ingest_dataset,
    sample_longitudinal_entities,
)


@pytest.fixture(autouse=True)
def isolate_registry(tmp_path, monkeypatch):
    root = tmp_path / "registry"
    paths = {
        "_CACHE_DIR": root,
        "_DATASETS_PATH": root / "datasets.json",
        "_DATASET_FILES_DIR": root / "datasets",
        "_UPLOADS_DIR": root / "uploads",
        "_RUNS_DIR": root / "runs",
        "_CLUSTER_ASSIGNMENTS_DIR": root / "cluster_assignments",
        "_LOCK_PATH": root / ".registry.lock",
    }
    for name, path in paths.items():
        monkeypatch.setattr(registry_module, name, path)
    for name in (
        "_CACHE_DIR",
        "_DATASET_FILES_DIR",
        "_UPLOADS_DIR",
        "_RUNS_DIR",
        "_CLUSTER_ASSIGNMENTS_DIR",
    ):
        getattr(registry_module, name).mkdir(parents=True, exist_ok=True)
    registry_module._LOCK_PATH.touch()
    _sessions.clear()
    yield
    _sessions.clear()


def test_sampling_is_deterministic_preserves_histories_and_parquet_dtypes(tmp_path):
    rows = []
    for entity, times in (
        ("0007", [0, 1, 2, 3]),
        ("0012", [0, 1, 3]),
        ("0025", [0, 0, 0, 1]),
        ("0040", [0, 1, 2]),
    ):
        for i, time in enumerate(times):
            rows.append(
                {
                    "patient_id": entity,
                    "hour": time,
                    "feature": float(i),
                    "count": i,
                }
            )
    source = pd.DataFrame(rows).astype(
        {
            "patient_id": "string",
            "hour": "int64",
            "feature": "float64",
            "count": "int64",
        }
    )
    source_path = tmp_path / "source.parquet"
    source.to_parquet(source_path, index=False)
    dataset = json.loads(asyncio.run(ingest_dataset(str(source_path))))

    session = _get_session(None)
    session.data = pd.DataFrame({"sentinel": [1]})
    session.model = object()
    session.clusters = pd.Series([8])
    session.dataset_id = "existing-dataset"
    static_state = (session.data, session.model, session.clusters, session.dataset_id)

    results = [
        json.loads(
            asyncio.run(
                sample_longitudinal_entities(
                    dataset["dataset_id"],
                    "patient_id",
                    max_entities=2,
                    min_observations=3,
                    time_column="hour",
                    min_time_points=3,
                )
            )
        )
        for _ in range(2)
    ]
    frames = [
        pd.read_parquet(_resolve_dataset_path(result["dataset_id"]))
        for result in results
    ]

    pd.testing.assert_frame_equal(frames[0], frames[1])
    assert results[0]["sampled_entities"] == 2
    assert len(set(frames[0]["patient_id"])) <= results[0]["max_entities"]
    assert "0025" not in set(frames[0]["patient_id"])
    assert set(frames[0]["patient_id"]).issubset({"0007", "0012", "0040"})
    assert frames[0]["patient_id"].dtype == source["patient_id"].dtype
    assert frames[0]["hour"].dtype == source["hour"].dtype
    assert frames[0]["feature"].dtype == source["feature"].dtype
    assert frames[0]["count"].dtype == source["count"].dtype
    for entity, history in frames[0].groupby("patient_id"):
        expected = source[source["patient_id"] == entity]
        pd.testing.assert_frame_equal(
            history.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=True,
        )
    assert session.data is static_state[0]
    assert session.model is static_state[1]
    assert session.clusters is static_state[2]
    assert session.dataset_id == static_state[3]
    assert not list(registry_module.registry.cache_dir.glob("sample_*.parquet"))


def test_min_time_points_requires_existing_time_column(tmp_path):
    source_path = tmp_path / "source.csv"
    pd.DataFrame(
        {"patient_id": ["0001", "0001", "0001"], "feature": [1.0, 2.0, 3.0]}
    ).to_csv(source_path, index=False)
    dataset = json.loads(asyncio.run(ingest_dataset(str(source_path))))

    result = json.loads(
        asyncio.run(
            sample_longitudinal_entities(
                dataset["dataset_id"], "patient_id", min_time_points=2
            )
        )
    )

    assert result["error_code"] == "INVALID_SAMPLE_SIZE"

    result = json.loads(
        asyncio.run(
            sample_longitudinal_entities(
                dataset["dataset_id"],
                "patient_id",
                time_column="missing",
                min_time_points=2,
            )
        )
    )

    assert result["error_code"] == "TIME_COLUMN_UNKNOWN"
