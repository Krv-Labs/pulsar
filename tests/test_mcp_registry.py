import json

import pytest

from pulsar.mcp import registry as registry_module
from pulsar.mcp.registry import MCPRegistry


def test_get_run_rejects_legacy_resolved_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(registry_module, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(registry_module, "_DATASETS_PATH", tmp_path / "datasets.json")
    monkeypatch.setattr(registry_module, "_DATASET_FILES_DIR", tmp_path / "datasets")
    monkeypatch.setattr(registry_module, "_UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(registry_module, "_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(registry_module, "_LOCK_PATH", tmp_path / ".registry.lock")

    registry = MCPRegistry()
    run_path = tmp_path / "runs" / "run_legacy.json"
    run_path.write_text(
        json.dumps(
            {
                "run_id": "run_legacy",
                "dataset_id": "ds_example",
                "config_yaml": "run:\n  name: legacy\n",
                "metrics": {},
                "resolved_threshold": 0.0,
                "graph_summary": {},
                "created_at": 0.0,
            }
        )
    )

    with pytest.raises(ValueError, match="resolved_threshold"):
        registry.get_run("run_legacy")
