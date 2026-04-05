import asyncio
import json
import re

import numpy as np
import pandas as pd

from pulsar.mcp import server


def _write_dataset(tmp_path, rows: int = 30, cols: int = 6) -> str:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        rng.standard_normal((rows, cols)),
        columns=[f"f{i}" for i in range(cols)],
    )
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_validate_config_reports_schema_mismatches(tmp_path):
    csv_path = _write_dataset(tmp_path)
    bad_yaml = f"""dataset:
  path: {csv_path}
preprocessing:
  drop: [f0]
  impute:
    f1: median
sweep:
  n_cubes: [5]
"""

    payload = asyncio.run(server.validate_config(bad_yaml))
    report = json.loads(payload)

    assert report["status"] == "error"
    assert report["error_code"] == "YAML_SCHEMA_MISMATCH"
    issue_paths = {issue["path"] for issue in report["details"]["issues"]}
    assert "dataset" in issue_paths
    assert "preprocessing.drop" in issue_paths
    assert "preprocessing.impute.f1" in issue_paths
    assert "sweep.n_cubes" in issue_paths


def test_validate_config_rejects_fenced_yaml(tmp_path):
    csv_path = _write_dataset(tmp_path)
    payload = asyncio.run(
        server.validate_config(
            f"```yaml\nrun:\n  data: {csv_path}\noutput:\n  n_reps: 1\n```"
        )
    )
    report = json.loads(payload)

    assert report["status"] == "error"
    assert report["error_code"] == "YAML_NOT_RAW"


def test_ingest_dataset_and_create_config_round_trip(tmp_path):
    csv_path = _write_dataset(tmp_path)

    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    config_yaml = asyncio.run(server.create_config(dataset["dataset_id"], "demo_sweep"))
    report = json.loads(
        asyncio.run(server.validate_config(config_yaml, dataset["dataset_id"]))
    )

    assert dataset["dataset_id"].startswith("ds_")
    assert "name: demo_sweep" in config_yaml
    assert report["status"] == "ok"
    assert report["resolved_dataset_path"] == csv_path
    assert "normalized_config_yaml" in report


def test_ingest_dataset_classifies_sandbox_local_missing_path():
    payload = asyncio.run(server.ingest_dataset("/home/claude/missing.csv"))
    report = json.loads(payload)

    assert report["status"] == "error"
    assert report["error_code"] == "HOST_PATH_NOT_VISIBLE"
    assert report["details"]["path_context"]["looks_like_sandbox_path"] is True


def test_ingest_dataset_classifies_host_missing_path():
    payload = asyncio.run(server.ingest_dataset("/definitely/not/real.csv"))
    report = json.loads(payload)

    assert report["status"] == "error"
    assert report["error_code"] == "FILE_NOT_FOUND"
    assert report["details"]["path_context"]["looks_like_sandbox_path"] is False


def test_run_topological_sweep_with_dataset_id_persists_run_summary(tmp_path):
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    config_yaml = asyncio.run(server.create_config(dataset["dataset_id"], "run_compare"))

    refined_a = json.loads(
        asyncio.run(
            server.refine_config(
                config_yaml,
                {"pca_dims": [2], "epsilon_values": [0.5], "n_reps": 1},
            )
        )
    )["config_yaml"]
    refined_b = json.loads(
        asyncio.run(
            server.refine_config(
                config_yaml,
                {"pca_dims": [3], "epsilon_values": [0.75], "n_reps": 1},
            )
        )
    )["config_yaml"]

    result_a = asyncio.run(
        server.run_topological_sweep(config_yaml=refined_a, dataset_id=dataset["dataset_id"])
    )
    result_b = asyncio.run(
        server.run_topological_sweep(config_yaml=refined_b, dataset_id=dataset["dataset_id"])
    )

    run_a = re.search(r"Run ID: (run_[a-z0-9]+)", result_a).group(1)
    run_b = re.search(r"Run ID: (run_[a-z0-9]+)", result_b).group(1)

    skeleton = json.loads(asyncio.run(server.get_topological_skeleton(run_a)))
    comparison = asyncio.run(server.compare_sweeps(run_a, run_b))

    assert skeleton["run_id"] == run_a
    assert skeleton["dataset_id"] == dataset["dataset_id"]
    assert skeleton["graph"]["node_count"] > 0
    assert "Sweep Comparison" in comparison
    assert run_a in comparison
    assert run_b in comparison


def test_unknown_handles_return_stable_codes():
    dataset_report = json.loads(asyncio.run(server.create_config("ds_missing")))
    run_report = json.loads(asyncio.run(server.get_topological_skeleton("run_missing")))

    assert dataset_report["error_code"] == "DATASET_ID_UNKNOWN"
    assert run_report["error_code"] == "RUN_ID_UNKNOWN"


def test_run_topological_sweep_missing_config_path_returns_structured_error():
    report = json.loads(
        asyncio.run(server.run_topological_sweep(config_path="/home/claude/missing.yaml"))
    )

    assert report["error_code"] == "HOST_PATH_NOT_VISIBLE"


def test_run_topological_sweep_missing_host_config_path_returns_config_code():
    report = json.loads(
        asyncio.run(server.run_topological_sweep(config_path="/definitely/not/real.yaml"))
    )

    assert report["error_code"] == "CONFIG_FILE_NOT_VISIBLE"
