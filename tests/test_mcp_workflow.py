import asyncio
import base64
import json

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


def _dataset_csv_content(rows: int = 30, cols: int = 6) -> str:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        rng.standard_normal((rows, cols)),
        columns=[f"f{i}" for i in range(cols)],
    )
    return df.to_csv(index=False)


def _extract_config_yaml(create_config_response: str) -> str:
    """Extract config_yaml from create_config's JSON response."""
    payload = json.loads(create_config_response)
    return payload["config_yaml"]


def _normalize_config_data_path(config_yaml: str) -> str:
    lines = []
    for line in config_yaml.splitlines():
        if line.startswith("  data: "):
            lines.append("  data: <normalized>")
        else:
            lines.append(line)
    return "\n".join(lines)


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
    create_response = json.loads(
        asyncio.run(server.create_config(dataset["dataset_id"], "demo_sweep"))
    )
    config_yaml = create_response["config_yaml"]
    report = json.loads(
        asyncio.run(server.validate_config(config_yaml, dataset["dataset_id"]))
    )

    assert dataset["dataset_id"].startswith("ds_")
    assert create_response["status"] == "ok"
    assert create_response["calibration_space"] in ("processed", "raw")
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
    assert "ingest_dataset_base64" in report["agent_action"]


def test_ingest_dataset_classifies_host_missing_path():
    payload = asyncio.run(server.ingest_dataset("/definitely/not/real.csv"))
    report = json.loads(payload)

    assert report["status"] == "error"
    assert report["error_code"] == "FILE_NOT_FOUND"
    assert report["details"]["path_context"]["looks_like_sandbox_path"] is False


def test_run_topological_sweep_with_dataset_id_persists_run_summary(tmp_path):
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(server.create_config(dataset["dataset_id"], "run_compare"))
    )

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

    result_a = json.loads(
        asyncio.run(
            server.run_topological_sweep(
                config_yaml=refined_a, dataset_id=dataset["dataset_id"]
            )
        )
    )
    result_b = json.loads(
        asyncio.run(
            server.run_topological_sweep(
                config_yaml=refined_b, dataset_id=dataset["dataset_id"]
            )
        )
    )

    assert result_a["status"] == "ok"
    assert result_b["status"] == "ok"
    run_a = result_a["run_id"]
    run_b = result_b["run_id"]

    skeleton = json.loads(asyncio.run(server.get_topological_skeleton(run_a)))
    comparison = asyncio.run(server.compare_sweeps(run_a, run_b))

    assert skeleton["run_id"] == run_a
    assert skeleton["dataset_id"] == dataset["dataset_id"]
    assert skeleton["graph"]["node_count"] > 0
    assert "Sweep Comparison" in comparison
    assert run_a in comparison
    assert run_b in comparison


def test_ingest_dataset_content_round_trip():
    dataset = json.loads(
        asyncio.run(
            server.ingest_dataset_content(
                "uploaded.csv",
                _dataset_csv_content(),
            )
        )
    )
    config_yaml = _extract_config_yaml(
        asyncio.run(server.create_config(dataset["dataset_id"], "content_sweep"))
    )
    report = json.loads(
        asyncio.run(server.validate_config(config_yaml, dataset["dataset_id"]))
    )

    assert dataset["dataset_id"].startswith("ds_")
    assert dataset["source"] == "content"
    assert dataset["name"] == "uploaded.csv"
    assert report["status"] == "ok"


def test_ingest_dataset_base64_round_trip():
    content = _dataset_csv_content().encode("utf-8")
    dataset = json.loads(
        asyncio.run(
            server.ingest_dataset_base64(
                "uploaded.csv",
                base64.b64encode(content).decode("ascii"),
            )
        )
    )
    config_yaml = _extract_config_yaml(
        asyncio.run(server.create_config(dataset["dataset_id"], "b64_sweep"))
    )
    report = json.loads(
        asyncio.run(server.validate_config(config_yaml, dataset["dataset_id"]))
    )

    assert dataset["dataset_id"].startswith("ds_")
    assert dataset["source"] == "base64"
    assert report["status"] == "ok"


def test_ingest_modes_produce_equivalent_downstream_configs(tmp_path):
    csv_path = _write_dataset(tmp_path)
    content = _dataset_csv_content().encode("utf-8")

    by_path = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    by_base64 = json.loads(
        asyncio.run(
            server.ingest_dataset_base64(
                "uploaded.csv",
                base64.b64encode(content).decode("ascii"),
            )
        )
    )
    upload = json.loads(
        asyncio.run(server.begin_dataset_upload("uploaded.csv", media_type="text/csv"))
    )
    chunk = base64.b64encode(content).decode("ascii")
    asyncio.run(server.append_dataset_chunk(upload["upload_id"], chunk))
    by_upload = json.loads(
        asyncio.run(server.finalize_dataset_upload(upload["upload_id"]))
    )

    path_cfg = _extract_config_yaml(
        asyncio.run(server.create_config(by_path["dataset_id"], "equiv"))
    )
    base64_cfg = _extract_config_yaml(
        asyncio.run(server.create_config(by_base64["dataset_id"], "equiv"))
    )
    upload_cfg = _extract_config_yaml(
        asyncio.run(server.create_config(by_upload["dataset_id"], "equiv"))
    )

    assert "name: equiv" in path_cfg
    assert (
        _normalize_config_data_path(path_cfg)
        == _normalize_config_data_path(base64_cfg)
        == _normalize_config_data_path(upload_cfg)
    )


def test_staged_dataset_upload_round_trip():
    upload = json.loads(
        asyncio.run(server.begin_dataset_upload("chunked.csv", media_type="text/csv"))
    )
    content = _dataset_csv_content()
    midpoint = len(content) // 2
    chunk_a = base64.b64encode(content[:midpoint].encode("utf-8")).decode("ascii")
    chunk_b = base64.b64encode(content[midpoint:].encode("utf-8")).decode("ascii")

    append_a = json.loads(
        asyncio.run(server.append_dataset_chunk(upload["upload_id"], chunk_a))
    )
    append_b = json.loads(
        asyncio.run(server.append_dataset_chunk(upload["upload_id"], chunk_b))
    )
    dataset = json.loads(
        asyncio.run(server.finalize_dataset_upload(upload["upload_id"]))
    )
    report = json.loads(
        asyncio.run(
            server.validate_config(
                _extract_config_yaml(
                    asyncio.run(server.create_config(dataset["dataset_id"], "chunked"))
                ),
                dataset["dataset_id"],
            )
        )
    )

    assert append_a["bytes_received"] > 0
    assert append_b["bytes_received"] > append_a["bytes_received"]
    assert dataset["source"] == "upload"
    assert report["status"] == "ok"


def test_upload_lifecycle_misuse_returns_stable_codes():
    upload = json.loads(
        asyncio.run(server.begin_dataset_upload("misuse.csv", media_type="text/csv"))
    )
    chunk = base64.b64encode(b"f0,f1\n1,2\n").decode("ascii")
    asyncio.run(server.append_dataset_chunk(upload["upload_id"], chunk))
    asyncio.run(server.finalize_dataset_upload(upload["upload_id"]))

    append_after = json.loads(
        asyncio.run(server.append_dataset_chunk(upload["upload_id"], chunk))
    )
    finalize_again = json.loads(
        asyncio.run(server.finalize_dataset_upload(upload["upload_id"]))
    )

    assert append_after["error_code"] == "UPLOAD_ID_UNKNOWN"
    assert finalize_again["error_code"] == "UPLOAD_ID_UNKNOWN"


def test_ingest_dataset_content_handles_bom_and_windows_newlines():
    content = "\ufefff0,f1\r\n1.0,2.0\r\n3.0,4.0\r\n"
    dataset = json.loads(
        asyncio.run(server.ingest_dataset_content("windows.csv", content))
    )
    result = json.loads(
        asyncio.run(server.characterize_dataset(dataset_id=dataset["dataset_id"]))
    )

    assert result["n_samples"] == 2
    assert result["n_features"] == 2


def test_unknown_handles_return_stable_codes():
    dataset_report = json.loads(asyncio.run(server.create_config("ds_missing")))
    run_report = json.loads(asyncio.run(server.get_topological_skeleton("run_missing")))
    upload_report = json.loads(
        asyncio.run(server.finalize_dataset_upload("upload_missing"))
    )

    assert dataset_report["error_code"] == "DATASET_ID_UNKNOWN"
    assert run_report["error_code"] == "RUN_ID_UNKNOWN"
    assert upload_report["error_code"] == "UPLOAD_ID_UNKNOWN"


def test_ingest_dataset_base64_rejects_bad_media_type():
    report = json.loads(
        asyncio.run(
            server.ingest_dataset_base64(
                "uploaded.csv",
                base64.b64encode(b"f0,f1\n1,2\n").decode("ascii"),
                media_type="application/vnd.ms-excel",
            )
        )
    )

    assert report["error_code"] == "UPLOAD_MEDIA_TYPE_UNSUPPORTED"


def test_ingest_dataset_base64_rejects_invalid_base64():
    report = json.loads(
        asyncio.run(
            server.ingest_dataset_base64(
                "uploaded.csv",
                "%%%not-base64%%%",
            )
        )
    )

    assert report["error_code"] == "UPLOAD_DECODE_FAILED"


def test_append_dataset_chunk_rejects_invalid_base64():
    upload = json.loads(
        asyncio.run(server.begin_dataset_upload("broken.csv", media_type="text/csv"))
    )
    report = json.loads(
        asyncio.run(
            server.append_dataset_chunk(
                upload["upload_id"],
                "%%%not-base64%%%",
            )
        )
    )

    assert report["error_code"] == "UPLOAD_DECODE_FAILED"


def test_run_topological_sweep_missing_config_path_returns_structured_error():
    report = json.loads(
        asyncio.run(
            server.run_topological_sweep(config_path="/home/claude/missing.yaml")
        )
    )

    assert report["error_code"] == "HOST_PATH_NOT_VISIBLE"


def test_run_topological_sweep_missing_host_config_path_returns_config_code():
    report = json.loads(
        asyncio.run(
            server.run_topological_sweep(config_path="/definitely/not/real.yaml")
        )
    )

    assert report["error_code"] == "CONFIG_FILE_NOT_VISIBLE"


# ---------------------------------------------------------------------------
# Processed-space calibration tests
# ---------------------------------------------------------------------------


def _write_mixed_dataset(tmp_path):
    """Dataset with numeric + categorical columns to test processed-space calibration."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 15, 100),
            "weight": rng.normal(70, 10, 100),
            "score": rng.standard_normal(100),
            "bp": rng.normal(120, 20, 100),
            "gender": rng.choice(["M", "F"], 100),
            "region": rng.choice(["North", "South", "East", "West"], 100),
        }
    )
    path = tmp_path / "mixed.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_create_config_returns_processed_calibration(tmp_path):
    """create_config should calibrate in processed space and report it."""
    csv_path = _write_mixed_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))

    response = json.loads(
        asyncio.run(server.create_config(dataset["dataset_id"], "mixed_test"))
    )

    assert response["status"] == "ok"
    assert response["calibration_space"] == "processed"
    assert "processed_feature_count" in response
    # One-hot expansion should increase feature count beyond raw numerics
    assert response["processed_feature_count"] > 4  # 4 raw numerics
    assert response["raw_to_processed_expansion_ratio"] > 1.0
    assert "calibration_note" in response

    # knn_distance_percentiles should be present and ordered
    pctiles = response["knn_distance_percentiles"]
    assert pctiles["p5"] <= pctiles["p25"] <= pctiles["p50"]
    assert pctiles["p50"] <= pctiles["p75"] <= pctiles["p95"]


def test_create_config_generates_wide_grid(tmp_path):
    """create_config should generate wide grids (multiple seeds, many epsilon steps)."""
    import yaml

    csv_path = _write_mixed_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    response = json.loads(asyncio.run(server.create_config(dataset["dataset_id"])))

    cfg = yaml.safe_load(response["config_yaml"])
    pca_dims = cfg["sweep"]["pca"]["dimensions"]["values"]
    pca_seeds = cfg["sweep"]["pca"]["seed"]["values"]
    eps = cfg["sweep"]["ball_mapper"]["epsilon"]["range"]

    # Should have multiple PCA dims for multi-scale aggregation
    assert len(pca_dims) >= 4
    # Should have multiple seeds
    assert len(pca_seeds) >= 2
    # Should have enough epsilon steps
    assert eps["steps"] >= 20


def test_create_config_numeric_only_still_works(tmp_path):
    """create_config on pure numeric data should still work (no expansion)."""
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))

    response = json.loads(
        asyncio.run(server.create_config(dataset["dataset_id"], "numeric_only"))
    )

    assert response["status"] == "ok"
    assert response["calibration_space"] == "processed"
    assert response["raw_to_processed_expansion_ratio"] == 1.0


# ---------------------------------------------------------------------------
# refine_config unknown key rejection
# ---------------------------------------------------------------------------


def test_refine_config_rejects_unknown_keys(tmp_path):
    """refine_config should error on unknown override keys."""
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(server.create_config(dataset["dataset_id"]))
    )

    result = json.loads(
        asyncio.run(
            server.refine_config(
                config_yaml,
                {"pca_dims": [2, 3], "bogus_key": True, "another_bad": 42},
            )
        )
    )

    assert result["status"] == "error"
    assert result["error_code"] == "UNKNOWN_OVERRIDE_KEY"
    assert "another_bad" in result["reason"]
    assert "bogus_key" in result["reason"]


def test_refine_config_returns_diff(tmp_path):
    """refine_config should return a diff of what changed."""
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(server.create_config(dataset["dataset_id"]))
    )

    result = json.loads(
        asyncio.run(
            server.refine_config(config_yaml, {"pca_dims": [2, 5], "n_reps": 2})
        )
    )

    assert result["status"] == "ok"
    assert "diff" in result
    assert len(result["diff"]) >= 2  # pca_dims and n_reps changed
    diff_paths = {d["path"] for d in result["diff"]}
    assert "sweep.pca.dimensions.values" in diff_paths
    assert "output.n_reps" in diff_paths


# ---------------------------------------------------------------------------
# JSON output structure tests
# ---------------------------------------------------------------------------


def test_run_topological_sweep_returns_json(tmp_path):
    """run_topological_sweep should return structured JSON, not Markdown."""
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(server.create_config(dataset["dataset_id"]))
    )
    refined = json.loads(
        asyncio.run(
            server.refine_config(
                config_yaml, {"pca_dims": [2], "epsilon_values": [0.5], "n_reps": 1}
            )
        )
    )["config_yaml"]

    result = json.loads(
        asyncio.run(
            server.run_topological_sweep(
                config_yaml=refined, dataset_id=dataset["dataset_id"]
            )
        )
    )

    assert result["status"] == "ok"
    assert "run_id" in result
    assert result["run_id"].startswith("run_")
    assert "metrics" in result
    assert "n_nodes" in result["metrics"]
    assert "n_edges" in result["metrics"]
    assert "diff" in result
    assert isinstance(result["diff"], list)
    assert "config_yaml_normalized" in result
    assert "data_shape" in result


