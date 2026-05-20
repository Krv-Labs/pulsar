import asyncio
import base64
import json

import numpy as np
import pandas as pd
import yaml

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


def _write_parquet_dataset(tmp_path, rows: int = 30, cols: int = 6) -> str:
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        rng.standard_normal((rows, cols)),
        columns=[f"f{i}" for i in range(cols)],
    )
    path = tmp_path / "dataset.parquet"
    df.to_parquet(path, index=False)
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
    assert "DO NOT use base64 or chunked uploads" in report["agent_action"]
    assert "get_runtime_context" in report["agent_action"]


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
    skeleton_edges = json.loads(
        asyncio.run(server.get_topological_skeleton(run_a, detail="edges", max_edges=5))
    )
    skeleton_full = json.loads(
        asyncio.run(server.get_topological_skeleton(run_a, detail="full"))
    )
    comparison = asyncio.run(server.compare_sweeps(run_a, run_b))

    assert skeleton["run_id"] == run_a
    assert skeleton["dataset_id"] == dataset["dataset_id"]
    assert skeleton["graph"]["node_count"] > 0
    assert skeleton["graph"]["detail"] == "summary"
    assert "nodes" not in skeleton["graph"]
    assert "edges" not in skeleton["graph"]
    assert skeleton["graph"]["nodes_returned"] == 0
    assert skeleton["graph"]["edges_returned"] == 0
    assert skeleton["config_yaml_omitted"] is True
    assert len(skeleton_edges["graph"]["edges"]) <= 5
    assert skeleton_edges["graph"]["edges_returned"] <= 5
    assert "nodes" not in skeleton_edges["graph"]
    assert "config_yaml" in skeleton_full
    assert "nodes" in skeleton_full["graph"]
    assert "edges" in skeleton_full["graph"]
    assert "Sweep Comparison" in comparison
    assert run_a in comparison
    assert run_b in comparison


def _chunked_upload(filename: str, content: bytes) -> dict:
    """Helper: drive the begin/append/finalize ingest path for a single-chunk upload."""
    upload = json.loads(
        asyncio.run(server.begin_dataset_upload(filename, media_type="text/csv"))
    )
    chunk = base64.b64encode(content).decode("ascii")
    asyncio.run(server.append_dataset_chunk(upload["upload_id"], chunk))
    return json.loads(asyncio.run(server.finalize_dataset_upload(upload["upload_id"])))


def test_chunked_upload_single_chunk_round_trip():
    content = _dataset_csv_content().encode("utf-8")
    dataset = _chunked_upload("uploaded.csv", content)
    config_yaml = _extract_config_yaml(
        asyncio.run(server.create_config(dataset["dataset_id"], "chunked_single"))
    )
    report = json.loads(
        asyncio.run(server.validate_config(config_yaml, dataset["dataset_id"]))
    )

    assert dataset["dataset_id"].startswith("ds_")
    assert dataset["source"] == "upload"
    assert dataset["name"] == "uploaded.csv"
    assert report["status"] == "ok"


def test_ingest_modes_produce_equivalent_downstream_configs(tmp_path):
    csv_path = _write_dataset(tmp_path)
    content = _dataset_csv_content().encode("utf-8")

    by_path = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    by_upload = _chunked_upload("uploaded.csv", content)

    path_cfg = _extract_config_yaml(
        asyncio.run(server.create_config(by_path["dataset_id"], "equiv"))
    )
    upload_cfg = _extract_config_yaml(
        asyncio.run(server.create_config(by_upload["dataset_id"], "equiv"))
    )

    assert "name: equiv" in path_cfg
    assert _normalize_config_data_path(path_cfg) == _normalize_config_data_path(
        upload_cfg
    )


def test_create_config_supports_parquet_dataset_id(tmp_path):
    parquet_path = _write_parquet_dataset(tmp_path)

    dataset = json.loads(asyncio.run(server.ingest_dataset(parquet_path)))
    create_response = json.loads(
        asyncio.run(server.create_config(dataset["dataset_id"], "parquet_sweep"))
    )
    report = json.loads(
        asyncio.run(
            server.validate_config(
                create_response["config_yaml"], dataset["dataset_id"]
            )
        )
    )

    assert create_response["status"] == "ok"
    assert "name: parquet_sweep" in create_response["config_yaml"]
    assert f"data: {parquet_path}" in create_response["config_yaml"]
    assert report["status"] == "ok"
    assert report["resolved_dataset_path"] == parquet_path


def test_probe_columns_reloads_for_requested_dataset_id(tmp_path):
    path_a = tmp_path / "dataset_a.csv"
    path_b = tmp_path / "dataset_b.csv"
    pd.DataFrame({"a_only": [1, 2], "shared": [3, 4]}).to_csv(path_a, index=False)
    pd.DataFrame({"b_only": [5, 6], "shared": [7, 8]}).to_csv(path_b, index=False)

    dataset_a = json.loads(asyncio.run(server.ingest_dataset(str(path_a))))
    dataset_b = json.loads(asyncio.run(server.ingest_dataset(str(path_b))))

    asyncio.run(server.characterize_dataset(dataset_id=dataset_a["dataset_id"]))
    payload = json.loads(
        asyncio.run(server.probe_columns(dataset_b["dataset_id"], ["b_only", "shared"]))
    )

    names = {profile["name"] for profile in payload["column_profiles"]}
    shared_profile = next(
        profile for profile in payload["column_profiles"] if profile["name"] == "shared"
    )

    assert payload["status"] == "ok"
    assert names == {"b_only", "shared"}
    assert shared_profile["sample_values"] == ["7", "8"]


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


def test_chunked_upload_handles_bom_and_windows_newlines():
    content = "\ufefff0,f1\r\n1.0,2.0\r\n3.0,4.0\r\n".encode("utf-8")
    dataset = _chunked_upload("windows.csv", content)
    result = json.loads(
        asyncio.run(server.characterize_dataset(dataset_id=dataset["dataset_id"]))
    )

    assert result["n_samples"] == 2
    assert result["n_features"] == 2
    assert result["recommended_next_tool"] == "probe_columns"


def test_unknown_handles_return_stable_codes():
    dataset_report = json.loads(asyncio.run(server.create_config("ds_missing")))
    run_report = json.loads(asyncio.run(server.get_topological_skeleton("run_missing")))
    upload_report = json.loads(
        asyncio.run(server.finalize_dataset_upload("upload_missing"))
    )

    assert dataset_report["error_code"] == "DATASET_ID_UNKNOWN"
    assert run_report["error_code"] == "RUN_ID_UNKNOWN"
    assert upload_report["error_code"] == "UPLOAD_ID_UNKNOWN"


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


def test_get_workflow_guide_returns_phase_prompt():
    guide = asyncio.run(server.get_workflow_guide())
    assert "PHASE I" in guide
    assert "PHASE II" in guide
    assert "PHASE III" in guide
    assert "ingest_dataset" in guide
    assert 'generate_cluster_dossier(detail="summary")' in guide
    assert 'get_topological_skeleton(detail="full")' in guide


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


def test_create_config_high_dimensional_baseline_uses_broad_tail_grid(tmp_path):
    """High-dimensional processed data should start broad before agents refine."""
    import yaml

    csv_path = _write_dataset(tmp_path, rows=80, cols=24)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    response = json.loads(asyncio.run(server.create_config(dataset["dataset_id"])))
    cfg = yaml.safe_load(response["config_yaml"])

    pca_dims = cfg["sweep"]["pca"]["dimensions"]["values"]
    pca_seeds = cfg["sweep"]["pca"]["seed"]["values"]
    eps_steps = cfg["sweep"]["ball_mapper"]["epsilon"]["range"]["steps"]

    assert pca_dims == [2, 5, 10, 15, 20]
    assert pca_seeds == [42, 7, 13]
    assert eps_steps == 24
    assert response["sweep_strategy"]["estimated_ball_maps"] == 360
    assert "compare_sweeps" in response["sweep_strategy"]["agent_guidance"]


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


def test_refine_config_accepts_dotted_paths(tmp_path):
    """refine_config should accept dotted YAML paths alongside flat shortcuts."""
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(server.create_config(dataset["dataset_id"]))
    )

    result = json.loads(
        asyncio.run(
            server.refine_config(
                config_yaml,
                {
                    "sweep.pca.dimensions.values": [3, 5, 7],
                    "output.n_reps": 4,
                },
            )
        )
    )

    assert result["status"] == "ok"
    diff_paths = {d["path"] for d in result["diff"]}
    assert "sweep.pca.dimensions.values" in diff_paths
    assert "output.n_reps" in diff_paths
    refined_cfg = yaml.safe_load(result["config_yaml"])
    assert refined_cfg["sweep"]["pca"]["dimensions"]["values"] == [3, 5, 7]
    assert refined_cfg["output"]["n_reps"] == 4


def test_refine_config_rejects_unknown_dotted_root(tmp_path):
    """Dotted paths under unknown roots should error."""
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(server.create_config(dataset["dataset_id"]))
    )

    result = json.loads(
        asyncio.run(server.refine_config(config_yaml, {"nonsense.path.here": True}))
    )

    assert result["status"] == "error"
    assert "nonsense" in result["reason"]


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
    server._sessions.clear()
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
    assert "is_connected" in result
    assert "singleton_fraction" in result
    assert "spectral_clustering_allowed" in result
    assert "graph_health" in result
    assert "recommended_next_action" in result
    assert result["pca_cached"] is False
    assert result["pca_cache_status"] == {
        "scope": "session",
        "status": "miss",
        "reason": "no_cached_embeddings",
    }


def test_run_topological_sweep_can_use_active_config_without_args(tmp_path):
    server._sessions.clear()
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    create_response = json.loads(
        asyncio.run(server.create_config(dataset["dataset_id"], "active_config"))
    )
    refined = json.loads(
        asyncio.run(
            server.refine_active_config(
                {"pca_dims": [2], "epsilon_values": [0.5], "n_reps": 1}
            )
        )
    )

    active_before = json.loads(asyncio.run(server.get_active_config()))
    result = json.loads(asyncio.run(server.run_topological_sweep()))

    assert create_response["status"] == "ok"
    assert refined["status"] == "ok"
    assert active_before["config_yaml"] == refined["config_yaml"]
    assert result["status"] == "ok"
    assert result["dataset_id"] == dataset["dataset_id"]
    assert result["config_yaml_normalized"] == refined["config_yaml"]


def test_dossier_inherits_construction_threshold(tmp_path):
    """generate_cluster_dossier inherits construction_threshold when arg omitted."""
    server._sessions.clear()
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(server.create_config(dataset["dataset_id"], "dossier_inherit"))
    )
    refined = json.loads(
        asyncio.run(
            server.refine_config(
                config_yaml, {"pca_dims": [2], "epsilon_values": [0.5], "n_reps": 1}
            )
        )
    )["config_yaml"]
    asyncio.run(
        server.run_topological_sweep(
            config_yaml=refined, dataset_id=dataset["dataset_id"]
        )
    )

    inherited = json.loads(asyncio.run(server.generate_cluster_dossier()))
    assert "construction_threshold" in inherited
    assert "interpretation_edge_weight_threshold" in inherited
    assert inherited["threshold_inherited"] is True
    assert (
        inherited["interpretation_edge_weight_threshold"]
        == inherited["construction_threshold"]
    )

    explicit = json.loads(
        asyncio.run(
            server.generate_cluster_dossier(interpretation_edge_weight_threshold=0.5)
        )
    )
    assert explicit["threshold_inherited"] is False
    assert explicit["interpretation_edge_weight_threshold"] == 0.5


def test_sweep_response_contains_stability_summary(tmp_path):
    """run_topological_sweep with auto threshold emits threshold_stability_summary."""
    server._sessions.clear()
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    # create_config defaults construction_threshold to "auto"
    config_yaml = _extract_config_yaml(
        asyncio.run(server.create_config(dataset["dataset_id"], "auto_threshold"))
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
    assert "construction_threshold" in result
    assert isinstance(result["construction_threshold"], float)
    assert "threshold_stability_summary" in result
    summary = result["threshold_stability_summary"]
    assert "selected_threshold" in summary
    assert "top_plateaus" in summary
    assert isinstance(summary["top_plateaus"], list)
    if summary["top_plateaus"]:
        first = summary["top_plateaus"][0]
        for key in ("start", "end", "midpoint", "length", "component_count"):
            assert key in first


def test_run_topological_sweep_reports_pca_cache_hit_on_repeat(tmp_path):
    server._sessions.clear()
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(server.ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(server.create_config(dataset["dataset_id"], "cache_hit"))
    )
    refined = json.loads(
        asyncio.run(
            server.refine_config(
                config_yaml, {"pca_dims": [2], "epsilon_values": [0.5], "n_reps": 1}
            )
        )
    )["config_yaml"]

    first = json.loads(
        asyncio.run(
            server.run_topological_sweep(
                config_yaml=refined, dataset_id=dataset["dataset_id"]
            )
        )
    )
    second = json.loads(
        asyncio.run(
            server.run_topological_sweep(
                config_yaml=refined, dataset_id=dataset["dataset_id"]
            )
        )
    )

    assert first["pca_cached"] is False
    assert first["pca_cache_status"]["reason"] == "no_cached_embeddings"
    assert second["pca_cached"] is True
    assert second["pca_cache_status"] == {
        "scope": "session",
        "status": "hit",
        "reason": "fingerprint_match",
    }
