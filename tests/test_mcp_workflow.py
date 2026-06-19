import asyncio
import base64
import importlib
import json
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from pulsar.mcp.session import SweepRecord, _sessions, _get_session
from pulsar.mcp.diagnostics import _finalization_gate
from pulsar.mcp.config_tools import _initial_pca_grid
from pulsar.mcp.payloads import sweep_payload_to_markdown
from pulsar.mcp.tools.ingestion import (
    ingest_dataset,
    begin_dataset_upload,
    append_dataset_chunk,
    finalize_dataset_upload,
)
from pulsar.mcp.tools.config import (
    create_config,
    validate_config,
    refine_config,
)
from pulsar.mcp.tools.meta import get_runtime_context
from pulsar.mcp.tools.sweeping import (
    get_sweep_history,
    run_topological_sweep,
    compare_sweeps,
)
from pulsar.mcp.thresholds import (
    agent_threshold_options,
    component_state_at_threshold,
    component_mass_profile,
    prepare_threshold_graph,
    structural_breakpoints,
)
from pulsar.mcp.tools.diagnostics import (
    _summary_structural_breakpoints,
    _threshold_agent_readout,
    _threshold_next_tool_lines,
    create_graph_artifact,
    diagnose_cosmic_graph,
    get_threshold_stability_curve,
    get_topological_skeleton,
)
from pulsar.mcp.tools.preprocessing import (
    recommend_preprocessing,
)
from pulsar.mcp.tools.meta import (
    characterize_dataset,
    get_workflow_guide,
)
from pulsar.mcp.tools.reporting import (
    probe_columns,
)
from pulsar.mcp.tools.clustering import (
    generate_cluster_dossier,
)


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


def test_default_mcp_tool_surface_is_curated(monkeypatch):
    monkeypatch.delenv("PULSAR_MCP_ENABLE_UPLOAD", raising=False)
    import pulsar.mcp.tools as tools

    tools = importlib.reload(tools)
    names = [tool.__name__ for tool in tools.ALL_TOOLS_LIST]

    assert len(names) == 25
    assert "get_sweep_history" in names
    assert "compare_clusters" in names
    assert "explain_suggestion" not in names
    assert "get_experiment_history" not in names
    assert "summarize_sweep_history" not in names
    assert "compare_clusters_tool" not in names
    assert "begin_dataset_upload" not in names
    assert "append_dataset_chunk" not in names
    assert "finalize_dataset_upload" not in names


def test_upload_tools_are_opt_in(monkeypatch):
    import pulsar.mcp.tools as tools

    monkeypatch.setenv("PULSAR_MCP_ENABLE_UPLOAD", "1")
    tools = importlib.reload(tools)
    names = [tool.__name__ for tool in tools.ALL_TOOLS_LIST]
    assert "begin_dataset_upload" in names
    assert "append_dataset_chunk" in names
    assert "finalize_dataset_upload" in names

    monkeypatch.delenv("PULSAR_MCP_ENABLE_UPLOAD", raising=False)
    importlib.reload(tools)


def test_get_sweep_history_consolidates_table_summary_and_full_modes():
    _sessions.clear()
    session = _get_session(None)
    session.sweep_history.append(
        SweepRecord(
            timestamp=1.0,
            dataset_id="ds_test",
            config_yaml="""
sweep:
  projection:
    dimensions:
      values: [2, 4]
  ball_mapper:
    epsilon:
      range:
        min: 0.1
        max: 0.3
""",
            metrics={
                "n_nodes": 30,
                "n_edges": 44,
                "component_count": 3,
                "giant_fraction": 0.8,
            },
        )
    )

    table = asyncio.run(get_sweep_history())
    assert "| Run | Projection Dims | Epsilon Range |" in table
    assert "[2, 4]" in table
    assert "80.00%" in table

    summary = json.loads(
        asyncio.run(get_sweep_history(detail="summary", response_format="json"))
    )
    assert summary["detail"] == "summary"
    assert summary["n_runs"] == 1
    assert "summary" in summary
    assert "runs" not in summary

    full = json.loads(
        asyncio.run(get_sweep_history(detail="full", response_format="json"))
    )
    assert full["detail"] == "full"
    assert full["runs"][0]["dataset_id"] == "ds_test"
    assert "summary" in full


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

    payload = asyncio.run(validate_config(bad_yaml))
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
        validate_config(f"```yaml\nrun:\n  data: {csv_path}\noutput:\n  n_reps: 1\n```")
    )
    report = json.loads(payload)

    assert report["status"] == "error"
    assert report["error_code"] == "YAML_NOT_RAW"


def test_validate_config_rejects_legacy_cosmic_graph_threshold(tmp_path):
    csv_path = _write_dataset(tmp_path)
    legacy_yaml = f"""run:
  data: {csv_path}
cosmic_graph:
  threshold: 0.0
"""

    payload = asyncio.run(validate_config(legacy_yaml))
    report = json.loads(payload)

    assert report["status"] == "error"
    assert report["error_code"] == "YAML_SCHEMA_MISMATCH"
    issue = report["details"]["issues"][0]
    assert issue["path"] == "cosmic_graph.threshold"
    assert "cosmic_graph.construction_threshold" in issue["message"]
    assert issue["expected"] == "cosmic_graph.construction_threshold"
    assert "construction_threshold" in issue["example_fix"]


def test_validate_config_normalizes_missing_construction_threshold_to_auto(tmp_path):
    csv_path = _write_dataset(tmp_path)
    config_yaml = f"""run:
  data: {csv_path}
output:
  n_reps: 1
"""

    payload = asyncio.run(validate_config(config_yaml))
    report = json.loads(payload)

    assert report["status"] == "ok"
    normalized = yaml.safe_load(report["normalized_config_yaml"])
    assert normalized["cosmic_graph"]["construction_threshold"] == "auto"


def test_validate_config_preserves_explicit_zero_construction_threshold(tmp_path):
    csv_path = _write_dataset(tmp_path)
    config_yaml = f"""run:
  data: {csv_path}
cosmic_graph:
  construction_threshold: 0.0
output:
  n_reps: 1
"""

    payload = asyncio.run(validate_config(config_yaml))
    report = json.loads(payload)

    assert report["status"] == "ok"
    normalized = yaml.safe_load(report["normalized_config_yaml"])
    assert normalized["cosmic_graph"]["construction_threshold"] == 0.0


def test_validate_config_rejects_out_of_range_construction_threshold(tmp_path):
    csv_path = _write_dataset(tmp_path)
    config_yaml = f"""run:
  data: {csv_path}
cosmic_graph:
  construction_threshold: 1.5
output:
  n_reps: 1
"""

    payload = asyncio.run(validate_config(config_yaml))
    report = json.loads(payload)

    assert report["status"] == "error"
    assert report["error_code"] == "YAML_SCHEMA_MISMATCH"
    issue = report["details"]["issues"][0]
    assert issue["path"] == "cosmic_graph.construction_threshold"
    assert "between 0.0 and 1.0" in issue["message"]


def test_ingest_dataset_and_create_config_round_trip(tmp_path):
    csv_path = _write_dataset(tmp_path)

    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    create_response = json.loads(
        asyncio.run(create_config(dataset["dataset_id"], "demo_sweep"))
    )
    config_yaml = create_response["config_yaml"]
    report = json.loads(
        asyncio.run(validate_config(config_yaml, dataset["dataset_id"]))
    )

    assert dataset["dataset_id"].startswith("ds_")
    assert create_response["status"] == "ok"
    assert create_response["calibration_space"] in ("processed", "raw")
    assert "name: demo_sweep" in config_yaml
    assert report["status"] == "ok"
    assert report["resolved_dataset_path"] == csv_path
    assert "normalized_config_yaml" in report


def test_ingest_dataset_classifies_sandbox_local_missing_path():
    payload = asyncio.run(ingest_dataset("/home/claude/missing.csv"))
    report = json.loads(payload)

    assert report["status"] == "error"
    assert report["error_code"] == "HOST_PATH_NOT_VISIBLE"
    assert report["details"]["path_context"]["looks_like_sandbox_path"] is True
    assert "DO NOT use base64 or chunked uploads" in report["agent_action"]
    assert "get_runtime_context" in report["agent_action"]


def test_ingest_dataset_classifies_host_missing_path():
    payload = asyncio.run(ingest_dataset("/definitely/not/real.csv"))
    report = json.loads(payload)

    assert report["status"] == "error"
    assert report["error_code"] == "FILE_NOT_FOUND"
    assert report["details"]["path_context"]["looks_like_sandbox_path"] is False


def test_run_topological_sweep_with_dataset_id_persists_run_summary(tmp_path):
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"], "run_compare"))
    )

    refined_a = json.loads(
        asyncio.run(
            refine_config(
                config_yaml,
                {"pca_dims": [2], "epsilon_values": [0.5], "n_reps": 1},
            )
        )
    )["config_yaml"]
    refined_b = json.loads(
        asyncio.run(
            refine_config(
                config_yaml,
                {"pca_dims": [3], "epsilon_values": [0.75], "n_reps": 1},
            )
        )
    )["config_yaml"]

    result_a = json.loads(
        asyncio.run(
            run_topological_sweep(
                config_yaml=refined_a,
                dataset_id=dataset["dataset_id"],
                response_format="json",
            )
        )
    )
    result_b = json.loads(
        asyncio.run(
            run_topological_sweep(
                config_yaml=refined_b,
                dataset_id=dataset["dataset_id"],
                response_format="json",
            )
        )
    )

    assert result_a["status"] == "ok"
    assert result_b["status"] == "ok"
    run_a = result_a["run_id"]
    run_b = result_b["run_id"]

    skeleton = json.loads(asyncio.run(get_topological_skeleton(run_a)))
    skeleton_edges = json.loads(
        asyncio.run(get_topological_skeleton(run_a, detail="edges", max_edges=5))
    )
    skeleton_full = json.loads(
        asyncio.run(get_topological_skeleton(run_a, detail="full"))
    )
    comparison = asyncio.run(compare_sweeps(run_a, run_b))

    assert skeleton["run_id"] == run_a
    assert skeleton["dataset_id"] == dataset["dataset_id"]
    assert skeleton["graph"]["node_count"] > 0
    assert skeleton["graph"]["detail"] == "summary"
    assert "component_sizes_summary" in skeleton["graph"]
    assert "component_sizes" not in skeleton["graph"]
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
    assert "component_sizes" in skeleton_full["graph"]
    assert "Sweep Comparison" in comparison
    assert run_a in comparison
    assert run_b in comparison


def _chunked_upload(filename: str, content: bytes) -> dict:
    """Helper: drive the begin/append/finalize ingest path for a single-chunk upload."""
    upload = json.loads(
        asyncio.run(begin_dataset_upload(filename, media_type="text/csv"))
    )
    chunk = base64.b64encode(content).decode("ascii")
    asyncio.run(append_dataset_chunk(upload["upload_id"], chunk))
    return json.loads(asyncio.run(finalize_dataset_upload(upload["upload_id"])))


def test_chunked_upload_single_chunk_round_trip():
    content = _dataset_csv_content().encode("utf-8")
    dataset = _chunked_upload("uploaded.csv", content)
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"], "chunked_single"))
    )
    report = json.loads(
        asyncio.run(validate_config(config_yaml, dataset["dataset_id"]))
    )

    assert dataset["dataset_id"].startswith("ds_")
    assert dataset["source"] == "upload"
    assert dataset["name"] == "uploaded.csv"
    assert report["status"] == "ok"


def test_ingest_modes_produce_equivalent_downstream_configs(tmp_path):
    csv_path = _write_dataset(tmp_path)
    content = _dataset_csv_content().encode("utf-8")

    by_path = json.loads(asyncio.run(ingest_dataset(csv_path)))
    by_upload = _chunked_upload("uploaded.csv", content)

    path_cfg = _extract_config_yaml(
        asyncio.run(create_config(by_path["dataset_id"], "equiv"))
    )
    upload_cfg = _extract_config_yaml(
        asyncio.run(create_config(by_upload["dataset_id"], "equiv"))
    )

    assert "name: equiv" in path_cfg
    assert _normalize_config_data_path(path_cfg) == _normalize_config_data_path(
        upload_cfg
    )


def test_create_config_supports_parquet_dataset_id(tmp_path):
    parquet_path = _write_parquet_dataset(tmp_path)

    dataset = json.loads(asyncio.run(ingest_dataset(parquet_path)))
    create_response = json.loads(
        asyncio.run(create_config(dataset["dataset_id"], "parquet_sweep"))
    )
    report = json.loads(
        asyncio.run(
            validate_config(create_response["config_yaml"], dataset["dataset_id"])
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

    dataset_a = json.loads(asyncio.run(ingest_dataset(str(path_a))))
    dataset_b = json.loads(asyncio.run(ingest_dataset(str(path_b))))

    asyncio.run(
        characterize_dataset(dataset_id=dataset_a["dataset_id"], response_format="json")
    )
    payload = json.loads(
        asyncio.run(
            probe_columns(
                dataset_b["dataset_id"],
                ["b_only", "shared"],
                response_format="json",
            )
        )
    )

    names = {profile["name"] for profile in payload["column_profiles"]}
    shared_profile = next(
        profile for profile in payload["column_profiles"] if profile["name"] == "shared"
    )

    assert payload["status"] == "ok"
    assert payload["columns_returned"] == 2
    assert payload["missing_columns"] == []
    assert names == {"b_only", "shared"}
    assert shared_profile["sample_values"] == ["7", "8"]


def test_staged_dataset_upload_round_trip():
    upload = json.loads(
        asyncio.run(begin_dataset_upload("chunked.csv", media_type="text/csv"))
    )
    content = _dataset_csv_content()
    midpoint = len(content) // 2
    chunk_a = base64.b64encode(content[:midpoint].encode("utf-8")).decode("ascii")
    chunk_b = base64.b64encode(content[midpoint:].encode("utf-8")).decode("ascii")

    append_a = json.loads(
        asyncio.run(append_dataset_chunk(upload["upload_id"], chunk_a))
    )
    append_b = json.loads(
        asyncio.run(append_dataset_chunk(upload["upload_id"], chunk_b))
    )
    dataset = json.loads(asyncio.run(finalize_dataset_upload(upload["upload_id"])))
    report = json.loads(
        asyncio.run(
            validate_config(
                _extract_config_yaml(
                    asyncio.run(create_config(dataset["dataset_id"], "chunked"))
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
        asyncio.run(begin_dataset_upload("misuse.csv", media_type="text/csv"))
    )
    chunk = base64.b64encode(b"f0,f1\n1,2\n").decode("ascii")
    asyncio.run(append_dataset_chunk(upload["upload_id"], chunk))
    asyncio.run(finalize_dataset_upload(upload["upload_id"]))

    append_after = json.loads(
        asyncio.run(append_dataset_chunk(upload["upload_id"], chunk))
    )
    finalize_again = json.loads(
        asyncio.run(finalize_dataset_upload(upload["upload_id"]))
    )

    assert append_after["error_code"] == "UPLOAD_ID_UNKNOWN"
    assert finalize_again["error_code"] == "UPLOAD_ID_UNKNOWN"


def test_chunked_upload_handles_bom_and_windows_newlines():
    content = "\ufefff0,f1\r\n1.0,2.0\r\n3.0,4.0\r\n".encode("utf-8")
    dataset = _chunked_upload("windows.csv", content)
    result = json.loads(
        asyncio.run(
            characterize_dataset(
                dataset_id=dataset["dataset_id"], response_format="json"
            )
        )
    )

    assert result["n_samples"] == 2
    assert result["n_features"] == 2
    assert result["recommended_next_tool"] == "create_config"


def test_characterize_dataset_returns_compact_wide_schema_summary(tmp_path):
    csv_path = _write_dataset(tmp_path, rows=40, cols=124)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))

    result = json.loads(
        asyncio.run(
            characterize_dataset(
                dataset_id=dataset["dataset_id"], response_format="json"
            )
        )
    )

    assert result["status"] == "ok"
    assert "column_profiles" not in result
    assert result["n_columns_total"] == 124
    assert result["schema_summary"]["numeric_columns"] == 124
    assert result["schema_summary"]["previewed_column_profiles"] == 0
    assert result["column_profile_preview"] == []
    assert result["column_name_preview"]["numeric"]["columns"]
    assert result["column_name_preview"]["numeric"]["omitted"] == 104
    assert result["omitted_column_profiles"] == 124
    assert "pca_cumulative_variance" in result["raw_numeric_geometry"]


def test_characterize_dataset_previews_interesting_columns(tmp_path):
    df = pd.DataFrame(
        {
            "complete_numeric": [1.0, 2.0, 3.0, 4.0],
            "missing_numeric": [1.0, None, 3.0, None],
            "category": ["a", "b", "a", "b"],
            "all_missing": [None, None, None, None],
            "identifier": ["id-1", "id-2", "id-3", "id-4"],
        }
    )
    path = tmp_path / "interesting.csv"
    df.to_csv(path, index=False)
    dataset = json.loads(asyncio.run(ingest_dataset(str(path))))

    result = json.loads(
        asyncio.run(
            characterize_dataset(
                dataset_id=dataset["dataset_id"], response_format="json"
            )
        )
    )
    preview_names = {cp["name"] for cp in result["column_profile_preview"]}

    assert "complete_numeric" not in preview_names
    assert {"missing_numeric", "category", "all_missing", "identifier"} <= preview_names
    assert result["schema_summary"]["columns_with_missing"] == 2
    assert result["schema_summary"]["categorical_columns"] == 2
    assert result["schema_summary"]["high_cardinality_columns"] == 1


def test_characterize_dataset_defaults_to_markdown(tmp_path):
    csv_path = _write_dataset(tmp_path, rows=10, cols=4)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))

    markdown = asyncio.run(characterize_dataset(dataset_id=dataset["dataset_id"]))

    assert "# Dataset Characterization" in markdown
    assert "## Column Name Preview" in markdown
    assert "## Next Tools" in markdown
    assert not markdown.lstrip().startswith("{")


def test_probe_columns_reports_missing_and_truncates_long_values(tmp_path):
    long_value = "x" * 120
    path = tmp_path / "long_values.csv"
    pd.DataFrame(
        {
            "text": [long_value, "short", long_value],
            "value": [1.0, 2.0, 3.0],
        }
    ).to_csv(path, index=False)
    dataset = json.loads(asyncio.run(ingest_dataset(str(path))))

    payload = json.loads(
        asyncio.run(
            probe_columns(
                dataset["dataset_id"],
                ["text", "missing"],
                response_format="json",
            )
        )
    )
    markdown = asyncio.run(probe_columns(dataset["dataset_id"], ["text"]))

    assert payload["columns_requested"] == 2
    assert payload["columns_returned"] == 1
    assert payload["missing_columns"] == ["missing"]
    text_profile = payload["column_profiles"][0]
    assert text_profile["truncation"]["sample_values_truncated"] > 0
    assert text_profile["truncation"]["top_values_truncated"] > 0
    assert "# Column Probe" in markdown
    assert not markdown.lstrip().startswith("{")


def test_recommend_preprocessing_defaults_to_markdown_and_samples_dirty_numeric(
    tmp_path,
):
    path = tmp_path / "dirty_numeric.csv"
    pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 3.0, 4.0, 5.0],
            "dirty": ["1.0", "2.5", "bad", "4.2"],
        }
    ).to_csv(path, index=False)
    dataset = json.loads(asyncio.run(ingest_dataset(str(path))))

    markdown = asyncio.run(recommend_preprocessing(dataset_id=dataset["dataset_id"]))
    payload = json.loads(
        asyncio.run(
            recommend_preprocessing(
                dataset_id=dataset["dataset_id"],
                response_format="json",
                detail="full",
            )
        )
    )

    assert "# Preprocessing Recommendation" in markdown
    assert "## Dirty Numeric Detection" in markdown
    assert not markdown.lstrip().startswith("{")
    assert payload["dirty_numeric_detection"]["status"] == "enabled"
    assert payload["dirty_numeric_detection"]["object_columns_sampled"] == 1
    assert any(
        warning["code"] == "DIRTY_NUMERIC_DETECTED" for warning in payload["warnings"]
    )
    assert "rationale" in payload


def test_recommend_preprocessing_summary_caps_rationale_and_warns_expansion(tmp_path):
    rows = 30
    data = {
        "x": list(range(rows)),
        "y": [value * 2 for value in range(rows)],
    }
    for idx in range(3):
        data[f"cat_{idx}"] = [f"v_{idx}_{value % 25}" for value in range(rows)]
    path = tmp_path / "wide_preprocessing.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    dataset = json.loads(asyncio.run(ingest_dataset(str(path))))

    payload = json.loads(
        asyncio.run(
            recommend_preprocessing(
                dataset_id=dataset["dataset_id"],
                response_format="json",
                rationale_limit=2,
            )
        )
    )

    assert payload["detail"] == "summary"
    assert len(payload["rationale_preview"]) == 2
    assert payload["rationale_omitted"] > 0
    assert "rationale" not in payload
    assert payload["expansion_estimate"] > 50
    assert any(
        warning["code"] == "HIGH_DIMENSION_EXPANSION" for warning in payload["warnings"]
    )


def test_recommend_preprocessing_surfaces_high_missingness_decisions(tmp_path):
    rows = 20
    path = tmp_path / "high_missingness.csv"
    pd.DataFrame(
        {
            "clean": list(range(rows)),
            "sparse_numeric": [None] * 17 + [1.0, 2.0, 3.0],
            "review_numeric": [None] * 13 + [float(i) for i in range(7)],
            "outcome_score": [None] * 17 + [0.0, 1.0, 1.0],
        }
    ).to_csv(path, index=False)
    dataset = json.loads(asyncio.run(ingest_dataset(str(path))))

    markdown = asyncio.run(recommend_preprocessing(dataset_id=dataset["dataset_id"]))
    payload = json.loads(
        asyncio.run(
            recommend_preprocessing(
                dataset_id=dataset["dataset_id"],
                response_format="json",
                detail="full",
            )
        )
    )

    block = yaml.safe_load(payload["preprocessing_yaml"])["preprocessing"]
    assert "sparse_numeric" in block["drop_columns"]
    assert "review_numeric" in block["impute"]
    assert "outcome_score" in block["impute"]
    assert payload["recommended_action"] == "probe_columns_first"
    assert payload["next_tool_call"]["tool"] == "probe_columns"
    assert {row["column"] for row in payload["high_missingness_columns_full"]} == {
        "sparse_numeric",
        "review_numeric",
        "outcome_score",
    }
    protected = {
        row["column"]: row["protected_name_hint"]
        for row in payload["high_missingness_columns_full"]
    }
    assert protected["outcome_score"] is True
    assert any(
        warning["code"] == "HIGH_MISSINGNESS_COLUMNS" for warning in payload["warnings"]
    )
    assert "## High Missingness Review" in markdown


def test_recommend_preprocessing_surfaces_numeric_coded_categories(tmp_path):
    rows = 24
    path = tmp_path / "numeric_codes.csv"
    pd.DataFrame(
        {
            "age": [40 + i for i in range(rows)],
            "has_diabetes": [0, 1] * (rows // 2),
            "severity_stage": [0, 1, 2, 3] * (rows // 4),
            "lab_value": [float(i) + 0.25 for i in range(rows)],
        }
    ).to_csv(path, index=False)
    dataset = json.loads(asyncio.run(ingest_dataset(str(path))))

    markdown = asyncio.run(recommend_preprocessing(dataset_id=dataset["dataset_id"]))
    payload = json.loads(
        asyncio.run(
            recommend_preprocessing(
                dataset_id=dataset["dataset_id"],
                response_format="json",
                detail="full",
            )
        )
    )

    block = yaml.safe_load(payload["preprocessing_yaml"])["preprocessing"]
    candidates = {
        row["column"]: row for row in payload["numeric_categorical_candidates_full"]
    }
    assert "has_diabetes" in candidates
    assert "severity_stage" in candidates
    assert "age" not in candidates
    assert "lab_value" not in candidates
    assert candidates["has_diabetes"]["supported_actions"] == [
        "keep_numeric",
        "one_hot_encode",
        "drop",
    ]
    assert "StandardScaler" in candidates["has_diabetes"]["encoding_note"]
    assert "has_diabetes" not in block["encode"]
    assert payload["recommended_action"] == "accept"
    assert payload["next_tool_call"]["tool"] == "create_config"
    assert "value counts" in candidates["has_diabetes"]["evidence_available"]
    assert any(
        warning["code"] == "NUMERIC_CODED_CATEGORICAL_CANDIDATES"
        and warning["severity"] == "info"
        for warning in payload["warnings"]
    )
    assert "## Numeric-Coded Category Review" in markdown
    assert "not a required next step" in markdown

    probe = json.loads(
        asyncio.run(
            probe_columns(
                dataset["dataset_id"],
                list(candidates),
                response_format="json",
            )
        )
    )
    has_diabetes = next(
        profile
        for profile in probe["column_profiles"]
        if profile["name"] == "has_diabetes"
    )
    assert has_diabetes["is_numeric"] is True
    assert has_diabetes["top_values"] == [["0", 12], ["1", 12]]


def test_unknown_handles_return_stable_codes():
    dataset_report = json.loads(asyncio.run(create_config("ds_missing")))
    run_report = json.loads(asyncio.run(get_topological_skeleton("run_missing")))
    upload_report = json.loads(asyncio.run(finalize_dataset_upload("upload_missing")))

    assert dataset_report["error_code"] == "DATASET_ID_UNKNOWN"
    assert run_report["error_code"] == "RUN_ID_UNKNOWN"
    assert upload_report["error_code"] == "UPLOAD_ID_UNKNOWN"


def test_append_dataset_chunk_rejects_invalid_base64():
    upload = json.loads(
        asyncio.run(begin_dataset_upload("broken.csv", media_type="text/csv"))
    )
    report = json.loads(
        asyncio.run(
            append_dataset_chunk(
                upload["upload_id"],
                "%%%not-base64%%%",
            )
        )
    )

    assert report["error_code"] == "UPLOAD_DECODE_FAILED"


def test_get_workflow_guide_returns_phase_prompt():
    guide = asyncio.run(get_workflow_guide())
    assert "PHASE I" in guide
    assert "PHASE II" in guide
    assert "PHASE III" in guide
    assert "ingest_dataset" in guide
    assert "get_sweep_history" in guide
    assert "get_experiment_history" not in guide
    assert "summarize_sweep_history" not in guide
    assert 'generate_cluster_dossier(detail="summary")' in guide
    assert 'get_topological_skeleton(detail="nodes")' in guide
    assert 'detail="full_nodes"' in guide


def test_run_topological_sweep_missing_config_path_returns_structured_error():
    report = json.loads(
        asyncio.run(run_topological_sweep(config_path="/home/claude/missing.yaml"))
    )

    assert report["error_code"] == "HOST_PATH_NOT_VISIBLE"


def test_run_topological_sweep_missing_host_config_path_returns_config_code():
    report = json.loads(
        asyncio.run(run_topological_sweep(config_path="/definitely/not/real.yaml"))
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
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))

    response = json.loads(
        asyncio.run(create_config(dataset["dataset_id"], "mixed_test"))
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
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    response = json.loads(asyncio.run(create_config(dataset["dataset_id"])))

    cfg = yaml.safe_load(response["config_yaml"])
    projection = cfg["sweep"]["projection"]
    projection_dims = projection["dimensions"]["values"]
    projection_seeds = projection["seed"]["values"]
    eps = cfg["sweep"]["ball_mapper"]["epsilon"]["range"]

    assert projection["method"] == "jl"
    # Should have multiple projection dims spanning a meaningful range
    assert len(projection_dims) >= 3
    assert max(projection_dims) > min(projection_dims)
    # Should have multiple seeds
    assert len(projection_seeds) >= 2
    # Should have enough epsilon steps
    assert eps["steps"] >= 20
    assert "pca_dimensions" not in response["sweep_strategy"]
    assert "pca_seeds" not in response["sweep_strategy"]


def test_create_config_high_dimensional_baseline_uses_broad_tail_grid(tmp_path):
    """High-dimensional processed data should start broad before agents refine."""
    import yaml

    csv_path = _write_dataset(tmp_path, rows=80, cols=24)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    response = json.loads(asyncio.run(create_config(dataset["dataset_id"])))
    cfg = yaml.safe_load(response["config_yaml"])

    projection = cfg["sweep"]["projection"]
    projection_dims = projection["dimensions"]["values"]
    projection_seeds = projection["seed"]["values"]
    legacy_pca_dims = cfg["sweep"]["pca"]["dimensions"]["values"]
    eps_steps = cfg["sweep"]["ball_mapper"]["epsilon"]["range"]["steps"]

    # Variance-frontier targeting: drops low-variance dims (e.g., 2, 5 captured
    # < 50% cumulative variance) in favor of signal-rich frontiers + elbow.
    assert projection["method"] == "jl"
    assert projection_dims == [10, 15, 16]
    assert legacy_pca_dims == projection_dims
    assert max(projection_dims) <= 16
    assert projection_seeds == [42, 7, 13]
    assert eps_steps == 24
    assert response["sweep_strategy"]["projection_method"] == "jl"
    assert response["sweep_strategy"]["projection_dimensions"] == [10, 15, 16]
    assert response["sweep_strategy"]["projection_seeds"] == [42, 7, 13]
    assert response["sweep_strategy"]["estimated_ball_maps"] == 216
    assert "compare_sweeps" in response["sweep_strategy"]["agent_guidance"]


def test_initial_pca_grid_expands_diffuse_high_dimensional_curve():
    grid = _initial_pca_grid(
        384,
        20,
        [
            (2, 0.01),
            (3, 0.015),
            (5, 0.025),
            (10, 0.05),
            (15, 0.075),
            (20, 0.10),
        ],
    )

    assert grid == [8, 12, 16]


def test_initial_pca_grid_respects_tiny_feature_ceiling():
    assert _initial_pca_grid(2, 2, [(2, 0.9)]) == [2]


def test_create_config_numeric_only_still_works(tmp_path):
    """create_config on pure numeric data should still work (no expansion)."""
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))

    response = json.loads(
        asyncio.run(create_config(dataset["dataset_id"], "numeric_only"))
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
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"]))
    )

    result = json.loads(
        asyncio.run(
            refine_config(
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
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"]))
    )

    result = json.loads(
        asyncio.run(
            refine_config(
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


def test_sweep_markdown_surfaces_resolved_construction_threshold():
    """The selected edge-weight cut must be visible in the markdown summary, not
    only in the JSON payload."""
    from pulsar.mcp.payloads import (
        build_sweep_summary_payload,
        sweep_payload_to_markdown,
    )

    response = {
        "status": "ok",
        "run_id": "run_abc",
        "dataset_id": "ds_abc",
        "construction_threshold": 0.4213,
        "graph_health": "connected",
        "analysis_status": "diagnostics_required",
        "constructed_graph_connected": False,
        "metrics": {
            "n_nodes": 1700,
            "n_edges": 916476,
            "component_count": 2,
            "giant_fraction": 0.96,
            "component_sizes": [1631, 9],
        },
    }
    md = sweep_payload_to_markdown(build_sweep_summary_payload(response))
    assert "Construction threshold (edge-weight cut): 0.4213" in md
    assert "Graph health status: usable_giant_component_with_tail" in md
    assert "Constructed graph connected: False" in md


def test_apply_overrides_remove_keys_deletes_nested_key():
    """remove_keys deletes a config entry entirely (vs. nulling it), is
    idempotent, and validates roots."""
    from pulsar.mcp.config_tools import (
        _MISSING,
        _remove_dotted_path,
        apply_overrides,
    )

    raw = {"preprocessing": {"encode": {"species": {"method": "onehot"}, "keep": 1}}}
    removed = _remove_dotted_path(raw, "preprocessing.encode.species")
    assert removed == {"method": "onehot"}
    assert raw["preprocessing"]["encode"] == {"keep": 1}
    # Idempotent: removing a missing path is a no-op sentinel, not an error.
    assert _remove_dotted_path(raw, "preprocessing.encode.species") is _MISSING

    # Validation fires before any config parsing.
    try:
        apply_overrides("run:\n  data: x\n", {"remove_keys": "notalist"})
        raise AssertionError("expected ValueError for non-list remove_keys")
    except ValueError as e:
        assert "remove_keys" in str(e)
    try:
        apply_overrides("run:\n  data: x\n", {"remove_keys": ["bogus.path"]})
        raise AssertionError("expected ValueError for unknown remove_keys root")
    except ValueError as e:
        assert "bogus" in str(e)


def test_refine_config_remove_keys_drops_preprocessing_entry(tmp_path):
    """End-to-end: a key injected under preprocessing can be deleted via
    remove_keys, and the deletion is reported in the diff with removed=true."""
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"]))
    )
    seeded = json.loads(
        asyncio.run(
            refine_config(
                config_yaml, {"preprocessing.impute.f0": {"method": "fill_mean"}}
            )
        )
    )["config_yaml"]
    assert "f0" in (
        yaml.safe_load(seeded).get("preprocessing", {}).get("impute", {}) or {}
    )

    removed = json.loads(
        asyncio.run(refine_config(seeded, {"remove_keys": ["preprocessing.impute.f0"]}))
    )
    assert removed["status"] == "ok"
    impute = (
        yaml.safe_load(removed["config_yaml"])
        .get("preprocessing", {})
        .get("impute", {})
        or {}
    )
    assert "f0" not in impute
    assert any(d.get("removed") for d in removed["diff"])


def test_refine_config_rejects_unknown_dotted_root(tmp_path):
    """Dotted paths under unknown roots should error."""
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"]))
    )

    result = json.loads(
        asyncio.run(refine_config(config_yaml, {"nonsense.path.here": True}))
    )

    assert result["status"] == "error"
    assert "nonsense" in result["reason"]


def test_refine_config_returns_diff(tmp_path):
    """refine_config should return a diff of what changed."""
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"]))
    )

    result = json.loads(
        asyncio.run(
            refine_config(config_yaml, {"projection_dimensions": [2, 5], "n_reps": 2})
        )
    )

    assert result["status"] == "ok"
    assert "diff" in result
    assert len(result["diff"]) >= 2  # projection_dimensions and n_reps changed
    diff_paths = {d["path"] for d in result["diff"]}
    assert "sweep.projection.dimensions.values" in diff_paths
    assert "output.n_reps" in diff_paths


def test_refine_config_supports_human_readable_outputs(tmp_path):
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"]))
    )

    markdown = asyncio.run(
        refine_config(
            config_yaml,
            {"projection_dimensions": [2], "n_reps": 2},
            response_format="markdown",
        )
    )
    raw_yaml = asyncio.run(
        refine_config(
            config_yaml,
            {"projection_dimensions": [2], "n_reps": 2},
            response_format="yaml",
        )
    )

    assert markdown.startswith("# Refined Config")
    assert "## Diff" in markdown
    assert "`sweep.projection.dimensions.values`" in markdown
    assert "```yaml" in markdown
    assert not markdown.lstrip().startswith("{")
    parsed = yaml.safe_load(raw_yaml)
    assert parsed["sweep"]["projection"]["dimensions"]["values"] == [2]
    assert parsed["output"]["n_reps"] == 2
    assert not raw_yaml.lstrip().startswith("{")


# ---------------------------------------------------------------------------
# JSON output structure tests
# ---------------------------------------------------------------------------


def test_run_topological_sweep_returns_json(tmp_path):
    """run_topological_sweep should return bounded structured JSON when requested."""
    _sessions.clear()
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"]))
    )
    refined = json.loads(
        asyncio.run(
            refine_config(
                config_yaml,
                {
                    "projection_method": "pca",
                    "pca_dims": [2],
                    "epsilon_values": [0.5],
                    "construction_threshold": 0.0,
                    "n_reps": 1,
                },
            )
        )
    )["config_yaml"]

    result = json.loads(
        asyncio.run(
            run_topological_sweep(
                config_yaml=refined,
                dataset_id=dataset["dataset_id"],
                response_format="json",
            )
        )
    )

    assert result["status"] == "ok"
    assert result["detail"] == "summary"
    assert "run_id" in result
    assert result["run_id"].startswith("run_")
    assert "key_metrics" in result
    assert "n_nodes" in result["key_metrics"]
    assert "n_edges" in result["key_metrics"]
    assert "metrics" not in result
    assert "diff_summary" in result
    assert "config_yaml_normalized" not in result
    assert result["config_yaml_included"] is False
    assert result["config_yaml_available_via"] == "get_runtime_context"
    assert "data_shape" in result
    assert "component_sizes_preview" in result
    assert "component_sizes_omitted" in result
    assert "graph_health" in result
    assert "recommended_next_action" not in result
    assert result["analysis_status"] == "diagnostics_required"
    assert result["next_required_check"] == "diagnose_cosmic_graph"
    assert result["pca_cached"] is False
    assert result["pca_cache_status"] == {
        "scope": "session",
        "status": "miss",
        "reason": "no_cached_embeddings",
    }

    full = json.loads(
        asyncio.run(
            run_topological_sweep(
                config_yaml=refined,
                dataset_id=dataset["dataset_id"],
                detail="full",
                response_format="json",
            )
        )
    )
    assert full["detail"] == "full"
    assert "metrics" in full
    assert "component_sizes" in full["metrics"]
    assert "config_yaml_normalized" in full

    # Verify that response defaults to markdown format
    markdown = asyncio.run(
        run_topological_sweep(
            config_yaml=refined,
            dataset_id=dataset["dataset_id"],
        )
    )
    assert "# Sweep Run Complete" in markdown
    assert "## Key Metrics" in markdown
    assert not markdown.lstrip().startswith("{")


def test_run_topological_sweep_can_use_active_config_without_args(tmp_path):
    _sessions.clear()
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    create_response = json.loads(
        asyncio.run(create_config(dataset["dataset_id"], "active_config"))
    )
    refined = json.loads(
        asyncio.run(
            refine_config(
                overrides={"pca_dims": [2], "epsilon_values": [0.5], "n_reps": 1}
            )
        )
    )

    active_before = json.loads(asyncio.run(get_runtime_context()))
    result = json.loads(asyncio.run(run_topological_sweep(response_format="json")))

    assert create_response["status"] == "ok"
    assert refined["status"] == "ok"
    assert active_before["active_config_yaml"] == refined["config_yaml"]
    assert result["status"] == "ok"
    assert result["dataset_id"] == dataset["dataset_id"]
    assert "config_yaml_normalized" not in result

    with_config = json.loads(
        asyncio.run(
            run_topological_sweep(include_config_yaml=True, response_format="json")
        )
    )
    assert with_config["config_yaml_normalized"] == refined["config_yaml"]


def test_refine_config_active_supports_nested_and_mixed_overrides(tmp_path):
    """refine_config (active mode) flattens nested, mixed, and flat overrides."""
    _sessions.clear()
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    asyncio.run(create_config(dataset["dataset_id"], "active_config"))

    # Pass a mixture of nested projection shorthand and flat keys (n_reps)
    refined = json.loads(
        asyncio.run(
            refine_config(
                overrides={
                    "sweep": {"projection_dimensions": [3], "epsilon_values": [0.6]},
                    "n_reps": 2,
                }
            )
        )
    )

    assert refined["status"] == "ok"
    assert "sweep.projection_dimensions" in refined["applied_overrides"]
    assert "sweep.epsilon_values" in refined["applied_overrides"]
    assert "output.n_reps" in refined["applied_overrides"]

    active_after = json.loads(asyncio.run(get_runtime_context()))
    assert "dimensions:" in active_after["active_config_yaml"]
    assert "epsilon:" in active_after["active_config_yaml"]
    assert "n_reps: 2" in active_after["active_config_yaml"]


def test_dossier_inherits_construction_threshold(tmp_path):
    """generate_cluster_dossier inherits construction_threshold when arg omitted."""
    _sessions.clear()
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"], "dossier_inherit"))
    )
    refined = json.loads(
        asyncio.run(
            refine_config(
                config_yaml,
                {
                    "projection_method": "pca",
                    "pca_dims": [2],
                    "epsilon_values": [0.5],
                    "construction_threshold": 0.0,
                    "n_reps": 1,
                },
            )
        )
    )["config_yaml"]
    asyncio.run(
        run_topological_sweep(config_yaml=refined, dataset_id=dataset["dataset_id"])
    )

    inherited = json.loads(
        asyncio.run(generate_cluster_dossier(response_format="json"))
    )
    assert "construction_threshold" in inherited
    assert "interpretation_edge_weight_threshold" in inherited
    assert inherited["cluster_assignment_id"].startswith("ca_")
    assert (
        inherited["cluster_assignment"]["cluster_assignment_id"]
        == inherited["cluster_assignment_id"]
    )
    assert inherited["threshold_inherited"] is True
    assert (
        inherited["interpretation_edge_weight_threshold"]
        == inherited["construction_threshold"]
    )

    explicit = json.loads(
        asyncio.run(
            generate_cluster_dossier(
                interpretation_edge_weight_threshold=0.5,
                response_format="json",
            )
        )
    )
    assert explicit["threshold_inherited"] is False
    assert explicit["interpretation_edge_weight_threshold"] == 0.5
    assert explicit["threshold_surface"]["status"] == "stricter_than_construction"


def test_dossier_rejects_invalid_interpretation_threshold(tmp_path):
    _sessions.clear()
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"], "bad_threshold"))
    )
    refined = json.loads(
        asyncio.run(
            refine_config(
                config_yaml, {"pca_dims": [2], "epsilon_values": [0.5], "n_reps": 1}
            )
        )
    )["config_yaml"]
    asyncio.run(
        run_topological_sweep(config_yaml=refined, dataset_id=dataset["dataset_id"])
    )

    payload = json.loads(
        asyncio.run(
            generate_cluster_dossier(
                interpretation_edge_weight_threshold=-0.1,
                response_format="json",
            )
        )
    )

    assert payload["status"] == "error"
    assert "between 0.0 and 1.0" in payload["reason"]


def test_dossier_explicit_spectral_method_is_not_overridden():
    _sessions.clear()
    session = _get_session(None)
    weighted = np.full((30, 30), 0.4, dtype=float)
    np.fill_diagonal(weighted, 0.0)
    weighted[:15, :15] = 0.9
    weighted[15:, 15:] = 0.9
    np.fill_diagonal(weighted, 0.0)
    session.model = SimpleNamespace(
        weighted_adjacency=weighted,
        cosmic_graph=nx.from_numpy_array((weighted > 0.2).astype(int)),
        resolved_construction_threshold=0.2,
    )
    session.data = pd.DataFrame(
        {
            "x": np.r_[np.ones(15), np.zeros(15)],
            "y": np.r_[np.zeros(15), np.ones(15)],
        }
    )

    payload = json.loads(
        asyncio.run(generate_cluster_dossier(method="spectral", response_format="json"))
    )

    assert payload["status"] == "ok"
    assert payload["cluster_result"]["method_used"] == "spectral"
    assert payload["interpretation_edge_weight_threshold"] == 0.0
    assert payload["threshold_inherited"] is False
    assert payload["threshold_source"] == "spectral_default_full_affinity"
    assert payload["threshold_surface"]["status"] == "full_affinity_spectral"

    sparse_payload = json.loads(
        asyncio.run(
            generate_cluster_dossier(
                method="spectral",
                interpretation_edge_weight_threshold=0.2,
                response_format="json",
            )
        )
    )

    assert sparse_payload["status"] == "ok"
    assert sparse_payload["cluster_result"]["method_used"] == "spectral"
    assert sparse_payload["interpretation_edge_weight_threshold"] == 0.2
    assert sparse_payload["threshold_surface"]["status"] == "matched_explicit"


def test_generate_cluster_dossier_spectral_failure_is_diagnostic():
    _sessions.clear()
    session = _get_session(None)
    session.model = SimpleNamespace(
        weighted_adjacency=np.zeros((5, 5), dtype=float),
        resolved_construction_threshold=0.0,
    )
    session.data = pd.DataFrame({"x": [0, 1, 2, 3, 4]})
    session.latest_run_id = "run_spectral_fail"

    markdown = asyncio.run(
        generate_cluster_dossier(
            method="spectral",
            max_k=4,
            response_format="markdown",
        )
    )
    summary = json.loads(
        asyncio.run(
            generate_cluster_dossier(
                method="spectral",
                max_k=4,
                response_format="json",
            )
        )
    )
    full = json.loads(
        asyncio.run(
            generate_cluster_dossier(
                method="spectral",
                max_k=4,
                detail="full",
                response_format="json",
            )
        )
    )

    assert markdown.startswith("# No Stable Spectral Cut")
    assert "Affinity components: 5" in markdown
    assert "k evaluated: 2-4" in markdown
    assert "candidate_scores" not in markdown
    assert summary["error_code"] == "NO_STABLE_SPECTRAL_CUT"
    assert summary["details"]["affinity_component_count"] == 5
    assert summary["details"]["giant_component_size"] == 1
    assert "candidate_scores" not in summary["details"]
    assert full["error_code"] == "NO_STABLE_SPECTRAL_CUT"
    assert "candidate_scores" in full["details"]


def test_sweep_response_contains_stability_summary(tmp_path):
    """run_topological_sweep with auto threshold emits threshold_stability_summary."""
    _sessions.clear()
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    # create_config defaults construction_threshold to "auto"
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"], "auto_threshold"))
    )
    refined = json.loads(
        asyncio.run(
            refine_config(
                config_yaml, {"pca_dims": [2], "epsilon_values": [0.5], "n_reps": 1}
            )
        )
    )["config_yaml"]

    result = json.loads(
        asyncio.run(
            run_topological_sweep(
                config_yaml=refined,
                dataset_id=dataset["dataset_id"],
                response_format="json",
            )
        )
    )

    assert result["status"] == "ok"
    assert "constructed_graph_connected" in result
    assert "full_affinity_connected" in result
    assert result["spectral_clustering_allowed"] == result["full_affinity_connected"]
    assert "construction_threshold" in result
    assert isinstance(result["construction_threshold"], float)
    assert "threshold_stability_summary" in result
    summary = result["threshold_stability_summary"]
    assert "selected_threshold" in summary
    assert "top_plateaus" in summary
    assert isinstance(summary["top_plateaus"], list)
    if summary["top_plateaus"]:
        first = summary["top_plateaus"][0]
        for key in (
            "start",
            "end",
            "midpoint",
            "length",
            "component_count",
            "singleton_count",
            "singleton_fraction",
        ):
            assert key in first


def test_threshold_stability_curve_defaults_to_sparse_summary(tmp_path):
    _sessions.clear()
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"], "threshold_curve"))
    )
    refined = json.loads(
        asyncio.run(
            refine_config(
                config_yaml, {"pca_dims": [2], "epsilon_values": [0.5], "n_reps": 1}
            )
        )
    )["config_yaml"]
    asyncio.run(
        run_topological_sweep(config_yaml=refined, dataset_id=dataset["dataset_id"])
    )

    markdown = asyncio.run(get_threshold_stability_curve())
    summary = json.loads(
        asyncio.run(get_threshold_stability_curve(response_format="json"))
    )
    full = json.loads(
        asyncio.run(
            get_threshold_stability_curve(detail="full", response_format="json")
        )
    )

    assert "# Threshold Stability" in markdown
    assert "## Threshold Morphology Sample" in markdown
    assert (
        "| Threshold | Components | Top sizes | Giant % | SLR | Singleton % |"
        in markdown
    )
    assert (
        "| Threshold | Components | Top sizes | Giant % | SLR | Singleton % | State |"
        not in markdown
    )
    assert "## Candidate Lenses" in markdown
    assert (
        "| Threshold | Kind | Tier | Components | Top sizes | Giant % | SLR | Singleton % | Why |"
        in markdown
    )
    assert "- H0 longest plateau:" in markdown
    assert "- Current construction morphology:" in markdown
    assert "- H0 plateau morphology:" in markdown
    assert "## Compatible Next Tools" in markdown
    assert not markdown.lstrip().startswith("{")
    assert summary["status"] == "ok"
    assert summary["detail"] == "summary"
    assert "thresholds" not in summary
    assert "component_counts" not in summary
    assert "singleton_counts" not in summary
    assert "resolved_construction_threshold" in summary
    assert "h0_longest_plateau_threshold" in summary
    assert "current_threshold_morphology" in summary
    assert "h0_longest_plateau_morphology" in summary
    assert "matches_current_threshold" in summary
    assert "h0_plateau_readout" in summary
    assert "agent_threshold_options" not in summary
    assert summary["threshold_candidate_policy"] == "balanced"
    assert len(summary["threshold_candidates"]) <= 3
    assert "threshold_candidates_omitted" in summary
    for candidate in summary["threshold_candidates"]:
        assert "morphology" in candidate
        assert "top_component_sizes" in candidate["morphology"]
        assert "giant_fraction" in candidate["morphology"]
        assert "second_largest_ratio" in candidate["morphology"]
        assert "singleton_fraction" in candidate["morphology"]
    assert "threshold_profiles" in summary
    assert len(summary["threshold_profiles"]) <= 10
    assert "threshold_profiles_omitted" in summary
    if summary["threshold_profiles"]:
        profile = summary["threshold_profiles"][0]
        assert "top_component_sizes" in profile
        assert "giant_fraction" in profile
        assert "second_largest_ratio" in profile
        assert "state_label" not in profile
    assert "structural_breakpoints" in summary
    assert len(summary["structural_breakpoints"]) <= 3
    assert all(
        row["event"] != "small_component_absorption"
        for row in summary["structural_breakpoints"]
    )
    assert "structural_breakpoints_omitted" in summary
    assert "structural_breakpoints_filter" in summary
    assert "curve_sample" not in summary
    assert "plateaus" not in summary
    assert "full_detail_available" in summary

    assert full["detail"] == "full"
    assert "resolved_construction_threshold" in full
    assert "thresholds" in full
    assert "component_counts" in full
    assert "singleton_counts" in full
    assert "agent_threshold_options" in full
    assert "threshold_candidates" in full
    for candidate in full["threshold_candidates"]:
        assert "morphology" in candidate
    assert "curve_sample" in full
    assert "plateaus" in full
    assert "threshold_profiles" in full
    assert len(full["threshold_profiles"]) == full["curve_point_count"]
    assert len(full["thresholds"]) == full["curve_point_count"]
    if full["plateaus"]:
        first_plateau = full["plateaus"][0]
        assert "singleton_count" in first_plateau
        assert "singleton_fraction" in first_plateau
        assert "component_mass_profile" in first_plateau


def test_diagnose_cosmic_graph_returns_structured_observables(tmp_path):
    _sessions.clear()
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"], "diagnose_task"))
    )
    refined = json.loads(
        asyncio.run(
            refine_config(
                config_yaml,
                {"pca_dims": [2], "epsilon_values": [0.5], "n_reps": 1},
            )
        )
    )["config_yaml"]
    asyncio.run(
        run_topological_sweep(
            config_yaml=refined,
            dataset_id=dataset["dataset_id"],
            response_format="json",
        )
    )

    objective = json.loads(asyncio.run(diagnose_cosmic_graph()))

    assert objective["status"] == "ok"
    assert objective["source"] == "live"
    assert objective["run_id"] is not None
    assert objective["graph_surface"]["kind"] == "constructed_cosmic_graph"
    assert objective["graph_surface"]["threshold_role"] == "construction"
    assert objective["graph_surface"]["preserves_current_fit"] is True
    assert "scale" in objective
    assert "component_morphology" in objective
    assert "weight_distribution" in objective
    assert "sweep_support" in objective
    assert "observed_patterns" in objective
    assert "risk_factors" in objective
    assert "recommendation" not in objective
    assert "finalization_gate" not in objective
    assert "component_sizes" not in objective["component_morphology"]
    morph = objective["component_morphology"]
    assert morph["nontrivial_component_floor"] >= 1
    assert 0.0 <= morph["nontrivial_mass_fraction"] <= 1.0
    assert 0.0 <= morph["component_size_entropy"] <= 1.0
    assert 0.0 <= morph["top_two_balance"] <= 1.0

    full = json.loads(asyncio.run(diagnose_cosmic_graph(detail="full")))
    assert "component_sizes" in full["component_morphology"]

    markdown = asyncio.run(diagnose_cosmic_graph(response_format="markdown"))
    assert markdown.startswith("# Cosmic Graph Diagnosis")
    assert "Observed patterns:" in markdown
    assert "Risk factors:" in markdown


def test_graph_artifact_estimate_build_and_staleness(tmp_path):
    _sessions.clear()
    csv_path = _write_dataset(tmp_path, rows=24)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"], "artifact"))
    )
    refined = json.loads(
        asyncio.run(
            refine_config(
                config_yaml,
                {"pca_dims": [2], "epsilon_values": [0.5], "n_reps": 1},
            )
        )
    )["config_yaml"]
    first = json.loads(
        asyncio.run(
            run_topological_sweep(
                config_yaml=refined,
                dataset_id=dataset["dataset_id"],
                response_format="json",
            )
        )
    )

    estimate = json.loads(asyncio.run(create_graph_artifact()))
    assert estimate["estimate_only"] is True
    assert estimate["build_call"]["args"]["estimate_only"] is False
    assert estimate["estimated_cost"]["runtime_s"] is not None

    # structural_backbone is a planned surface, not a callable artifact kind.
    unsupported = json.loads(
        asyncio.run(
            create_graph_artifact(kind="structural_backbone", run_id=first["run_id"])
        )
    )
    assert unsupported["status"] == "error"
    assert unsupported["error_code"] == "GRAPH_ARTIFACT_KIND_UNSUPPORTED"

    built = json.loads(
        asyncio.run(
            create_graph_artifact(
                run_id=first["run_id"],
                sketch_dim=2,
                sample_count=12,
                estimate_only=False,
            )
        )
    )
    assert built["status"] == "ok"
    assert built["artifact_id"].startswith("graph_")

    artifact = json.loads(
        asyncio.run(
            get_topological_skeleton(
                surface="artifact",
                artifact_id=built["artifact_id"],
            )
        )
    )
    assert artifact["artifact_staleness"]["stale"] is False
    artifact_diagnosis = json.loads(
        asyncio.run(
            diagnose_cosmic_graph(
                surface="artifact",
                artifact_id=built["artifact_id"],
            )
        )
    )
    assert artifact_diagnosis["source"] == "artifact"
    assert artifact_diagnosis["graph_surface"]["artifact_id"] == built["artifact_id"]
    assert artifact_diagnosis["artifact_staleness"]["stale"] is False

    asyncio.run(
        run_topological_sweep(
            config_yaml=refined,
            dataset_id=dataset["dataset_id"],
            response_format="json",
        )
    )
    stale = json.loads(
        asyncio.run(
            get_topological_skeleton(
                surface="artifact",
                artifact_id=built["artifact_id"],
            )
        )
    )
    assert stale["artifact_staleness"]["stale"] is True
    old_curve = json.loads(
        asyncio.run(
            get_threshold_stability_curve(
                run_id=built["run_id"],
                detail="full",
                response_format="json",
            )
        )
    )
    assert old_curve["status"] == "error"
    assert old_curve["error_code"] == "RUN_NOT_LIVE"


def test_threshold_mass_profiles_capture_large_structural_transition():
    adj = np.array(
        [
            [0.0, 0.9, 0.6, 0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.88, 0.0, 0.0],
            [0.0, 0.0, 0.88, 0.0, 0.55, 0.0],
            [0.0, 0.0, 0.0, 0.55, 0.0, 0.87],
            [0.0, 0.0, 0.0, 0.0, 0.87, 0.0],
        ],
        dtype=np.float64,
    )

    profile = component_mass_profile(adj, 0.8)

    assert profile["n_nodes"] == 6
    assert profile["component_count"] == 3
    assert profile["top_component_sizes"] == [2, 2, 2]
    assert "top_component_fractions" not in profile
    assert profile["largest_component_fraction"] == 0.3333
    assert profile["singleton_count"] == 0

    breakpoints = structural_breakpoints(
        adj,
        thresholds=[1.0, 0.8, 0.5, 0.0],
        component_counts=[6, 3, 1, 1],
    )

    assert breakpoints[0]["event"] == "large_component_transition"
    assert breakpoints[0]["threshold"] == 0.5
    assert breakpoints[0]["before_top_component_sizes"] == [2, 2, 2]
    assert breakpoints[0]["after_top_component_sizes"] == [6]
    assert breakpoints[0]["absorbed_mass_fraction"] == 0.6667
    assert breakpoints[0]["resulting_component_fraction"] == 1.0
    assert breakpoints[0]["affected_mass_fraction"] == 0.6667


def test_summary_structural_breakpoints_hide_dust_absorption():
    breakpoints = [
        {
            "threshold": 0.7,
            "event": "small_component_absorption",
            "component_count_before": 20,
            "component_count_after": 18,
        },
        {
            "threshold": 0.5,
            "event": "large_component_transition",
            "component_count_before": 3,
            "component_count_after": 1,
        },
        {
            "threshold": 0.4,
            "event": "mixed_component_transition",
            "component_count_before": 5,
            "component_count_after": 3,
        },
    ]

    summary = _summary_structural_breakpoints(breakpoints)

    assert [row["event"] for row in summary] == [
        "large_component_transition",
        "mixed_component_transition",
    ]


def test_threshold_readout_calls_singleton_plateau_dust():
    readout = _threshold_agent_readout(
        {
            "largest_component_fraction": 0.001,
            "singleton_fraction": 0.995,
            "small_component_mass_fraction": 0.999,
        },
        [],
    )

    assert "singleton-dominated" in readout
    assert "not a construction or cohort threshold" in readout


def test_threshold_next_tools_are_contextual_for_giant_tail():
    lines = _threshold_next_tool_lines(
        {
            "current_threshold_morphology": {
                "giant_fraction": 0.945,
                "second_largest_ratio": 0.002,
                "singleton_fraction": 0.046,
            }
        }
    )

    assert lines[0].startswith('- `generate_cluster_dossier(method="spectral")`')
    assert "dominant component" in lines[0]
    assert "only when the tail/outlier components" in lines[1]


def test_threshold_next_tools_prefer_clean_component_lens():
    lines = _threshold_next_tool_lines(
        {
            "current_threshold_morphology": {
                "giant_fraction": 0.96,
                "second_largest_ratio": 0.002,
                "singleton_fraction": 0.04,
            },
            "threshold_candidates": [
                {
                    "interpretability_tier": "balanced",
                    "threshold": 0.2246,
                    "why": "Candidate exposes nontrivial components.",
                }
            ],
        }
    )

    assert lines[0].startswith(
        '- `generate_cluster_dossier(method="components", '
        "interpretation_edge_weight_threshold=0.2246)`"
    )
    assert "natural H0 components" in lines[0]
    assert "spectral" in lines[1]


def test_threshold_next_tools_ignore_low_tier_candidates_for_giant_tail():
    # Candidates exist but none clear the report_ready/balanced bar, so there is
    # no clean component lens to recommend: guidance must fall through to the
    # giant-tail branch (spectral first), not fabricate a components lens.
    lines = _threshold_next_tool_lines(
        {
            "current_threshold_morphology": {
                "giant_fraction": 0.96,
                "second_largest_ratio": 0.002,
                "singleton_fraction": 0.04,
            },
            "threshold_candidates": [
                {
                    "interpretability_tier": "giant_component_with_dust",
                    "threshold": 0.31,
                },
                {"interpretability_tier": "exploratory", "threshold": 0.42},
            ],
        }
    )

    assert 'method="spectral"' in lines[0]
    assert "only when the tail/outlier components" in lines[1]


def test_prepared_threshold_graph_matches_dense_component_partitions():
    adj = np.array(
        [
            [0.0, 0.9, 0.6, 0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.88, 0.0, 0.0],
            [0.0, 0.0, 0.88, 0.0, 0.55, 0.0],
            [0.0, 0.0, 0.0, 0.55, 0.0, 0.87],
            [0.0, 0.0, 0.0, 0.0, 0.87, 0.0],
        ],
        dtype=np.float64,
    )
    prepared = prepare_threshold_graph(adj)

    dense_sizes, dense_labels = component_state_at_threshold(adj, 0.8)
    sparse_sizes, sparse_labels = component_state_at_threshold(prepared, 0.8)

    assert sparse_sizes == dense_sizes
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            assert (sparse_labels[i] == sparse_labels[j]) == (
                dense_labels[i] == dense_labels[j]
            )


def test_prepared_threshold_graph_matches_dense_threshold_payloads():
    adj = np.array(
        [
            [0.0, 0.9, 0.6, 0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.88, 0.0, 0.0],
            [0.0, 0.0, 0.88, 0.0, 0.55, 0.0],
            [0.0, 0.0, 0.0, 0.55, 0.0, 0.87],
            [0.0, 0.0, 0.0, 0.0, 0.87, 0.0],
        ],
        dtype=np.float64,
    )
    prepared = prepare_threshold_graph(adj)
    thresholds = [1.0, 0.8, 0.5, 0.0]
    component_counts = [6, 3, 1, 1]

    assert component_mass_profile(prepared, 0.8) == component_mass_profile(adj, 0.8)
    assert structural_breakpoints(
        prepared,
        thresholds=thresholds,
        component_counts=component_counts,
    ) == structural_breakpoints(
        adj,
        thresholds=thresholds,
        component_counts=component_counts,
    )


def test_threshold_options_are_policy_aware_without_excess_payload():
    adj = np.array(
        [
            [0.0, 0.9, 0.6, 0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.88, 0.0, 0.0],
            [0.0, 0.0, 0.88, 0.0, 0.55, 0.0],
            [0.0, 0.0, 0.0, 0.55, 0.0, 0.87],
            [0.0, 0.0, 0.0, 0.0, 0.87, 0.0],
        ],
        dtype=np.float64,
    )
    from pulsar._pulsar import find_stable_thresholds

    stability = find_stable_thresholds(adj, num_bins=10)
    options = agent_threshold_options(
        adj,
        stability.top_k_plateaus(10),
        [float(t) for t in stability.thresholds],
        [int(c) for c in stability.component_counts],
        policy="detail_seeking",
        max_candidates=5,
    )

    assert options["policy"] == "detail_seeking"
    assert options["selection_strategy"] == "transition_adjacent"
    assert len(options["candidates"]) <= 5
    assert "recommended_action" in options
    assert options["transition_adjacent_candidates"]
    assert options["candidates"][0]["candidate_kind"] == "transition_adjacent"
    for candidate in options["candidates"]:
        assert "threshold" in candidate
        assert "candidate_kind" in candidate
        assert "mass_shape" in candidate
        assert "best_for" in candidate
        assert "avoid_for" in candidate
        assert "why" in candidate


def test_report_ready_threshold_options_do_not_recommend_giant_dust():
    adj = np.zeros((101, 101), dtype=np.float64)
    for node in range(99):
        adj[node, node + 1] = 0.9
        adj[node + 1, node] = 0.9
    from pulsar._pulsar import find_stable_thresholds

    stability = find_stable_thresholds(adj, num_bins=10)
    options = agent_threshold_options(
        adj,
        stability.top_k_plateaus(10),
        [float(t) for t in stability.thresholds],
        [int(c) for c in stability.component_counts],
        policy="report_ready",
        max_candidates=5,
    )

    assert options["policy"] == "report_ready"
    assert all(
        candidate["interpretability_tier"] != "giant_component_with_dust"
        for candidate in options["candidates"]
    )


def test_threshold_breakpoints_rank_absorbed_mass_not_giant_result():
    n = 101
    adj = np.zeros((n, n), dtype=np.float64)
    for node in range(99):
        adj[node, node + 1] = 0.9
        adj[node + 1, node] = 0.9
    adj[99, 100] = 0.5
    adj[100, 99] = 0.5

    breakpoints = structural_breakpoints(
        adj,
        thresholds=[0.8, 0.4],
        component_counts=[2, 1],
    )

    assert breakpoints[0]["event"] == "small_component_absorption"
    assert breakpoints[0]["before_top_component_sizes"] == [100, 1]
    assert breakpoints[0]["after_top_component_sizes"] == [101]
    assert breakpoints[0]["absorbed_mass_fraction"] == 0.0099
    assert breakpoints[0]["resulting_component_fraction"] == 1.0


def test_finalization_gate_blocks_early_dominant_component():
    config_yaml = """
sweep:
  pca:
    dimensions:
      values: [10, 20]
"""
    gate = _finalization_gate(
        {
            "giant_fraction": 0.9724,
            "density": 0.77,
            "n_ball_maps": 32,
        },
        sweep_count=2,
        config_yaml=config_yaml,
    )

    assert gate["status"] == "blocked"
    assert gate["code"] == "UNRESOLVED_DOMINANT_COMPONENT"
    assert gate["suggested_refinement"]["projection_dimensions"] == [10, 12, 14, 16]
    assert gate["suggested_refinement"]["projection_seeds"] == [42, 7, 13]
    assert gate["suggested_refinement"]["epsilon_steps_min"] == 24


def test_finalization_gate_warns_when_clean_component_lens_exists():
    gate = _finalization_gate(
        {
            "giant_fraction": 0.9724,
            "density": 0.77,
            "n_ball_maps": 32,
        },
        sweep_count=2,
        config_yaml="sweep: {}",
        threshold_curve_summary={
            "threshold_candidates": [
                {
                    # Stable-plateau candidates carry their cut under "threshold",
                    # not "midpoint" -- mirror the real candidate schema.
                    "interpretability_tier": "report_ready",
                    "threshold": 0.2246,
                }
            ]
        },
    )

    assert gate["status"] == "caution"
    assert gate["code"] == "DOMINANT_COMPONENT_HAS_COMPONENT_LENS"
    action = gate["recommended_action"]
    assert action["tool"] == "generate_cluster_dossier"
    assert action["method"] == "components"
    assert action["interpretation_edge_weight_threshold"] == 0.2246


def test_finalization_gate_blocks_when_only_dusty_lens_exists():
    # A giant-dominated slice that splits off only dust is tagged
    # giant_component_with_dust, NOT report_ready/balanced. The tier itself is
    # the (scale-invariant) substance floor, so such a candidate must NOT
    # downgrade the hard block. Locks in that guarantee.
    gate = _finalization_gate(
        {
            "giant_fraction": 0.9724,
            "density": 0.77,
            "n_ball_maps": 32,
        },
        sweep_count=2,
        config_yaml="sweep: {}",
        threshold_curve_summary={
            "threshold_candidates": [
                {
                    "interpretability_tier": "giant_component_with_dust",
                    "threshold": 0.31,
                },
                {"interpretability_tier": "weak_candidate", "threshold": 0.4},
            ]
        },
    )

    assert gate["status"] == "blocked"
    assert gate["code"] == "UNRESOLVED_DOMINANT_COMPONENT"
    assert "recommended_action" not in gate


def test_finalization_gate_allows_resolved_dominant_component():
    gate = _finalization_gate(
        {
            "giant_fraction": 0.96,
            "density": 0.2,
            "n_ball_maps": 72,
        },
        sweep_count=3,
        config_yaml="sweep: {}",
    )

    assert gate["status"] == "ok"


def test_sweep_markdown_renders_caution_gate_recommendation():
    markdown = sweep_payload_to_markdown(
        {
            "run_id": "run_test",
            "dataset_id": "ds_test",
            "finalization_gate": {
                "status": "caution",
                "code": "DOMINANT_COMPONENT_HAS_COMPONENT_LENS",
                "message": "A stricter component lens is available.",
                "recommended_action": {
                    "tool": "generate_cluster_dossier",
                    "method": "components",
                    "interpretation_edge_weight_threshold": 0.2246,
                    "reason": "Use the clean H0 component slice.",
                },
            },
            "key_metrics": {},
            "component_sizes_preview": [],
            "next_tools": [],
        }
    )

    assert "- Status: `caution`" in markdown
    assert "- Code: `DOMINANT_COMPONENT_HAS_COMPONENT_LENS`" in markdown
    assert (
        '- Recommended action: `generate_cluster_dossier(method="components", '
        "interpretation_edge_weight_threshold=0.2246)`"
    ) in markdown


def test_run_topological_sweep_reports_pca_cache_hit_on_repeat(tmp_path):
    _sessions.clear()
    csv_path = _write_dataset(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(csv_path)))
    config_yaml = _extract_config_yaml(
        asyncio.run(create_config(dataset["dataset_id"], "cache_hit"))
    )
    refined = json.loads(
        asyncio.run(
            refine_config(
                config_yaml, {"pca_dims": [2], "epsilon_values": [0.5], "n_reps": 1}
            )
        )
    )["config_yaml"]

    first = json.loads(
        asyncio.run(
            run_topological_sweep(
                config_yaml=refined,
                dataset_id=dataset["dataset_id"],
                response_format="json",
            )
        )
    )
    second = json.loads(
        asyncio.run(
            run_topological_sweep(
                config_yaml=refined,
                dataset_id=dataset["dataset_id"],
                response_format="json",
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
