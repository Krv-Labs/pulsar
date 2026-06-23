import asyncio
import json

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from pulsar.mcp.registry import registry
from pulsar.mcp.session import _get_session, _sessions
from pulsar.mcp.tools.reporting import (
    _normalize_cluster_names,
    export_html_report,
    export_labeled_data,
)


def test_normalize_cluster_names_accepts_flat_string_mapping():
    names = _normalize_cluster_names(
        {"0": "Moderate Desert", "1": "Baseline"},
        valid_cluster_ids={0, 1},
        require_all=True,
    )

    assert names == {0: "Moderate Desert", 1: "Baseline"}


def test_normalize_cluster_names_rejects_non_integer_keys():
    with pytest.raises(ToolError, match="integer cluster IDs"):
        _normalize_cluster_names(
            {"cluster_0": "Moderate Desert"},
            valid_cluster_ids={0},
            require_all=True,
        )


def test_export_labeled_data_uses_flat_cluster_name_mapping(tmp_path):
    _sessions.clear()
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"fips": ["001", "002", "003"]}).to_csv(input_path, index=False)
    dataset = registry.register_dataset(str(input_path))
    assignment = registry.save_cluster_assignment(
        run_id="run_export_test",
        dataset_id=dataset.dataset_id,
        method="components",
        interpretation_edge_weight_threshold=0.25,
        threshold_source="explicit",
        construction_threshold=0.25,
        labels=[0, 1, 0],
    )
    output_path = tmp_path / "labeled.csv"

    result = json.loads(
        asyncio.run(
            export_labeled_data(
                assignment.cluster_assignment_id,
                {"0": "Moderate Desert", "1": "Baseline"},
                str(output_path),
                ctx=None,
            )
        )
    )

    assert result["status"] == "ok"
    assert result["cluster_assignment_id"] == assignment.cluster_assignment_id
    exported = pd.read_csv(output_path, dtype={"fips": str})
    assert exported["topological_cluster_name"].tolist() == [
        "Moderate Desert",
        "Baseline",
        "Moderate Desert",
    ]


def test_export_labeled_data_requires_known_assignment_handle(tmp_path):
    _sessions.clear()

    result = json.loads(
        asyncio.run(
            export_labeled_data(
                "ca_missing",
                {"0": "Moderate Desert"},
                str(tmp_path / "labeled.csv"),
                ctx=None,
            )
        )
    )

    assert result["status"] == "error"
    assert result["error_code"] == "CLUSTER_ASSIGNMENT_ID_UNKNOWN"
    assert result["details"]["cluster_assignment_id"] == "ca_missing"


def test_export_labeled_data_reports_undurable_assignment_data(tmp_path):
    _sessions.clear()
    assignment = registry.save_cluster_assignment(
        run_id="run_without_dataset",
        dataset_id=None,
        method="components",
        interpretation_edge_weight_threshold=0.25,
        threshold_source="explicit",
        construction_threshold=0.25,
        labels=[0, 1, 0],
    )

    result = json.loads(
        asyncio.run(
            export_labeled_data(
                assignment.cluster_assignment_id,
                {"0": "Moderate Desert", "1": "Baseline"},
                str(tmp_path / "labeled.csv"),
                ctx=None,
            )
        )
    )

    assert result["status"] == "error"
    assert result["error_code"] == "CLUSTER_ASSIGNMENT_DATA_UNAVAILABLE"


def test_export_labeled_data_can_use_session_data_for_undurable_assignment(tmp_path):
    _sessions.clear()
    session = _get_session(None)
    session.data = pd.DataFrame({"fips": ["001", "002", "003"]})
    session.latest_run_id = "run_without_dataset_loaded"
    assignment = registry.save_cluster_assignment(
        run_id=session.latest_run_id,
        dataset_id=None,
        method="components",
        interpretation_edge_weight_threshold=0.25,
        threshold_source="explicit",
        construction_threshold=0.25,
        labels=[0, 1, 0],
    )
    output_path = tmp_path / "labeled.csv"

    result = json.loads(
        asyncio.run(
            export_labeled_data(
                assignment.cluster_assignment_id,
                {"0": "Moderate Desert", "1": "Baseline"},
                str(output_path),
                ctx=None,
            )
        )
    )

    assert result["status"] == "ok"
    exported = pd.read_csv(output_path, dtype={"fips": str})
    assert exported["topological_cluster_name"].tolist() == [
        "Moderate Desert",
        "Baseline",
        "Moderate Desert",
    ]


def test_export_labeled_data_rejects_changed_dataset_file(tmp_path):
    _sessions.clear()
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"fips": ["001", "002", "003"]}).to_csv(input_path, index=False)
    dataset = registry.register_dataset(str(input_path))
    assignment = registry.save_cluster_assignment(
        run_id="run_changed_file",
        dataset_id=dataset.dataset_id,
        method="components",
        interpretation_edge_weight_threshold=0.25,
        threshold_source="explicit",
        construction_threshold=0.25,
        labels=[0, 1, 0],
    )

    # Edit the backing file after registration: now 4 rows, new size/mtime.
    pd.DataFrame({"fips": ["001", "002", "003", "004"]}).to_csv(input_path, index=False)

    result = json.loads(
        asyncio.run(
            export_labeled_data(
                assignment.cluster_assignment_id,
                {"0": "Moderate Desert", "1": "Baseline"},
                str(tmp_path / "labeled.csv"),
                ctx=None,
            )
        )
    )

    assert result["status"] == "error"
    assert result["error_code"] == "CLUSTER_ASSIGNMENT_DATASET_STALE"


def test_export_labeled_data_does_not_rebind_session_data(tmp_path):
    _sessions.clear()
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"fips": ["001", "002", "003"]}).to_csv(input_path, index=False)
    dataset = registry.register_dataset(str(input_path))
    assignment = registry.save_cluster_assignment(
        run_id="run_no_rebind",
        dataset_id=dataset.dataset_id,
        method="components",
        interpretation_edge_weight_threshold=0.25,
        threshold_source="explicit",
        construction_threshold=0.25,
        labels=[0, 1, 0],
    )

    # An unrelated dataset is active in the session; export must read the
    # assignment's dataset without clobbering it or its derived caches.
    session = _get_session(None)
    active_data = pd.DataFrame({"other": [9, 9]})
    session.data = active_data
    session.data_dataset_id = "ds_other_active"
    session.clusters = pd.Series([0, 1])
    session.clusters_run_id = "run_other_active"

    result = json.loads(
        asyncio.run(
            export_labeled_data(
                assignment.cluster_assignment_id,
                {"0": "Moderate Desert", "1": "Baseline"},
                str(tmp_path / "labeled.csv"),
                ctx=None,
            )
        )
    )

    assert result["status"] == "ok"
    assert session.data is active_data
    assert session.data_dataset_id == "ds_other_active"
    assert session.clusters is not None
    assert session.clusters_run_id == "run_other_active"


def test_export_html_report_missing_session_state_gives_recovery_action():
    _sessions.clear()
    session = _get_session(None)
    session.latest_run_id = "run_previous"
    session.dataset_id = "ds_previous"

    payload = json.loads(asyncio.run(export_html_report(ctx=None)))

    assert payload["status"] == "error"
    assert payload["error_code"] == "SESSION_STATE_MISSING"
    assert payload["details"]["latest_run_id"] == "run_previous"
    assert payload["details"]["latest_dataset_id"] == "ds_previous"
    assert "get_runtime_context" in payload["agent_action"]
    assert "run_topological_sweep" in payload["agent_action"]
