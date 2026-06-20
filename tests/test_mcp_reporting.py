import asyncio
import json

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

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
    session = _get_session(None)
    session.data = pd.DataFrame({"fips": ["001", "002", "003"]})
    session.clusters = pd.Series([0, 1, 0])
    output_path = tmp_path / "labeled.csv"

    result = asyncio.run(
        export_labeled_data(
            {"0": "Moderate Desert", "1": "Baseline"},
            str(output_path),
            ctx=None,
        )
    )

    assert "Successfully exported" in result
    exported = pd.read_csv(output_path, dtype={"fips": str})
    assert exported["topological_cluster_name"].tolist() == [
        "Moderate Desert",
        "Baseline",
        "Moderate Desert",
    ]


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
