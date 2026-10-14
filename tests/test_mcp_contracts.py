"""Catalog-wide MCP contract tests.

Every tool registered by ``register_server_tools`` is checked on the wire (through
the in-memory ``fastmcp.Client``) for annotations, an output schema and described
parameters. Two end-to-end loops (static and longitudinal) then run in a single
client session, which makes FastMCP validate each result's ``structured_content``
against the tool's declared schema.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client, FastMCP

from pulsar.mcp import server as server_mod
from pulsar.mcp.server import register_server_tools
from pulsar.mcp.tools import ALL_TOOLS_LIST
from tests.mcp_helpers import payload_of
from tests.test_mcp_longitudinal import _config_yaml, _write_long
from tests.test_mcp_surface_diagnostics_clustering import (  # noqa: F401
    _isolate_mcp_state,
    _write_blobs_csv,
)

CATALOG_BUDGET_CHARS = 48_000  # ~12k tokens; raise deliberately, not by accident


def _server() -> FastMCP:
    server = FastMCP("t")
    register_server_tools(server)
    return server


async def _list_tools() -> list:
    async with Client(_server()) as client:
        return await client.list_tools()


def test_registration_raises_for_tool_without_meta(monkeypatch):
    async def untagged_tool() -> str:
        """No metadata."""
        return "x"

    monkeypatch.setattr(server_mod, "ALL_TOOLS_LIST", [*ALL_TOOLS_LIST, untagged_tool])
    with pytest.raises(RuntimeError, match="untagged_tool"):
        register_server_tools(FastMCP("t"))


def test_every_tool_declares_contract_on_the_wire():
    tools = asyncio.run(_list_tools())
    assert len(tools) == len(ALL_TOOLS_LIST)
    for tool in tools:
        assert tool.annotations is not None, tool.name
        assert tool.annotations.open_world_hint is False, tool.name
        assert tool.annotations.read_only_hint is not None, tool.name
        assert tool.output_schema, tool.name
        assert (tool.description or "").strip(), tool.name
        for prop, schema in tool.input_schema.get("properties", {}).items():
            assert (schema.get("description") or "").strip(), (tool.name, prop)


def test_tools_list_is_deterministic():
    async def go():
        async with Client(_server()) as client:
            first = [t.model_dump_json() for t in await client.list_tools()]
            second = [t.model_dump_json() for t in await client.list_tools()]
        return first, second

    first, second = asyncio.run(go())
    assert first == second


def test_catalog_budget():
    tools = asyncio.run(_list_tools())
    total = sum(
        len(t.description or "") + len(json.dumps(t.input_schema)) for t in tools
    )
    print(
        f"\nMCP catalog budget: {total} chars (~{total // 4} tokens), limit "
        f"{CATALOG_BUDGET_CHARS}"
    )
    assert total < CATALOG_BUDGET_CHARS


async def _ok(client: Client, tool: str, args: dict[str, Any] | None = None):
    res = await client.call_tool(tool, args or {}, raise_on_error=False)
    assert res.is_error is False, (tool, args, res.content)
    assert isinstance(res.structured_content, dict), tool
    return res


def test_canonical_loop(tmp_path: Path):
    csv = _write_blobs_csv(tmp_path / "blobs.csv")
    out = tmp_path / "labeled.csv"

    async def go():
        async with Client(_server()) as client:
            ing = payload_of(await _ok(client, "ingest_dataset", {"path": str(csv)}))
            dataset_id = ing["dataset_id"]
            await _ok(client, "characterize_dataset", {"dataset_id": dataset_id})
            cfg = payload_of(
                await _ok(client, "create_config", {"dataset_id": dataset_id})
            )
            await _ok(
                client,
                "validate_config",
                {"config_yaml": cfg["config_yaml"], "dataset_id": dataset_id},
            )
            sweep = payload_of(
                await _ok(
                    client,
                    "run_topological_sweep",
                    {"dataset_id": dataset_id, "response_format": "json"},
                )
            )
            run_id = sweep["run_id"]
            await _ok(client, "diagnose_cosmic_graph", {"run_id": run_id})
            await _ok(client, "get_threshold_stability_curve", {"run_id": run_id})
            await _ok(client, "get_topological_skeleton", {"run_id": run_id})
            dossier = payload_of(
                await _ok(
                    client,
                    "generate_cluster_dossier",
                    {"method": "spectral", "detail": "summary", "run_id": run_id},
                )
            )
            cluster_ids = [c["cluster_id"] for c in dossier["clusters"]]
            assert len(cluster_ids) >= 2
            first, second = cluster_ids[:2]
            n_clusters = dossier["clusters_returned"] + dossier["clusters_omitted"]
            await _ok(client, "get_cluster_profile", {"cluster_id": first})
            await _ok(client, "get_feature_signal", {"feature_names": ["f1"]})
            await _ok(client, "get_cluster_signal_matrix")
            await _ok(
                client, "compare_clusters", {"cluster_a": first, "cluster_b": second}
            )
            await _ok(client, "get_sweep_history")
            await _ok(
                client,
                "export_labeled_data",
                {
                    "cluster_assignment_id": dossier["cluster_assignment_id"],
                    "cluster_names": {str(c): f"cohort {c}" for c in range(n_clusters)},
                    "output_path": str(out),
                    "run_id": run_id,
                },
            )
            await _ok(client, "get_runtime_context")
            await _ok(client, "get_workflow_guide")

    asyncio.run(go())
    assert out.exists()


def test_longitudinal_loop(tmp_path: Path):
    async def go():
        async with Client(_server()) as client:
            ing = payload_of(
                await _ok(client, "ingest_dataset", {"path": _write_long(tmp_path)})
            )
            sampled = payload_of(
                await _ok(
                    client,
                    "sample_longitudinal_entities",
                    {
                        "dataset_id": ing["dataset_id"],
                        "entity_column": "patient_id",
                        "time_column": "hour",
                    },
                )
            )
            dataset_id = sampled["dataset_id"]
            await _ok(client, "create_config", {"dataset_id": dataset_id})
            built = payload_of(
                await _ok(
                    client,
                    "build_longitudinal_graph",
                    {
                        "dataset_id": dataset_id,
                        "entity_column": "patient_id",
                        "time_column": "hour",
                        "config_yaml": _config_yaml(),
                    },
                )
            )
            lid = built["longitudinal_id"]
            await _ok(client, "diagnose_longitudinal_graph", {"longitudinal_id": lid})
            await _ok(client, "get_trajectory_archetypes", {"longitudinal_id": lid})
            # Omitting longitudinal_id addresses the most recent panel.
            await _ok(client, "get_cross_time_neighbors", {"entity_id": "p0", "t": 1})
            await _ok(client, "classify_trajectories")

    asyncio.run(go())
