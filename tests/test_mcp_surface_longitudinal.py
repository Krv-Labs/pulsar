"""Wire-level contract tests for the longitudinal MCP tools.

FastMCP validates ``structured_content`` against ``output_schema`` only when a tool
is called through a server, so every tool x representation x response_format x
detail combination is exercised through an in-memory ``fastmcp.Client``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client, FastMCP

from pulsar.mcp import registry as registry_mod
from pulsar.mcp import session as session_mod
from pulsar.mcp.server import register_server_tools
from pulsar.mcp.toolmeta import tool_meta_of
from pulsar.mcp.tools.longitudinal import (
    build_longitudinal_graph,
    classify_trajectories,
    diagnose_longitudinal_graph,
    get_cross_time_neighbors,
    get_trajectory_archetypes,
)
from tests.test_mcp_longitudinal import _config_yaml, _write_long

TOOLS = {
    "build_longitudinal_graph": (build_longitudinal_graph, False, None),
    "diagnose_longitudinal_graph": (diagnose_longitudinal_graph, True, True),
    "get_trajectory_archetypes": (get_trajectory_archetypes, True, True),
    "get_cross_time_neighbors": (get_cross_time_neighbors, True, True),
    "classify_trajectories": (classify_trajectories, True, True),
}
FORMATS = ["json", "markdown"]
DETAILS = ["summary", "full"]


@pytest.fixture(autouse=True)
def _isolate_mcp_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache_dir = tmp_path / "pulsar_mcp_cache"
    monkeypatch.setattr(registry_mod, "_CACHE_DIR", cache_dir)
    monkeypatch.setattr(registry_mod, "_DATASETS_PATH", cache_dir / "datasets.json")
    monkeypatch.setattr(registry_mod, "_DATASET_FILES_DIR", cache_dir / "datasets")
    monkeypatch.setattr(registry_mod, "_UPLOADS_DIR", cache_dir / "uploads")
    monkeypatch.setattr(registry_mod, "_RUNS_DIR", cache_dir / "runs")
    monkeypatch.setattr(
        registry_mod, "_CLUSTER_ASSIGNMENTS_DIR", cache_dir / "cluster_assignments"
    )
    monkeypatch.setattr(registry_mod, "_LOCK_PATH", cache_dir / ".registry.lock")
    registry_mod.MCPRegistry()
    session_mod._sessions.clear()
    yield
    session_mod._sessions.clear()


def _server() -> FastMCP:
    server = FastMCP("t")
    register_server_tools(server)
    return server


def _ok(result: Any) -> dict:
    assert result.is_error is False, result.content
    assert isinstance(result.structured_content, dict)
    return result.structured_content


async def _sample(client: Client, tmp_path: Path) -> str:
    ingested = _ok(
        await client.call_tool("ingest_dataset", {"path": _write_long(tmp_path)})
    )
    sampled = _ok(
        await client.call_tool(
            "sample_longitudinal_entities",
            {
                "dataset_id": ingested["dataset_id"],
                "entity_column": "patient_id",
                "time_column": "hour",
            },
        )
    )
    return sampled["dataset_id"]


async def _build(client: Client, dataset_id: str, **kw: Any) -> dict:
    args = {
        "dataset_id": dataset_id,
        "entity_column": "patient_id",
        "time_column": "hour",
        "config_yaml": _config_yaml(),
    }
    args.update(kw)
    return _ok(await client.call_tool("build_longitudinal_graph", args))


def _text(result: Any) -> str:
    return result.content[0].text


def test_wire_annotations_schema_and_descriptions():
    asyncio.run(_wire())


async def _wire():
    async with Client(_server()) as client:
        tools = {t.name: t for t in await client.list_tools()}
    for name, (fn, read_only, idempotent) in TOOLS.items():
        tool = tools[name]
        assert tool.annotations.read_only_hint is read_only, name
        assert tool.annotations.open_world_hint is False, name
        assert tool.annotations.destructive_hint is False, name
        if idempotent is not None:
            assert tool.annotations.idempotent_hint is idempotent, name
        assert tool.output_schema is not None and tool.output_schema["type"] == "object"
        assert tool.output_schema["properties"]["longitudinal_id"]
        props = tool.input_schema["properties"]
        assert "ctx" not in props
        for pname, prop in props.items():
            assert prop.get("description"), f"{name}.{pname} lacks a description"
        assert len(tool.description) < 700, name
        meta = tool_meta_of(fn)
        assert meta is not None and meta.read_only is read_only
    assert tool_meta_of(build_longitudinal_graph).timeout == 1800


@pytest.mark.parametrize("representation", ["trajectory", "temporal", "both"])
@pytest.mark.parametrize("response_format", FORMATS)
@pytest.mark.parametrize("detail", DETAILS)
def test_build_variants(tmp_path, representation, response_format, detail):
    async def run():
        async with Client(_server()) as client:
            dataset_id = await _sample(client, tmp_path)
            args = {
                "dataset_id": dataset_id,
                "entity_column": "patient_id",
                "time_column": "hour",
                "config_yaml": _config_yaml(),
                "representation": representation,
                "response_format": response_format,
                "detail": detail,
            }
            res = await client.call_tool("build_longitudinal_graph", args)
            payload = _ok(res)
            assert payload["longitudinal_id"].startswith("lng_")
            assert payload["representation"] == representation
            assert payload["detail"] == detail
            assert ("config_yaml" in payload) == (detail == "full")
            assert ("trajectory" in payload["graph_surface"]) == (
                representation != "temporal"
            )
            assert ("temporal" in payload["graph_surface"]) == (
                representation != "trajectory"
            )
            if response_format == "json":
                assert json.loads(_text(res)) == payload
            else:
                assert _text(res).startswith("# Longitudinal Panel")

    asyncio.run(run())


@pytest.mark.parametrize("representation", ["auto", "trajectory", "temporal"])
@pytest.mark.parametrize("response_format", FORMATS)
@pytest.mark.parametrize("detail", DETAILS)
def test_diagnose_variants(tmp_path, representation, response_format, detail):
    async def run():
        async with Client(_server()) as client:
            built = await _build(
                client, await _sample(client, tmp_path), representation="both"
            )
            res = await client.call_tool(
                "diagnose_longitudinal_graph",
                {
                    "longitudinal_id": built["longitudinal_id"],
                    "representation": representation,
                    "response_format": response_format,
                    "detail": detail,
                },
            )
            payload = _ok(res)
            assert payload["longitudinal_id"] == built["longitudinal_id"]
            surfaces = payload["surfaces"]
            assert set(surfaces) == (
                {"trajectory", "temporal"}
                if representation == "auto"
                else {representation}
            )
            if "trajectory" in surfaces:
                assert ("component_morphology" in surfaces["trajectory"]) == (
                    detail == "full"
                )
            if response_format == "markdown":
                assert _text(res).startswith("# Longitudinal Diagnosis")

    asyncio.run(run())


@pytest.mark.parametrize("response_format", FORMATS)
@pytest.mark.parametrize("detail", DETAILS)
def test_archetype_variants(tmp_path, response_format, detail):
    async def run():
        async with Client(_server()) as client:
            await _build(client, await _sample(client, tmp_path))
            # longitudinal_id omitted: most recent panel is used.
            res = await client.call_tool(
                "get_trajectory_archetypes",
                {"response_format": response_format, "detail": detail},
            )
            payload = _ok(res)
            assert payload["longitudinal_id"].startswith("lng_")
            assert payload["archetypes"]
            if response_format == "markdown":
                assert _text(res).startswith("# Trajectory Archetypes")

    asyncio.run(run())


@pytest.mark.parametrize("response_format", FORMATS)
@pytest.mark.parametrize("direction", ["any", "forward", "backward"])
def test_cross_time_variants(tmp_path, response_format, direction):
    async def run():
        async with Client(_server()) as client:
            await _build(client, await _sample(client, tmp_path))
            by_entity = _ok(
                await client.call_tool(
                    "get_cross_time_neighbors",
                    {
                        "entity_id": "p0",
                        "t": 1,
                        "direction": direction,
                        "response_format": response_format,
                    },
                )
            )
            assert by_entity["source"]["entity_id"] == "p0"
            by_obs = _ok(
                await client.call_tool(
                    "get_cross_time_neighbors",
                    {
                        "observation_id": by_entity["source"]["obs_id"],
                        "direction": direction,
                        "response_format": response_format,
                    },
                )
            )
            assert by_obs["source"] == by_entity["source"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "method", ["complexity", "transition", "sequence", "levenshtein", "dtw"]
)
@pytest.mark.parametrize("response_format", FORMATS)
def test_classify_variants(tmp_path, method, response_format):
    async def run():
        async with Client(_server()) as client:
            await _build(client, await _sample(client, tmp_path))
            res = await client.call_tool(
                "classify_trajectories",
                {"method": method, "response_format": response_format},
            )
            payload = _ok(res)
            assert payload["method"] == method
            assert payload["detail"] == "summary"
            assert "classification" not in payload
            n = payload["n_entities"]
            assert payload["entities_returned"] == len(
                payload["classification_preview"]
            )
            assert payload["entities_returned"] <= 25
            assert payload["entities_returned"] + payload["entities_omitted"] == n
            if response_format == "markdown":
                assert _text(res).startswith("# Trajectory Classification")

    asyncio.run(run())


def test_classify_full_detail_returns_all_entities(tmp_path):
    async def run():
        async with Client(_server()) as client:
            await _build(client, await _sample(client, tmp_path))
            summary = _ok(
                await client.call_tool(
                    "classify_trajectories", {"response_format": "json"}
                )
            )
            full = _ok(
                await client.call_tool(
                    "classify_trajectories",
                    {"detail": "full", "response_format": "json"},
                )
            )
            assert full["detail"] == "full"
            assert len(full["classification"]) == full["n_entities"]
            assert full["entities_returned"] == full["n_entities"]
            assert full["entities_omitted"] == 0
            # Preview is the leading slice of the full map.
            assert (
                list(summary["classification_preview"])
                == list(full["classification"])[
                    : len(summary["classification_preview"])
                ]
            )

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Error paths (is_error results are not schema-validated; envelope is checked)
# ---------------------------------------------------------------------------
def _assert_error(res: Any, tool: str, code: str) -> dict:
    assert res.is_error is True
    env = res.structured_content
    assert env["status"] == "error" and env["tool"] == tool
    assert env["error_code"] == code, env
    assert env["agent_action"]
    assert json.loads(_text(res)) == env
    return env


@pytest.mark.parametrize(
    "tool",
    [
        "diagnose_longitudinal_graph",
        "get_trajectory_archetypes",
        "get_cross_time_neighbors",
        "classify_trajectories",
    ],
)
def test_unknown_longitudinal_id(tmp_path, tool):
    async def run():
        async with Client(_server()) as client:
            # Nothing built yet, and an explicit wrong id after one build.
            args = {"longitudinal_id": "lng_missing"}
            res = await client.call_tool(tool, {}, raise_on_error=False)
            _assert_error(res, tool, "LONGITUDINAL_ID_UNKNOWN")
            await _build(client, await _sample(client, tmp_path))
            res = await client.call_tool(tool, args, raise_on_error=False)
            env = _assert_error(res, tool, "LONGITUDINAL_ID_UNKNOWN")
            assert env["details"] == {"longitudinal_id": "lng_missing"}

    asyncio.run(run())


def test_build_errors(tmp_path):
    async def run():
        async with Client(_server()) as client:
            tool = "build_longitudinal_graph"
            res = await client.call_tool(tool, {}, raise_on_error=False)
            _assert_error(res, tool, "PANEL_KEYS_MISSING")
            res = await client.call_tool(
                tool,
                {
                    "dataset_id": "ds_nope",
                    "entity_column": "patient_id",
                    "time_column": "hour",
                    "config_yaml": _config_yaml(),
                },
                raise_on_error=False,
            )
            _assert_error(res, tool, "DATASET_ID_UNKNOWN")
            dataset_id = await _sample(client, tmp_path)
            res = await client.call_tool(
                tool,
                {
                    "dataset_id": dataset_id,
                    "entity_column": "patient_id",
                    "time_column": "hour",
                    "config_yaml": _config_yaml(),
                    "similarity_threshold": -1.0,
                },
                raise_on_error=False,
            )
            _assert_error(res, tool, "INVALID_ARGUMENT")

    asyncio.run(run())


def test_diagnose_missing_surface_and_trajectory_required(tmp_path):
    async def run():
        async with Client(_server()) as client:
            built = await _build(
                client, await _sample(client, tmp_path), representation="temporal"
            )
            lid = built["longitudinal_id"]
            res = await client.call_tool(
                "diagnose_longitudinal_graph",
                {"longitudinal_id": lid, "representation": "trajectory"},
                raise_on_error=False,
            )
            _assert_error(
                res, "diagnose_longitudinal_graph", "REPRESENTATION_NOT_BUILT"
            )
            for tool in (
                "get_trajectory_archetypes",
                "get_cross_time_neighbors",
                "classify_trajectories",
            ):
                res = await client.call_tool(
                    tool, {"longitudinal_id": lid}, raise_on_error=False
                )
                _assert_error(res, tool, "REPRESENTATION_NOT_BUILT")

    asyncio.run(run())


def test_query_argument_errors(tmp_path):
    async def run():
        async with Client(_server()) as client:
            await _build(client, await _sample(client, tmp_path))
            cases = [
                (
                    "get_trajectory_archetypes",
                    {"max_archetypes": 0},
                    "INVALID_ARGUMENT",
                ),
                ("get_trajectory_archetypes", {"threshold": -1}, "INVALID_ARGUMENT"),
                ("get_cross_time_neighbors", {"max_neighbors": 0}, "INVALID_ARGUMENT"),
                ("get_cross_time_neighbors", {}, "OBSERVATION_NOT_SPECIFIED"),
                (
                    "get_cross_time_neighbors",
                    {"entity_id": "p0", "t": 999},
                    "OBSERVATION_NOT_FOUND",
                ),
                (
                    "get_cross_time_neighbors",
                    {"observation_id": 10**6},
                    "OBSERVATION_NOT_FOUND",
                ),
                ("classify_trajectories", {"threshold": -0.5}, "INVALID_ARGUMENT"),
            ]
            for tool, args, code in cases:
                res = await client.call_tool(tool, args, raise_on_error=False)
                _assert_error(res, tool, code)

    asyncio.run(run())
