"""Surface contract tests for the diagnostics and clustering MCP tools.

FastMCP validates ``structured_content`` against ``output_schema`` only when a
tool is called through a server, so every supported ``detail`` x
``response_format`` (x ``surface``) combination is exercised here through an
in-memory ``fastmcp.Client``.
"""

from __future__ import annotations

import asyncio
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pulsar.mcp.toolmeta import default_tool_timeout
from fastmcp import Client, FastMCP

from pulsar.mcp import registry as registry_mod
from pulsar.mcp import session as session_mod
from pulsar.mcp.server import register_server_tools
from pulsar.mcp.tools import diagnostics as diag_mod
from tests.mcp_helpers import payload_of, text_of

DIAGNOSTICS_TOOLS = {
    "diagnose_cosmic_graph": dict(read_only=True, idempotent=True),
    "get_threshold_stability_curve": dict(read_only=True, idempotent=True),
    "get_topological_skeleton": dict(read_only=True, idempotent=True),
    "create_graph_artifact": dict(read_only=False, idempotent=False),
}
CLUSTERING_TOOLS = {
    "generate_cluster_dossier": dict(read_only=False, idempotent=False),
    "get_cluster_profile": dict(read_only=True, idempotent=True),
    "get_feature_signal": dict(read_only=True, idempotent=True),
    "get_cluster_signal_matrix": dict(read_only=True, idempotent=True),
    "compare_clusters": dict(read_only=True, idempotent=True),
}
ALL_OWNED = {**DIAGNOSTICS_TOOLS, **CLUSTERING_TOOLS}


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
    session_mod.model_cache.clear()
    yield
    session_mod._sessions.clear()
    session_mod.model_cache.clear()


def _write_blobs_csv(path: Path, n_rows: int = 40) -> Path:
    """Two well-separated numeric blobs plus a categorical column."""
    rng = np.random.default_rng(0)
    lines = ["f1,f2,f3,cat"]
    for i in range(n_rows):
        center = 0.0 if i < n_rows // 2 else 8.0
        f1, f2, f3 = rng.normal(center, 0.4, 3)
        lines.append(f"{f1:.4f},{f2:.4f},{f3:.4f},{'a' if i % 2 == 0 else 'b'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _server() -> FastMCP:
    server = FastMCP("t")
    register_server_tools(server)
    return server


async def _call(client: Client, tool: str, args: dict[str, Any] | None = None):
    """Call a tool, returning the raw result (errors included, not raised)."""
    return await client.call_tool(tool, args or {}, raise_on_error=False)


async def _ok(client: Client, tool: str, args: dict[str, Any] | None = None):
    res = await _call(client, tool, args)
    assert res.is_error is False, (tool, args, res.content)
    assert isinstance(res.structured_content, dict), (tool, args)
    return res


async def _sweep(client: Client, csv: Path) -> tuple[str, str]:
    """Ingest + create_config + sweep; return (dataset_id, run_id)."""
    ing = await _ok(client, "ingest_dataset", {"path": str(csv)})
    dataset_id = payload_of(ing)["dataset_id"]
    await _ok(client, "create_config", {"dataset_id": dataset_id})
    sweep = await _ok(
        client,
        "run_topological_sweep",
        {"dataset_id": dataset_id, "response_format": "json"},
    )
    return dataset_id, payload_of(sweep)["run_id"]


async def _sweep_and_dossier(client: Client, csv: Path) -> dict[str, Any]:
    dataset_id, run_id = await _sweep(client, csv)
    res = await _ok(
        client,
        "generate_cluster_dossier",
        {
            "method": "spectral",
            "response_format": "json",
        },
    )
    dossier = payload_of(res)
    cluster_ids = [c["cluster_id"] for c in dossier["clusters"]]
    assert len(cluster_ids) >= 2, dossier["clusters"]
    return {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "cluster_ids": cluster_ids,
        "dossier": dossier,
    }


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Wire contract
# ---------------------------------------------------------------------------


def test_wire_annotations_schema_and_param_descriptions():
    async def go():
        async with Client(_server()) as client:
            tools = {t.name: t for t in await client.list_tools()}
            for name, expected in ALL_OWNED.items():
                tool = tools[name]
                assert tool.annotations is not None, name
                assert tool.annotations.read_only_hint is expected["read_only"], name
                assert tool.annotations.idempotent_hint is expected["idempotent"], name
                assert tool.annotations.destructive_hint is False, name
                assert tool.annotations.open_world_hint is False, name
                assert tool.output_schema, name
                assert tool.output_schema["type"] == "object", name
                props = tool.input_schema.get("properties", {})
                assert props, name
                for pname, pschema in props.items():
                    assert pschema.get("description"), f"{name}.{pname} undescribed"
                assert len(tool.description or "") < 700, name
            # run_id is on every tool that reads run/cluster state.
            for name in ALL_OWNED:
                assert "run_id" in tools[name].input_schema["properties"], name
            # Only the dossier carries the long-tool timeout.

    _run(go())


def test_dossier_timeout_from_env(monkeypatch: pytest.MonkeyPatch):
    from pulsar.mcp.toolmeta import tool_meta_of
    from pulsar.mcp.tools.clustering import generate_cluster_dossier

    meta = tool_meta_of(generate_cluster_dossier)
    assert meta.timeout == default_tool_timeout()
    monkeypatch.setenv("PULSAR_MCP_TOOL_TIMEOUT_S", "12.5")
    assert default_tool_timeout() == 12.5
    monkeypatch.setenv("PULSAR_MCP_TOOL_TIMEOUT_S", "garbage")
    assert default_tool_timeout() == 1800.0
    for fn in (diag_mod.diagnose_cosmic_graph, diag_mod.get_topological_skeleton):
        assert tool_meta_of(fn).timeout is None


# ---------------------------------------------------------------------------
# Every variant validates against its output schema (FastMCP enforces this)
# ---------------------------------------------------------------------------


def test_diagnostics_tools_all_variants_validate(tmp_path: Path):
    async def go():
        csv = _write_blobs_csv(tmp_path / "blobs.csv")
        async with Client(_server()) as client:
            _, run_id = await _sweep(client, csv)

            # diagnose: detail x response_format on the constructed surface.
            for detail, fmt in itertools.product(
                ("summary", "full"), ("json", "markdown")
            ):
                res = await _ok(
                    client,
                    "diagnose_cosmic_graph",
                    {"detail": detail, "response_format": fmt},
                )
                body = payload_of(res)
                assert body["source"] == "live" and body["detail"] == detail
                assert ("component_sizes" in body["component_morphology"]) == (
                    detail == "full"
                )
                if fmt == "markdown":
                    assert text_of(res).startswith("# Cosmic Graph Diagnosis")
            # Same structured payload regardless of response_format.
            a = payload_of(await _ok(client, "diagnose_cosmic_graph", {}))
            b = payload_of(
                await _ok(
                    client, "diagnose_cosmic_graph", {"response_format": "markdown"}
                )
            )
            assert a == b

            # create_graph_artifact: estimate, then build (the artifact surface).
            est = payload_of(
                await _ok(client, "create_graph_artifact", {"epsilon": 1.0})
            )
            assert est["estimate_only"] is True and est["run_id"] == run_id
            built = payload_of(
                await _ok(
                    client,
                    "create_graph_artifact",
                    {
                        "estimate_only": False,
                        "sketch_dim": 2,
                        "sample_count": 12,
                    },
                )
            )
            artifact_id = built["artifact_id"]
            for detail, fmt in itertools.product(
                ("summary", "full"), ("json", "markdown")
            ):
                res = await _ok(
                    client,
                    "diagnose_cosmic_graph",
                    {
                        "surface": "artifact",
                        "artifact_id": artifact_id,
                        "detail": detail,
                        "response_format": fmt,
                    },
                )
                assert payload_of(res)["source"] == "artifact"

            # threshold curve: detail x source x response_format.
            for detail, source, fmt in itertools.product(
                ("summary", "full"), ("persisted", "live"), ("markdown", "json")
            ):
                res = await _ok(
                    client,
                    "get_threshold_stability_curve",
                    {"detail": detail, "source": source, "response_format": fmt},
                )
                body = payload_of(res)
                assert body["detail"] in {"summary", "full"} or "source" in body
            for policy in (
                "balanced",
                "report_ready",
                "detail_seeking",
                "outlier_mining",
            ):
                await _ok(
                    client,
                    "get_threshold_stability_curve",
                    {
                        "source": "live",
                        "threshold_candidate_policy": policy,
                        "response_format": "json",
                    },
                )

            # skeleton: every detail on both surfaces.
            for detail in ("summary", "nodes", "edges", "full", "full_nodes"):
                res = await _ok(
                    client,
                    "get_topological_skeleton",
                    {"detail": detail, "max_edges": 5, "max_nodes": 5},
                )
                assert payload_of(res)["run_id"] == run_id
                res = await _ok(
                    client,
                    "get_topological_skeleton",
                    {
                        "surface": "artifact",
                        "artifact_id": artifact_id,
                        "detail": detail,
                        "max_edges": 5,
                        "max_nodes": 5,
                    },
                )
                assert payload_of(res)["artifact_id"] == artifact_id

    _run(go())


def test_clustering_tools_all_variants_validate(tmp_path: Path):
    async def go():
        csv = _write_blobs_csv(tmp_path / "blobs.csv")
        async with Client(_server()) as client:
            _, run_id = await _sweep(client, csv)

            # dossier: detail x response_format (spectral finds the two blobs).
            for detail, fmt in itertools.product(
                ("summary", "standard", "full"), ("json", "markdown")
            ):
                res = await _ok(
                    client,
                    "generate_cluster_dossier",
                    {"method": "spectral", "detail": detail, "response_format": fmt},
                )
                body = payload_of(res)
                assert body["clusters_run_id"] == run_id
                assert body["cluster_assignment_id"]
            cluster_ids = [c["cluster_id"] for c in body["clusters"]]
            assert len(cluster_ids) >= 2
            first, second = cluster_ids[0], cluster_ids[1]

            for detail, fmt in itertools.product(
                ("summary", "standard", "full"), ("json", "markdown")
            ):
                res = await _ok(
                    client,
                    "get_cluster_profile",
                    {"cluster_id": first, "detail": detail, "response_format": fmt},
                )
                assert payload_of(res)["clusters_run_id"] == run_id

            for detail, fmt in itertools.product(
                ("summary", "standard", "full"), ("json", "markdown")
            ):
                res = await _ok(
                    client,
                    "get_feature_signal",
                    {
                        "feature_names": ["f1", "cat=a"],
                        "detail": detail,
                        "response_format": fmt,
                    },
                )
                assert payload_of(res)["clusters_run_id"] == run_id

            for include_ctx, md in itertools.product((False, True), (True, False)):
                res = await _ok(
                    client,
                    "get_cluster_signal_matrix",
                    {"include_context_tier": include_ctx, "return_markdown": md},
                )
                assert payload_of(res)["clusters_run_id"] == run_id
            # structured_content is identical for both text renderings.
            m_true = payload_of(
                await _ok(
                    client, "get_cluster_signal_matrix", {"return_markdown": True}
                )
            )
            m_false = payload_of(
                await _ok(
                    client, "get_cluster_signal_matrix", {"return_markdown": False}
                )
            )
            assert m_true == m_false

            res = await _ok(
                client,
                "compare_clusters",
                {"cluster_a": first, "cluster_b": second},
            )
            body = payload_of(res)
            assert body["clusters_run_id"] == run_id
            assert body["results"] and body["features_compared"] >= len(body["results"])
            assert "# Cluster Comparison" in text_of(res)
            small = payload_of(
                await _ok(
                    client,
                    "compare_clusters",
                    {"cluster_a": first, "cluster_b": second, "max_features": 1},
                )
            )
            assert len(small["results"]) == 1
            assert small["features_omitted"] == small["features_compared"] - 1

    _run(go())


# ---------------------------------------------------------------------------
# run_id behaviour
# ---------------------------------------------------------------------------


def test_run_id_round_trip_reports_first_run(tmp_path: Path):
    async def go():
        small = _write_blobs_csv(tmp_path / "small.csv", n_rows=24)
        large = _write_blobs_csv(tmp_path / "large.csv", n_rows=40)
        async with Client(_server()) as client:
            _, first = await _sweep(client, small)
            _, second = await _sweep(client, large)
            assert first != second
            latest = payload_of(await _ok(client, "diagnose_cosmic_graph", {}))
            assert latest["run_id"] == second and latest["scale"]["n_nodes"] == 40
            old = payload_of(
                await _ok(client, "diagnose_cosmic_graph", {"run_id": first})
            )
            assert old["run_id"] == first and old["scale"]["n_nodes"] == 24
            # Live threshold curve on the latest run named explicitly works...
            curve = payload_of(
                await _ok(
                    client,
                    "get_threshold_stability_curve",
                    {"run_id": second, "source": "live", "response_format": "json"},
                )
            )
            assert curve["run_id"] == second
            # ...an artifact estimate is a pure read, so an older run works...
            est = payload_of(
                await _ok(client, "create_graph_artifact", {"run_id": first})
            )
            assert est["estimate_only"] is True and est["run_id"] == first
            assert est["input"]["n_nodes"] == 24
            # ...but building (a write) for the older run is refused.
            res = await _call(
                client,
                "create_graph_artifact",
                {"run_id": first, "estimate_only": False},
            )
            assert res.is_error is True
            assert res.structured_content["error_code"] == "SESSION_REVISION_CONFLICT"

    _run(go())


@pytest.mark.parametrize(
    "tool,args",
    [
        ("diagnose_cosmic_graph", {}),
        ("get_threshold_stability_curve", {"source": "live"}),
        ("get_topological_skeleton", {}),
        ("create_graph_artifact", {}),
        ("generate_cluster_dossier", {}),
    ],
)
def test_unknown_run_id_is_run_id_unknown(tool, args, tmp_path: Path):
    async def go():
        async with Client(_server()) as client:
            res = await _call(client, tool, {**args, "run_id": "run_does_not_exist"})
            assert res.is_error is True
            env = res.structured_content
            assert env["error_code"] == "RUN_ID_UNKNOWN"
            assert env["tool"] == tool
            assert env["agent_action"]
            assert env["details"] == {"run_id": "run_does_not_exist"}

    _run(go())


def test_dossier_with_non_latest_run_id_is_refused(tmp_path: Path):
    async def go():
        csv = _write_blobs_csv(tmp_path / "blobs.csv")
        async with Client(_server()) as client:
            _, first = await _sweep(client, csv)
            _, second = await _sweep(client, csv)
            res = await _call(
                client,
                "generate_cluster_dossier",
                {"run_id": first, "method": "spectral"},
            )
            assert res.is_error is True
            env = res.structured_content
            assert env["error_code"] == "SESSION_REVISION_CONFLICT"
            assert "latest run" in env["agent_action"]
            # Nothing was published: cluster tools still say there are no clusters.
            res = await _call(client, "get_cluster_profile", {"cluster_id": 0})
            assert res.structured_content["error_code"] == "CLUSTERS_NOT_BUILT"
            # The latest run is accepted when named explicitly.
            ok = await _ok(
                client,
                "generate_cluster_dossier",
                {"run_id": second, "method": "spectral", "response_format": "json"},
            )
            assert payload_of(ok)["run_id"] == second

    _run(go())


def test_cluster_tools_report_clusters_stale_for_other_run(tmp_path: Path):
    async def go():
        csv = _write_blobs_csv(tmp_path / "blobs.csv")
        async with Client(_server()) as client:
            state = await _sweep_and_dossier(client, csv)
            first_run = state["run_id"]
            a, b = state["cluster_ids"][:2]
            calls = {
                "get_cluster_profile": {"cluster_id": a},
                "get_feature_signal": {"feature_names": ["f1"]},
                "get_cluster_signal_matrix": {},
                "compare_clusters": {"cluster_a": a, "cluster_b": b},
            }
            for tool, args in calls.items():
                # Naming a run the clusters were not built for is refused.
                res = await _call(client, tool, {**args, "run_id": "run_other"})
                assert res.is_error is True
                env = res.structured_content
                assert env["error_code"] == "CLUSTERS_STALE", tool
                assert env["tool"] == tool
                assert env["details"]["clusters_run_id"] == first_run
                assert "dossier" in env["agent_action"]
                # Naming the clusters' own run serves them, labelled with that run.
                res = await _ok(client, tool, {**args, "run_id": first_run})
                assert payload_of(res)["clusters_run_id"] == first_run
            # A new sweep invalidates the cached cluster evidence: reads without
            # run_id are refused until the dossier is rebuilt.
            await _sweep(client, csv)
            for tool, args in calls.items():
                res = await _call(client, tool, args)
                assert res.is_error is True
                assert res.structured_content["error_code"] in {
                    "CLUSTERS_NOT_BUILT",
                    "CLUSTER_CACHE_STALE",
                }

    _run(go())


# ---------------------------------------------------------------------------
# One error path per tool
# ---------------------------------------------------------------------------


def _assert_envelope(res, tool: str, code: str):
    assert res.is_error is True, res.content
    env = res.structured_content
    assert env["status"] == "error"
    assert env["tool"] == tool
    assert env["error_code"] == code, env
    assert env["agent_action"]
    return env


def test_error_paths_without_a_model():
    async def go():
        async with Client(_server()) as client:
            for tool, args in (
                ("diagnose_cosmic_graph", {}),
                ("get_threshold_stability_curve", {"source": "live"}),
                ("get_topological_skeleton", {}),
                ("create_graph_artifact", {}),
                ("generate_cluster_dossier", {}),
                ("get_cluster_profile", {"cluster_id": 0}),
                ("get_feature_signal", {"feature_names": ["f1"]}),
                ("get_cluster_signal_matrix", {}),
                ("compare_clusters", {"cluster_a": 0, "cluster_b": 1}),
            ):
                res = await _call(client, tool, args)
                _assert_envelope(res, tool, "MODEL_NOT_LOADED")
                assert "run_topological_sweep" in res.structured_content["agent_action"]

    _run(go())


def test_error_paths_with_a_model(tmp_path: Path):
    async def go():
        csv = _write_blobs_csv(tmp_path / "blobs.csv")
        async with Client(_server()) as client:
            state = await _sweep_and_dossier(client, csv)
            a, b = state["cluster_ids"][:2]

            _assert_envelope(
                await _call(
                    client,
                    "diagnose_cosmic_graph",
                    {"surface": "artifact", "artifact_id": ""},
                ),
                "diagnose_cosmic_graph",
                "ARTIFACT_ID_REQUIRED",
            )
            env = _assert_envelope(
                await _call(
                    client,
                    "diagnose_cosmic_graph",
                    {"surface": "artifact", "artifact_id": "graph_nope"},
                ),
                "diagnose_cosmic_graph",
                "ARTIFACT_ID_UNKNOWN",
            )
            assert env["details"] == {"artifact_id": "graph_nope"}
            _assert_envelope(
                await _call(
                    client,
                    "get_topological_skeleton",
                    {"surface": "artifact", "artifact_id": "graph_nope"},
                ),
                "get_topological_skeleton",
                "ARTIFACT_ID_UNKNOWN",
            )
            _assert_envelope(
                await _call(client, "get_topological_skeleton", {"max_edges": 0}),
                "get_topological_skeleton",
                "INVALID_ARGUMENT",
            )
            _assert_envelope(
                await _call(client, "create_graph_artifact", {"epsilon": 0.0}),
                "create_graph_artifact",
                "INVALID_ARGUMENT",
            )
            _assert_envelope(
                await _call(
                    client,
                    "get_threshold_stability_curve",
                    {"run_id": "run_missing"},
                ),
                "get_threshold_stability_curve",
                "RUN_ID_UNKNOWN",
            )
            _assert_envelope(
                await _call(client, "generate_cluster_dossier", {"max_clusters": 0}),
                "generate_cluster_dossier",
                "INVALID_ARGUMENT",
            )
            _assert_envelope(
                await _call(client, "get_cluster_profile", {"cluster_id": 9999}),
                "get_cluster_profile",
                "CLUSTER_ID_UNKNOWN",
            )
            _assert_envelope(
                await _call(
                    client, "get_cluster_profile", {"cluster_id": a, "max_features": 0}
                ),
                "get_cluster_profile",
                "INVALID_ARGUMENT",
            )
            _assert_envelope(
                await _call(
                    client,
                    "get_feature_signal",
                    {"feature_names": ["f1"], "max_clusters": 0},
                ),
                "get_feature_signal",
                "INVALID_ARGUMENT",
            )
            _assert_envelope(
                await _call(client, "get_cluster_signal_matrix", {"max_clusters": 0}),
                "get_cluster_signal_matrix",
                "INVALID_ARGUMENT",
            )
            _assert_envelope(
                await _call(
                    client,
                    "compare_clusters",
                    {"cluster_a": a, "cluster_b": 9999},
                ),
                "compare_clusters",
                "COMPARISON_EMPTY",
            )
            _assert_envelope(
                await _call(
                    client,
                    "compare_clusters",
                    {"cluster_a": a, "cluster_b": b, "max_features": 0},
                ),
                "compare_clusters",
                "INVALID_ARGUMENT",
            )

    _run(go())


@pytest.mark.parametrize("bad", [1.5, -0.1])
def test_dossier_out_of_range_interpretation_threshold_is_invalid_argument(
    bad, tmp_path: Path
):
    async def go():
        csv = _write_blobs_csv(tmp_path / "blobs.csv")
        async with Client(_server()) as client:
            await _sweep(client, csv)
            res = await _call(
                client,
                "generate_cluster_dossier",
                {"interpretation_edge_weight_threshold": bad},
            )
            env = _assert_envelope(res, "generate_cluster_dossier", "INVALID_ARGUMENT")
            assert "0.0" in env["agent_action"] and "1.0" in env["agent_action"]
            assert env["details"]["interpretation_edge_weight_threshold"] == bad

    _run(go())


def test_dossier_param_and_feature_signal_descriptions_are_precise():
    async def go():
        async with Client(_server()) as client:
            return {t.name: t for t in await client.list_tools()}

    tools = _run(go())
    props = tools["generate_cluster_dossier"].input_schema["properties"]
    assert "resolved interpretation threshold" in props["method"]["description"]
    assert "2..max_k" in props["max_k"]["description"]
    run_desc = tools["diagnose_cosmic_graph"].input_schema["properties"]["run_id"][
        "description"
    ]
    assert run_desc.count("Omit for the latest run") == 1
    detail = tools["get_feature_signal"].input_schema["properties"]["detail"]
    assert "context-tier" in detail["description"]


@pytest.mark.parametrize("fmt", ["markdown", "json"])
def test_spectral_no_cut_error_text_is_the_json_envelope(
    fmt, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import json

    from pulsar.mcp.interpreter import SpectralClusterCutError
    from pulsar.mcp.tools import clustering as clustering_mod

    def boom(*args, **kwargs):
        raise SpectralClusterCutError(
            {
                "k_min": 2,
                "max_k": 4,
                "best_k": 3,
                "best_silhouette_score": 0.01,
                "accepted_silhouette_min": 0.05,
                "candidate_scores": [],
            }
        )

    monkeypatch.setattr(clustering_mod, "resolve_clusters", boom)

    async def go():
        csv = _write_blobs_csv(tmp_path / "blobs.csv")
        async with Client(_server()) as client:
            await _sweep(client, csv)
            res = await _call(
                client,
                "generate_cluster_dossier",
                {"method": "spectral", "response_format": fmt},
            )
            env = _assert_envelope(
                res, "generate_cluster_dossier", "NO_STABLE_SPECTRAL_CUT"
            )
            assert json.loads(res.content[0].text) == env
            assert "k=2-4" in env["reason"]
            assert env["details"]["best_k"] == 3

    _run(go())
