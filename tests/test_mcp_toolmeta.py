import json
import asyncio
import logging

from fastmcp import Client, FastMCP
from fastmcp.tools import ToolResult
from mcp.types import ToolAnnotations

from pulsar.mcp.errors import mcp_error, path_access_error, unknown_handle_error
from pulsar.mcp.results import mcp_result
from pulsar.mcp.server import ArgumentSanitizerMiddleware
from pulsar.mcp.toolmeta import (
    BOOL,
    ERROR_ENVELOPE_SCHEMA,
    INT,
    NUM,
    STR,
    arr,
    enum,
    nullable,
    obj,
    tool_meta,
    tool_meta_of,
)

SCHEMA = obj({"n": INT, "label": STR}, ["n"])


def test_mcp_result_defaults_text_to_json():
    r = mcp_result({"a": 1})
    assert isinstance(r, ToolResult)
    assert r.structured_content == {"a": 1}
    assert r.content[0].text == json.dumps({"a": 1}, indent=2)
    assert not r.is_error


def test_mcp_result_custom_text():
    r = mcp_result({"a": 1}, text="# hi")
    assert r.content[0].text == "# hi"
    assert r.structured_content == {"a": 1}


def test_mcp_error_envelope():
    r = mcp_error(
        "t", "boom", {"m": 1}, error_code="X", agent_action="do", details={"d": 2}
    )
    env = {
        "status": "error",
        "tool": "t",
        "reason": "boom",
        "error_code": "X",
        "agent_action": "do",
        "details": {"d": 2},
        "metrics": {"m": 1},
    }
    assert r.is_error is True
    assert r.structured_content == env
    assert r.content[0].text == json.dumps(env, indent=2)


def test_path_and_handle_errors():
    p = path_access_error("t", "/nope/x")
    assert p.is_error and p.structured_content["error_code"] == "FILE_NOT_FOUND"
    s = path_access_error("t", "/mnt/x")
    assert s.structured_content["error_code"] == "HOST_PATH_NOT_VISIBLE"
    d = unknown_handle_error("t", "dataset_id", "abc")
    assert d.structured_content["error_code"] == "DATASET_ID_UNKNOWN"
    r = unknown_handle_error("t", "run_id", "r1")
    assert r.is_error
    assert r.structured_content["error_code"] == "RUN_ID_UNKNOWN"
    assert "re-run" in r.structured_content["agent_action"].lower()
    assert r.structured_content["details"] == {"run_id": "r1"}


def test_tool_meta_attachment():
    @tool_meta(
        read_only=True, idempotent=True, output_schema=SCHEMA, timeout=5, tags={"a"}
    )
    def f():
        return None

    def g():
        return None

    m = tool_meta_of(f)
    assert m.read_only and m.idempotent and m.timeout == 5
    assert m.tags == frozenset({"a"}) and m.destructive is False
    assert tool_meta_of(g) is None


def test_schema_builders():
    assert (STR, INT, NUM, BOOL) == (
        {"type": "string"},
        {"type": "integer"},
        {"type": "number"},
        {"type": "boolean"},
    )
    assert obj({"a": STR}) == {
        "type": "object",
        "properties": {"a": STR},
        "additionalProperties": False,
    }
    o = obj({"a": STR}, ["a"], additional=True)
    assert o["required"] == ["a"] and o["additionalProperties"] is True
    assert arr(INT) == {"type": "array", "items": INT}
    assert nullable(STR) == {"anyOf": [STR, {"type": "null"}]}
    assert enum(["a", "b"]) == {"enum": ["a", "b"]}
    assert ERROR_ENVELOPE_SCHEMA["required"] == ["status", "tool", "reason"]


def _meta_server() -> FastMCP:
    srv = FastMCP("T", middleware=[ArgumentSanitizerMiddleware()])

    @tool_meta(read_only=True, idempotent=True, output_schema=SCHEMA)
    def ok_tool(n: int) -> ToolResult:
        return mcp_result({"n": n})

    @tool_meta(read_only=True, output_schema=SCHEMA)
    def bad_tool(n: int) -> ToolResult:
        return mcp_error("bad_tool", "nope", error_code="NOPE", agent_action="retry")

    for fn in (ok_tool, bad_tool):
        m = tool_meta_of(fn)
        srv.tool(
            annotations=ToolAnnotations(
                read_only_hint=m.read_only,
                destructive_hint=m.destructive,
                idempotent_hint=m.idempotent,
                open_world_hint=False,
            ),
            output_schema=m.output_schema,
        )(fn)
    return srv


def test_wire_annotations_schema_and_results():
    asyncio.run(_test_wire_annotations_schema_and_results())


async def _test_wire_annotations_schema_and_results():
    async with Client(_meta_server()) as client:
        tools = {t.name: t for t in await client.list_tools()}
        t = tools["ok_tool"]
        assert t.annotations.read_only_hint is True
        assert t.annotations.open_world_hint is False
        assert t.annotations.idempotent_hint is True
        assert t.output_schema["properties"]["n"] == INT
        ok = await client.call_tool("ok_tool", {"n": 3})
        assert ok.is_error is False and ok.structured_content == {"n": 3}
        bad = await client.call_tool("bad_tool", {"n": 1}, raise_on_error=False)
        assert bad.is_error is True
        assert bad.structured_content["error_code"] == "NOPE"
        assert json.loads(bad.content[0].text) == bad.structured_content


def test_sanitizer_strips_and_logs(caplog):
    asyncio.run(_test_sanitizer_strips_and_logs(caplog))


async def _test_sanitizer_strips_and_logs(caplog):
    async with Client(_meta_server()) as client:
        with caplog.at_level(logging.WARNING):
            res = await client.call_tool(
                "ok_tool", {"n": 2, "wait_for_previous": True, "bogus": 1}
            )
    assert res.structured_content == {"n": 2}
    msgs = [r.message for r in caplog.records if "Stripped" in r.message]
    assert len(msgs) == 1 and "bogus" in msgs[0] and "wait_for_previous" not in msgs[0]
