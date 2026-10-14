import asyncio
import logging

from fastmcp import Client, FastMCP

from pulsar.mcp.server import ArgumentSanitizerMiddleware


def _server() -> FastMCP:
    srv = FastMCP("TestServer", middleware=[ArgumentSanitizerMiddleware()])

    @srv.tool()
    def my_tool(x: int) -> str:
        return f"value:{x}"

    return srv


def test_sanitizer_strips_orchestration_keys():
    asyncio.run(_test_sanitizer_strips_orchestration_keys())


async def _test_sanitizer_strips_orchestration_keys():
    async with Client(_server()) as client:
        res = await client.call_tool("my_tool", {"x": 42, "wait_for_previous": True})
    assert "value:42" in str(res)


def test_sanitizer_logs_unrecognized_keys(caplog):
    asyncio.run(_test_sanitizer_logs_unrecognized_keys(caplog))


async def _test_sanitizer_logs_unrecognized_keys(caplog):
    async with Client(_server()) as client:
        with caplog.at_level(logging.WARNING):
            res = await client.call_tool(
                "my_tool", {"x": 42, "const_threshold": "dummy"}
            )
    assert "value:42" in str(res)
    assert any(
        "Stripped unrecognized parameters from tool my_tool" in r.message
        and "const_threshold" in r.message
        for r in caplog.records
    )
