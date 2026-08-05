import asyncio
import logging
from pulsar.mcp.server import AgnosticFastMCP


def test_agnostic_mcp_strips_orchestration_keys():
    mcp_test = AgnosticFastMCP("TestServer")

    @mcp_test.tool()
    def my_tool(x: int) -> str:
        return f"value:{x}"

    # Verify that call_tool successfully intercepts and strips wait_for_previous
    res = asyncio.run(
        mcp_test.call_tool("my_tool", {"x": 42, "wait_for_previous": True})
    )
    # In FastMCP, call_tool returns a CallToolResult whose content list contains TextContent
    # Let's inspect the return type, but we can also check text representation.
    assert "value:42" in str(res)


def test_agnostic_mcp_logs_unrecognized_keys(caplog):
    mcp_test = AgnosticFastMCP("TestServer")

    @mcp_test.tool()
    def my_tool(x: int) -> str:
        return f"value:{x}"

    with caplog.at_level(logging.WARNING):
        res = asyncio.run(
            mcp_test.call_tool("my_tool", {"x": 42, "const_threshold": "dummy"})
        )

    assert "value:42" in str(res)
    assert any(
        "Stripped unrecognized parameters from tool my_tool" in record.message
        for record in caplog.records
    )
    assert any("const_threshold" in record.message for record in caplog.records)


def test_mcp_server_main_cli(monkeypatch):
    from unittest.mock import MagicMock
    from pulsar.mcp.server import main, mcp

    mock_run = MagicMock()
    monkeypatch.setattr(mcp, "run", mock_run)

    # Stdio default
    main([])
    mock_run.assert_called_with(transport="stdio")

    # SSE with custom host, port, path
    mock_run.reset_mock()
    main(["--transport", "sse", "--host", "127.0.0.1", "--port", "9000", "--path", "/mcp"])
    mock_run.assert_called_with(
        transport="sse",
        host="127.0.0.1",
        port=9000,
        allowed_hosts=["*"],
        path="/mcp",
    )

