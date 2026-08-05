"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client-Agnostic Server Subclass to manage non-compliant MCP clients
# ---------------------------------------------------------------------------
_KNOWN_ORCHESTRATION_KEYS = frozenset({"wait_for_previous"})


class AgnosticFastMCP(FastMCP):
    """Custom FastMCP subclass that gracefully strips client orchestration parameters."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ):
        if arguments:
            tool = await self.get_tool(name, version=kwargs.get("version"))
            if tool is not None:
                valid_keys = set(tool.parameters.get("properties", {}).keys())
                unexpected = set(arguments.keys()) - valid_keys
                if unexpected:
                    unrecognized = unexpected - _KNOWN_ORCHESTRATION_KEYS
                    if unrecognized:
                        logger.warning(
                            "Stripped unrecognized parameters from tool %s: %s",
                            name,
                            sorted(unrecognized),
                        )
                arguments = {k: v for k, v in arguments.items() if k in valid_keys}
        return await super().call_tool(name, arguments, *args, **kwargs)


# ---------------------------------------------------------------------------
# Initialize Client-Agnostic FastMCP
# ---------------------------------------------------------------------------
mcp = AgnosticFastMCP(
    "Pulsar",
    instructions=(
        "Manifold discovery and topological data analysis for tabular datasets. Call "
        "`get_workflow_guide` once for the end-to-end procedure and tool map.\n"
        "Shared params across tools: `detail` ('summary' default; 'full' for "
        "audit/debug) and `response_format` ('json' for structured payloads; "
        "'markdown' where a readable rendering is supported)."
    ),
)

# ---------------------------------------------------------------------------
# Dynamic Registration of All Tools
# ---------------------------------------------------------------------------
from pulsar.mcp.tools import ALL_TOOLS_LIST  # noqa: E402

for tool_fn in ALL_TOOLS_LIST:
    mcp.tool()(tool_fn)


def main(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Pulsar FastMCP Server")
    parser.add_argument(
        "--transport",
        type=str,
        default=os.environ.get("PULSAR_MCP_TRANSPORT", "stdio"),
        choices=["stdio", "sse", "http", "streamable-http"],
        help="MCP transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("PULSAR_MCP_HOST", "0.0.0.0"),
        help="Host address to bind to for HTTP/SSE transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PULSAR_MCP_PORT", "8000")),
        help="Port to bind to for HTTP/SSE transport (default: 8000)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=os.environ.get("PULSAR_MCP_PATH"),
        help="Custom path endpoint for HTTP/SSE transport (e.g. /sse or /mcp)",
    )
    parser.add_argument(
        "--allowed-hosts",
        type=str,
        default=os.environ.get("PULSAR_MCP_ALLOWED_HOSTS", "*"),
        help="Comma-separated allowed hosts for HTTP transport guard (default: *)",
    )

    parsed = parser.parse_args(args)

    if parsed.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        allowed_hosts_list = [
            h.strip() for h in parsed.allowed_hosts.split(",") if h.strip()
        ]
        run_kwargs: dict[str, Any] = {
            "transport": parsed.transport,
            "host": parsed.host,
            "port": parsed.port,
            "allowed_hosts": allowed_hosts_list,
        }
        if parsed.path:
            run_kwargs["path"] = parsed.path
        mcp.run(**run_kwargs)


if __name__ == "__main__":
    main()

