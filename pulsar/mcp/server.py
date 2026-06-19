"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

from __future__ import annotations

import logging
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


def main():
    mcp.run()


if __name__ == "__main__":
    main()
