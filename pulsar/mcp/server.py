"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

from __future__ import annotations

import logging
from typing import Any
from fastmcp import FastMCP
from fastmcp.server.extensions import ServerExtension
from fastmcp.utilities.tasks import TASKS_EXTENSION_ID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client-Agnostic Server Subclass to manage non-compliant MCP clients
# ---------------------------------------------------------------------------
_KNOWN_ORCHESTRATION_KEYS = frozenset({"wait_for_previous"})


def _sanitize_tool_arguments(
    tool: Any,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    valid_keys = set(tool.parameters.get("properties", {}).keys())
    unexpected = set(arguments.keys()) - valid_keys
    unrecognized = unexpected - _KNOWN_ORCHESTRATION_KEYS
    if unrecognized:
        logger.warning(
            "Stripped unrecognized parameters from tool %s: %s",
            name,
            sorted(unrecognized),
        )
    return {k: v for k, v in arguments.items() if k in valid_keys}


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
                arguments = _sanitize_tool_arguments(tool, name, arguments)
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
from pulsar.mcp.tools import (  # noqa: E402
    ALL_TOOLS_LIST,
    _tasks_enabled,
    get_tool_task_config,
)


class _PulsarTasksExtension(ServerExtension):
    """Fallback tasks extension declaration when fastmcp-tasks is not installed."""

    identifier = TASKS_EXTENSION_ID


def _ensure_tasks_extension(server: FastMCP) -> None:
    if TASKS_EXTENSION_ID in server._extensions:
        return
    try:
        from fastmcp_tasks import TasksExtension  # type: ignore[import-not-found]

        server.add_extension(TasksExtension())
    except ImportError:
        logger.warning(
            "fastmcp-tasks is not installed; registering fallback tasks capability declaration."
        )
        server.add_extension(_PulsarTasksExtension())


def register_server_tools(server: FastMCP) -> None:
    if _tasks_enabled():
        _ensure_tasks_extension(server)
    for tool_fn in ALL_TOOLS_LIST:
        server.tool(task=get_tool_task_config(tool_fn))(tool_fn)


register_server_tools(mcp)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
