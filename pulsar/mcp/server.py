"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mcp.types as mt
from fastmcp import FastMCP
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.server.middleware.response_limiting import ResponseLimitingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware
from fastmcp.tools import ToolResult
from mcp.types import ToolAnnotations

import pulsar
from pulsar.mcp.toolmeta import tool_meta_of
from fastmcp.utilities.tasks import TASKS_EXTENSION_ID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument sanitising for non-compliant MCP clients
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


class ArgumentSanitizerMiddleware(Middleware):
    """Strip client orchestration keys and unknown arguments before dispatch."""

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        message = context.message
        if message.arguments and context.fastmcp_context is not None:
            tool = await context.fastmcp_context.fastmcp.get_tool(message.name)
            if tool is not None:
                message.arguments = _sanitize_tool_arguments(
                    tool, message.name, message.arguments
                )
        return await call_next(context)


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Pulsar",
    instructions=(
        "Manifold discovery and topological data analysis for tabular datasets. Call "
        "`get_workflow_guide` once for the end-to-end procedure and tool map.\n"
        "Shared params across tools: `detail` ('summary' default; 'full' for "
        "audit/debug) and `response_format` ('json' for structured payloads; "
        "'markdown' where a readable rendering is supported).\n"
        "Results carry structured content; errors set the MCP error flag with a "
        "JSON envelope (error_code, agent_action)."
    ),
    version=pulsar.__version__,
    website_url="https://github.com/Krv-Labs/pulsar",
    cache_ttl=3600,
    middleware=[
        ArgumentSanitizerMiddleware(),
        ResponseLimitingMiddleware(
            max_size=int(os.environ.get("PULSAR_MCP_MAX_RESPONSE_BYTES", "400000"))
        ),
        TimingMiddleware(logger=logging.getLogger("pulsar.mcp")),
    ],
)

# ---------------------------------------------------------------------------
# Dynamic Registration of All Tools
# ---------------------------------------------------------------------------
from pulsar.mcp.tools import (  # noqa: E402
    ALL_TOOLS_LIST,
    _tasks_enabled,
    get_tool_task_config,
)


def _ensure_tasks_extension(server: FastMCP) -> bool:
    """Register the real tasks extension; return False if it cannot be provided.

    Never advertises a tasks capability it cannot serve: when ``fastmcp-tasks``
    is missing, no extension is registered and every tool stays task-forbidden.
    """
    if TASKS_EXTENSION_ID in server._extensions:
        return True
    try:
        from fastmcp_tasks import TasksExtension
    except ImportError:
        logger.error(
            "PULSAR_MCP_ENABLE_TASKS is set but fastmcp-tasks is not installed; "
            "background tasks are DISABLED. Install it with "
            "`pip install 'fastmcp[tasks]>=4.0.10,<5'` (or `pip install "
            "'thema-pulsar[mcp]'`)."
        )
        return False
    server.add_extension(TasksExtension())
    return True


def register_server_tools(server: FastMCP) -> None:
    tasks_available = _tasks_enabled() and _ensure_tasks_extension(server)
    for tool_fn in ALL_TOOLS_LIST:
        kwargs: dict[str, Any] = {
            "task": get_tool_task_config(tool_fn, tasks_available)
        }
        meta = tool_meta_of(tool_fn)
        if meta is None:
            raise RuntimeError(
                f"MCP tool {getattr(tool_fn, '__name__', tool_fn)!r} has no "
                "@tool_meta; every tool must declare annotations and an "
                "output schema."
            )
        kwargs["annotations"] = ToolAnnotations(
            title=meta.title,
            read_only_hint=meta.read_only,
            destructive_hint=meta.destructive,
            idempotent_hint=meta.idempotent,
            open_world_hint=False,
        )
        kwargs["output_schema"] = meta.output_schema
        if meta.timeout is not None:
            kwargs["timeout"] = meta.timeout
        if meta.tags:
            kwargs["tags"] = set(meta.tags)
        server.tool(**kwargs)(tool_fn)


register_server_tools(mcp)


def main():
    # mcp.run() with no argument uses fastmcp.settings.transport (FASTMCP_TRANSPORT),
    # so a non-stdio transport can be selected by the environment.
    import fastmcp

    transport = fastmcp.settings.transport
    if transport != "stdio":
        logger.warning(
            "Pulsar MCP is starting on transport %r. In-memory fitted-model "
            "continuity between tool calls is guaranteed only on stdio and "
            "legacy-session HTTP; stateless HTTP isolates clients but does not "
            "preserve the fitted model between calls. Use dataset_id / run_id "
            "handles as the durable state.",
            transport,
        )
    mcp.run()


if __name__ == "__main__":
    main()
