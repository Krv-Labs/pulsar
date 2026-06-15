"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse

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
        "audit/debug) and `response_format` ('markdown' default; 'json' for "
        "structured payloads)."
    ),
)


# ---------------------------------------------------------------------------
# Health check (unauthenticated) + bearer auth for Streamable HTTP transport
# ---------------------------------------------------------------------------
@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "pulsar-mcp"})


class _BearerAuthASGI:
    """Pure-ASGI bearer gate for Streamable HTTP (does NOT buffer streaming responses).

    Guards every path except those in ``exempt``. Requires
    ``Authorization: Bearer <INTERNAL_MCP_TOKEN>``; a missing/wrong token => 401.
    If no token is configured the gate is open (stdio/dev only).
    """

    def __init__(self, app, token: str, exempt: tuple[str, ...] = ("/healthz",)) -> None:
        self.app = app
        self.token = token
        self.exempt = exempt

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not self.token:
            return await self.app(scope, receive, send)
        path = (scope.get("path", "") or "/").rstrip("/") or "/"
        if path in self.exempt:
            return await self.app(scope, receive, send)
        provided = dict(scope.get("headers") or []).get(b"authorization", b"").decode()
        if provided == f"Bearer {self.token}":
            return await self.app(scope, receive, send)
        await JSONResponse({"error": "Unauthorized"}, status_code=401)(scope, receive, send)


# ---------------------------------------------------------------------------
# Tool registration — env-gated allowlist (build-spec §3.2)
#   PULSAR_TOOLSET=full (default; stdio dev — all tools)
#                 |curated (artifact-based HTTP production surface)
# ---------------------------------------------------------------------------
_TOOLSET = os.environ.get("PULSAR_TOOLSET", "full").lower()
if _TOOLSET == "curated":
    from pulsar.mcp.tools.curated import CURATED_TOOLS_LIST  # noqa: E402

    _TOOLS = CURATED_TOOLS_LIST
else:
    from pulsar.mcp.tools import ALL_TOOLS_LIST  # noqa: E402

    _TOOLS = ALL_TOOLS_LIST

for tool_fn in _TOOLS:
    mcp.tool()(tool_fn)


def main():
    transport = os.environ.get("PULSAR_TRANSPORT", "stdio").lower()
    if transport in ("http", "streamable-http"):
        token = os.environ.get("INTERNAL_MCP_TOKEN", "")
        host = os.environ.get("PULSAR_MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("PULSAR_MCP_PORT", os.environ.get("PORT", "8000")))
        path = os.environ.get("PULSAR_MCP_PATH", "/mcp")
        if not token:
            logger.warning(
                "PULSAR_TRANSPORT=http but INTERNAL_MCP_TOKEN is empty — bearer gate OPEN (dev only)."
            )
        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
            path=path,
            middleware=[Middleware(_BearerAuthASGI, token=token)],
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
