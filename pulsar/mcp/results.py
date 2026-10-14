"""Result helpers: tools return ``ToolResult`` with text and structured content."""

from __future__ import annotations

import json
from typing import Any

from fastmcp.tools import ToolResult
from mcp.types import TextContent


def mcp_result(payload: dict[str, Any], *, text: str | None = None) -> ToolResult:
    """Wrap a success payload as a ``ToolResult``.

    ``text`` is the human/LLM-facing rendering (e.g. markdown); it defaults to
    ``json.dumps(payload, indent=2)``. ``structured_content`` is ``payload``.
    """
    if text is None:
        text = json.dumps(payload, indent=2)
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=payload,
    )
