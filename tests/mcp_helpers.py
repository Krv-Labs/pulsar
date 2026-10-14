"""Helpers for reading MCP tool results regardless of return type.

Tool functions return either a legacy JSON ``str`` or a ``fastmcp.tools.ToolResult``;
in-memory clients return a ``CallToolResult``. These helpers let tests read all three.
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp.tools import ToolResult


def _first_text(content: Any) -> str:
    for block in content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
    raise ValueError("tool result has no text content block")


def _is_result_like(result: Any) -> bool:
    return isinstance(result, ToolResult) or (
        hasattr(result, "content") and hasattr(result, "structured_content")
    )


def payload_of(result: Any) -> dict:
    """Return the JSON payload of a tool result (str, ToolResult, or CallToolResult)."""
    if isinstance(result, str):
        return json.loads(result)
    if _is_result_like(result):
        structured = result.structured_content
        if isinstance(structured, dict):
            # Legacy str-returning tools are wrapped by FastMCP as {"result": "<json>"}.
            if set(structured) == {"result"} and isinstance(structured["result"], str):
                return json.loads(structured["result"])
            return structured
        return json.loads(_first_text(result.content))
    raise TypeError(f"Unsupported tool result type: {type(result).__name__}")


def text_of(result: Any) -> str:
    """Return the text view of a tool result."""
    if isinstance(result, str):
        return result
    if _is_result_like(result):
        return _first_text(result.content)
    raise TypeError(f"Unsupported tool result type: {type(result).__name__}")


def is_error_of(result: Any) -> bool | None:
    """Return ``is_error`` for result objects, or ``None`` for legacy ``str`` results."""
    if isinstance(result, str):
        return None
    if _is_result_like(result):
        return bool(getattr(result, "is_error", False))
    raise TypeError(f"Unsupported tool result type: {type(result).__name__}")
