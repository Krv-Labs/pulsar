import json
from typing import Any, Dict

def mcp_error(tool: str, reason: str, metrics: Dict[str, Any] = None) -> str:
    """
    Standardized error response for all MCP tools.
    Never raise exceptions directly in tools; return this structured JSON string instead.
    """
    error_obj = {
        "status": "error",
        "tool": tool,
        "reason": reason,
        "metrics": metrics or {}
    }
    return json.dumps(error_obj, indent=2)
