from __future__ import annotations

import json
from typing import Any


_SANDBOX_PATH_PREFIXES = (
    "/home/claude",
    "/mnt/",
    "/workspace/",
    "/tmp/claude",
    "/System/Volumes/Data/home/claude",
)


def mcp_error(
    tool: str,
    reason: str,
    metrics: dict[str, Any] | None = None,
    *,
    error_code: str | None = None,
    agent_action: str | None = None,
    details: dict[str, Any] | None = None,
) -> str:
    """Return the canonical MCP error envelope for Pulsar tools."""
    error_obj = {
        "status": "error",
        "tool": tool,
        "reason": reason,
        "error_code": error_code,
        "agent_action": agent_action,
        "details": details or {},
        "metrics": metrics or {},
    }
    return json.dumps(error_obj, indent=2)


def classify_path(path: str) -> dict[str, Any]:
    """Classify a path as likely host-visible or sandbox-local."""
    matched_prefix = next(
        (prefix for prefix in _SANDBOX_PATH_PREFIXES if path.startswith(prefix)),
        None,
    )
    return {
        "attempted_path": path,
        "looks_like_sandbox_path": matched_prefix is not None,
        "matched_prefix": matched_prefix,
    }


def path_access_error(
    tool: str,
    path: str,
    *,
    missing_code: str = "FILE_NOT_FOUND",
    missing_reason: str = "Path does not exist on the MCP host filesystem.",
    missing_action: str = (
        "Provide a host-visible absolute path, or call get_runtime_context first."
    ),
) -> str:
    """Return a structured error for path visibility or existence failures."""
    path_context = classify_path(path)
    if path_context["looks_like_sandbox_path"]:
        return mcp_error(
            tool,
            "Path is not visible to the MCP server host filesystem.",
            error_code="HOST_PATH_NOT_VISIBLE",
            agent_action=(
                "Do not use bash/cp as a bridge. Ask the user for a host-visible "
                "absolute path, or call get_runtime_context first."
            ),
            details={"path_context": path_context},
        )

    return mcp_error(
        tool,
        missing_reason,
        error_code=missing_code,
        agent_action=missing_action,
        details={"path_context": path_context},
    )


def unknown_handle_error(tool: str, handle_name: str, handle_value: str) -> str:
    """Return a structured error for unknown dataset/run handles."""
    return mcp_error(
        tool,
        f"Unknown {handle_name} '{handle_value}'.",
        error_code=f"{handle_name.upper()}_UNKNOWN",
        agent_action=(
            f"Create or retrieve a valid {handle_name} before retrying this tool."
        ),
        details={handle_name: handle_value},
    )
