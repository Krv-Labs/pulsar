"""Per-tool MCP metadata (annotations, output schema, timeout, tags) and
small JSON-Schema builders for declaring success payloads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_META_ATTR = "__pulsar_tool_meta__"


@dataclass(frozen=True)
class PulsarToolMeta:
    read_only: bool
    output_schema: dict[str, Any]
    idempotent: bool | None = None
    destructive: bool = False
    timeout: float | None = None
    tags: frozenset[str] | None = None
    title: str | None = None


def tool_meta(
    *,
    read_only: bool,
    output_schema: dict[str, Any],
    idempotent: bool | None = None,
    destructive: bool = False,
    timeout: float | None = None,
    tags: set[str] | None = None,
    title: str | None = None,
) -> Callable[[F], F]:
    """Attach a ``PulsarToolMeta`` to a tool function."""
    meta = PulsarToolMeta(
        read_only=read_only,
        output_schema=output_schema,
        idempotent=idempotent,
        destructive=destructive,
        timeout=timeout,
        tags=frozenset(tags) if tags else None,
        title=title,
    )

    def decorate(fn: F) -> F:
        setattr(fn, _META_ATTR, meta)
        return fn

    return decorate


def tool_meta_of(fn: Callable[..., Any]) -> PulsarToolMeta | None:
    return getattr(fn, _META_ATTR, None)


# ---------------------------------------------------------------------------
# Schema builders
# ---------------------------------------------------------------------------
STR: dict[str, Any] = {"type": "string"}
INT: dict[str, Any] = {"type": "integer"}
NUM: dict[str, Any] = {"type": "number"}
BOOL: dict[str, Any] = {"type": "boolean"}


def obj(
    properties: dict[str, Any],
    required: list[str] | None = None,
    *,
    additional: bool = False,
) -> dict[str, Any]:
    """Object schema. ``required`` lists mandatory keys; ``additional=True``
    allows undeclared keys (open sub-objects such as free-form metrics)."""
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = list(required)
    schema["additionalProperties"] = bool(additional)
    return schema


def arr(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def enum(values: list[Any]) -> dict[str, Any]:
    return {"enum": list(values)}


# Documentation only: error results (is_error=True) are not schema-validated.
ERROR_ENVELOPE_SCHEMA: dict[str, Any] = obj(
    {
        "status": enum(["error"]),
        "tool": STR,
        "reason": STR,
        "error_code": nullable(STR),
        "agent_action": nullable(STR),
        "details": obj({}, additional=True),
        "metrics": obj({}, additional=True),
    },
    required=["status", "tool", "reason"],
)
