from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import Literal
import uuid

from fastmcp import Context

from pulsar.mcp.errors import mcp_error, path_access_error, unknown_handle_error
from pulsar.mcp.registry import registry
from pulsar.mcp.session import _get_session, _read_dataset_file, _resolve_dataset_path

logger = logging.getLogger(__name__)


def _select_longitudinal_entities(
    frame,
    entity_column: str,
    max_entities: int,
    min_observations: int,
    time_column: str | None,
    min_time_points: int | None,
) -> tuple[list, int]:
    import pandas as pd

    counts = frame.groupby(entity_column, dropna=False).size()
    eligible = [
        entity
        for entity, count in counts.items()
        if count >= min_observations and pd.notna(entity)
    ]
    if min_time_points is not None:
        time_counts = frame.groupby(entity_column, dropna=False)[time_column].nunique(
            dropna=True
        )
        eligible = [
            entity
            for entity in eligible
            if time_counts.get(entity, 0) >= min_time_points
        ]

    def entity_order(entity):
        identity = f"{type(entity).__module__}.{type(entity).__qualname__}:{entity!r}"
        return hashlib.sha256(identity.encode("utf-8")).digest()

    return sorted(eligible, key=entity_order)[:max_entities], len(eligible)


async def ingest_dataset(path: str, ctx: Context = None) -> str:
    """Register a host-visible absolute dataset path; returns a `dataset_id`."""
    try:
        record = registry.register_dataset(path)
        session = _get_session(ctx)
        session.dataset_id = record.dataset_id
        return json.dumps(dataclasses.asdict(record), indent=2)
    except FileNotFoundError:
        return path_access_error(
            "ingest_dataset",
            path,
            missing_action=(
                "Ask the user for a host-visible absolute dataset path, then call "
                "ingest_dataset again."
            ),
            sandbox_action=(
                "Your file is isolated in a sandbox. DO NOT use base64 or chunked uploads. "
                "Run a bash script to copy the file to the `cache_dir` (call `get_runtime_context` "
                "to find it), then retry `ingest_dataset(path)` with the new path."
            ),
        )
    except PermissionError:
        return mcp_error(
            "ingest_dataset",
            "Dataset path exists but is not readable by the MCP server.",
            error_code="FILE_PERMISSION_DENIED",
            agent_action="Provide a readable host-visible dataset path.",
            details={"path_context": {"attempted_path": path}},
        )
    except Exception as e:
        return mcp_error("ingest_dataset", str(e))


async def sample_longitudinal_entities(
    dataset_id: str,
    entity_column: str,
    max_entities: int = 2000,
    min_observations: int = 3,
    ctx: Context = None,
    time_column: str | None = None,
    min_time_points: int | None = None,
) -> str:
    """Create a deterministic, row-preserving sample of longitudinal entities.

    The source must already be ingested. The result is a new ``dataset_id`` that
    can be passed to ``create_config`` and ``build_longitudinal_graph``; no rows
    or identifier values are returned. ``min_observations`` counts rows; when
    ``min_time_points`` is set, entities must also have enough distinct values
    in ``time_column``.
    """
    if (
        max_entities < 1
        or min_observations < 1
        or (min_time_points is not None and min_time_points < 1)
        or (min_time_points is not None and time_column is None)
    ):
        return mcp_error(
            "sample_longitudinal_entities",
            "max_entities and min_observations must be positive, and min_time_points requires a positive value and time_column.",
            error_code="INVALID_SAMPLE_SIZE",
            agent_action="Use positive values and provide time_column when setting min_time_points.",
        )

    try:
        source_path = _resolve_dataset_path(dataset_id)
    except LookupError:
        return unknown_handle_error(
            "sample_longitudinal_entities", "dataset_id", dataset_id
        )

    try:
        frame = await asyncio.to_thread(_read_dataset_file, source_path)
        if entity_column not in frame.columns:
            return mcp_error(
                "sample_longitudinal_entities",
                f"Entity column '{entity_column}' is not present in the dataset.",
                error_code="ENTITY_COLUMN_UNKNOWN",
                agent_action="Call characterize_dataset or probe_columns and pass an existing entity column.",
                details={"dataset_id": dataset_id, "entity_column": entity_column},
            )
        if time_column is not None and time_column not in frame.columns:
            return mcp_error(
                "sample_longitudinal_entities",
                f"Time column '{time_column}' is not present in the dataset.",
                error_code="TIME_COLUMN_UNKNOWN",
                agent_action="Pass an existing discrete time column or omit time_column.",
                details={"dataset_id": dataset_id, "time_column": time_column},
            )

        selected, eligible_count = _select_longitudinal_entities(
            frame,
            entity_column,
            max_entities,
            min_observations,
            time_column,
            min_time_points,
        )
        sampled = frame[frame[entity_column].isin(selected)]
        if sampled.empty:
            return mcp_error(
                "sample_longitudinal_entities",
                "No entities meet the requested minimum observation and timepoint counts.",
                error_code="NO_ELIGIBLE_ENTITIES",
                agent_action="Lower the minimum counts or use different entity and time columns.",
                details={
                    "dataset_id": dataset_id,
                    "entity_column": entity_column,
                    "min_observations": min_observations,
                    "time_column": time_column,
                    "min_time_points": min_time_points,
                },
            )

        # CSV sources stay CSV: pyarrow rejects the mixed-type object columns
        # read_csv can produce, and a CSV round-trip re-reads like the source.
        suffix = ".parquet" if source_path.lower().endswith(".parquet") else ".csv"
        writer = sampled.to_parquet if suffix == ".parquet" else sampled.to_csv
        staging_path = registry.cache_dir / f"sample_{uuid.uuid4().hex}{suffix}"
        try:
            await asyncio.to_thread(writer, staging_path, index=False)
            record = registry.register_dataset_from_file(
                f"{Path(source_path).stem}_sample{suffix}",
                staging_path,
                source="entity_sample",
            )
        finally:
            staging_path.unlink(missing_ok=True)
        return json.dumps(
            {
                "dataset_id": record.dataset_id,
                "source_dataset_id": dataset_id,
                "entity_column": entity_column,
                "max_entities": max_entities,
                "min_observations": min_observations,
                "time_column": time_column,
                "min_time_points": min_time_points,
                "eligible_entities": eligible_count,
                "sampled_entities": len(selected),
                "sampled_rows": int(len(sampled)),
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Could not sample longitudinal entities")
        return mcp_error("sample_longitudinal_entities", str(e))


async def begin_dataset_upload(
    filename: str,
    media_type: str = "text/csv",
    ctx: Context = None,
) -> str:
    """Begin staged upload (sandboxed clients only). Then append chunks + finalize."""
    try:
        record = registry.begin_upload(filename, media_type=media_type)
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("begin_dataset_upload", str(e))


async def append_dataset_chunk(
    upload_id: str,
    chunk: str,
    encoding: Literal["base64", "utf-8"] = "base64",
    ctx: Context = None,
) -> str:
    """Append one chunk to a staged upload (base64 default for binary safety)."""
    try:
        if encoding == "base64":
            try:
                chunk_bytes = base64.b64decode(chunk, validate=True)
            except Exception:
                return mcp_error(
                    "append_dataset_chunk",
                    "Chunk payload could not be decoded from base64.",
                    error_code="UPLOAD_DECODE_FAILED",
                    agent_action="Retry with valid base64 chunk data.",
                )
        elif encoding == "utf-8":
            chunk_bytes = chunk.encode("utf-8")
        else:
            return mcp_error(
                "append_dataset_chunk",
                f"Unsupported chunk encoding '{encoding}'.",
                error_code="UPLOAD_ENCODING_UNSUPPORTED",
                agent_action="Use encoding='base64' for binary-safe chunk transport.",
            )

        record = registry.append_upload_chunk(upload_id, chunk_bytes)
        if record is None:
            return unknown_handle_error("append_dataset_chunk", "upload_id", upload_id)
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("append_dataset_chunk", str(e))


async def finalize_dataset_upload(upload_id: str, ctx: Context = None) -> str:
    """Finalize a staged upload; returns a `dataset_id`."""
    try:
        record = registry.finalize_upload(upload_id)
        if record is None:
            return unknown_handle_error(
                "finalize_dataset_upload", "upload_id", upload_id
            )
        session = _get_session(ctx)
        session.dataset_id = record.dataset_id
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("finalize_dataset_upload", str(e))
