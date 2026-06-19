from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import Any, Literal

import yaml
from fastmcp import Context

from pulsar.config import load_config
from pulsar.preprocessing import preprocess_dataframe
from pulsar.mcp.errors import mcp_error
from pulsar.mcp.preprocessing import (
    _preprocessing_block_to_yaml,
    _recommend_preprocessing_block,
    build_preprocessing_recommendation_payload,
    enrich_dirty_numeric_samples,
    preprocessing_recommendation_to_markdown,
    repair_config,
)
from pulsar.mcp.session import (
    _get_session,
    _load_session_dataframe,
    _resolve_dataset_path,
)

logger = logging.getLogger(__name__)


async def recommend_preprocessing(
    dataset_id: str = "",
    detail: Literal["summary", "full"] = "summary",
    response_format: Literal["markdown", "json"] = "markdown",
    rationale_limit: int = 20,
    ctx: Context = None,
) -> str:
    """Preprocessing recommendations from column profiles. Returns
    `preprocessing_yaml`, rationale, and expansion estimate.
    """
    if rationale_limit < 1:
        return mcp_error(
            "recommend_preprocessing",
            f"rationale_limit must be >= 1, got '{rationale_limit}'",
        )
    if not dataset_id:
        return mcp_error(
            "recommend_preprocessing",
            "dataset_id is required.",
            error_code="MISSING_INPUT",
            agent_action="Pass dataset_id from ingest_dataset.",
        )
    try:
        from pulsar.analysis.characterization import characterize_dataset as _char

        session = _get_session(ctx)
        df, normalized_path = await _load_session_dataframe(
            session,
            dataset_id=dataset_id,
        )
        result = await asyncio.to_thread(_char, normalized_path, dataframe=df)
        geo = dataclasses.asdict(result)

        n_samples = geo.get("n_samples", 0)
        column_profiles = geo.get("column_profiles", [])

        column_profiles, dirty_numeric_detection = enrich_dirty_numeric_samples(
            column_profiles, df
        )
        drop, impute, encode, rationale = _recommend_preprocessing_block(
            column_profiles, n_samples
        )

        preprocessing_yaml = _preprocessing_block_to_yaml(drop, impute, encode)

        # Estimate expansion: each encoded column adds ~n_unique dummy columns
        expansion_estimate = 0
        for raw_cp in column_profiles:
            cp = (
                raw_cp
                if isinstance(raw_cp, dict)
                else {"name": raw_cp.name, "n_unique": raw_cp.n_unique}
            )
            name = cp["name"]
            if name in encode:
                expansion_estimate += cp.get("n_unique", 2)

        payload = build_preprocessing_recommendation_payload(
            drop=drop,
            impute=impute,
            encode=encode,
            rationale=rationale,
            column_profiles=column_profiles,
            preprocessing_yaml=preprocessing_yaml,
            expansion_estimate=expansion_estimate,
            dirty_numeric_detection=dirty_numeric_detection,
            detail=detail,
            rationale_limit=rationale_limit,
        )
        if response_format == "markdown":
            return preprocessing_recommendation_to_markdown(payload)
        return json.dumps(payload, indent=2)

    except Exception as e:
        logger.error(f"Error in recommend_preprocessing: {e}")
        return mcp_error("recommend_preprocessing", str(e))


async def repair_preprocessing_config(
    error_message: str,
    config_yaml: str,
    dataset_id: str,
    ctx: Context = None,
) -> str:
    """Given a sweep preprocessing error, return corrected `config_yaml`
    with a change log. Handles NaN remaining, non-numeric columns, coercion
    failure, all-missing columns, and cardinality violations."""
    try:
        from pulsar.analysis.characterization import characterize_dataset as _char

        session = _get_session(ctx)
        dataset_path = _resolve_dataset_path(dataset_id)
        df, _ = await _load_session_dataframe(session, dataset_id=dataset_id)
        result = await asyncio.to_thread(_char, dataset_path, dataframe=df)
        geo = dataclasses.asdict(result)

        profiles_by_name: dict[str, Any] = {}
        for cp in geo.get("column_profiles", []):
            pname = cp["name"] if isinstance(cp, dict) else cp.name
            profiles_by_name[pname] = cp
        return repair_config(config_yaml, error_message, profiles_by_name)
    except Exception as e:
        logger.error(f"Error in repair_preprocessing_config: {e}")
        return mcp_error("repair_preprocessing_config", str(e))


async def validate_preprocessing_config(config_yaml: str, ctx: Context) -> str:
    """Dry-run preprocessing only against session data (no PCA/BallMapper/sweep
    cost). Requires prior `run_topological_sweep` or `characterize_dataset` to
    load data into the session. Returns PASS+schema summary or structured error.
    """
    session = _get_session(ctx)

    if session.data is None:
        return mcp_error(
            "validate_preprocessing_config",
            "No data in session. Run run_topological_sweep (even with a minimal config) or characterize_dataset first to load data into the session.",
        )

    try:
        config_dict = yaml.safe_load(config_yaml)
        if not isinstance(config_dict, dict):
            return mcp_error(
                "validate_preprocessing_config",
                "config_yaml must be a valid YAML mapping.",
            )

        cfg = load_config(config_dict)
        df_out, layout = await asyncio.to_thread(
            preprocess_dataframe, session.data, cfg
        )

        # Compute expansion diagnostics
        input_cols = set(session.data.columns)
        output_names = layout.feature_names
        dummy_count = sum(
            1 for name in output_names if "_" in name and name not in input_cols
        )
        missingness_flag_count = sum(
            1 for name in output_names if name.endswith("_was_missing")
        )
        high_cardinality_encoded = [
            col for col, cats in layout.vocab.items() if len(cats) > 20
        ]

        col_preview = list(output_names[:8])
        if len(output_names) > 8:
            col_preview.append(f"... +{len(output_names) - 8} more")

        return json.dumps(
            {
                "status": "ok",
                "valid": True,
                "input_rows": len(session.data),
                "output_rows": layout.n_rows,
                "output_feature_count": len(output_names),
                "dummy_expansion_count": dummy_count,
                "missingness_flag_count": missingness_flag_count,
                "high_cardinality_encoded_columns": high_cardinality_encoded,
                "feature_names_preview": list(col_preview),
                "nan_remaining": 0,
            },
            indent=2,
        )

    except (ValueError, TypeError) as e:
        return json.dumps(
            {
                "status": "error",
                "valid": False,
                "error": str(e),
                "agent_action": (
                    "Call repair_preprocessing_config(error_message=..., "
                    "config_yaml=..., dataset_id=...) to fix this automatically."
                ),
            },
            indent=2,
        )
    except Exception as e:
        logger.error(f"Error in validate_preprocessing_config: {e}")
        return mcp_error("validate_preprocessing_config", str(e))
