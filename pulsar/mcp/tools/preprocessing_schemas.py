from __future__ import annotations

from typing import Any

from pulsar.mcp.preprocessing import EVENT_FLAG_HINT_DESCRIPTION
from pulsar.mcp.toolmeta import (
    ANY,
    BOOL,
    INT,
    NUM,
    STR,
    arr,
    enum,
    nullable,
    obj,
)


def _strict(props: dict[str, Any]) -> dict[str, Any]:
    return obj(props, list(props))


_OPEN = obj({}, additional=True)

_DECIDED_BY_SCHEMA = _strict({"default": INT, "agent": INT, "user": INT})

_COUNTS_PROPS: dict[str, Any] = {
    "columns": INT,
    "feature_columns": INT,
    "dropped": INT,
    "open_entries": INT,
    "open_questions": INT,
    "decided_by": _DECIDED_BY_SCHEMA,
}
_EDIT_COUNTS: dict[str, Any] = {
    **_COUNTS_PROPS,
    "confirmed_without_change": INT,
}

_COUNTS_SCHEMA = _strict(_COUNTS_PROPS)
_EDIT_COUNTS_SCHEMA = _strict(_EDIT_COUNTS)

_PROCESSED_SPACE_SCHEMA = nullable(
    _strict(
        {
            "feature_count": INT,
            "missing_marker_count": INT,
            "dummy_column_count": INT,
        }
    )
)

_MARKER_LEGEND_ITEM_SCHEMA = _strict(
    {
        "marker_column": STR,
        "columns_preview": arr(STR),
        "columns_count": INT,
    }
)

_QUESTION_ITEM_SCHEMA = _strict(
    {
        "question_id": STR,
        "topic": enum(
            [
                "blank_meaning",
                "code_columns",
                "conditional_column",
                "date_columns",
                "duplicate_signal",
                "mixed_values",
                "mostly_blank",
                "outcome_columns",
            ]
        ),
        "text": STR,
        "applies_to_preview": arr(STR),
        "applies_to_count": INT,
        "rows_affected": INT,
        "options": arr(
            _strict(
                {"option_id": STR, "label": STR, "effect": STR},
            )
        ),
        "default_option_id": STR,
        "stakes": nullable(
            _strict(
                {
                    "neighbors_kept": NUM,
                    "feature_count_delta": INT,
                    "rows_sampled": INT,
                    "k": INT,
                }
            )
        ),
    }
)

_TARGET_PROXY_ITEM_SCHEMA = _strict(
    {
        "column": STR,
        "target": STR,
        "marker_rows": INT,
        "share_with_target": NUM,
        "target_base_rate": NUM,
    }
)

_EVIDENCE_SCHEMA = _strict(
    {
        "missing_groups": arr(
            _strict(
                {
                    "group_id": STR,
                    "columns_preview": arr(STR),
                    "columns_count": INT,
                    "n_missing": INT,
                    "zero_free_columns_preview": arr(STR),
                }
            )
        ),
        "missing_groups_omitted": INT,
        "mask_dependencies": arr(
            _strict(
                {
                    "column": STR,
                    "partner": STR,
                    "relation": STR,
                    "n_missing": INT,
                    "n_other_partners": INT,
                }
            )
        ),
        "mask_dependencies_omitted": INT,
        "relations": arr(
            _strict(
                {
                    "kind": STR,
                    "target": STR,
                    "sources": arr(STR),
                    "value": NUM,
                    "rows_used": INT,
                }
            )
        ),
        "relations_omitted": INT,
        "rare_binary_leverage": arr(
            _strict(
                {"column": STR, "minority_fraction": NUM, "max_abs_z": NUM},
            )
        ),
        "rare_binary_leverage_omitted": INT,
        "constant_columns_preview": arr(STR),
        "constant_columns_count": INT,
        "constant_columns_omitted": INT,
        "datetime_columns_preview": arr(STR),
        "datetime_columns_count": INT,
        "datetime_columns_omitted": INT,
        "integer_coded_preview": arr(_strict({"column": STR, "n_unique": INT})),
        "integer_coded_omitted": INT,
        "target_proxies": arr(_TARGET_PROXY_ITEM_SCHEMA),
        "target_proxies_omitted": INT,
        "most_missing_columns": arr(
            _strict(
                {
                    "column": STR,
                    "missing_fraction": NUM,
                    "n_missing": INT,
                    "missing_group": nullable(STR),
                    "plan_role": STR,
                    "plan_impute_method": nullable(STR),
                }
            )
        ),
        "most_missing_columns_omitted": INT,
    }
)

_TARGET_PROXY_WARNING_SCHEMA = _strict(
    {
        "code": STR,
        "target": STR,
        "columns": arr(STR),
        "message": STR,
    }
)

_EDIT_EVIDENCE_SCHEMA = _strict(
    {
        "target_proxies": arr(_TARGET_PROXY_ITEM_SCHEMA),
        "target_proxies_omitted": INT,
    }
)

DRAFT_PLAN_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "plan_id": STR,
        "dataset_id": STR,
        "revision": INT,
        "plan_fingerprint": STR,
        "detail": enum(["summary", "full"]),
        "counts": _COUNTS_SCHEMA,
        "processed_space": _PROCESSED_SPACE_SCHEMA,
        "marker_legend": arr(_MARKER_LEGEND_ITEM_SCHEMA),
        "marker_legend_omitted": INT,
        "questions": arr(_QUESTION_ITEM_SCHEMA),
        "questions_omitted": INT,
        "evidence": _EVIDENCE_SCHEMA,
        "warnings": arr(_TARGET_PROXY_WARNING_SCHEMA),
        "weak_hints": _strict(
            {
                "event_flag_columns": {
                    **arr(STR),
                    "description": EVENT_FLAG_HINT_DESCRIPTION,
                },
                "event_flag_columns_omitted": INT,
                "description": STR,
            }
        ),
        "preprocessing_yaml_chars": INT,
        "cost": _OPEN,
        "how_to_use": STR,
        "next_tools": arr(STR),
        # detail="full" only
        "preprocessing_yaml": STR,
        "entries": _OPEN,
    },
    [
        "status",
        "plan_id",
        "dataset_id",
        "revision",
        "plan_fingerprint",
        "detail",
        "counts",
        "processed_space",
        "marker_legend",
        "marker_legend_omitted",
        "questions",
        "questions_omitted",
        "evidence",
        "warnings",
        "weak_hints",
        "preprocessing_yaml_chars",
        "cost",
        "how_to_use",
        "next_tools",
    ],
)

EDIT_PLAN_SCHEMA = _strict(
    {
        "status": enum(["ok"]),
        "plan_id": STR,
        "dataset_id": STR,
        "revision": INT,
        "plan_fingerprint": STR,
        "changes": arr(
            _strict(
                {"column": STR, "field": STR, "was": ANY, "now": ANY},
            )
        ),
        "changes_omitted": INT,
        "change_summary": arr(
            _strict(
                {"field": STR, "was": ANY, "now": ANY, "count": INT},
            )
        ),
        "change_summary_omitted": INT,
        "counts": _EDIT_COUNTS_SCHEMA,
        "processed_space": _PROCESSED_SPACE_SCHEMA,
        "marker_legend": arr(_MARKER_LEGEND_ITEM_SCHEMA),
        "marker_legend_omitted": INT,
        "evidence": _EDIT_EVIDENCE_SCHEMA,
        "warnings": arr(_TARGET_PROXY_WARNING_SCHEMA),
        "validation": _strict(
            {"valid": BOOL, "error": nullable(STR)},
        ),
        "open_questions_preview": arr(_QUESTION_ITEM_SCHEMA),
        "open_questions_omitted": INT,
        "preprocessing_yaml_chars": INT,
        "next_tools": arr(STR),
    }
)

VALIDATE_SCHEMA = _strict(
    {
        "status": enum(["ok"]),
        "valid": BOOL,
        "input_rows": INT,
        "output_rows": INT,
        "output_feature_count": INT,
        "dummy_expansion_count": INT,
        "missingness_flag_count": INT,
        "high_cardinality_encoded_columns": arr(STR),
        "feature_names_preview": arr(STR),
        "nan_remaining": INT,
    }
)
