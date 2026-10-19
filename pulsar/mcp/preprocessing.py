"""Preprocessing plan payload formatting and processed-space calibration helpers
for Pulsar MCP tools."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Literal

import numpy as np

from pulsar.analysis.characterization import profile_numeric_matrix
from pulsar.config import load_config
from pulsar.mcp.preprocessing_plan import (
    PreprocessingPlan,
    build_marker_legend,
    plan_fingerprint,
    plan_to_preprocessing_dict,
    plan_to_preprocessing_yaml,
    summarize_plan_counts,
)
from pulsar.preprocessing import preprocess_dataframe

logger = logging.getLogger(__name__)

_HOW_TO_USE = (
    "Plan contains measured evidence and labeled defaults, not a verdict. "
    "Resolve questions you can infer from context via "
    "edit_preprocessing_plan(source='agent'); ask the user in their own domain "
    "words only when the answer cannot be inferred and stakes are material, "
    "then record with edit_preprocessing_plan(source='user'). "
    "To hold out outcome columns, pass edits=[{'column': '<col>', 'role': 'target'}] "
    "to edit_preprocessing_plan (or target_columns=[...] to draft_preprocessing_plan). "
    "To remove TARGET_PROXY_COLUMNS, set indicator='none' or role='drop' on the source columns in edits. "
    "For code_columns that are labels rather than amounts, pass edits=[{'column': '<col>', 'encode': 'one_hot'}] or role='drop'."
)

EVENT_FLAG_HINT_DESCRIPTION = (
    "complete 0/1 or count columns whose value exactly explains "
    "other columns' blanks (often events or exposures; only you "
    "can say whether any is an outcome)"
)


def _build_target_proxy_warnings(
    target_proxies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_target: dict[str, list[str]] = {}
    for item in target_proxies:
        if item.get("share_with_target") == 1.0:
            by_target.setdefault(str(item.get("target")), []).append(
                str(item["column"])
            )
    return [
        {
            "code": "TARGET_PROXY_COLUMNS",
            "target": tgt,
            "columns": cols,
            "message": (
                f"these columns are 1 only for rows where {tgt} is set; "
                "they would let the map encode the outcome"
            ),
        }
        for tgt, cols in by_target.items()
    ]


def _format_question_item(
    q: Any, *, col_preview_limit: int | None = 5
) -> dict[str, Any]:
    return {
        "question_id": q.question_id,
        "topic": q.topic,
        "text": q.text,
        "applies_to_preview": _cap_list(q.applies_to, col_preview_limit)[0],
        "applies_to_count": len(q.applies_to),
        "rows_affected": int(q.rows_affected),
        "options": [
            {
                "option_id": opt.option_id,
                "label": opt.label,
                "effect": opt.effect,
            }
            for opt in q.options
        ],
        "default_option_id": q.default_option_id,
        "stakes": copy.deepcopy(q.stakes),
    }


def _cap_list(items: Any, limit: int | None) -> tuple[list[Any], int]:
    if limit is None:
        return list(items), 0
    capped = list(items[:limit])
    return capped, len(items) - len(capped)


def _capped(
    key: str,
    items: list[Any],
    cap: int | None,
    *,
    preview_key: str | None = None,
    count_key: str | None = None,
) -> dict[str, Any]:
    out, omitted = _cap_list(items, cap)
    result: dict[str, Any] = {preview_key or key: out}
    if count_key is not None:
        result[count_key] = len(items)
    result[f"{key}_omitted"] = omitted
    return result


def _plan_header(plan: PreprocessingPlan) -> dict[str, Any]:
    return {
        "status": "ok",
        "plan_id": plan.plan_id,
        "dataset_id": plan.dataset_id,
        "revision": int(plan.revision),
        "plan_fingerprint": plan_fingerprint(plan),
    }


def _open_question_items(
    plan: PreprocessingPlan, *, col_preview_limit: int | None = 5
) -> list[dict[str, Any]]:
    return [
        _format_question_item(q, col_preview_limit=col_preview_limit)
        for q in plan.questions.values()
        if q.status == "open"
    ]


def _target_proxy_fields(plan: PreprocessingPlan, cap: int | None) -> dict[str, Any]:
    proxies_all = [copy.deepcopy(p) for p in plan.target_proxies]
    return _capped("target_proxies", proxies_all, cap)


def _missing_group_item(mg: Any, col_cap: int | None) -> dict[str, Any]:
    return {
        "group_id": mg.group_id,
        "columns_preview": _cap_list(mg.columns, col_cap)[0],
        "columns_count": len(mg.columns),
        "n_missing": int(mg.n_missing),
        "zero_free_columns_preview": _cap_list(mg.zero_free_columns, col_cap)[0],
    }


def _most_missing_item(plan: PreprocessingPlan, col_ev: Any) -> dict[str, Any]:
    entry = plan.entries[col_ev.name]
    return {
        "column": col_ev.name,
        "missing_fraction": float(col_ev.missing_fraction),
        "n_missing": int(col_ev.n_missing),
        "missing_group": col_ev.missing_group,
        "plan_role": entry.role,
        "plan_impute_method": (
            entry.impute.get("method") if isinstance(entry.impute, dict) else None
        ),
    }


def _mask_dependency_items(ev: Any) -> list[dict[str, Any]]:
    ev_by_col = {c.name: c for c in ev.columns}
    return [
        {
            "column": md.column,
            "partner": md.partner,
            "relation": md.relation,
            "n_missing": (
                int(ev_by_col[md.column].n_missing) if md.column in ev_by_col else 0
            ),
            "n_other_partners": int(md.n_other_partners),
        }
        for md in ev.mask_dependencies
    ]


def _rare_binary_items(ev: Any) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "column": col_ev.name,
                "minority_fraction": float(col_ev.minority_fraction),
                "max_abs_z": float(col_ev.max_abs_z),
            }
            for col_ev in ev.columns
            if col_ev.minority_fraction is not None and col_ev.max_abs_z is not None
        ],
        key=lambda item: (-item["max_abs_z"], item["column"]),
    )


def _integer_coded_items(ev: Any) -> list[dict[str, Any]]:
    return [
        {"column": col_ev.name, "n_unique": int(col_ev.n_unique)}
        for col_ev in ev.columns
        if col_ev.kind == "numeric"
        and col_ev.integer_valued is True
        and not col_ev.is_constant
        and col_ev.n_unique > 2
    ]


def _most_missing_items(plan: PreprocessingPlan, ev: Any) -> list[dict[str, Any]]:
    return sorted(
        [
            _most_missing_item(plan, col_ev)
            for col_ev in ev.columns
            if col_ev.n_missing > 0 and col_ev.name in plan.entries
        ],
        key=lambda item: (-item["missing_fraction"], item["column"]),
    )


def _draft_evidence(
    plan: PreprocessingPlan, *, ev_cap: int | None, col_cap: int | None
) -> dict[str, Any]:
    ev = plan.evidence
    mg_all = [_missing_group_item(mg, col_cap) for mg in ev.missing_groups]
    rel_all = [
        {
            "kind": rel.kind,
            "target": rel.target,
            "sources": list(rel.sources),
            "value": float(rel.value),
            "rows_used": int(rel.rows_used),
        }
        for rel in ev.relations
    ]
    const_all = [col_ev.name for col_ev in ev.columns if col_ev.is_constant]
    date_all = [col_ev.name for col_ev in ev.columns if col_ev.kind == "datetime_like"]
    return {
        **_capped("missing_groups", mg_all, ev_cap),
        **_capped("mask_dependencies", _mask_dependency_items(ev), ev_cap),
        **_capped("relations", rel_all, ev_cap),
        **_capped("rare_binary_leverage", _rare_binary_items(ev), ev_cap),
        **_capped(
            "constant_columns",
            const_all,
            ev_cap,
            preview_key="constant_columns_preview",
            count_key="constant_columns_count",
        ),
        **_capped(
            "datetime_columns",
            date_all,
            ev_cap,
            preview_key="datetime_columns_preview",
            count_key="datetime_columns_count",
        ),
        **_capped(
            "integer_coded",
            _integer_coded_items(ev),
            ev_cap,
            preview_key="integer_coded_preview",
        ),
        **_target_proxy_fields(plan, ev_cap),
        **_capped("most_missing_columns", _most_missing_items(plan, ev), ev_cap),
    }


def _entries_full(plan: PreprocessingPlan) -> dict[str, dict[str, Any]]:
    return {col: copy.deepcopy(entry.to_dict()) for col, entry in plan.entries.items()}


def build_draft_plan_payload(
    plan: PreprocessingPlan,
    *,
    processed_space: dict[str, int] | None,
    detail: Literal["summary", "full"] = "summary",
    question_limit: int = 8,
    evidence_limit: int = 12,
) -> dict[str, Any]:
    """Build the structured payload for ``draft_preprocessing_plan``."""
    is_full = detail == "full"
    q_cap = None if is_full else max(1, int(question_limit))
    ev_cap = None if is_full else max(1, int(evidence_limit))
    col_cap = None if is_full else 5

    open_questions = _open_question_items(plan, col_preview_limit=col_cap)
    legend_all = build_marker_legend(
        plan, preview_limit=5 if col_cap is not None else 10_000
    )

    seen_partners = list(
        dict.fromkeys(md.partner for md in plan.evidence.mask_dependencies)
    )

    yaml_str = plan_to_preprocessing_yaml(plan, include_header=True)
    payload: dict[str, Any] = {
        **_plan_header(plan),
        "detail": detail,
        "counts": summarize_plan_counts(plan),
        "processed_space": copy.deepcopy(processed_space),
        **_capped("marker_legend", legend_all, ev_cap),
        **_capped("questions", open_questions, q_cap),
        "evidence": _draft_evidence(plan, ev_cap=ev_cap, col_cap=col_cap),
        "warnings": _build_target_proxy_warnings(plan.target_proxies),
        "weak_hints": {
            **_capped("event_flag_columns", seen_partners, ev_cap),
            "description": EVENT_FLAG_HINT_DESCRIPTION,
        },
        "preprocessing_yaml_chars": len(yaml_str),
        "cost": copy.deepcopy(plan.cost),
        "how_to_use": _HOW_TO_USE,
        "next_tools": ["edit_preprocessing_plan", "create_config", "probe_columns"],
    }

    if is_full:
        payload["preprocessing_yaml"] = yaml_str
        payload["entries"] = _entries_full(plan)

    return payload


def _warning_lines(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for w in payload.get("warnings", []):
        cols_str = ", ".join(f"`{c}`" for c in w.get("columns", []))
        lines.append(
            f"- Warning `{w.get('code')}` (`{w.get('target')}`): {w.get('message')} ({cols_str})"
        )
    return lines


def _format_question_lines(q: dict[str, Any]) -> list[str]:
    stakes = q.get("stakes")
    if isinstance(stakes, dict) and stakes.get("neighbors_kept") is not None:
        stakes_str = (
            f"neighbors_kept={stakes['neighbors_kept']:.2f}, "
            f"feature_delta={stakes.get('feature_count_delta', 0):+d}"
        )
    else:
        stakes_str = "stakes=n/a"
    lines = [
        f"### `{q['question_id']}` [{q['topic']}] "
        f"({q['rows_affected']} rows; {stakes_str})",
        q["text"],
    ]
    for opt in q.get("options", []):
        default_tag = (
            " *(default)*" if opt["option_id"] == q.get("default_option_id") else ""
        )
        lines.append(
            f"- `{opt['option_id']}` — **{opt['label']}**{default_tag}: {opt['effect']}"
        )
    lines.append("")
    return lines


def _open_questions_markdown(payload: dict[str, Any]) -> list[str]:
    questions = payload.get("questions", [])
    lines = ["", "## Open Questions", ""]
    if not questions:
        lines.append("- None. All entries are resolved.")
        return lines
    for q in questions:
        lines.extend(_format_question_lines(q))
    if payload.get("questions_omitted", 0):
        lines.append(
            f"*... +{payload['questions_omitted']} additional open question(s) omitted.*"
        )
    return lines


def draft_plan_payload_to_markdown(payload: dict[str, Any]) -> str:
    """Render a compact Markdown view of ``draft_preprocessing_plan``."""
    counts = payload.get("counts", {})
    proc = payload.get("processed_space")
    lines = [
        f"# Preprocessing Plan `{payload.get('plan_id')}` (rev {payload.get('revision')})",
        "",
        f"- Dataset: `{payload.get('dataset_id')}` | Fingerprint: `{payload.get('plan_fingerprint')}`",
        (
            f"- Columns: {counts.get('columns', 0)} "
            f"(feature columns: {counts.get('feature_columns', 0)}, "
            f"dropped/held-out: {counts.get('dropped', 0)}, "
            f"open questions: {counts.get('open_questions', 0)})"
        ),
    ]
    if proc is not None:
        lines.append(
            f"- Processed space: {proc.get('feature_count', 0)} features "
            f"({proc.get('missing_marker_count', 0)} missing markers, "
            f"{proc.get('dummy_column_count', 0)} dummy columns)"
        )
    else:
        lines.append("- Processed space: dry-run unavailable")

    lines.extend(_warning_lines(payload))
    lines.extend(_open_questions_markdown(payload))

    ev = payload.get("evidence", {})
    lines.extend(["", "## Evidence Headline", ""])
    for label, key in (
        ("Shared missing groups", "missing_groups"),
        ("Conditional mask dependencies", "mask_dependencies"),
        ("Exact relations (duplicate/sum_of)", "relations"),
        ("Most-missing columns", "most_missing_columns"),
    ):
        lines.append(
            f"- {label}: {len(ev.get(key, []))} (+{ev.get(f'{key}_omitted', 0)} omitted)"
        )
    lines.append(f"- Constant columns dropped: {ev.get('constant_columns_count', 0)}")
    weak_hints = payload.get("weak_hints", {})
    hints = weak_hints.get("event_flag_columns", [])
    hints_omitted = weak_hints.get("event_flag_columns_omitted", 0)
    if hints:
        omitted_suffix = f" (+{hints_omitted} omitted)" if hints_omitted else ""
        lines.append(
            "- Event-flag weak hints: "
            + ", ".join(f"`{c}`" for c in hints)
            + omitted_suffix
        )

    if "preprocessing_yaml" in payload:
        lines.extend(
            [
                "",
                "## Preprocessing YAML",
                "",
                "```yaml",
                str(payload.get("preprocessing_yaml", "")).strip(),
                "```",
            ]
        )
    else:
        lines.append(
            f"- Preprocessing YAML: {payload.get('preprocessing_yaml_chars', 0)} chars "
            "(pass `detail='full'` or call `create_config` to view)"
        )
    lines.extend(["", payload.get("how_to_use", _HOW_TO_USE)])
    return "\n".join(lines).strip()


def build_edit_plan_payload(
    plan: PreprocessingPlan,
    *,
    changes: list[dict[str, Any]],
    processed_space: dict[str, int] | None,
    validation: dict[str, Any],
    question_limit: int = 8,
    change_limit: int = 20,
    evidence_limit: int = 12,
) -> dict[str, Any]:
    """Build the structured payload for ``edit_preprocessing_plan``."""
    effective_changes = [
        ch
        for ch in changes
        if ch.get("field") in ("role", "impute", "indicator", "encode")
    ]

    summary_groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ch in effective_changes:
        fld = str(ch.get("field"))
        was_val = ch.get("was")
        now_val = ch.get("now")
        key = (
            fld,
            json.dumps(was_val, sort_keys=True, default=str),
            json.dumps(now_val, sort_keys=True, default=str),
        )
        grp = summary_groups.setdefault(
            key,
            {
                "field": fld,
                "was": copy.deepcopy(was_val),
                "now": copy.deepcopy(now_val),
                "count": 0,
            },
        )
        grp["count"] += 1
    summary_all = list(summary_groups.values())

    open_questions = _open_question_items(plan, col_preview_limit=5)
    legend_all = build_marker_legend(plan, preview_limit=5)

    effective_changed_cols = {ch["column"] for ch in effective_changes}
    provenance_changed_cols = {
        ch["column"] for ch in changes if ch.get("field") in ("status", "decided_by")
    }
    counts = summarize_plan_counts(plan)
    counts["confirmed_without_change"] = len(
        provenance_changed_cols - effective_changed_cols
    )

    change_cap = max(1, int(change_limit))
    ev_cap = max(1, int(evidence_limit))
    yaml_str = plan_to_preprocessing_yaml(plan, include_header=True)
    next_tools = (
        ["edit_preprocessing_plan", "create_config"]
        if open_questions
        else ["create_config", "validate_preprocessing_config"]
    )
    return {
        **_plan_header(plan),
        **_capped("changes", effective_changes, change_cap),
        **_capped("change_summary", summary_all, change_cap),
        "counts": counts,
        "processed_space": copy.deepcopy(processed_space),
        **_capped("marker_legend", legend_all, ev_cap),
        "evidence": _target_proxy_fields(plan, ev_cap),
        "warnings": _build_target_proxy_warnings(plan.target_proxies),
        "validation": {
            "valid": bool(validation.get("valid", False)),
            "error": validation.get("error"),
        },
        **_capped(
            "open_questions",
            open_questions,
            max(1, int(question_limit)),
            preview_key="open_questions_preview",
        ),
        "preprocessing_yaml_chars": len(yaml_str),
        "next_tools": next_tools,
    }


def edit_plan_payload_to_markdown(payload: dict[str, Any]) -> str:
    """Render a compact Markdown view of ``edit_preprocessing_plan``."""
    counts = payload.get("counts", {})
    val = payload.get("validation", {})
    lines = [
        f"# Updated Preprocessing Plan `{payload.get('plan_id')}` (rev {payload.get('revision')})",
        "",
        f"- Fingerprint: `{payload.get('plan_fingerprint')}` | Valid: `{val.get('valid')}`",
        (
            f"- Counts: {counts.get('feature_columns', 0)} feature columns, "
            f"{counts.get('dropped', 0)} dropped/held-out, "
            f"{counts.get('open_questions', 0)} open questions"
        ),
        f"- Preprocessing YAML: {payload.get('preprocessing_yaml_chars', 0)} chars",
    ]
    open_preview = payload.get("open_questions_preview", [])
    if open_preview:
        q_tags = ", ".join(f"`{q['question_id']}` [{q['topic']}]" for q in open_preview)
        omitted = payload.get("open_questions_omitted", 0)
        if omitted:
            q_tags += f" (+{omitted} more)"
        lines.append(f"- Open questions: {q_tags}")
    if val.get("error"):
        lines.append(f"- Validation error: `{val['error']}`")
    lines.extend(_warning_lines(payload))

    changes = payload.get("changes", [])
    lines.extend(["", "## Changes", ""])
    if changes:
        lines.extend(["| Column | Field | Was | Now |", "|---|---|---|---|"])
        for ch in changes:
            was_str = json.dumps(ch.get("was"), default=str)
            now_str = json.dumps(ch.get("now"), default=str)
            lines.append(
                f"| `{ch.get('column')}` | `{ch.get('field')}` | `{was_str}` | `{now_str}` |"
            )
        if payload.get("changes_omitted", 0):
            lines.append(
                f"\n*... +{payload['changes_omitted']} additional change(s) omitted.*"
            )
    else:
        lines.append("- No effective entry changes.")

    return "\n".join(lines).strip()


def _calibrate_processed_space(
    df: Any,
    plan: PreprocessingPlan,
    data_path: str = "in_memory",
) -> Any:
    """Run the plan's preprocessing + scaling, then profile the resulting numeric matrix.

    Returns ``None`` if preprocessing fails or fewer than 2 features remain.
    """
    from sklearn.preprocessing import StandardScaler as SkScaler

    try:
        pre_dict = plan_to_preprocessing_dict(plan)
        temp_cfg = load_config(
            {
                "run": {"name": "calibrate", "data": data_path or "in_memory"},
                "preprocessing": pre_dict,
            }
        )
        df_processed, _ = preprocess_dataframe(df, temp_cfg)
    except Exception as exc:
        logger.info(
            "Processed-space calibration failed (%s); falling back to raw geometry",
            exc,
        )
        return None

    X = df_processed.to_numpy(dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 2:
        return None
    X_scaled = SkScaler().fit_transform(X)
    return profile_numeric_matrix(X_scaled)
