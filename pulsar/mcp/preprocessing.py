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
    "to edit_preprocessing_plan (or target_columns=[...] to draft_preprocessing_plan)."
)


def _format_question_item(
    q: Any, *, col_preview_limit: int | None = 5
) -> dict[str, Any]:
    applies = list(q.applies_to)
    preview = (
        applies if col_preview_limit is None else list(applies[:col_preview_limit])
    )
    return {
        "question_id": q.question_id,
        "topic": q.topic,
        "text": q.text,
        "applies_to_preview": preview,
        "applies_to_count": len(applies),
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


def _cap_list(items: list[Any], limit: int | None) -> tuple[list[Any], int]:
    if limit is None:
        return list(items), 0
    capped = list(items[:limit])
    return capped, max(0, len(items) - len(capped))


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

    open_questions = [
        _format_question_item(q, col_preview_limit=col_cap)
        for q in plan.questions.values()
        if q.status == "open"
    ]
    questions_out, questions_omitted = _cap_list(open_questions, q_cap)

    ev = plan.evidence
    mg_all = [
        {
            "group_id": mg.group_id,
            "columns_preview": (
                list(mg.columns) if col_cap is None else list(mg.columns[:col_cap])
            ),
            "columns_count": len(mg.columns),
            "n_missing": int(mg.n_missing),
            "zero_free_columns_preview": (
                list(mg.zero_free_columns)
                if col_cap is None
                else list(mg.zero_free_columns[:col_cap])
            ),
        }
        for mg in ev.missing_groups
    ]
    mg_out, mg_omitted = _cap_list(mg_all, ev_cap)

    ev_by_col = {c.name: c for c in ev.columns}
    md_all = [
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
    md_out, md_omitted = _cap_list(md_all, ev_cap)

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
    rel_out, rel_omitted = _cap_list(rel_all, ev_cap)

    rare_all = sorted(
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
    rare_out, rare_omitted = _cap_list(rare_all, ev_cap)

    const_all = [col_ev.name for col_ev in ev.columns if col_ev.is_constant]
    const_out, const_omitted = _cap_list(const_all, ev_cap)

    date_all = [col_ev.name for col_ev in ev.columns if col_ev.kind == "datetime_like"]
    date_out, date_omitted = _cap_list(date_all, ev_cap)

    int_coded_all = [
        {"column": col_ev.name, "n_unique": int(col_ev.n_unique)}
        for col_ev in ev.columns
        if col_ev.kind == "numeric"
        and col_ev.integer_valued is True
        and not col_ev.is_constant
        and col_ev.n_unique > 2
    ]
    int_coded_out, int_coded_omitted = _cap_list(int_coded_all, ev_cap)

    legend_all = build_marker_legend(
        plan, preview_limit=5 if col_cap is not None else 10_000
    )
    legend_out, legend_omitted = _cap_list(legend_all, ev_cap)

    seen_partners: list[str] = []
    for md in ev.mask_dependencies:
        if md.partner not in seen_partners:
            seen_partners.append(md.partner)
    partners_out, partners_omitted = _cap_list(seen_partners, ev_cap)

    payload: dict[str, Any] = {
        "status": "ok",
        "plan_id": plan.plan_id,
        "dataset_id": plan.dataset_id,
        "revision": int(plan.revision),
        "plan_fingerprint": plan_fingerprint(plan),
        "detail": detail,
        "counts": summarize_plan_counts(plan),
        "processed_space": copy.deepcopy(processed_space),
        "marker_legend": legend_out,
        "marker_legend_omitted": legend_omitted,
        "questions": questions_out,
        "questions_omitted": questions_omitted,
        "evidence": {
            "missing_groups": mg_out,
            "missing_groups_omitted": mg_omitted,
            "mask_dependencies": md_out,
            "mask_dependencies_omitted": md_omitted,
            "relations": rel_out,
            "relations_omitted": rel_omitted,
            "rare_binary_leverage": rare_out,
            "rare_binary_leverage_omitted": rare_omitted,
            "constant_columns_preview": const_out,
            "constant_columns_count": len(const_all),
            "constant_columns_omitted": const_omitted,
            "datetime_columns_preview": date_out,
            "datetime_columns_count": len(date_all),
            "datetime_columns_omitted": date_omitted,
            "integer_coded_preview": int_coded_out,
            "integer_coded_omitted": int_coded_omitted,
        },
        "weak_hints": {
            "event_flag_columns": partners_out,
            "event_flag_columns_omitted": partners_omitted,
            "description": (
                "complete 0/1 or count columns whose value exactly explains "
                "other columns' blanks (often events or exposures; only you "
                "can say whether any is an outcome)"
            ),
        },
        "preprocessing_yaml_chars": len(
            yaml_str := plan_to_preprocessing_yaml(plan, include_header=True)
        ),
        "cost": copy.deepcopy(plan.cost),
        "how_to_use": _HOW_TO_USE,
        "next_tools": ["edit_preprocessing_plan", "create_config", "probe_columns"],
    }

    if is_full:
        payload["preprocessing_yaml"] = yaml_str
        payload["entries"] = {
            col: {
                "column": entry.column,
                "role": entry.role,
                "impute": copy.deepcopy(entry.impute),
                "indicator": entry.indicator,
                "encode": entry.encode,
                "decided_by": entry.decided_by,
                "status": entry.status,
                "reason": entry.reason,
                "question_id": entry.question_id,
            }
            for col, entry in plan.entries.items()
        }

    return payload


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
            f"(features: {counts.get('features', 0)}, "
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

    questions = payload.get("questions", [])
    lines.extend(["", "## Open Questions", ""])
    if questions:
        for q in questions:
            stakes = q.get("stakes")
            if isinstance(stakes, dict) and stakes.get("neighbors_kept") is not None:
                stakes_str = (
                    f"neighbors_kept={stakes['neighbors_kept']:.2f}, "
                    f"feature_delta={stakes.get('feature_count_delta', 0):+d}"
                )
            else:
                stakes_str = "stakes=n/a"
            lines.append(
                f"### `{q['question_id']}` [{q['topic']}] "
                f"({q['rows_affected']} rows; {stakes_str})"
            )
            lines.append(q["text"])
            for opt in q.get("options", []):
                default_tag = (
                    " *(default)*"
                    if opt["option_id"] == q.get("default_option_id")
                    else ""
                )
                lines.append(
                    f"- `{opt['option_id']}` — **{opt['label']}**{default_tag}: {opt['effect']}"
                )
            lines.append("")
        if payload.get("questions_omitted", 0):
            lines.append(
                f"*... +{payload['questions_omitted']} additional open question(s) omitted.*"
            )
    else:
        lines.append("- None. All entries are resolved.")

    ev = payload.get("evidence", {})
    lines.extend(["", "## Evidence Headline", ""])
    lines.append(
        f"- Shared missing groups: {len(ev.get('missing_groups', []))} "
        f"(+{ev.get('missing_groups_omitted', 0)} omitted)"
    )
    lines.append(
        f"- Conditional mask dependencies: {len(ev.get('mask_dependencies', []))} "
        f"(+{ev.get('mask_dependencies_omitted', 0)} omitted)"
    )
    lines.append(
        f"- Exact relations (duplicate/sum_of): {len(ev.get('relations', []))} "
        f"(+{ev.get('relations_omitted', 0)} omitted)"
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
) -> dict[str, Any]:
    """Build the structured payload for ``edit_preprocessing_plan``."""
    effective_changes = [
        ch
        for ch in changes
        if ch.get("field") in ("role", "impute", "indicator", "encode")
    ]
    changes_out, changes_omitted = _cap_list(
        effective_changes, max(1, int(change_limit))
    )

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
        if key not in summary_groups:
            summary_groups[key] = {
                "field": fld,
                "was": copy.deepcopy(was_val),
                "now": copy.deepcopy(now_val),
                "count": 1,
            }
        else:
            summary_groups[key]["count"] += 1
    summary_all = list(summary_groups.values())
    summary_out, summary_omitted = _cap_list(summary_all, max(1, int(change_limit)))

    open_questions = [
        _format_question_item(q, col_preview_limit=5)
        for q in plan.questions.values()
        if q.status == "open"
    ]
    open_out, open_omitted = _cap_list(open_questions, max(1, int(question_limit)))
    legend_all = build_marker_legend(plan, preview_limit=5)
    legend_out, legend_omitted = _cap_list(legend_all, 12)

    effective_changed_cols = {ch["column"] for ch in effective_changes}
    provenance_changed_cols = {
        ch["column"] for ch in changes if ch.get("field") in ("status", "decided_by")
    }
    counts = summarize_plan_counts(plan)
    counts["confirmed_without_change"] = len(
        provenance_changed_cols - effective_changed_cols
    )

    yaml_str = plan_to_preprocessing_yaml(plan, include_header=True)
    next_tools = (
        ["edit_preprocessing_plan", "create_config"]
        if open_questions
        else ["create_config", "validate_preprocessing_config"]
    )
    return {
        "status": "ok",
        "plan_id": plan.plan_id,
        "dataset_id": plan.dataset_id,
        "revision": int(plan.revision),
        "plan_fingerprint": plan_fingerprint(plan),
        "changes": changes_out,
        "changes_omitted": changes_omitted,
        "change_summary": summary_out,
        "change_summary_omitted": summary_omitted,
        "counts": counts,
        "processed_space": copy.deepcopy(processed_space),
        "marker_legend": legend_out,
        "marker_legend_omitted": legend_omitted,
        "validation": {
            "valid": bool(validation.get("valid", False)),
            "error": validation.get("error"),
        },
        "open_questions_preview": open_out,
        "open_questions_omitted": open_omitted,
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
            f"- Counts: {counts.get('features', 0)} features, "
            f"{counts.get('dropped', 0)} dropped/held-out, "
            f"{counts.get('open_questions', 0)} open questions"
        ),
        f"- Preprocessing YAML: {payload.get('preprocessing_yaml_chars', 0)} chars",
    ]
    if val.get("error"):
        lines.append(f"- Validation error: `{val['error']}`")

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

    pre_dict = plan_to_preprocessing_dict(plan)
    try:
        temp_cfg = load_config(
            {
                "run": {"name": "calibrate", "data": data_path or "in_memory"},
                "preprocessing": pre_dict,
            }
        )
        df_processed, _ = preprocess_dataframe(df, temp_cfg)
    except (ValueError, TypeError, KeyError) as exc:
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
