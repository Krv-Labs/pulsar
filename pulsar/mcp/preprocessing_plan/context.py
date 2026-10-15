"""Shared mutable state passed to default-entry and question builders during draft_plan."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import pandas as pd

from pulsar.analysis.preprocessing_evidence import (
    ColumnEvidence,
    PreprocessingEvidence,
)
from pulsar.mcp.preprocessing_plan.model import PlanEntry


@dataclass
class DraftContext:
    """Shared context populated by ``defaults`` and consumed by question builders."""

    df: pd.DataFrame
    evidence: PreprocessingEvidence
    ev_by_col: dict[str, ColumnEvidence]
    missing_groups: dict[str, list[str]]
    target_list: list[str]
    target_set: set[str]
    entity_col: str
    time_col: str
    n_rows: int
    cat_threshold: float
    entries: dict[str, PlanEntry] = field(default_factory=dict)
    date_cols: list[str] = field(default_factory=list)
    mixed_value_cols: list[str] = field(default_factory=list)
    conditional_by_partner: dict[str, list[str]] = field(default_factory=dict)
    rule7b_cols: list[str] = field(default_factory=list)
    grouped_7b: dict[str, list[str]] = field(default_factory=dict)
    blank_group_cols: set[str] = field(default_factory=set)

    @classmethod
    def from_inputs(
        cls,
        df: pd.DataFrame,
        evidence: PreprocessingEvidence,
        *,
        target_list: list[str],
        entity_col: str,
        time_col: str,
    ) -> DraftContext:
        """Initialize a :class:`DraftContext` from measured evidence and role arguments."""
        ev_by_col = {c.name: c for c in evidence.columns}
        missing_groups = {
            mg.group_id: list(mg.columns) for mg in evidence.missing_groups
        }
        n_rows = evidence.n_rows
        return cls(
            df=df,
            evidence=evidence,
            ev_by_col=ev_by_col,
            missing_groups=missing_groups,
            target_list=target_list,
            target_set=set(target_list),
            entity_col=entity_col,
            time_col=time_col,
            n_rows=n_rows,
            cat_threshold=math.sqrt(max(1, n_rows)),
        )
