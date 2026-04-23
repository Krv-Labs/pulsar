"""Value formatting and styling helpers shared across report sections."""

from __future__ import annotations

from html import escape
from typing import Any, List

import numpy as np

from pulsar.mcp.interpreter import ClusterProfile


def _escape_html(value: Any) -> str:
    """Escape arbitrary content before inserting it into HTML."""
    if value is None:
        return ""
    return escape(str(value), quote=True)


def _format_value(value: Any, digits: int = 3) -> str:
    """Format scalar values consistently for report display."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NaN"
        return f"{float(value):.{digits}f}"
    return str(value)


def _format_heatmap_value(value: Any) -> str:
    """Format heatmap values with restrained precision."""
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isnan(value):
            return "NaN"
        abs_value = abs(value)
        if abs_value >= 100:
            return f"{value:,.0f}"
        if abs_value >= 10:
            return f"{value:,.1f}"
        return f"{value:,.2f}"
    return _escape_html(value)


def _signal_tone(z_score: float) -> tuple[str, str]:
    """Return semantic label and CSS class for a z-score."""
    if z_score > 1.0:
        return "Enriched", "signal-pos"
    if z_score < -1.0:
        return "Depleted", "signal-neg"
    return "Baseline", "signal-neu"


def _cluster_palette(clusters: List[ClusterProfile]) -> dict[int, str]:
    """Muted palette for cohort accents in figures and cards."""
    colors = [
        "#1a73e8",
        "#669df6",
        "#147d64",
        "#8ab4f8",
        "#f29900",
        "#5f6368",
        "#00acc1",
        "#7baaf7",
        "#c58a00",
        "#7c8b95",
    ]
    return {
        profile.cluster_id: colors[i % len(colors)]
        for i, profile in enumerate(clusters)
    }


def _cluster_trait_summary(profile: ClusterProfile) -> str:
    """Create a short textual summary for overview cards."""
    fragments = []
    for feature in profile.numeric_features[:3]:
        direction = (
            "high"
            if feature["z_score"] > 0.5
            else "low"
            if feature["z_score"] < -0.5
            else "stable"
        )
        fragments.append(f"{direction} {feature['column']}")
    return ", ".join(fragments) if fragments else "No dominant numeric signal captured."
