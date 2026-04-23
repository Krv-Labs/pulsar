"""Top-level HTML renderer that composes the dossier sections."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pulsar.mcp.interpreter import TopologicalDossier
from pulsar.mcp.report.fonts import _render_font_links
from pulsar.mcp.report.formatting import _cluster_palette
from pulsar.mcp.report.script import _render_report_script
from pulsar.mcp.report.sections import (
    _render_cluster_section,
    _render_config_appendix,
    _render_heatmap_section,
    _render_nav,
    _render_summary_section,
    _render_topology_section,
)
from pulsar.mcp.report.styles import _render_report_styles

if TYPE_CHECKING:
    from pulsar.pipeline import ThemaRS


def dossier_to_html(
    dossier: TopologicalDossier,
    model: ThemaRS | None = None,
    config_yaml: str | None = None,
) -> str:
    """Render the dossier as a polished standalone HTML research report."""
    graph_metrics = dossier.global_stats.get("graph_metrics", {})
    palette = _cluster_palette(dossier.clusters)
    sections = [
        _render_summary_section(dossier, graph_metrics),
        _render_topology_section(dossier, model, palette),
        _render_heatmap_section(dossier),
    ]
    sections.extend(
        _render_cluster_section(profile, palette[profile.cluster_id])
        for profile in dossier.clusters
    )
    sections.append(_render_config_appendix(config_yaml))
    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "<title>Pulsar Topological Dossier</title>",
            _render_font_links(),
            "<style>",
            _render_report_styles(),
            "</style>",
            "</head>",
            "<body>",
            "<div class='report-shell'>",
            "<main class='report-main'>",
            "<div class='report-stack'>",
            *[section for section in sections if section],
            "</div>",
            "</main>",
            _render_nav(dossier),
            "</div>",
            "<script>",
            _render_report_script(),
            "</script>",
            "</body>",
            "</html>",
        ]
    )
