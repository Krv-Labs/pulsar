"""Section renderers for the topological dossier HTML report."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd
import networkx as nx

from pulsar.mcp.interpreter import ClusterProfile, TopologicalDossier
from pulsar.mcp.report.formatting import (
    _cluster_trait_summary,
    _escape_html,
    _format_heatmap_value,
    _format_value,
    _signal_tone,
)

if TYPE_CHECKING:
    from pulsar.pipeline import ThemaRS


_MAX_SIGNAL_MATRIX_NUMERIC = 10
_MAX_SIGNAL_MATRIX_CATEGORICAL = 5


def _build_key_findings(
    dossier: TopologicalDossier, gm: Dict[str, Any]
) -> list[tuple[str, str]]:
    """Generate concise top-level findings for the hero panel."""
    findings: list[tuple[str, str]] = []
    if dossier.clusters:
        dominant = max(dossier.clusters, key=lambda profile: profile.size)
        findings.append(
            (
                "Largest cohort",
                f"{dominant.semantic_name} accounts for {dominant.size_pct:.1f}% of the population.",
            )
        )

        strongest_feature = None
        strongest_cluster = None
        strongest_score = -1.0
        for profile in dossier.clusters:
            for feature in profile.numeric_features:
                score = abs(float(feature["z_score"]))
                if score > strongest_score:
                    strongest_feature = feature
                    strongest_cluster = profile
                    strongest_score = score
        if strongest_feature is not None and strongest_cluster is not None:
            findings.append(
                (
                    "Sharpest shift",
                    f"{strongest_cluster.semantic_name} shows the strongest deviation on "
                    f"{strongest_feature['column']} ({strongest_feature['z_score']:+.2f} z).",
                )
            )

    if gm:
        findings.append(
            (
                "Topology footprint",
                f"{gm.get('component_count', 'N/A')} stable components across "
                f"{gm.get('n_nodes', 'N/A')} graph nodes and {gm.get('n_edges', 'N/A')} edges.",
            )
        )
    return findings[:3]


def _render_nav(dossier: TopologicalDossier) -> str:
    """Render the persistent navigation rail."""
    parts = [
        "<aside class='report-nav'>",
        "<div class='report-nav__brand'>",
        "<span class='report-nav__eyebrow brand-mark'>Quick links</span>",
        "<h1>Pulsar topological report</h1>",
        f"<p>{dossier.n_total:,} observations · {dossier.n_clusters} cohorts</p>",
        "</div>",
        "<nav aria-label='Section navigation'>",
        "<div class='nav-group'>",
        "<div class='nav-label report-nav__eyebrow'>Sections</div>",
        "<a class='nav-link is-active' href='#summary'>Summary</a>",
        "<a class='nav-link' href='#topology'>Topology figure</a>",
        "<a class='nav-link' href='#heatmap'>Feature drift</a>",
        "<a class='nav-link' href='#config'>Run parameters</a>",
        "</div>",
        "<div class='nav-group'>",
        "<div class='nav-label report-nav__eyebrow'>Clusters</div>",
    ]
    for profile in dossier.clusters:
        parts.append(
            f"<a class='nav-link' href='#cluster-{profile.cluster_id}' data-cluster-id='{profile.cluster_id}'>"
            f"C{profile.cluster_id:02d} — {_escape_html(profile.semantic_name)}</a>"
        )
    parts.extend(
        [
            "</div>",
            "</nav>",
            "</aside>",
        ]
    )
    return "".join(parts)


def _render_summary_section(dossier: TopologicalDossier, gm: Dict[str, Any]) -> str:
    """Render the executive summary section."""
    findings = _build_key_findings(dossier, gm)
    column_count = len(dossier.global_stats.get("columns", []))
    metrics = [
        ("Observations", f"{dossier.n_total:,}"),
        ("Cohorts", f"{dossier.n_clusters}"),
        ("Graph nodes", _format_value(gm.get("n_nodes", "N/A"), digits=0)),
        ("Graph edges", _format_value(gm.get("n_edges", "N/A"), digits=0)),
        ("Columns profiled", f"{column_count}"),
    ]
    parts = [
        "<section id='summary' class='report-section'>",
        "<div class='hero-grid'>",
        "<div class='hero-copy'>",
        "<span class='section-eyebrow'>Summary</span>",
        "<h2>Latent structure report</h2>",
        (
            f"<p>{dossier.n_total:,} observations resolve into <strong>{dossier.n_clusters}</strong> cohorts. "
            f"The supporting topology contains <strong>{_escape_html(_format_value(gm.get('n_nodes', 'N/A'), digits=0))}</strong> nodes, "
            f"<strong>{_escape_html(_format_value(gm.get('n_edges', 'N/A'), digits=0))}</strong> edges, and "
            f"<strong>{_escape_html(_format_value(gm.get('component_count', 'N/A'), digits=0))}</strong> stable components.</p>"
        ),
        "</div>",
        "</div>",
        "<div class='hero-metrics'>",
    ]
    for label, value in metrics:
        parts.append(
            "<div class='metric-card'>"
            f"<div class='metric-value'>{_escape_html(value)}</div>"
            f"<div class='metric-label'>{_escape_html(label)}</div>"
            "</div>"
        )
    parts.append("</div>")
    if findings:
        parts.append("<div class='findings-grid'><ul class='hero-list'>")
        for title, body in findings:
            parts.append(
                f"<li><strong>{_escape_html(title)}.</strong> {_escape_html(body)}</li>"
            )
        parts.append("</ul></div>")
    parts.append("</section>")
    return "".join(parts)


def _generate_cosmic_graph_svg(
    model: ThemaRS,
    width: int = 920,
    height: int = 500,
    cluster_labels: pd.Series | None = None,
    palette: dict[int, str] | None = None,
) -> str:
    """Generate a sampled SVG topology figure with persistent node state for interaction."""
    G = model.cosmic_graph
    if G.number_of_nodes() == 0:
        return ""

    budget = 1000
    if G.number_of_nodes() > budget:
        sampled = set()
        if cluster_labels is not None:
            for cid in sorted(cluster_labels.unique()):
                c_mask = cluster_labels == cid
                c_nodes = cluster_labels.index[c_mask].tolist()
                c_nodes = [node for node in c_nodes if G.has_node(node)]
                if not c_nodes:
                    continue
                n_sample = max(10, int(len(c_nodes) * 0.05))
                sampled.update(
                    sorted(c_nodes, key=lambda node: G.degree(node), reverse=True)[
                        :n_sample
                    ]
                )

        remaining = budget - len(sampled)
        if remaining > 0:
            global_sorted = sorted(
                G.nodes(), key=lambda node: G.degree(node), reverse=True
            )
            for node in global_sorted:
                if node not in sampled:
                    sampled.add(node)
                    remaining -= 1
                if remaining <= 0:
                    break
        G_sub = G.subgraph(sampled)
    else:
        G_sub = G

    pos = nx.spring_layout(
        G_sub, seed=42, k=1.5 / np.sqrt(max(1, G_sub.number_of_nodes()))
    )
    x_vals = [point[0] for point in pos.values()]
    y_vals = [point[1] for point in pos.values()]
    if not x_vals:
        return ""

    xmin, xmax, ymin, ymax = min(x_vals), max(x_vals), min(y_vals), max(y_vals)
    padding = 40

    def scale(
        value: float, value_min: float, value_max: float, target_max: int
    ) -> float:
        return padding + (value - value_min) / (value_max - value_min + 1e-9) * (
            target_max - 2 * padding
        )

    svg = [
        f"<svg id='cosmicGraph' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg' role='img' aria-label='Topological skeleton projection'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff' rx='24' />",
    ]

    opacity_bands: dict[float, list[str]] = {}
    for u, v, data in G_sub.edges(data=True):
        x1 = scale(pos[u][0], xmin, xmax, width)
        y1 = scale(pos[u][1], ymin, ymax, height)
        x2 = scale(pos[v][0], xmin, xmax, width)
        y2 = scale(pos[v][1], ymin, ymax, height)
        band = round(max(0.10, 0.08 + data.get("weight", 0.1) * 0.22), 2)
        opacity_bands.setdefault(band, []).append(
            f"M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f}"
        )

    for opacity, segments in sorted(opacity_bands.items()):
        svg.append(
            f"<path d='{' '.join(segments)}' stroke='#a8afb8' stroke-width='0.65' "
            f"stroke-opacity='{opacity}' fill='none' />"
        )

    for node in G_sub.nodes():
        x = scale(pos[node][0], xmin, xmax, width)
        y = scale(pos[node][1], ymin, ymax, height)
        degree = G.degree(node)
        radius = 1.8 + (degree / max(1, G.number_of_nodes()) * 7.5)
        cluster_id = None
        if cluster_labels is not None and node in cluster_labels.index:
            cluster_id = int(cluster_labels[node])
        base_fill = palette.get(cluster_id, "#2563eb") if palette else "#2563eb"
        svg.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{radius:.2f}' fill='{base_fill}' fill-opacity='0.78' "
            f"class='graph-node' data-cluster='{'' if cluster_id is None else cluster_id}' "
            f"data-base-radius='{radius:.2f}' data-base-fill='{base_fill}' data-base-opacity='0.78'>"
            f"<title>Node {node} (degree {degree}, cluster {'' if cluster_id is None else cluster_id})</title>"
            "</circle>"
        )

    svg.append("</svg>")
    return "".join(svg)


def _render_topology_section(
    dossier: TopologicalDossier,
    model: ThemaRS | None,
    palette: dict[int, str],
) -> str:
    """Render the topology figure and legend."""
    parts = [
        "<section id='topology' class='report-section'>",
        "<div class='section-header'>",
        "<div>",
        "<span class='section-eyebrow'>Figure 1</span>",
        "<h2>Topological skeleton projection</h2>",
        "</div>",
        "<p class='section-subtitle'>Sampled from the cosmic graph for responsiveness. Cohort colors stay consistent across the full report.</p>",
        "</div>",
        "<div class='figure-frame'>",
        "<div class='figure-toolbar'>",
        "<div>",
        "<div class='figure-title'>Global manifold view</div>",
        "<p class='figure-subtitle'>Edges are intentionally quiet. The emphasis stays on node structure and cohort support.</p>",
        "</div>",
        "<div class='legend'>",
    ]
    for profile in dossier.clusters:
        parts.append(
            f"<span class='legend-chip' style='--cluster-accent:{palette[profile.cluster_id]}'>"
            f"C{profile.cluster_id:02d} · {_escape_html(profile.semantic_name)}</span>"
        )
    parts.extend(["</div>", "</div>", "<div class='graph-wrapper'>"])
    if model is not None:
        parts.append(
            _generate_cosmic_graph_svg(
                model,
                cluster_labels=dossier.cluster_labels,
                palette=palette,
            )
        )
        parts.append(
            "<p class='figure-caption'><strong>Figure note.</strong> Nodes are sampled manifold landmarks. Edges encode topological consensus. Hover a cohort row below to isolate its sampled support.</p>"
        )
    else:
        parts.append(
            "<div class='empty-state'>Graph view unavailable for this report.</div>"
        )
    parts.extend(["</div>", "</div>", "</section>"])
    return "".join(parts)


def _render_heatmap_section(dossier: TopologicalDossier) -> str:
    """Render a bounded cohort signal matrix with numeric and categorical groups."""
    numeric_scores: dict[str, float] = {}
    categorical_scores: dict[str, float] = {}

    for profile in dossier.clusters:
        for feature in profile.numeric_features:
            numeric_scores[feature["column"]] = numeric_scores.get(
                feature["column"], 0.0
            ) + abs(float(feature.get("z_score", 0.0)))
        for feature in profile.categorical_features:
            column_name = str(feature["column"])
            prevalence = float(
                feature.get("in_cluster_prevalence", feature.get("concentration", 0.0))
            )
            global_recall = float(feature.get("global_recall", 0.0))
            score = prevalence + (0.25 * global_recall)
            categorical_scores[column_name] = (
                categorical_scores.get(column_name, 0.0) + score
            )

    numeric_features = [
        name
        for name, _score in sorted(
            numeric_scores.items(), key=lambda item: (-item[1], item[0])
        )[:_MAX_SIGNAL_MATRIX_NUMERIC]
    ]
    categorical_columns = [
        name
        for name, _score in sorted(
            categorical_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:_MAX_SIGNAL_MATRIX_CATEGORICAL]
    ]

    if not numeric_features and not categorical_columns:
        return ""

    parts = [
        "<section id='heatmap' class='report-section'>",
        "<div class='section-header'>",
        "<div>",
        "<span class='section-eyebrow'>Figure 2</span>",
        "<h2>Cohort signal matrix</h2>",
        "</div>",
        f"<p class='section-subtitle'>Bounded for readability: up to {_MAX_SIGNAL_MATRIX_NUMERIC} numeric features and {_MAX_SIGNAL_MATRIX_CATEGORICAL} categorical columns. Numeric values are cohort means, and the smaller line below each numeric value is the z-shift.</p>",
        "</div>",
        "<div class='heatmap-shell'>",
        "<table class='heatmap-table'><thead>",
        "<tr>",
        "<th rowspan='2'>Cohort</th>",
    ]
    if numeric_features:
        parts.append(f"<th colspan='{len(numeric_features)}'>Numeric means</th>")
    if categorical_columns:
        parts.append(
            f"<th colspan='{len(categorical_columns)}'>Dominant categories</th>"
        )
    parts.append("</tr><tr>")
    for feature_name in numeric_features:
        parts.append(f"<th>{_escape_html(feature_name)}</th>")
    for column_name in categorical_columns:
        parts.append(f"<th>{_escape_html(column_name)}</th>")
    parts.append("</tr></thead><tbody>")
    for profile in dossier.clusters:
        parts.append(
            f"<tr data-cluster-id='{profile.cluster_id}'>"
            "<td>"
            f"<div><strong>C{profile.cluster_id:02d} — {_escape_html(profile.semantic_name)}</strong></div>"
            f"<div class='cluster-card__meta'>n = {profile.size:,} · {profile.size_pct:.1f}%</div>"
            "</td>"
        )
        numeric_map = {
            feature["column"]: feature for feature in profile.numeric_features
        }
        for feature_name in numeric_features:
            feature = numeric_map.get(feature_name)
            if feature is None:
                parts.append(
                    "<td class='heat-cell'><span class='heat-subvalue'>—</span></td>"
                )
                continue
            z_score = float(feature.get("z_score", 0.0))
            mean_value = feature.get("mean")
            alpha = min(0.38, abs(z_score) * 0.14)
            tone_class = (
                "heat-positive"
                if z_score > 0.5
                else "heat-negative"
                if z_score < -0.5
                else ""
            )
            style = f" style='--heat-alpha:{alpha:.2f}'" if tone_class else ""
            parts.append(
                f"<td class='heat-cell {tone_class}'{style}>"
                f"<span class='heat-value'>{_escape_html(_format_heatmap_value(mean_value))}</span>"
                f"<span class='heat-subvalue'>z {z_score:+.2f}</span>"
                "</td>"
            )
        categorical_map: dict[str, list[dict[str, Any]]] = {}
        for feature in profile.categorical_features:
            categorical_map.setdefault(str(feature["column"]), []).append(feature)
        for column_name in categorical_columns:
            column_features = categorical_map.get(column_name, [])
            if not column_features:
                parts.append(
                    "<td class='heat-cell'>"
                    "<span class='heat-value'>—</span>"
                    "<span class='heat-subvalue'>no match</span>"
                    "</td>"
                )
                continue
            feature = max(
                column_features,
                key=lambda item: (
                    float(
                        item.get(
                            "in_cluster_prevalence", item.get("concentration", 0.0)
                        )
                    ),
                    int(item.get("count", 0)),
                ),
            )
            prevalence = float(
                feature.get("in_cluster_prevalence", feature.get("concentration", 0.0))
            )
            count = int(feature.get("count", 0))
            if count <= 0 or prevalence <= 0:
                parts.append(
                    "<td class='heat-cell'>"
                    "<span class='heat-value'>—</span>"
                    "<span class='heat-subvalue'>no match</span>"
                    "</td>"
                )
                continue
            alpha = min(0.38, prevalence * 0.0038)
            tone_class = "heat-positive" if prevalence >= 50.0 else ""
            style = f" style='--heat-alpha:{alpha:.2f}'" if prevalence > 0 else ""
            parts.append(
                f"<td class='heat-cell {tone_class}'{style}>"
                f"<span class='heat-value'>{_escape_html(feature['value'])}</span>"
                f"<span class='heat-subvalue'>{count}/{profile.size} · {prevalence:.1f}%</span>"
                "</td>"
            )
        parts.append("</tr>")
    parts.extend(["</tbody></table>", "</div>", "</section>"])
    return "".join(parts)


def _render_cluster_overview_section(
    dossier: TopologicalDossier, palette: dict[int, str]
) -> str:
    """Render the cohort overview cards."""
    parts = [
        "<section id='clusters' class='report-section'>",
        "<div class='section-header'>",
        "<div>",
        "<span class='section-eyebrow'>Cohort overview</span>",
        "<h2>Population structure, reduced to the essentials</h2>",
        "</div>",
        "<p class='section-subtitle'>Each row is a compact abstract: cohort identity, share, and the few traits most worth carrying into the detailed read.</p>",
        "</div>",
        "<div class='search-shell'>",
        "<input id='clusterSearch' class='cluster-search' type='search' placeholder='Search clusters, signals, or traits…' aria-label='Search cohorts'>",
        "<p class='overview-note'>The overview is deliberately spare. Click through only when a cohort looks meaningfully distinct.</p>",
        "</div>",
        "<div class='cluster-grid'>",
    ]
    for profile in dossier.clusters:
        trait_chips = []
        for feature in profile.numeric_features[:3]:
            trait_chips.append(
                f"<span class='trait-chip'>{_escape_html(feature['column'])}</span>"
            )
        parts.append(
            f"<a class='cluster-card' href='#cluster-{profile.cluster_id}' data-cluster-id='{profile.cluster_id}' "
            f"style='--cluster-accent:{palette[profile.cluster_id]}'>"
            "<div>"
            "<div class='cluster-card__topline'>"
            f"<span class='cluster-id'>Cohort {profile.cluster_id:02d}</span>"
            f"<span class='legend-chip' style='--cluster-accent:{palette[profile.cluster_id]}'>"
            f"{_escape_html(profile.semantic_name)}</span>"
            "</div>"
            f"<p class='cluster-card__summary'>{_escape_html(_cluster_trait_summary(profile))}</p>"
            f"<div class='cluster-card__traits'>{''.join(trait_chips)}</div>"
            "</div>"
            "<div>"
            f"<div class='cluster-share'>{profile.size_pct:.1f}%</div>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{profile.size_pct:.2f}%;'></div></div>"
            f"<p class='cluster-card__meta'>{profile.size:,} rows</p>"
            "</div>"
            "</a>"
        )
    parts.extend(["</div>", "</section>"])
    return "".join(parts)


def _render_numeric_panel(profile: ClusterProfile) -> str:
    """Render the numeric signal table for a cluster."""
    parts = [
        "<section class='panel'>",
        "<h3>Dominant numeric signals</h3>",
        "<p class='panel-note'>Top features are ranked by shift magnitude adjusted for within-cohort homogeneity.</p>",
        "<table class='data-table'><thead><tr><th>Feature</th><th>Distribution</th><th>Mean</th><th>Z-score</th><th>Signal</th></tr></thead><tbody>",
    ]
    for feature in profile.numeric_features:
        label, tone_class = _signal_tone(float(feature["z_score"]))
        parts.append(
            "<tr>"
            f"<td><strong>{_escape_html(feature['column'])}</strong></td>"
            f"<td class='sparkline'>{_escape_html(feature.get('sparkline', ''))}</td>"
            f"<td><code>{float(feature['mean']):.3f}</code></td>"
            f"<td><code>{float(feature['z_score']):+.2f}</code></td>"
            f"<td><span class='signal-badge {tone_class}'>{label}</span></td>"
            "</tr>"
        )
    parts.extend(["</tbody></table>", "</section>"])
    return "".join(parts)


def _render_categorical_panel(profile: ClusterProfile, accent_color: str) -> str:
    """Render categorical concentrations for a cluster."""
    if not profile.categorical_features:
        return "<section class='panel'><h3>Categorical concentrations</h3><div class='empty-state'>No categorical enrichment was captured for this cohort.</div></section>"

    parts = [
        "<section class='panel'>",
        "<h3>Categorical concentrations</h3>",
        "<p class='panel-note'>Prevalence reports what share of this cohort has the value. Global recall reports how much of the overall value population this cohort captures.</p>",
        "<table class='data-table'><thead><tr><th>Feature</th><th>Value</th><th>Prevalence</th><th>Global recall</th></tr></thead><tbody>",
    ]
    for feature in profile.categorical_features:
        prevalence = float(
            feature.get("in_cluster_prevalence", feature.get("concentration", 0.0))
        )
        global_recall = float(feature.get("global_recall", 0.0))
        count = int(feature.get("count", 0))
        cluster_size = int(feature.get("cluster_size", profile.size))
        global_count = int(feature.get("global_count", count))
        parts.append(
            "<tr>"
            f"<td>{_escape_html(feature['column'])}</td>"
            f"<td><code>{_escape_html(feature['value'])}</code></td>"
            "<td>"
            f"{prevalence:.1f}% <span class='cluster-card__meta'>({count}/{cluster_size})</span>"
            "<div class='bar-track' style='margin-top:0.45rem;'>"
            f"<div class='bar-fill' style='width:{prevalence:.2f}%; --cluster-accent:{accent_color};'></div>"
            "</div>"
            "</td>"
            f"<td>{global_recall:.1f}% <span class='cluster-card__meta'>({count}/{global_count})</span></td>"
            "</tr>"
        )
    parts.extend(["</tbody></table>", "</section>"])
    return "".join(parts)


def _render_instances_panel(profile: ClusterProfile) -> str:
    """Render representative instances as compact cards."""
    parts = [
        "<section class='panel'>",
        "<h3>Representative instances</h3>",
        "<p class='panel-note'>Rows chosen from the cohort core so the values stay close to the local manifold center.</p>",
    ]
    if not profile.central_rows:
        parts.append(
            "<div class='empty-state'>No representative rows available for this cohort.</div></section>"
        )
        return "".join(parts)

    parts.append("<div class='instances-grid'>")
    for index, row in enumerate(profile.central_rows, start=1):
        details = []
        for key, value in row.items():
            details.append(
                f"<span class='detail-chip'><strong>{_escape_html(key)}</strong>: {_escape_html(_format_value(value))}</span>"
            )
        parts.append(
            "<article class='instance-card'>"
            f"<div class='instance-label'>Example {index:02d}</div>"
            f"<div class='instance-details'>{''.join(details)}</div>"
            "</article>"
        )
    parts.extend(["</div>", "</section>"])
    return "".join(parts)


def _render_cluster_section(profile: ClusterProfile, accent_color: str) -> str:
    """Render a detailed cluster section."""
    traits = "".join(
        f"<span class='trait-chip'>{_escape_html(feature['column'])}</span>"
        for feature in profile.numeric_features[:5]
    )
    return "".join(
        [
            f"<section id='cluster-{profile.cluster_id}' class='report-section cluster-section' "
            f"data-cluster-id='{profile.cluster_id}' style='--cluster-accent:{accent_color}'>",
            "<div class='cluster-section__heading'>",
            "<div class='cluster-section__topline'>",
            f"<span class='cluster-id'>C{profile.cluster_id:02d}</span>",
            f"<span class='legend-chip' style='--cluster-accent:{accent_color}'>n = {profile.size:,} · {profile.size_pct:.1f}%</span>",
            "</div>",
            f"<h2>C{profile.cluster_id:02d} — {_escape_html(profile.semantic_name)}</h2>",
            (
                f"<p class='cluster-section__summary'>This cohort contains <strong>{profile.size:,}</strong> observations "
                f"and represents <strong>{profile.size_pct:.1f}%</strong> of the analyzed population. "
                "Treat the numeric and categorical tables below as the evidence sheet for this slice of the manifold.</p>"
            ),
            f"<div class='cluster-section__traits'>{traits}</div>",
            "</div>",
            "<div class='cluster-detail-grid'>",
            _render_numeric_panel(profile),
            "<div class='cluster-secondary-stack'>",
            _render_categorical_panel(profile, accent_color),
            _render_instances_panel(profile),
            "</div>",
            "</div>",
            "</section>",
        ]
    )


def _render_config_appendix(config_yaml: str | None) -> str:
    """Render a copyable params appendix at the bottom of the report."""
    if not config_yaml:
        return ""

    return "".join(
        [
            "<section id='config' class='report-section config-appendix'>",
            "<div class='section-header'>",
            "<div>",
            "<span class='section-eyebrow'>Appendix</span>",
            "<h2>Run parameters</h2>",
            "</div>",
            "<p class='section-subtitle'>Normalized params.yaml used for the exported run.</p>",
            "</div>",
            "<div class='config-toolbar'>",
            "<p>Readonly, formatted for direct copy and reuse.</p>",
            "<button type='button' class='copy-button' data-copy-target='paramsYaml'>Copy YAML</button>",
            "</div>",
            f"<textarea id='paramsYaml' class='config-textarea' readonly spellcheck='false'>{_escape_html(config_yaml)}</textarea>",
            "</section>",
        ]
    )
