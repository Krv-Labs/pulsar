from types import SimpleNamespace

import networkx as nx
import pandas as pd

from pulsar.mcp.interpreter import ClusterProfile, TopologicalDossier, dossier_to_html


def _sample_dossier() -> TopologicalDossier:
    clusters = [
        ClusterProfile(
            cluster_id=0,
            size=12,
            size_pct=60.0,
            semantic_name='<script>alert("x")</script>',
            numeric_features=[
                {
                    "column": "flipper_length_mm",
                    "mean": 210.123,
                    "global_mean": 198.321,
                    "z_score": 2.4,
                    "homogeneity": 0.45,
                    "importance": 4.1,
                    "sparkline": "▁▂▃▆█",
                },
                {
                    "column": "body_mass_g",
                    "mean": 5100.0,
                    "global_mean": 4300.0,
                    "z_score": 1.8,
                    "homogeneity": 0.7,
                    "importance": 2.1,
                    "sparkline": "▂▃▄▆█",
                },
            ],
            categorical_features=[
                {
                    "column": "species",
                    "value": "<img src=x onerror=alert(1)>",
                    "count": 8,
                    "concentration": 88.0,
                }
            ],
            central_rows=[
                {"unsafe_value": "<b>bold</b>", 'quoted"field': "x & y"},
            ],
        ),
        ClusterProfile(
            cluster_id=1,
            size=8,
            size_pct=40.0,
            semantic_name="Baseline cohort",
            numeric_features=[
                {
                    "column": "bill_depth_mm",
                    "mean": 18.2,
                    "global_mean": 17.1,
                    "z_score": -1.3,
                    "homogeneity": 0.9,
                    "importance": 1.2,
                    "sparkline": "█▇▅▃▁",
                }
            ],
            categorical_features=[],
            central_rows=[{"stable": "row"}],
        ),
    ]
    return TopologicalDossier(
        n_total=20,
        n_clusters=2,
        clusters=clusters,
        global_stats={
            "columns": ["flipper_length_mm", "body_mass_g", "bill_depth_mm", "species"],
            "graph_metrics": {"n_nodes": 3, "n_edges": 2, "component_count": 1},
        },
        cluster_labels=pd.Series([0, 0, 1], index=[0, 1, 2]),
    )


def _sample_model():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=0.7)
    graph.add_edge(1, 2, weight=0.5)
    return SimpleNamespace(cosmic_graph=graph)


def test_dossier_to_html_escapes_untrusted_content():
    html = dossier_to_html(_sample_dossier(), model=None)

    assert '<script>alert("x")</script>' not in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "<b>bold</b>" not in html
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "&lt;b&gt;bold&lt;/b&gt;" in html
    assert "quoted&quot;field" in html
    assert "x &amp; y" in html


def test_dossier_to_html_renders_research_report_shell_and_graph_state():
    html = dossier_to_html(_sample_dossier(), model=_sample_model())

    assert "class='report-shell'" in html
    assert "Population structure, reduced to the essentials" in html
    assert "Topological skeleton projection" in html
    assert "class='cluster-card'" in html
    assert "data-base-radius" in html
    assert "data-base-fill" in html
    assert "const resetGraphNodes" in html
    assert "Feature drift matrix" in html
