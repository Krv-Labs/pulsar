from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd

from pulsar.mcp.interpreter import (
    ClusterProfile,
    TopologicalDossier,
    build_dossier,
)
from pulsar.mcp.report import dossier_to_html


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
                    "column": "sex",
                    "value": "female",
                    "count": 12,
                    "concentration": 100.0,
                    "in_cluster_prevalence": 100.0,
                    "global_recall": 80.0,
                    "cluster_size": 12,
                    "global_count": 15,
                },
                {
                    "column": "species",
                    "value": "<img src=x onerror=alert(1)>",
                    "count": 8,
                    "concentration": 100.0,
                    "in_cluster_prevalence": 100.0,
                    "global_recall": 88.0,
                    "cluster_size": 8,
                    "global_count": 9,
                },
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
            categorical_features=[
                {
                    "column": "sex",
                    "value": "male",
                    "count": 7,
                    "concentration": 87.5,
                    "in_cluster_prevalence": 87.5,
                    "global_recall": 70.0,
                    "cluster_size": 8,
                    "global_count": 10,
                }
            ],
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
    weighted_adjacency = np.array(
        [
            [0.0, 0.9, 0.6, 0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.88, 0.0, 0.0],
            [0.0, 0.0, 0.88, 0.0, 0.55, 0.0],
            [0.0, 0.0, 0.0, 0.55, 0.0, 0.87],
            [0.0, 0.0, 0.0, 0.0, 0.87, 0.0],
        ],
        dtype=np.float64,
    )
    return SimpleNamespace(
        cosmic_graph=graph,
        weighted_adjacency=weighted_adjacency,
        stability_result=None,
        resolved_construction_threshold=0.5,
    )


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
    html = dossier_to_html(
        _sample_dossier(),
        model=_sample_model(),
        config_yaml="run:\n  name: demo\noutput:\n  n_reps: 1\n",
    )

    assert "class='report-shell'" in html
    assert "Cohort signal matrix" in html
    assert "Numeric means" in html
    assert "Dominant categories" in html
    assert "z +" in html or "z -" in html
    assert "Quick links" in html
    assert "C00 —" in html
    assert "Topological skeleton projection" in html
    assert "Threshold transition map" in html
    assert "Auto threshold, simply:" in html
    assert "stable graph cut" in html
    assert "top component sizes" in html
    assert "threshold-breakpoint-card--exploratory" in html
    assert "not final cohort cuts" in html
    assert "threshold-split-dot" in html
    assert "data-cluster-id='0'" in html
    assert "data-base-radius" in html
    assert "data-base-fill" in html
    assert "const resetGraphNodes" in html
    assert "fonts.googleapis.com" in html
    assert "Run parameters" in html
    assert "Copy YAML" in html
    assert "paramsYaml" in html
    assert "name: demo" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "no match" in html
    assert "<th>sex</th>" in html
    assert "female" in html
    assert "male" in html


def test_build_dossier_reports_prevalence_and_global_recall_for_categories():
    data = pd.DataFrame(
        {
            "x": [0.1, 0.2, 2.1, 2.2],
            "sex": ["female", "female", "female", "male"],
        }
    )
    clusters = pd.Series([0, 0, 1, 1], name="cluster")
    model = SimpleNamespace(cosmic_graph=nx.Graph([(0, 1), (2, 3)]))

    dossier = build_dossier(model, data, clusters)
    cluster_zero = next(
        profile for profile in dossier.clusters if profile.cluster_id == 0
    )
    sex_feature = next(
        feature
        for feature in cluster_zero.categorical_features
        if feature["value"] == "female"
    )

    assert sex_feature["count"] == 2
    assert sex_feature["cluster_size"] == 2
    assert sex_feature["global_count"] == 3
    assert sex_feature["in_cluster_prevalence"] == 100.0
    assert round(sex_feature["global_recall"], 1) == 66.7
    assert sex_feature["concentration"] == sex_feature["in_cluster_prevalence"]
