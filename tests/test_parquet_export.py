import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pulsar.config import load_config
from pulsar.pipeline import ThemaRS


@pytest.fixture
def minimal_config():
    """Minimal config for quick fitting."""
    return {
        "run": {"name": "test_export"},
        "preprocessing": {"drop_columns": [], "impute": {}},
        "sweep": {
            "pca": {
                "dimensions": {"values": [2]},
                "seed": {"values": [42]},
            },
            "ball_mapper": {
                "epsilon": {"values": [0.5, 1.0]},
            },
        },
        "cosmic_graph": {"construction_threshold": "0.1"},
        "output": {"n_reps": 1},
    }


@pytest.fixture
def simple_data():
    """Simple test dataset."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((30, 3)),
        columns=["x", "y", "z"],
    )


@pytest.fixture
def fitted_model(minimal_config, simple_data):
    """A ThemaRS model fit on `simple_data`, ready for export helper tests."""
    cfg = load_config(minimal_config)
    model = ThemaRS(cfg)
    model.fit(data=simple_data)
    return model


def test_export_before_fit_raises(minimal_config):
    cfg = load_config(minimal_config)
    model = ThemaRS(cfg)

    with pytest.raises(RuntimeError, match="Call fit.* first"):
        model.data

    with pytest.raises(RuntimeError, match="Call fit.* first"):
        model.clean_data()

    with pytest.raises(RuntimeError, match="Call fit.* first"):
        model.cosmic_edges()

    with pytest.raises(RuntimeError, match="Call fit.* first"):
        model.snapshot_edges()

    with pytest.raises(RuntimeError, match="Call fit.* first"):
        model.node_table(cluster_labels=np.zeros(30))

    with pytest.raises(RuntimeError, match="Call fit.* first"):
        model.group_table(cluster_labels=np.zeros(30))

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError, match="Call fit.* first"):
            model.export_dataset_bundle(
                tmpdir, "test-slug", cluster_labels=np.zeros(30)
            )

    with pytest.raises(RuntimeError, match="Call fit.* first"):
        model.clean_embedding_at_center()


def test_tabular_helpers_and_edges(fitted_model):
    # data
    df_raw = fitted_model.data
    assert isinstance(df_raw, pd.DataFrame)
    assert len(df_raw) == 30

    # cosmic_edges
    df_cosmic = fitted_model.cosmic_edges(threshold=0.0)
    assert list(df_cosmic.columns) == ["source_id", "target_id", "weight"]
    assert df_cosmic["source_id"].dtype == np.uint32
    assert df_cosmic["target_id"].dtype == np.uint32
    assert df_cosmic["weight"].dtype == np.float32
    # Check undirected contract: i < j
    assert (df_cosmic["source_id"] < df_cosmic["target_id"]).all()

    # snapshot_edges (using default threshold)
    df_snap = fitted_model.snapshot_edges()
    assert list(df_snap.columns) == ["source_id", "target_id", "weight"]
    assert len(df_snap) <= len(df_cosmic)


def test_node_table_projection_layout(fitted_model):
    cluster_labels = np.array([0] * 15 + [1] * 15)
    cluster_names = {0: "Zeroes", 1: "Ones"}

    df = fitted_model.node_table(
        cluster_labels=cluster_labels,
        cluster_names=cluster_names,
        layout="projection",
        extra_columns=["x", "z"],
    )
    assert list(df.columns[:8]) == [
        "node_id",
        "group_id",
        "val",
        "archetype",
        "ex",
        "ey",
        "ez",
        "is_live",
    ]
    assert df["node_id"].dtype == np.uint32
    assert df["group_id"].dtype == np.uint32
    assert df["val"].dtype == np.float32
    assert df["is_live"].dtype == bool
    assert (df["archetype"].iloc[:15] == "Zeroes").all()
    assert (df["archetype"].iloc[15:] == "Ones").all()
    # Check extra columns passthrough
    assert "x" in df.columns
    assert "z" in df.columns
    assert "y" not in df.columns


def test_node_table_zeros_layout(fitted_model):
    cluster_labels = np.array([0] * 15 + [1] * 15)
    df = fitted_model.node_table(cluster_labels=cluster_labels, layout="zeros")
    assert (df["ex"] == 0.0).all()
    assert (df["ey"] == 0.0).all()
    assert (df["ez"] == 0.0).all()


def test_node_table_spectral_layout(fitted_model):
    cluster_labels = np.array([0] * 15 + [1] * 15)
    df = fitted_model.node_table(cluster_labels=cluster_labels, layout="spectral")
    assert len(df) == 30


def test_group_table_schema(fitted_model):
    cluster_labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    df_groups = fitted_model.group_table(cluster_labels=cluster_labels)
    assert list(df_groups.columns) == ["group_id", "name", "desc", "member_ids"]
    assert len(df_groups) == 3
    assert df_groups["group_id"].dtype == np.uint32


@pytest.mark.parametrize(
    "row_idx, expected",
    [
        (
            0,
            {
                "group_id": 0,
                "name": "Zero",
                "desc": "The first ten",
                "member_ids": list(range(10)),
            },
        ),
        (
            2,
            {
                "group_id": 2,
                "name": "Cluster 2",  # no name configured for gid 2
                "desc": "The last ten",
                "member_ids": list(range(20, 30)),
            },
        ),
    ],
)
def test_group_table_row_contents(fitted_model, row_idx, expected):
    cluster_labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    cluster_names = {0: "Zero", 1: "One"}
    cluster_descriptions = {0: "The first ten", 1: "The middle ten", 2: "The last ten"}

    df_groups = fitted_model.group_table(
        cluster_labels=cluster_labels,
        cluster_names=cluster_names,
        cluster_descriptions=cluster_descriptions,
    )
    row = df_groups.iloc[row_idx]
    assert row["group_id"] == expected["group_id"]
    assert row["name"] == expected["name"]
    assert row["desc"] == expected["desc"]
    assert row["member_ids"] == expected["member_ids"]


@pytest.fixture
def exported_bundle(fitted_model):
    """Writes a full export bundle once; yields (tmpdir, slug, manifest)."""
    cluster_labels = np.array([0] * 15 + [1] * 15)
    with tempfile.TemporaryDirectory() as tmpdir:
        slug = "test-slug"
        manifest = fitted_model.export_dataset_bundle(
            output_dir=tmpdir,
            slug=slug,
            cluster_labels=cluster_labels,
            layout="projection",
        )
        yield tmpdir, slug, manifest


def test_export_dataset_bundle_manifest(exported_bundle):
    _, _, manifest = exported_bundle
    assert isinstance(manifest, dict)
    assert "pulsar_version" in manifest
    assert "config_hash" in manifest


def test_export_dataset_bundle_file_layout(exported_bundle):
    tmpdir, slug, _ = exported_bundle
    base_path = os.path.join(tmpdir, slug)
    assert os.path.exists(base_path)
    assert os.path.exists(os.path.join(base_path, "tabular", "raw.parquet"))

    graph_dir = os.path.join(base_path, "graph")
    assert os.path.exists(os.path.join(graph_dir, "nodes.parquet"))
    assert os.path.exists(os.path.join(graph_dir, "edges.parquet"))
    assert os.path.exists(os.path.join(graph_dir, "cosmic.parquet"))
    assert os.path.exists(os.path.join(graph_dir, "groups.parquet"))

    assert os.path.exists(os.path.join(base_path, "export_manifest.json"))


def test_export_dataset_bundle_nodes_schema(exported_bundle):
    tmpdir, slug, _ = exported_bundle
    graph_dir = os.path.join(tmpdir, slug, "graph")
    tbl_nodes = pq.read_table(os.path.join(graph_dir, "nodes.parquet"))
    assert tbl_nodes.schema.field("node_id").type == pa.uint32()
    assert tbl_nodes.schema.field("group_id").type == pa.uint32()
    assert tbl_nodes.schema.field("val").type == pa.float32()
    assert tbl_nodes.schema.field("archetype").type == pa.string()
    assert tbl_nodes.schema.field("ex").type == pa.float32()
    assert tbl_nodes.schema.field("ey").type == pa.float32()
    assert tbl_nodes.schema.field("ez").type == pa.float32()
    assert tbl_nodes.schema.field("is_live").type == pa.bool_()


def test_export_dataset_bundle_edges_schema(exported_bundle):
    tmpdir, slug, _ = exported_bundle
    graph_dir = os.path.join(tmpdir, slug, "graph")
    tbl_edges = pq.read_table(os.path.join(graph_dir, "edges.parquet"))
    assert tbl_edges.schema.field("source_id").type == pa.uint32()
    assert tbl_edges.schema.field("target_id").type == pa.uint32()
    assert tbl_edges.schema.field("weight").type == pa.float32()


def test_export_dataset_bundle_groups_schema(exported_bundle):
    tmpdir, slug, _ = exported_bundle
    graph_dir = os.path.join(tmpdir, slug, "graph")
    tbl_groups = pq.read_table(os.path.join(graph_dir, "groups.parquet"))
    assert tbl_groups.schema.field("group_id").type == pa.uint32()
    assert tbl_groups.schema.field("name").type == pa.string()
    assert tbl_groups.schema.field("desc").type == pa.string()
    assert tbl_groups.schema.field("member_ids").type == pa.list_(pa.uint32())


def test_clean_data_returns_cached_preprocessed_data(fitted_model):
    """clean_data() is currently a stub: it just returns the preprocessed table
    already cached by fit(). No 'phil' dependency is required for this path
    (see TODO(#27, #28) in pulsar/exports.py for the future multi-imputation
    fusion and ECT-based selection)."""
    # Works even with 'phil' completely absent from sys.modules.
    original_phil = sys.modules.pop("phil", None)
    try:
        result = fitted_model.clean_data()
    finally:
        if original_phil is not None:
            sys.modules["phil"] = original_phil

    assert result is fitted_model.preprocessed_data


def test_clean_embedding_at_center(fitted_model):
    emb = fitted_model.clean_embedding_at_center()
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    # Minimal config has pca dimensions values: [2], so output projection should be 2D
    assert len(emb) == 2
