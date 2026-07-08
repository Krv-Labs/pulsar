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


def test_tabular_helpers_and_edges(minimal_config, simple_data):
    cfg = load_config(minimal_config)
    model = ThemaRS(cfg)
    model.fit(data=simple_data)

    # data
    df_raw = model.data
    assert isinstance(df_raw, pd.DataFrame)
    assert len(df_raw) == 30

    # cosmic_edges
    df_cosmic = model.cosmic_edges(threshold=0.0)
    assert list(df_cosmic.columns) == ["source_id", "target_id", "weight"]
    assert df_cosmic["source_id"].dtype == np.uint32
    assert df_cosmic["target_id"].dtype == np.uint32
    assert df_cosmic["weight"].dtype == np.float32
    # Check undirected contract: i < j
    assert (df_cosmic["source_id"] < df_cosmic["target_id"]).all()

    # snapshot_edges (using default threshold)
    df_snap = model.snapshot_edges()
    assert list(df_snap.columns) == ["source_id", "target_id", "weight"]
    assert len(df_snap) <= len(df_cosmic)


def test_node_table(minimal_config, simple_data):
    cfg = load_config(minimal_config)
    model = ThemaRS(cfg)
    model.fit(data=simple_data)

    cluster_labels = np.array([0] * 15 + [1] * 15)
    cluster_names = {0: "Zeroes", 1: "Ones"}

    # Layout: projection
    df_nodes_proj = model.node_table(
        cluster_labels=cluster_labels,
        cluster_names=cluster_names,
        layout="projection",
        extra_columns=["x", "z"],
    )
    assert list(df_nodes_proj.columns[:8]) == [
        "node_id",
        "group_id",
        "val",
        "archetype",
        "ex",
        "ey",
        "ez",
        "is_live",
    ]
    assert df_nodes_proj["node_id"].dtype == np.uint32
    assert df_nodes_proj["group_id"].dtype == np.uint32
    assert df_nodes_proj["val"].dtype == np.float32
    assert df_nodes_proj["is_live"].dtype == bool
    assert (df_nodes_proj["archetype"].iloc[:15] == "Zeroes").all()
    assert (df_nodes_proj["archetype"].iloc[15:] == "Ones").all()
    # Check extra columns passthrough
    assert "x" in df_nodes_proj.columns
    assert "z" in df_nodes_proj.columns
    assert "y" not in df_nodes_proj.columns

    # Layout: zeros
    df_nodes_zeros = model.node_table(
        cluster_labels=cluster_labels,
        layout="zeros",
    )
    assert (df_nodes_zeros["ex"] == 0.0).all()
    assert (df_nodes_zeros["ey"] == 0.0).all()
    assert (df_nodes_zeros["ez"] == 0.0).all()

    # Layout: spectral
    df_nodes_spec = model.node_table(
        cluster_labels=cluster_labels,
        layout="spectral",
    )
    assert len(df_nodes_spec) == 30


def test_group_table(minimal_config, simple_data):
    cfg = load_config(minimal_config)
    model = ThemaRS(cfg)
    model.fit(data=simple_data)

    cluster_labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    cluster_names = {0: "Zero", 1: "One"}
    cluster_descriptions = {0: "The first ten", 1: "The middle ten", 2: "The last ten"}

    df_groups = model.group_table(
        cluster_labels=cluster_labels,
        cluster_names=cluster_names,
        cluster_descriptions=cluster_descriptions,
    )
    assert list(df_groups.columns) == ["group_id", "name", "desc", "member_ids"]
    assert len(df_groups) == 3
    assert df_groups["group_id"].dtype == np.uint32

    # Row 0 (gid 0)
    row0 = df_groups.iloc[0]
    assert row0["group_id"] == 0
    assert row0["name"] == "Zero"
    assert row0["desc"] == "The first ten"
    assert row0["member_ids"] == list(range(10))

    # Row 2 (gid 2, missing name)
    row2 = df_groups.iloc[2]
    assert row2["group_id"] == 2
    assert row2["name"] == "Cluster 2"
    assert row2["desc"] == "The last ten"
    assert row2["member_ids"] == list(range(20, 30))


def test_export_dataset_bundle_success(minimal_config, simple_data):
    cfg = load_config(minimal_config)
    model = ThemaRS(cfg)
    model.fit(data=simple_data)

    cluster_labels = np.array([0] * 15 + [1] * 15)

    with tempfile.TemporaryDirectory() as tmpdir:
        slug = "test-slug"
        manifest = model.export_dataset_bundle(
            output_dir=tmpdir,
            slug=slug,
            cluster_labels=cluster_labels,
            layout="projection",
        )

        assert isinstance(manifest, dict)
        assert "pulsar_version" in manifest
        assert "config_hash" in manifest

        base_path = os.path.join(tmpdir, slug)
        assert os.path.exists(base_path)

        # Tabular folder
        tab_dir = os.path.join(base_path, "tabular")
        assert os.path.exists(os.path.join(tab_dir, "raw.parquet"))

        # Graph folder
        graph_dir = os.path.join(base_path, "graph")
        assert os.path.exists(os.path.join(graph_dir, "nodes.parquet"))
        assert os.path.exists(os.path.join(graph_dir, "edges.parquet"))
        assert os.path.exists(os.path.join(graph_dir, "cosmic.parquet"))
        assert os.path.exists(os.path.join(graph_dir, "groups.parquet"))

        # Manifest
        assert os.path.exists(os.path.join(base_path, "export_manifest.json"))

        # Verify Node parquet schema
        tbl_nodes = pq.read_table(os.path.join(graph_dir, "nodes.parquet"))
        assert tbl_nodes.schema.field("node_id").type == pa.uint32()
        assert tbl_nodes.schema.field("group_id").type == pa.uint32()
        assert tbl_nodes.schema.field("val").type == pa.float32()
        assert tbl_nodes.schema.field("archetype").type == pa.string()
        assert tbl_nodes.schema.field("ex").type == pa.float32()
        assert tbl_nodes.schema.field("ey").type == pa.float32()
        assert tbl_nodes.schema.field("ez").type == pa.float32()
        assert tbl_nodes.schema.field("is_live").type == pa.bool_()

        # Verify Edges parquet schema
        tbl_edges = pq.read_table(os.path.join(graph_dir, "edges.parquet"))
        assert tbl_edges.schema.field("source_id").type == pa.uint32()
        assert tbl_edges.schema.field("target_id").type == pa.uint32()
        assert tbl_edges.schema.field("weight").type == pa.float32()

        # Verify Groups parquet schema
        tbl_groups = pq.read_table(os.path.join(graph_dir, "groups.parquet"))
        assert tbl_groups.schema.field("group_id").type == pa.uint32()
        assert tbl_groups.schema.field("name").type == pa.string()
        assert tbl_groups.schema.field("desc").type == pa.string()
        assert tbl_groups.schema.field("member_ids").type == pa.list_(pa.uint32())


def test_clean_data_returns_cached_preprocessed_data(minimal_config, simple_data):
    """clean_data() is currently a stub: it just returns the preprocessed table
    already cached by fit(). No 'phil' dependency is required for this path
    (see TODO(#26) in pulsar/exports.py for the future ECT-based selection)."""
    cfg = load_config(minimal_config)
    model = ThemaRS(cfg)
    model.fit(data=simple_data)

    # Works even with 'phil' completely absent from sys.modules.
    original_phil = sys.modules.pop("phil", None)
    try:
        result = model.clean_data()
    finally:
        if original_phil is not None:
            sys.modules["phil"] = original_phil

    assert result is model.preprocessed_data


def test_clean_embedding_at_center(minimal_config, simple_data):
    cfg = load_config(minimal_config)
    model = ThemaRS(cfg)
    model.fit(data=simple_data)

    emb = model.clean_embedding_at_center()
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    # Minimal config has pca dimensions values: [2], so output projection should be 2D
    assert len(emb) == 2
