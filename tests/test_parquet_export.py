import os
import tempfile
import numpy as np
import pandas as pd
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


def test_parquet_export_before_fit_raises(minimal_config):
    cfg = load_config(minimal_config)
    model = ThemaRS(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        edge_path = os.path.join(tmpdir, "edges.parquet")
        node_path = os.path.join(tmpdir, "nodes.parquet")

        with pytest.raises(RuntimeError, match="Call fit.* first"):
            model.export_edges_parquet(edge_path)

        with pytest.raises(RuntimeError, match="Call fit.* first"):
            model.export_nodes_parquet(node_path)


def test_parquet_export_success(minimal_config, simple_data):
    cfg = load_config(minimal_config)
    model = ThemaRS(cfg)
    model.fit(data=simple_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        edge_path = os.path.join(tmpdir, "edges.parquet")
        node_path = os.path.join(tmpdir, "nodes.parquet")

        # Test default export
        model.export_edges_parquet(edge_path)
        assert os.path.exists(edge_path)

        df_edges = pd.read_parquet(edge_path)
        assert list(df_edges.columns) == ["row", "col", "weight"]
        assert len(df_edges) > 0

        # Test custom threshold for edge export
        model.export_edges_parquet(edge_path, threshold=0.5)
        df_edges_high = pd.read_parquet(edge_path)
        assert len(df_edges_high) <= len(df_edges)

        # Test node export with default thresholds (list(float))
        model.export_nodes_parquet(node_path)
        assert os.path.exists(node_path)

        df_nodes = pd.read_parquet(node_path)
        # Should have original columns + cc_label_<threshold>
        expected_cc_col = (
            f"cc_label_{model.resolved_construction_threshold:.4f}".replace(".", "_")
        )
        assert expected_cc_col in df_nodes.columns
        assert all(c in df_nodes.columns for c in ["x", "y", "z"])
        assert df_nodes[expected_cc_col].dtype in [np.int32, np.int64]

        # Test node export with multiple custom thresholds
        custom_thresholds = [0.05, 0.2]
        model.export_nodes_parquet(node_path, thresholds=custom_thresholds)
        df_nodes_custom = pd.read_parquet(node_path)
        for t in custom_thresholds:
            col = f"cc_label_{t:.4f}".replace(".", "_")
            assert col in df_nodes_custom.columns
            assert df_nodes_custom[col].dtype in [np.int32, np.int64]
