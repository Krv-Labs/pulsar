import numpy as np
from unittest import mock
import yaml

from pulsar._pulsar import JLProjection, jl_grid
from pulsar.config import config_to_yaml, load_config
from pulsar.pipeline import ThemaRS
from pulsar.runtime.fingerprint import pca_fingerprint, projection_fingerprint


def test_jl_projection_shape_and_reproducibility(small_array):
    jl1 = JLProjection(n_components=3, seed=42)
    jl2 = JLProjection(n_components=3, seed=42)
    jl3 = JLProjection(n_components=3, seed=7)

    out1 = np.array(jl1.fit_transform(small_array))
    out2 = np.array(jl2.fit_transform(small_array))
    out3 = np.array(jl3.fit_transform(small_array))

    assert out1.shape == (small_array.shape[0], 3)
    np.testing.assert_allclose(out1, out2)
    assert not np.allclose(out1, out3)
    np.testing.assert_allclose(out1, np.array(jl1.transform(small_array)))


def test_jl_grid_seed_outer_dimension_inner_order(small_array):
    dims = [2, 3]
    seeds = [42, 7]
    grid = [np.array(emb) for emb in jl_grid(small_array, dims, seeds)]

    assert [emb.shape[1] for emb in grid] == [2, 3, 2, 3]
    assert not np.allclose(grid[0], grid[2])
    assert not np.allclose(grid[1], grid[3])


def test_load_config_projection_default_from_legacy_pca():
    cfg = load_config(
        {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [2, 4]},
                    "seed": {"values": [42, 7]},
                }
            }
        }
    )

    assert cfg.projection.method == "jl"
    assert cfg.projection.dimensions == [2, 4]
    assert cfg.projection.seeds == [42, 7]
    assert cfg.pca.dimensions == [2, 4]


def test_config_roundtrip_preserves_projection():
    cfg = load_config(
        {
            "run": {"name": "roundtrip", "data": "data.csv"},
            "sweep": {
                "projection": {
                    "method": "pca",
                    "dimensions": {"values": [3]},
                    "seed": {"values": [5]},
                    "center": False,
                },
                "ball_mapper": {"epsilon": {"values": [0.4]}},
            },
            "cosmic_graph": {"construction_threshold": "0.0"},
        }
    )

    yaml_str = config_to_yaml(cfg)
    assert "projection:" in yaml_str
    assert "method: pca" in yaml_str
    cfg2 = load_config(yaml.safe_load(yaml_str))
    assert cfg2.projection.method == "pca"
    assert cfg2.projection.dimensions == [3]
    assert cfg2.projection.seeds == [5]
    assert cfg2.projection.center is False


def test_pipeline_default_uses_jl_grid():
    data = np.random.default_rng(0).standard_normal((30, 3))
    import pandas as pd

    cfg = load_config(
        {
            "preprocessing": {},
            "sweep": {
                "pca": {
                    "dimensions": {"values": [2]},
                    "seed": {"values": [42]},
                },
                "ball_mapper": {"epsilon": {"values": [0.8]}},
            },
            "cosmic_graph": {"construction_threshold": "0.0"},
        }
    )

    with mock.patch("pulsar.pipeline.jl_grid") as mock_jl:
        mock_jl.return_value = [data[:, :2].copy()]
        ThemaRS(cfg).fit(data=pd.DataFrame(data, columns=["a", "b", "c"]))
        assert mock_jl.called


def test_pipeline_explicit_pca_uses_pca_grid():
    data = np.random.default_rng(1).standard_normal((30, 3))
    import pandas as pd

    cfg = load_config(
        {
            "preprocessing": {},
            "sweep": {
                "projection": {
                    "method": "pca",
                    "dimensions": {"values": [2]},
                    "seed": {"values": [42]},
                },
                "ball_mapper": {"epsilon": {"values": [0.8]}},
            },
            "cosmic_graph": {"construction_threshold": "0.0"},
        }
    )

    with mock.patch("pulsar.pipeline.pca_grid") as mock_pca:
        mock_pca.return_value = [data[:, :2].copy()]
        ThemaRS(cfg).fit(data=pd.DataFrame(data, columns=["a", "b", "c"]))
        assert mock_pca.called


def test_projection_fingerprint_alias_and_method_sensitivity():
    cfg_jl = load_config({"sweep": {"projection": {"method": "jl"}}})
    cfg_pca = load_config({"sweep": {"projection": {"method": "pca"}}})

    assert projection_fingerprint(cfg_jl, 10) == pca_fingerprint(cfg_jl, 10)
    assert projection_fingerprint(cfg_jl, 10) != projection_fingerprint(cfg_pca, 10)


def test_pipeline_cosmic_graph_sparsified_when_enabled():
    # Spectral sparsification is opt-in (sparsify=True below); it is no longer the
    # default. With it enabled, cosmic_rust is the sparsified graph.
    import pandas as pd

    rng = np.random.default_rng(4)
    data = pd.DataFrame(
        rng.standard_normal((40, 4)),
        columns=["a", "b", "c", "d"],
    )
    cfg = load_config(
        {
            "preprocessing": {},
            "sweep": {
                "projection": {
                    "method": "jl",
                    "dimensions": {"values": [2]},
                    "seed": {"values": [42]},
                },
                "ball_mapper": {"epsilon": {"values": [100.0]}},
            },
            "cosmic_graph": {
                "construction_threshold": 0.0,
                "sparsify": True,
                "sparsify_epsilon": 1.0,
                "sparsify_seed": 42,
                "sparsify_sketch_dim": 4,
                "sparsify_sample_count": 80,
            },
        }
    )

    model = ThemaRS(cfg).fit(data=data)

    assert model.cosmic_rust.n_edges <= model.dense_cosmic_rust.n_edges
    assert model.cosmic_graph.number_of_edges() == len(model.weighted_edges())
    assert model.cosmic_graph.number_of_edges() <= 80
    assert np.count_nonzero(model.weighted_adjacency) // 2 == model.cosmic_rust.n_edges


def test_pipeline_spectral_sparsify_update_refreshes_thresholded_graph():
    import pandas as pd

    rng = np.random.default_rng(5)
    data = pd.DataFrame(rng.standard_normal((30, 3)), columns=["a", "b", "c"])
    cfg = load_config(
        {
            "preprocessing": {},
            "sweep": {
                "projection": {
                    "method": "jl",
                    "dimensions": {"values": [2]},
                    "seed": {"values": [42]},
                },
                "ball_mapper": {"epsilon": {"values": [1.0]}},
            },
            "cosmic_graph": {"construction_threshold": "auto", "sparsify": False},
        }
    )

    model = ThemaRS(cfg).fit(data=data)
    before = model.cosmic_rust.n_edges
    sparse = model.spectral_sparsify(
        epsilon=1.0,
        seed=7,
        sketch_dim=4,
        sample_count=30,
        update=True,
    )

    assert model.cosmic_rust is sparse
    assert model.cosmic_rust.n_edges <= before
    assert all(
        w > model.resolved_construction_threshold for _, _, w in model.weighted_edges()
    )
