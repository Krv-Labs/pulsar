"""
Tests for TemporalCosmicGraph functionality.
"""

import numpy as np
import pytest

from pulsar.config import load_config
from pulsar.representations import TemporalCosmicGraph


class TestTemporalCosmicGraphConstruction:
    """Test TemporalCosmicGraph initialization and validation."""

    def test_init_valid_tensor(self):
        """Valid 3D tensor should initialize correctly."""
        tensor = np.random.rand(10, 10, 5)
        tcg = TemporalCosmicGraph(tensor, threshold=0.1)

        assert tcg.n == 10
        assert tcg.T == 5
        assert tcg.shape == (10, 10, 5)
        assert tcg._threshold == 0.1

    def test_init_rejects_2d_tensor(self):
        """2D tensor should raise ValueError."""
        tensor = np.random.rand(10, 10)
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            TemporalCosmicGraph(tensor)

    def test_init_rejects_non_square(self):
        """Non-square slices should raise ValueError."""
        tensor = np.random.rand(10, 5, 3)
        with pytest.raises(ValueError, match="Expected square slices"):
            TemporalCosmicGraph(tensor)

    def test_init_converts_dtype(self):
        """Input should be converted to float64."""
        tensor = np.random.randint(0, 100, (5, 5, 3)).astype(np.int32)
        tcg = TemporalCosmicGraph(tensor)
        assert tcg.tensor.dtype == np.float64


class TestPersistenceGraph:
    """Test persistence graph aggregation."""

    def test_persistence_all_above_threshold(self):
        """All values above threshold → persistence = 1.0."""
        tensor = np.ones((3, 3, 10)) * 0.5
        tcg = TemporalCosmicGraph(tensor, threshold=0.1)

        result = tcg.persistence_graph()

        # All entries (including diagonal) should be 1.0 since 0.5 > 0.1
        expected = np.ones((3, 3))
        np.testing.assert_array_almost_equal(result, expected)

    def test_persistence_all_below_threshold(self):
        """All values below threshold → persistence = 0.0."""
        tensor = np.ones((3, 3, 10)) * 0.05
        tcg = TemporalCosmicGraph(tensor, threshold=0.1)

        result = tcg.persistence_graph()
        np.testing.assert_array_almost_equal(result, np.zeros((3, 3)))

    def test_persistence_half_above(self):
        """Half above threshold → persistence = 0.5."""
        tensor = np.zeros((2, 2, 10))
        tensor[0, 1, :5] = 0.5  # Above threshold for 5/10 steps
        tensor[1, 0, :5] = 0.5
        tcg = TemporalCosmicGraph(tensor, threshold=0.1)

        result = tcg.persistence_graph()

        assert result[0, 1] == 0.5
        assert result[1, 0] == 0.5

    def test_persistence_custom_threshold(self):
        """Custom threshold should be used."""
        tensor = np.ones((2, 2, 10)) * 0.3
        tcg = TemporalCosmicGraph(tensor, threshold=0.1)

        # With default threshold (0.1), all above
        result_default = tcg.persistence_graph()
        assert result_default[0, 1] == 1.0

        # With higher threshold, all below
        result_high = tcg.persistence_graph(threshold=0.5)
        assert result_high[0, 1] == 0.0


class TestMeanGraph:
    """Test mean graph aggregation."""

    def test_mean_constant_values(self):
        """Constant values should return that constant."""
        tensor = np.ones((3, 3, 5)) * 0.7
        tcg = TemporalCosmicGraph(tensor)

        result = tcg.mean_graph()
        np.testing.assert_array_almost_equal(result, np.ones((3, 3)) * 0.7)

    def test_mean_varying_values(self):
        """Mean should be computed correctly."""
        tensor = np.zeros((2, 2, 4))
        tensor[0, 1, :] = [0.1, 0.2, 0.3, 0.4]  # Mean = 0.25
        tensor[1, 0, :] = [0.1, 0.2, 0.3, 0.4]
        tcg = TemporalCosmicGraph(tensor)

        result = tcg.mean_graph()
        assert abs(result[0, 1] - 0.25) < 1e-10


class TestRecencyGraph:
    """Test recency-weighted graph aggregation."""

    def test_recency_recent_only(self):
        """Recent values still receive the highest per-step weight."""
        tensor = np.zeros((2, 2, 10))
        tensor[0, 1, -1] = 1.0  # Only last time step has value
        tensor[1, 0, -1] = 1.0
        tcg = TemporalCosmicGraph(tensor)

        result = tcg.recency_graph(decay=0.99)

        # Most recent step has highest weight
        # For decay=0.99, T=10: weight[9] = 0.99^0 = 1
        # Sum of weights = 1 + 0.99 + 0.99^2 + ... = (1 - 0.99^10) / (1 - 0.99) ≈ 9.56
        # So result ≈ 1.0 / 9.56 ≈ 0.1046
        assert result[0, 1] > 0.1  # Just needs to be positive and meaningful

    def test_recency_old_only(self):
        """Old values get less weight with high decay."""
        tensor = np.zeros((2, 2, 10))
        tensor[0, 1, 0] = 1.0  # Only first time step has value
        tensor[1, 0, 0] = 1.0
        tcg = TemporalCosmicGraph(tensor)

        result = tcg.recency_graph(decay=0.9)

        # First step gets weight 0.9^9 ≈ 0.387, much less than recent
        assert result[0, 1] < 0.2

    def test_recency_invalid_decay(self):
        """Decay outside (0, 1) should raise error."""
        tensor = np.ones((2, 2, 5))
        tcg = TemporalCosmicGraph(tensor)

        with pytest.raises(ValueError, match="decay must be in"):
            tcg.recency_graph(decay=1.0)

        with pytest.raises(ValueError, match="decay must be in"):
            tcg.recency_graph(decay=0.0)


class TestVolatilityGraph:
    """Test volatility graph aggregation."""

    def test_volatility_constant(self):
        """Constant values should have zero volatility."""
        tensor = np.ones((3, 3, 10)) * 0.5
        tcg = TemporalCosmicGraph(tensor)

        result = tcg.volatility_graph()
        np.testing.assert_array_almost_equal(result, np.zeros((3, 3)))

    def test_volatility_varying(self):
        """Varying values should have positive volatility."""
        tensor = np.zeros((2, 2, 4))
        tensor[0, 1, :] = [0.0, 1.0, 0.0, 1.0]  # High variance
        tensor[1, 0, :] = [0.0, 1.0, 0.0, 1.0]
        tcg = TemporalCosmicGraph(tensor)

        result = tcg.volatility_graph()

        # Variance of [0, 1, 0, 1] = 0.25
        assert abs(result[0, 1] - 0.25) < 1e-10


class TestTrendGraph:
    """Test trend graph aggregation."""

    def test_trend_increasing(self):
        """Increasing values should have positive trend."""
        tensor = np.zeros((2, 2, 5))
        tensor[0, 1, :] = [0.0, 0.25, 0.5, 0.75, 1.0]  # Linear increase
        tensor[1, 0, :] = [0.0, 0.25, 0.5, 0.75, 1.0]
        tcg = TemporalCosmicGraph(tensor)

        result = tcg.trend_graph()

        # Slope should be positive
        assert result[0, 1] > 0

    def test_trend_decreasing(self):
        """Decreasing values should have negative trend."""
        tensor = np.zeros((2, 2, 5))
        tensor[0, 1, :] = [1.0, 0.75, 0.5, 0.25, 0.0]  # Linear decrease
        tensor[1, 0, :] = [1.0, 0.75, 0.5, 0.25, 0.0]
        tcg = TemporalCosmicGraph(tensor)

        result = tcg.trend_graph()

        # Slope should be negative
        assert result[0, 1] < 0

    def test_trend_constant(self):
        """Constant values should have zero trend."""
        tensor = np.ones((2, 2, 10)) * 0.5
        tcg = TemporalCosmicGraph(tensor)

        result = tcg.trend_graph()
        np.testing.assert_array_almost_equal(result, np.zeros((2, 2)))

    def test_trend_single_timestep(self):
        """Single time step should return zeros (no trend)."""
        tensor = np.ones((2, 2, 1))
        tcg = TemporalCosmicGraph(tensor)

        result = tcg.trend_graph()
        np.testing.assert_array_almost_equal(result, np.zeros((2, 2)))


class TestChangePointGraph:
    """Test change-point graph aggregation."""

    def test_changepoint_no_change(self):
        """Constant values should have zero change-point score."""
        tensor = np.ones((2, 2, 10)) * 0.5
        tcg = TemporalCosmicGraph(tensor)

        result = tcg.change_point_graph()
        np.testing.assert_array_almost_equal(result, np.zeros((2, 2)))

    def test_changepoint_sudden_jump(self):
        """Sudden jump should be detected."""
        tensor = np.zeros((2, 2, 10))
        tensor[0, 1, :5] = 0.0
        tensor[0, 1, 5:] = 1.0  # Jump from 0 to 1 at t=5
        tensor[1, 0, :5] = 0.0
        tensor[1, 0, 5:] = 1.0
        tcg = TemporalCosmicGraph(tensor)

        result = tcg.change_point_graph()

        # Max change is 1.0 (from 0 to 1)
        assert abs(result[0, 1] - 1.0) < 1e-10

    def test_changepoint_single_timestep(self):
        """Single time step should return zeros."""
        tensor = np.ones((2, 2, 1))
        tcg = TemporalCosmicGraph(tensor)

        result = tcg.change_point_graph()
        np.testing.assert_array_almost_equal(result, np.zeros((2, 2)))


class TestSlice:
    """Test time-range slicing."""

    def test_slice_subset(self):
        """Slicing should extract correct time range."""
        tensor = np.arange(3 * 3 * 10).reshape(3, 3, 10).astype(float)
        tcg = TemporalCosmicGraph(tensor)

        sliced = tcg.slice(start=2, end=5)

        assert sliced.T == 3
        assert sliced.n == 3
        np.testing.assert_array_equal(sliced.tensor, tensor[:, :, 2:5])

    def test_slice_default_end(self):
        """Default end should be T."""
        tensor = np.random.rand(3, 3, 10)
        tcg = TemporalCosmicGraph(tensor)

        sliced = tcg.slice(start=5)

        assert sliced.T == 5
        np.testing.assert_array_equal(sliced.tensor, tensor[:, :, 5:])


class TestToNetworkX:
    """Test NetworkX conversion."""

    def test_to_networkx_basic(self):
        """Basic conversion should work."""
        tensor = np.ones((3, 3, 5)) * 0.5
        tcg = TemporalCosmicGraph(tensor, threshold=0.1)

        G = tcg.to_networkx(aggregation="mean")

        assert G.number_of_nodes() == 3
        assert G.number_of_edges() > 0

    def test_to_networkx_invalid_aggregation(self):
        """Invalid aggregation name should raise error."""
        tensor = np.ones((3, 3, 5))
        tcg = TemporalCosmicGraph(tensor)

        with pytest.raises(ValueError, match="Unknown aggregation"):
            tcg.to_networkx(aggregation="invalid")

    def test_to_networkx_threshold_filters_edges(self):
        """Threshold should filter edges."""
        tensor = np.zeros((3, 3, 5))
        tensor[0, 1, :] = 0.1
        tensor[1, 0, :] = 0.1
        tensor[0, 2, :] = 0.5
        tensor[2, 0, :] = 0.5
        tcg = TemporalCosmicGraph(tensor)

        # Low threshold: both edges
        G_low = tcg.to_networkx(aggregation="mean", threshold=0.05)
        assert G_low.number_of_edges() == 2

        # High threshold: only high-weight edge
        G_high = tcg.to_networkx(aggregation="mean", threshold=0.3)
        assert G_high.number_of_edges() == 1

    def test_to_networkx_recency_kwargs_passthrough(self):
        """Aggregation kwargs should be passed to the selected method."""
        tensor = np.zeros((3, 3, 4))
        tensor[0, 1, :] = [0.0, 0.0, 0.0, 1.0]
        tensor[1, 0, :] = [0.0, 0.0, 0.0, 1.0]
        tcg = TemporalCosmicGraph(tensor)

        G = tcg.to_networkx(aggregation="recency", threshold=0.2, decay=0.2)
        assert G.has_edge(0, 1)


class TestFromSnapshots:
    """Test TemporalCosmicGraph.from_snapshots end-to-end behavior."""

    def test_from_snapshots_builds_symmetric_tensor(self):
        """from_snapshots should produce an (n, n, T) symmetric tensor with zero diagonal."""
        snapshots = [
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [0.1, 0.0, 0.0],
                    [1.1, 0.0, 0.0],
                    [0.0, 1.1, 0.0],
                    [1.0, 1.2, 0.0],
                ],
                dtype=np.float64,
            ),
        ]

        config = load_config(
            {
                "run": {"name": "test_temporal_from_snapshots"},
                "preprocessing": {"drop_columns": [], "impute": {}},
                "sweep": {
                    "pca": {
                        "dimensions": {"values": [2]},
                        "seed": {"values": [42]},
                    },
                    "ball_mapper": {
                        "epsilon": {"values": [0.8]},
                    },
                },
                "cosmic_graph": {"threshold": 0.0},
                "output": {"n_reps": 1},
            }
        )

        tcg = TemporalCosmicGraph.from_snapshots(snapshots=snapshots, config=config)

        assert tcg.shape == (4, 4, 2)
        np.testing.assert_array_equal(np.diagonal(tcg.tensor, axis1=0, axis2=1), 0.0)
        np.testing.assert_allclose(tcg.tensor, np.swapaxes(tcg.tensor, 0, 1))


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        """__repr__ should include key info."""
        tensor = np.random.rand(10, 10, 5)
        tcg = TemporalCosmicGraph(tensor, threshold=0.15)

        repr_str = repr(tcg)

        assert "n=10" in repr_str
        assert "T=5" in repr_str
        assert "0.15" in repr_str
