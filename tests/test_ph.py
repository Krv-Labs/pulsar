"""Unit tests for persistent homology / threshold stability functions."""

import numpy as np
from pulsar._pulsar import Plateau, StabilityResult, find_stable_thresholds


def make_two_cluster_adj():
    """Two well-separated clusters: nodes 0-1 connected, nodes 2-3 connected."""
    return np.array(
        [
            [0.0, 0.9, 0.1, 0.1],
            [0.9, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.0, 0.8],
            [0.1, 0.1, 0.8, 0.0],
        ],
        dtype=np.float64,
    )


def make_uniform_adj():
    """Fully connected with uniform weights."""
    return np.array(
        [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )


def make_empty_adj(n=3):
    """No edges (all zeros)."""
    return np.zeros((n, n), dtype=np.float64)


class TestStabilityResultOutput:
    """Tests for StabilityResult output structure."""

    def test_returns_stability_result(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        assert isinstance(result, StabilityResult)

    def test_optimal_threshold_in_range(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        assert 0.0 <= result.optimal_threshold <= 1.0

    def test_thresholds_descending(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        thresholds = np.array(result.thresholds)
        # Thresholds should be in descending order (1.0 → 0.0)
        assert thresholds[0] == 1.0
        for i in range(1, len(thresholds)):
            assert thresholds[i] <= thresholds[i - 1]

    def test_component_counts_positive(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        counts = np.array(result.component_counts)
        assert np.all(counts > 0)

    def test_component_counts_bounded_by_n(self):
        w = make_two_cluster_adj()
        n = w.shape[0]
        result = find_stable_thresholds(w)
        counts = np.array(result.component_counts)
        assert np.all(counts <= n)

    def test_thresholds_and_counts_same_length(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        assert len(result.thresholds) == len(result.component_counts)


class TestPlateauStructure:
    """Tests for Plateau objects."""

    def test_plateaus_not_empty(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        assert len(result.plateaus) > 0

    def test_plateau_has_required_attributes(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        plateau = result.plateaus[0]
        assert isinstance(plateau, Plateau)
        assert hasattr(plateau, "start_threshold")
        assert hasattr(plateau, "end_threshold")
        assert hasattr(plateau, "component_count")
        assert hasattr(plateau, "length")
        assert hasattr(plateau, "midpoint")

    def test_plateau_thresholds_valid(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        for plateau in result.plateaus:
            assert 0.0 <= plateau.end_threshold <= plateau.start_threshold <= 1.0

    def test_plateau_length_nonnegative(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        for plateau in result.plateaus:
            assert plateau.length >= 0.0

    def test_plateau_midpoint_in_range(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        for plateau in result.plateaus:
            assert plateau.end_threshold <= plateau.midpoint <= plateau.start_threshold

    def test_plateaus_sorted_by_length(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        lengths = [p.length for p in result.plateaus]
        for i in range(1, len(lengths)):
            assert lengths[i] <= lengths[i - 1]


class TestTwoClusterGraph:
    """Tests specific to two-cluster graph structure."""

    def test_identifies_two_components_plateau(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w, num_bins=100)
        component_counts = [p.component_count for p in result.plateaus]
        assert 2 in component_counts

    def test_longest_plateau_has_two_components(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w, num_bins=100)
        assert result.plateaus[0].component_count == 2

    def test_optimal_threshold_in_stable_region(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w, num_bins=100)
        # Optimal threshold should be between the cluster weights (0.1) and
        # intra-cluster weights (0.8-0.9)
        assert 0.1 < result.optimal_threshold < 0.9


class TestUniformGraph:
    """Tests for uniformly-weighted fully-connected graphs."""

    def test_component_evolution(self):
        w = make_uniform_adj()
        result = find_stable_thresholds(w)
        counts = np.array(result.component_counts)
        # Should have n components at high threshold, 1 at low
        assert 3 in counts
        assert 1 in counts

    def test_single_transition(self):
        w = make_uniform_adj()
        result = find_stable_thresholds(w)
        counts = np.array(result.component_counts)
        # Uniform weights mean a single transition from n to 1
        unique_counts = set(counts)
        assert unique_counts == {1, 3}


class TestEmptyGraph:
    """Tests for graphs with no edges."""

    def test_single_plateau(self):
        w = make_empty_adj(n=4)
        result = find_stable_thresholds(w)
        assert len(result.plateaus) == 1

    def test_constant_component_count(self):
        n = 4
        w = make_empty_adj(n=n)
        result = find_stable_thresholds(w)
        assert result.plateaus[0].component_count == n

    def test_plateau_spans_full_range(self):
        w = make_empty_adj()
        result = find_stable_thresholds(w)
        assert result.plateaus[0].start_threshold == 1.0
        assert result.plateaus[0].end_threshold == 0.0


class TestSingleNode:
    """Tests for single-node graphs."""

    def test_single_node_result(self):
        w = np.array([[0.0]], dtype=np.float64)
        result = find_stable_thresholds(w)
        assert result.optimal_threshold == 0.5
        assert len(result.plateaus) == 1
        assert result.plateaus[0].component_count == 1


class TestEmptyMatrix:
    """Tests for empty (0x0) matrices."""

    def test_empty_matrix_result(self):
        w = np.zeros((0, 0), dtype=np.float64)
        result = find_stable_thresholds(w)
        assert result.optimal_threshold == 0.5
        assert len(result.plateaus) == 0


class TestNumBinsParameter:
    """Tests for the num_bins parameter."""

    def test_default_num_bins(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        # Should work with default bins
        assert isinstance(result, StabilityResult)

    def test_custom_num_bins(self):
        w = make_two_cluster_adj()
        for bins in [10, 50, 100, 500]:
            result = find_stable_thresholds(w, num_bins=bins)
            assert isinstance(result, StabilityResult)

    def test_higher_bins_more_thresholds(self):
        w = make_two_cluster_adj()
        result_coarse = find_stable_thresholds(w, num_bins=10)
        result_fine = find_stable_thresholds(w, num_bins=100)
        # Note: after deduplication, this isn't strictly true, but generally
        # finer bins should have at least as many thresholds
        assert len(result_fine.thresholds) >= len(result_coarse.thresholds) - 2


class TestTopKMethods:
    """Tests for top_k_* methods."""

    def test_top_k_plateaus_default(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w, num_bins=100)
        top3 = result.top_k_plateaus()
        assert len(top3) <= 3
        assert all(isinstance(p, Plateau) for p in top3)

    def test_top_k_plateaus_custom(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w, num_bins=100)
        top1 = result.top_k_plateaus(k=1)
        assert len(top1) <= 1

    def test_top_k_thresholds_default(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w, num_bins=100)
        thresholds = np.array(result.top_k_thresholds())
        assert len(thresholds) <= 3
        assert np.all(thresholds >= 0.0)
        assert np.all(thresholds <= 1.0)

    def test_top_k_thresholds_are_midpoints(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w, num_bins=100)
        top_thresholds = np.array(result.top_k_thresholds(k=2))
        top_plateaus = result.top_k_plateaus(k=2)
        expected = [p.midpoint for p in top_plateaus]
        np.testing.assert_array_almost_equal(top_thresholds, expected)


class TestRepr:
    """Tests for string representation."""

    def test_stability_result_repr(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        repr_str = repr(result)
        assert "StabilityResult" in repr_str
        assert "optimal_threshold" in repr_str

    def test_plateau_repr(self):
        w = make_two_cluster_adj()
        result = find_stable_thresholds(w)
        plateau = result.plateaus[0]
        repr_str = repr(plateau)
        assert "Plateau" in repr_str
        assert "start" in repr_str
        assert "end" in repr_str
