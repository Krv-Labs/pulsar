"""
Correctness tests for ph.rs — threshold stability analysis.

These tests compare the Rust implementation against a Python reference
implementation of the connected component analysis algorithm. The Python
version uses a pure BFS connected-component counter for correctness verification.

The algorithm:
1. Sweep threshold τ from 1.0 → 0.0
2. At each τ, include edges with weight > τ
3. Count connected components
4. Identify plateaus (stable regions)
5. Return optimal threshold = midpoint of longest plateau
"""

import numpy as np
from pulsar._pulsar import find_stable_thresholds


def py_connected_components(adj_matrix):
    """Count connected components using BFS — pure Python reference.

    Args:
        adj_matrix: Binary adjacency matrix (n x n)

    Returns:
        Number of connected components
    """
    n = adj_matrix.shape[0]
    if n == 0:
        return 0

    visited = [False] * n
    num_components = 0

    for start in range(n):
        if visited[start]:
            continue

        # BFS from this node
        num_components += 1
        queue = [start]
        visited[start] = True

        while queue:
            node = queue.pop(0)
            for neighbor in range(n):
                if adj_matrix[node, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

    return num_components


def py_find_stable_thresholds(weighted_adj, num_bins=256):
    """Python reference implementation of threshold stability analysis.

    Args:
        weighted_adj: Weighted adjacency matrix (n x n), values in [0, 1]
        num_bins: Number of quantization bins

    Returns:
        dict with:
            - optimal_threshold: midpoint of longest plateau
            - plateaus: list of (start, end, count) tuples
            - thresholds: list of threshold values
            - component_counts: list of component counts
    """
    n = weighted_adj.shape[0]

    if n == 0:
        return {
            "optimal_threshold": 0.5,
            "plateaus": [],
            "thresholds": [],
            "component_counts": [],
        }

    if n == 1:
        return {
            "optimal_threshold": 0.5,
            "plateaus": [(1.0, 0.0, 1)],
            "thresholds": [1.0, 0.0],
            "component_counts": [1, 1],
        }

    # Collect edges with their weights
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = weighted_adj[i, j]
            if w > 0:
                bin_idx = min(int(w * num_bins), num_bins - 1)
                edges.append((bin_idx, i, j))

    # Check for no edges
    if not edges:
        return {
            "optimal_threshold": 0.5,
            "plateaus": [(1.0, 0.0, n)],
            "thresholds": [1.0, 0.0],
            "component_counts": [n, n],
        }

    # Group edges by bin
    bins = [[] for _ in range(num_bins)]
    for bin_idx, i, j in edges:
        bins[bin_idx].append((i, j))

    # Sweep from high to low threshold
    adj = np.zeros((n, n), dtype=np.int32)
    thresholds = [1.0]
    component_counts = [n]

    for bin_idx in range(num_bins - 1, -1, -1):
        # Add edges in this bin
        for i, j in bins[bin_idx]:
            adj[i, j] = 1
            adj[j, i] = 1

        threshold = bin_idx / num_bins
        count = py_connected_components(adj)
        thresholds.append(threshold)
        component_counts.append(count)

    # Identify plateaus
    plateaus = []
    plateau_start_idx = 0

    for i in range(1, len(component_counts)):
        if component_counts[i] != component_counts[plateau_start_idx]:
            plateaus.append(
                (
                    thresholds[plateau_start_idx],
                    thresholds[i - 1],
                    component_counts[plateau_start_idx],
                )
            )
            plateau_start_idx = i

    # Final plateau
    plateaus.append(
        (
            thresholds[plateau_start_idx],
            thresholds[-1],
            component_counts[plateau_start_idx],
        )
    )

    # Sort by length (descending)
    plateaus.sort(key=lambda p: p[0] - p[1], reverse=True)

    # Optimal threshold
    if plateaus:
        optimal_threshold = (plateaus[0][0] + plateaus[0][1]) / 2
    else:
        optimal_threshold = 0.5

    # Deduplicate consecutive identical counts
    dedup_thresholds = []
    dedup_counts = []
    for i in range(len(thresholds)):
        if not dedup_counts or dedup_counts[-1] != component_counts[i]:
            dedup_thresholds.append(thresholds[i])
            dedup_counts.append(component_counts[i])

    return {
        "optimal_threshold": optimal_threshold,
        "plateaus": plateaus,
        "thresholds": dedup_thresholds,
        "component_counts": dedup_counts,
    }


def make_two_cluster_adj():
    """Two well-separated clusters."""
    return np.array(
        [
            [0.0, 0.9, 0.1, 0.1],
            [0.9, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.0, 0.8],
            [0.1, 0.1, 0.8, 0.0],
        ],
        dtype=np.float64,
    )


def make_hierarchical_adj():
    """Hierarchical structure with multiple cluster levels."""
    return np.array(
        [
            [0.0, 0.95, 0.4, 0.4, 0.1, 0.1],
            [0.95, 0.0, 0.4, 0.4, 0.1, 0.1],
            [0.4, 0.4, 0.0, 0.9, 0.1, 0.1],
            [0.4, 0.4, 0.9, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.0, 0.85],
            [0.1, 0.1, 0.1, 0.1, 0.85, 0.0],
        ],
        dtype=np.float64,
    )


class TestComponentCountMatches:
    """Verify component counts match Python reference."""

    def test_two_cluster_counts(self):
        w = make_two_cluster_adj()
        num_bins = 100

        rust_result = find_stable_thresholds(w, num_bins=num_bins)
        py_result = py_find_stable_thresholds(w, num_bins=num_bins)

        # Component counts should contain the same unique values
        rust_counts = set(rust_result.component_counts)
        py_counts = set(py_result["component_counts"])
        assert rust_counts == py_counts

    def test_hierarchical_counts(self):
        w = make_hierarchical_adj()
        num_bins = 100

        rust_result = find_stable_thresholds(w, num_bins=num_bins)
        py_result = py_find_stable_thresholds(w, num_bins=num_bins)

        rust_counts = set(rust_result.component_counts)
        py_counts = set(py_result["component_counts"])
        assert rust_counts == py_counts

    def test_random_graph_counts(self):
        rng = np.random.default_rng(42)
        n = 20
        w = rng.random((n, n))
        w = (w + w.T) / 2  # Symmetrize
        np.fill_diagonal(w, 0)  # Zero diagonal

        num_bins = 50
        rust_result = find_stable_thresholds(w, num_bins=num_bins)
        py_result = py_find_stable_thresholds(w, num_bins=num_bins)

        rust_counts = set(rust_result.component_counts)
        py_counts = set(py_result["component_counts"])
        assert rust_counts == py_counts


class TestPlateauDetection:
    """Verify plateau detection matches Python reference."""

    def test_two_cluster_longest_plateau(self):
        w = make_two_cluster_adj()
        num_bins = 100

        rust_result = find_stable_thresholds(w, num_bins=num_bins)
        py_result = py_find_stable_thresholds(w, num_bins=num_bins)

        # Longest plateau should have same component count
        assert rust_result.plateaus[0].component_count == py_result["plateaus"][0][2]

    def test_hierarchical_longest_plateau(self):
        w = make_hierarchical_adj()
        num_bins = 100

        rust_result = find_stable_thresholds(w, num_bins=num_bins)
        py_result = py_find_stable_thresholds(w, num_bins=num_bins)

        assert rust_result.plateaus[0].component_count == py_result["plateaus"][0][2]

    def test_plateau_count_matches(self):
        w = make_two_cluster_adj()
        num_bins = 100

        rust_result = find_stable_thresholds(w, num_bins=num_bins)
        py_result = py_find_stable_thresholds(w, num_bins=num_bins)

        assert len(rust_result.plateaus) == len(py_result["plateaus"])


class TestOptimalThreshold:
    """Verify optimal threshold matches Python reference."""

    def test_two_cluster_optimal_threshold(self):
        w = make_two_cluster_adj()
        num_bins = 100

        rust_result = find_stable_thresholds(w, num_bins=num_bins)
        py_result = py_find_stable_thresholds(w, num_bins=num_bins)

        # Should match within bin resolution
        tolerance = 2.0 / num_bins
        assert (
            abs(rust_result.optimal_threshold - py_result["optimal_threshold"])
            < tolerance
        )

    def test_hierarchical_optimal_threshold(self):
        w = make_hierarchical_adj()
        num_bins = 100

        rust_result = find_stable_thresholds(w, num_bins=num_bins)
        py_result = py_find_stable_thresholds(w, num_bins=num_bins)

        tolerance = 2.0 / num_bins
        assert (
            abs(rust_result.optimal_threshold - py_result["optimal_threshold"])
            < tolerance
        )

    def test_random_graph_optimal_threshold(self):
        rng = np.random.default_rng(123)
        n = 15
        w = rng.random((n, n))
        w = (w + w.T) / 2
        np.fill_diagonal(w, 0)

        num_bins = 50
        rust_result = find_stable_thresholds(w, num_bins=num_bins)
        py_result = py_find_stable_thresholds(w, num_bins=num_bins)

        tolerance = 2.0 / num_bins
        assert (
            abs(rust_result.optimal_threshold - py_result["optimal_threshold"])
            < tolerance
        )


class TestEdgeCases:
    """Verify edge cases match Python reference."""

    def test_empty_graph_matches(self):
        w = np.zeros((5, 5), dtype=np.float64)
        num_bins = 50

        rust_result = find_stable_thresholds(w, num_bins=num_bins)
        py_result = py_find_stable_thresholds(w, num_bins=num_bins)

        assert len(rust_result.plateaus) == len(py_result["plateaus"])
        assert rust_result.plateaus[0].component_count == 5

    def test_single_node_matches(self):
        w = np.array([[0.0]], dtype=np.float64)

        rust_result = find_stable_thresholds(w)
        py_result = py_find_stable_thresholds(w)

        assert rust_result.optimal_threshold == py_result["optimal_threshold"]

    def test_complete_graph_matches(self):
        n = 5
        w = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(w, 0)

        num_bins = 50
        rust_result = find_stable_thresholds(w, num_bins=num_bins)
        py_result = py_find_stable_thresholds(w, num_bins=num_bins)

        rust_counts = set(rust_result.component_counts)
        py_counts = set(py_result["component_counts"])
        assert rust_counts == py_counts


class TestPlateauBoundaries:
    """Verify plateau boundaries are computed correctly."""

    def test_plateau_start_end_order(self):
        w = make_two_cluster_adj()
        num_bins = 100

        rust_result = find_stable_thresholds(w, num_bins=num_bins)
        py_result = py_find_stable_thresholds(w, num_bins=num_bins)

        for rust_p, py_p in zip(rust_result.plateaus, py_result["plateaus"]):
            # Start should be >= end (thresholds decrease)
            assert rust_p.start_threshold >= rust_p.end_threshold
            # Should match Python within tolerance
            tolerance = 2.0 / num_bins
            assert abs(rust_p.start_threshold - py_p[0]) < tolerance
            assert abs(rust_p.end_threshold - py_p[1]) < tolerance

    def test_plateau_lengths_match(self):
        w = make_hierarchical_adj()
        num_bins = 100

        rust_result = find_stable_thresholds(w, num_bins=num_bins)
        py_result = py_find_stable_thresholds(w, num_bins=num_bins)

        tolerance = 2.0 / num_bins
        for rust_p, py_p in zip(rust_result.plateaus, py_result["plateaus"]):
            rust_len = rust_p.length
            py_len = py_p[0] - py_p[1]
            assert abs(rust_len - py_len) < tolerance


class TestUnionFind:
    """Verify the union-find implementation is correct via connected components."""

    def test_chain_graph(self):
        """Chain: 0-1-2-3-4 should become 1 component when all edges included."""
        n = 5
        w = np.zeros((n, n), dtype=np.float64)
        for i in range(n - 1):
            w[i, i + 1] = 0.5
            w[i + 1, i] = 0.5

        result = find_stable_thresholds(w, num_bins=100)
        assert 1 in result.component_counts
        assert n in result.component_counts

    def test_star_graph(self):
        """Star: center node 0 connected to all others."""
        n = 6
        w = np.zeros((n, n), dtype=np.float64)
        for i in range(1, n):
            w[0, i] = 0.5
            w[i, 0] = 0.5

        result = find_stable_thresholds(w, num_bins=100)
        assert 1 in result.component_counts
        assert n in result.component_counts

    def test_two_disconnected_cliques(self):
        """Two 3-cliques that are disconnected."""
        n = 6
        w = np.zeros((n, n), dtype=np.float64)
        # First clique: 0-1-2
        for i in range(3):
            for j in range(i + 1, 3):
                w[i, j] = 0.8
                w[j, i] = 0.8
        # Second clique: 3-4-5
        for i in range(3, 6):
            for j in range(i + 1, 6):
                w[i, j] = 0.8
                w[j, i] = 0.8

        result = find_stable_thresholds(w, num_bins=100)
        assert 2 in result.component_counts  # Two disconnected cliques
        assert 6 in result.component_counts  # All disconnected at high threshold


class TestQuantizationConsistency:
    """Verify quantization behaves consistently."""

    def test_coarse_and_fine_bins_same_structure(self):
        """Different bin counts should find the same topological structure."""
        w = make_two_cluster_adj()

        result_coarse = find_stable_thresholds(w, num_bins=10)
        result_fine = find_stable_thresholds(w, num_bins=100)

        # Both should find the same unique component counts
        coarse_counts = set(result_coarse.component_counts)
        fine_counts = set(result_fine.component_counts)
        assert coarse_counts == fine_counts

    def test_increasing_bins_monotonic_threshold_count(self):
        """More bins should generally give more (or equal) threshold checkpoints."""
        w = make_hierarchical_adj()

        prev_len = 0
        for bins in [10, 20, 50, 100]:
            result = find_stable_thresholds(w, num_bins=bins)
            # Allow some flexibility due to deduplication
            assert len(result.thresholds) >= prev_len - 1
            prev_len = len(result.thresholds)


class TestSymmetry:
    """Verify results are independent of input symmetry."""

    def test_upper_vs_lower_triangle(self):
        """Results should be the same whether we use upper or lower triangle."""
        rng = np.random.default_rng(99)
        n = 10

        # Create from upper triangle
        upper = rng.random((n, n))
        upper = np.triu(upper, k=1)
        w1 = upper + upper.T

        # Create from lower triangle
        lower = upper.T
        w2 = lower + lower.T

        result1 = find_stable_thresholds(w1, num_bins=50)
        result2 = find_stable_thresholds(w2, num_bins=50)

        np.testing.assert_array_equal(
            result1.component_counts, result2.component_counts
        )
        assert result1.optimal_threshold == result2.optimal_threshold
