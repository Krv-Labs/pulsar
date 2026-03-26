import numpy as np
import pytest
import sys
import os

from pulsar._pulsar import pseudo_laplacian


def make_nodes(n, memberships):
    """memberships: list of sets of point indices → ball membership lists"""
    return [list(s) for s in memberships]


def test_diagonal_equals_membership_count():
    n = 5
    # Point 0 is in balls 0 and 1; point 1 is only in ball 0; etc.
    nodes = [[0, 1, 2], [0, 3, 4]]
    L = np.array(pseudo_laplacian(nodes, n))

    # Point 0 appears in 2 balls → L[0,0] == 2
    assert L[0, 0] == 2
    # Points 1,2 appear in 1 ball → L[i,i] == 1
    assert L[1, 1] == 1
    assert L[2, 2] == 1
    # Points 3,4 appear in 1 ball → L[i,i] == 1
    assert L[3, 3] == 1
    assert L[4, 4] == 1


def test_offdiag_negative_shared_memberships():
    n = 5
    nodes = [[0, 1, 2], [0, 3, 4]]
    L = np.array(pseudo_laplacian(nodes, n))

    # Points 1 and 2 share ball 0 → L[1,2] == -1
    assert L[1, 2] == -1
    assert L[2, 1] == -1

    # Points 0 and 1 share ball 0 → L[0,1] == -1
    assert L[0, 1] == -1


def test_no_cross_ball_negative():
    n = 4
    # Two disjoint balls
    nodes = [[0, 1], [2, 3]]
    L = np.array(pseudo_laplacian(nodes, n))
    # Points in different balls have no shared membership → L[0,2] == 0
    assert L[0, 2] == 0
    assert L[1, 3] == 0


def test_symmetry():
    n = 6
    nodes = [[0, 1, 2], [1, 2, 3], [3, 4, 5]]
    L = np.array(pseudo_laplacian(nodes, n))
    np.testing.assert_array_equal(L, L.T)


def test_matches_thema_reference():
    """Compare against the Thema starHelpers.mapper_pseudo_laplacian implementation."""
    thema_path = os.path.join(os.path.dirname(__file__), "..", "Thema")
    if not os.path.isdir(thema_path):
        pytest.skip("Thema submodule not available")

    sys.path.insert(0, thema_path)
    try:
        from thema.multiverse.universe.utils.starHelpers import mapper_pseudo_laplacian
    except ImportError:
        pytest.skip("Cannot import Thema starHelpers")

    n = 8
    nodes_list = [[0, 1, 2], [1, 2, 3], [4, 5], [5, 6, 7]]
    nodes_dict = {i: nodes_list[i] for i in range(len(nodes_list))}

    pulsar_L = np.array(pseudo_laplacian(nodes_list, n))

    # Thema's interface: complex = {"nodes": {id: [members]}}
    thema_complex = {"nodes": nodes_dict}
    thema_L = mapper_pseudo_laplacian(thema_complex, n, components={}, neighborhood="node")

    np.testing.assert_array_equal(pulsar_L, thema_L)


def test_empty_nodes_zero_matrix():
    n = 4
    L = np.array(pseudo_laplacian([], n))
    np.testing.assert_array_equal(L, np.zeros((n, n), dtype=np.int64))
