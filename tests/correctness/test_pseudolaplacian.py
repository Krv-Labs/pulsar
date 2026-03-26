"""
Correctness tests for pseudolaplacian.rs — pseudo_laplacian.

The Python reference is a literal translation of the Rust nested loop:

    L = zeros(n, n)
    for members in nodes:
        for i in members:
            for j in members:
                L[i, j] += 1 if i == j else -1

Both implementations are integer arithmetic on the same inputs, so the
outputs must be bit-identical (assert_array_equal, not allclose).
"""
import numpy as np
import pytest

from pulsar._pulsar import pseudo_laplacian


# ---------------------------------------------------------------------------
# Python reference implementation
# ---------------------------------------------------------------------------

def py_pseudo_laplacian(nodes, n):
    """Pseudo-Laplacian from Ball Mapper node membership.

    For each ball (list of member point indices):
      - Increment L[i, i] by 1 for every member i  (diagonal: how many balls
        contain point i)
      - Decrement L[i, j] by 1 for every pair i≠j  (off-diagonal: negated
        count of balls containing both i and j)

    This mirrors the Rust implementation exactly.
    """
    L = np.zeros((n, n), dtype=np.int64)
    for members in nodes:
        for i in members:
            for j in members:
                if i == j:
                    L[i, j] += 1
                else:
                    L[i, j] -= 1
    return L


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_matches_reference_simple():
    """Smallest non-trivial case: two overlapping balls."""
    nodes = [[0, 1, 2], [1, 2, 3]]
    n = 4
    expected = py_pseudo_laplacian(nodes, n)
    actual = np.array(pseudo_laplacian(nodes, n))
    np.testing.assert_array_equal(actual, expected,
                                  err_msg="Rust output differs from Python reference")


def test_matches_reference_disjoint_balls():
    """Two disjoint balls: cross-ball entries should be zero."""
    nodes = [[0, 1], [2, 3]]
    n = 4
    expected = py_pseudo_laplacian(nodes, n)
    actual = np.array(pseudo_laplacian(nodes, n))
    np.testing.assert_array_equal(actual, expected)


def test_matches_reference_single_ball():
    nodes = [[0, 1, 2, 3]]
    n = 4
    expected = py_pseudo_laplacian(nodes, n)
    actual = np.array(pseudo_laplacian(nodes, n))
    np.testing.assert_array_equal(actual, expected)


def test_matches_reference_many_balls():
    """Larger random case: 8 points, 4 overlapping balls."""
    nodes = [[0, 1, 2], [1, 2, 3], [4, 5], [5, 6, 7]]
    n = 8
    expected = py_pseudo_laplacian(nodes, n)
    actual = np.array(pseudo_laplacian(nodes, n))
    np.testing.assert_array_equal(actual, expected)


def test_matches_reference_empty_nodes():
    n = 5
    expected = py_pseudo_laplacian([], n)
    actual = np.array(pseudo_laplacian([], n))
    np.testing.assert_array_equal(actual, expected)


def test_symmetry():
    """The pseudo-Laplacian is always symmetric: L[i,j] == L[j,i]."""
    nodes = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    n = 5
    actual = np.array(pseudo_laplacian(nodes, n))
    np.testing.assert_array_equal(actual, actual.T,
                                  err_msg="pseudo-Laplacian must be symmetric")


def test_diagonal_is_membership_count():
    """L[i, i] must equal the number of balls that contain point i."""
    nodes = [[0, 1, 2], [0, 3], [0, 1]]
    n = 4
    actual = np.array(pseudo_laplacian(nodes, n))

    # Point 0 is in all three balls → L[0,0] = 3
    assert actual[0, 0] == 3, f"L[0,0] = {actual[0,0]}, expected 3"
    # Points 1 is in balls 0 and 2 → L[1,1] = 2
    assert actual[1, 1] == 2, f"L[1,1] = {actual[1,1]}, expected 2"
    # Point 3 is only in ball 1 → L[3,3] = 1
    assert actual[3, 3] == 1, f"L[3,3] = {actual[3,3]}, expected 1"


def test_offdiagonal_is_negative_shared_count():
    """L[i,j] (i≠j) must equal the negative number of balls containing both i and j."""
    nodes = [[0, 1, 2], [0, 1, 3]]  # points 0 and 1 share 2 balls
    n = 4
    actual = np.array(pseudo_laplacian(nodes, n))

    assert actual[0, 1] == -2, f"L[0,1] = {actual[0,1]}, expected -2"
    assert actual[1, 0] == -2, f"L[1,0] = {actual[1,0]}, expected -2"
    # Point 2 and 3 share no ball
    assert actual[2, 3] == 0, f"L[2,3] = {actual[2,3]}, expected 0"


def test_accumulation_matches_summed_reference():
    """Accumulating pseudo-Laplacians across multiple BallMappers must equal
    the reference sum."""
    nodes_list = [
        [[0, 1], [1, 2]],
        [[0, 2], [2, 3]],
        [[1, 3]],
    ]
    n = 4
    galactic_L = np.zeros((n, n), dtype=np.int64)
    for nodes in nodes_list:
        galactic_L += np.array(pseudo_laplacian(nodes, n))

    expected = sum(py_pseudo_laplacian(nodes, n) for nodes in nodes_list)
    np.testing.assert_array_equal(galactic_L, expected)
