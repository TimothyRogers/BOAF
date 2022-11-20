import pytest

import numpy as np
from scipy.spatial.distance import pdist, squareform
from boaf.utils import ind_from_condensed, col_from_condensed, binary_tree_DFS

np.random.seed(1)


def test_ind_from_condensed():

    N = 4
    condensed = pdist(np.arange(N)[:, None])
    A = squareform(condensed)

    for idx in range(condensed.shape[0]):
        i, j = ind_from_condensed(idx, N)
        assert A[i, j] == condensed[idx]


def test_col_from_condensed():

    N = 5
    condensed = pdist(np.random.standard_normal((N, 2)))
    A = squareform(condensed)

    for j in range(N):
        assert np.all(A[:, j] == col_from_condensed(condensed, j, N))


def test_binary_tree_DFS():

    N = 4  # 3 leaves

    # Draw the tree defined by L below
    #        6
    #       / \
    #      5   \
    #     / \   \
    #    4   \   \
    #   / \   \   \
    #  0   1   2    3

    L = np.array([[0, 1, 1], [4, 2, 1], [5, 3, 1]])

    # Full preorder
    preorder_correct = [6, 5, 4, 0, 1, 2, 3]
    # All leaves for leaves only
    all_leaves = [0, 1, 2, 3]

    # Full tree traversal
    idx = binary_tree_DFS(L, 6, 4, leaves_only=False)
    assert np.all(preorder_correct == idx)

    # Only leaves full tree
    idx = binary_tree_DFS(L, 6, 4, leaves_only=True)
    assert np.all(idx == binary_tree_DFS(L, 6, 4))
    assert np.all(idx == all_leaves)

    # Partial tree traversal
    preorder_correct = [5, 4, 0, 1, 2]
    idx = binary_tree_DFS(L, 5, 4, leaves_only=False)
    assert np.all(preorder_correct == idx)
    all_leaves = [0, 1, 2]
    idx = binary_tree_DFS(L, 5, 4)
    assert np.all(all_leaves == idx)
