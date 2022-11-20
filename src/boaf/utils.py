import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, List


def cond_ind(
    i: Union[int, NDArray], j: Union[int, NDArray], N: int
) -> NDArray[np.int32]:
    """Indices from condensed distance array

    Args:
        i (Union[int, NDArray]): row index/indices
        j (Union[int, NDArray]): column index/indices
        N (int): full square matrix size N x N

    Returns:
        NDArray[np.int32]: index/indices in condensed array
    """

    return np.array(N * i + j - ((i + 1) * (i + 2)) // 2, dtype=np.int32)


def col_cond_ind(
    j:int, N: int
) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Indices of matrix column in condensed distance array

    Since only the upper triangular (off diagonal) matrix is stored, the column indices rely on the 
    symmetry of the distance matrix. Therefore, we return indices for the top part of the column, 
    i.e. from the upper triangular matrix and then bottom indices from the lower triangular matrix.

    To be fully explicit, for an N x N matrix A, idx_top would contain indices to retrieve A[:j,j]
    and idx_bot would contain indices for A[j+1:,j].

    Args:
        j (int): column index/indices
        N (int): full square matrix size N x N

    Returns:
        Tuple[NDArray[np.int32], NDArray[np.int32]]: indices above diagonal, indices below diagonal
    """
    idx_top = cond_ind(np.arange(j), j, N)
    idx_bot = cond_ind(j, np.arange(j + 1, N), N)
    return idx_top, idx_bot


def ind_from_condensed(idx: int, N: int) -> Tuple[int, int]:
    """Matrix indices from condensed matrix index

    Args:
        idx (int): index in condensed matrix list
        N (int): full square matrix size N x N

    Returns:
        Tuple[int, int]: row and column index in full matrix
    """
    i = int(np.ceil(0.5 * (2 * N - 1 - np.sqrt(-8 * idx + 4 * N**2 - 4 * N - 7)) - 1))
    j = int(N - ((i + 1) * (N - 1 - (i + 1)) + ((i + 1) * (i + 2)) // 2) + idx)
    return i, j


def col_from_condensed(
    condensed: NDArray[np.float64], j: int, N: int
) -> NDArray[np.float64]:
    """Recover column of distance matrix from condensed form

    Using the column indices as described above we can return one column of the full square
    distance matrix from the condensed matrix form.

    Args:
        condensed (NDArray[np.float64]): condensed distance matrix
        j (int): column index
        N (int): full square matrix size N x N

    Returns:
        NDArray[np.float64]: column of distance matrix containing distances to item j
    """

    idx_top, idx_bot = col_cond_ind(j, N)
    col = np.concatenate(
        (condensed[idx_top], np.array([0]), condensed[idx_bot]), axis=0
    )

    return col


def binary_tree_DFS(
    L: NDArray[np.float64], ind: int, N: int, leaves_only: bool = True
) -> List[int]:
    """Depth First Serach of Binary Tree

    Perform a depth first search of a binary tree represented in the standard dendrogram form.
    Return either all nodes below a stated one on the tree or only all leaf nodes below that given
    node.

    Args:
        L (NDArray[np.float64]): stepwise dendrogram matrix
        ind (int): index of root node to search from
        N (int): number of leaf nodes
        leaves_only (bool, optional): return only leaf node indices? Defaults to True.

    Returns:
        List[int]: list of nodes traversed
    """
    def preorder_traversal(node: int) -> int:
        """Recursive preorder traversal

        Args:
            node (int): current root node index

        Yields:
            int: root node, then left node, then right node
        """
        if leaves_only and node < N:
            yield node
        elif not leaves_only:
            yield node
        if node - N >= 0:
            yield from preorder_traversal(int(L[node - N, 0]))
            yield from preorder_traversal(int(L[node - N, 1]))

    idx = [p for p in preorder_traversal(ind)]

    return idx


class NoPredictException(Exception):
    """Exception when Clustering Cannot Predict"""

    def __str__(self):
        return "This clustering approach has no predict method"
