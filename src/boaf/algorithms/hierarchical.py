import numpy as np
from numpy.typing import NDArray
import warnings

from scipy.spatial.distance import pdist

from .base import Cluster
from ..utils import (
    cond_ind,
    col_cond_ind,
    ind_from_condensed,
    col_from_condensed,
    binary_tree_DFS,
)


class Hierarchical(Cluster):
    """Provide Hierarchical Agglomerative Clustering

    Implemented are currently a naive O(N^3) algorithm for HAG and O(N^2) for single linkage

    Theory for this has been used from https://arxiv.org/pdf/1109.2378.pdf, except UnionFind
    from https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    """

    def __init__(self, opts: dict) -> None:
        """Initialise HAG

        Args:
            opts (dict): Dictionary of options with args:
                linkage: String indicating linkage function see below;
                algorithm: Compute algorithm (primitive or mst_single)
        """
        super().__init__(opts)

    def learn(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Learn HAG clustering

        For the HAG the learning process is to construct the dendrogram which provides
        the cluster linkages.

        Args:
            data (NDArray[np.float64]): An array of training data size (N,D) with N observations in
                D dimensions.

        Returns:
            NDArray[np.float64]: Stepwise dendrogram, an (N-1, 3) array.
        """

        # Available algorithm selector
        algorithms = {
            "primitive": self._primitive,
            "mst_single": self._mst_single_linkage,
        }

        # First comput dissimilarity - TODO replace with direct dissimilarity input option
        dissimilarity = pdist(data)

        # Call HAG
        alg = algorithms[self.opts["algorithm"]]
        return alg(np.arange(data.shape[0]), dissimilarity)

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.int16]:
        """Predict clusters

        Not yet implemented...

        Args:
            data (NDArray[np.float64]): Null

        Returns:
            NDArray[np.int16]: Null
        """
        return super().predict(data)

    def linkage(self, *args):
        """Linkage Functions

        Args:
            d_ax (NDArray[np.float64]): distances from point a to all others
            d_ax (NDArray[np.float64]): distances from point b to all others
            d_ab (float): distance from point a to b
            s_a (int): size of subtree a
            s_b (int): size of subtree b
            sizes (NDArray[np.int16]): sizes of all other subtrees

        Returns:
            NDArray[np.float64]: linkages of new point to all remaining in S \ {a,b}

        """

        def single(args):
            # Single linkage
            return np.minimum(args[0], args[1])

        def complete(args):
            # Complete linkage
            return np.maximum(args[0], args[1])

        def average(args):
            # Average linkage
            return (args[3] * args[0] + args[4] * args[1]) / (args[3] + args[4])

        def weighted(args):
            # Weighted linkage
            return (args[0] + args[1]) / 2

        def ward(args):
            # Ward linkage
            return np.sqrt(
                (
                    (args[3] + args[5]) * args[0]
                    + (args[4] + args[5]) * args[1]
                    - args[5] * args[2]
                )
                / (args[3] + args[4] + args[5])
            )

        def centroid(args):
            # Centroid linkage
            return np.sqrt(
                (args[3] * args[0] + args[4] * args[1]) / (args[3] + args[4])
                - (args[3] * args[4] * args[2]) / (args[3] + args[4]) ** 2
            )

        def median(args):
            # Median Linkage
            return np.sqrt(0.5 * args[0] + 0.5 * args[1] - 0.25 * args[2])

        linkages = {
            "single": single(args),
            "complete": complete(args),
            "average": average(args),
            "weighted": weighted(args),
            "ward": ward(args),
            "centroid": centroid(args),
            "median": median(args),
        }

        return linkages[self.opts["linkage"]]

    def _primitive(
        self, y: NDArray[np.int16], dis: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Primitive HAG clustering

        O(N^3) algorithm as found in Figure 1 of https://arxiv.org/pdf/1109.2378.pdf

        Args:
            y (NDArray[np.int16]): data labels array size ( N, )
            dis (NDArray[np.float64]): dissimilarity matrix in condensed form ( N * (N-1) // 2, )

        Returns:
            NDArray[np.float64]: stepwise dendrogram as ( N-1, 3 ) array
        """

        N = y.shape[0]
        NN = N
        sizes = np.ones((N,), dtype=np.int16)
        L = np.empty((N - 1, 3))

        for k in range(N - 1):

            # (a,b) <- argmin_{S x S \ del} dis
            idx = np.argmin(dis)
            # Get directly from condensed form matrix
            a, b = ind_from_condensed(idx, N)

            # Append to L
            d_ab = dis[idx]
            L[k, :] = [y[a], y[b], d_ab]

            # Remove a,b from S
            mask = np.ones_like(y)
            mask[[a, b]] = False
            y = y[mask == True]

            # Update d
            d_n = self.linkage(
                col_from_condensed(dis, a, N)[mask == True],
                col_from_condensed(dis, b, N)[mask == True],
                d_ab,
                sizes[a],
                sizes[b],
                sizes[mask == True],
            )

            dis = self.rem_and_update_dissimilarity(dis, d_n, a, b, N)

            # Size of pseudonode n
            sn = np.sum(sizes[[a, b]])[None]
            sizes = sizes[mask == True]
            sizes = np.concatenate((sizes, sn), axis=0)

            if k < NN - 2:
                # Update current set
                y = np.concatenate((y, np.array([NN + k])))
                # Decrement set size
                N -= 1

        # idx = binary_tree_DFS(L, 2*(NN-1), NN)
        return L

    def _mst_single_linkage(
        self, y: NDArray[np.int16], dis: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """_summary_

        O(N^2) algorithm as found in Figure 6 of https://arxiv.org/pdf/1109.2378.pdf
        Only valid for single linkage.

        Args:
            y (NDArray[np.int16]): data labels array size ( N, )
            dis (NDArray[np.float64]): dissimilarity matrix in condensed form ( N * (N-1) // 2, )

        Returns:
            NDArray[np.float64]: stepwise dendrogram as ( N-1, 3 ) array
        """

        if self.opts["linkage"] != "single":
            warnings.warn(
                "MST Single Linkage Algorithm is only compatible with single linkage type HAG, ignoring linkage option",
                RuntimeWarning,
            )

        N = y.shape[0]
        NN = N
        L = np.empty((N - 1, 3))

        # ==== MST-LINKAGE-CORE ====
        c = 0  # doesn't matter where we start, first element as good as any

        # distances
        d_i = np.ones_like(y) * np.inf
        mask = np.ones_like(y)

        # Find edges of minimal spanning tree
        for k in np.arange(1, NN):

            # Keep track of what nodes are still in play
            mask[c] = False
            y_in = y[mask == True]

            # Update single linkage distances
            d_i = np.minimum(d_i, col_from_condensed(dis, c, N))

            # Find next nearest neighbour, update L and c
            n = np.argmin(d_i[mask == True])
            L[k - 1, :] = np.array([c, y_in[n], d_i[y_in[n]]])
            c = y_in[n]

        # ==== END MST-LINKAGE-CORE ====

        # Tidying up, in fig 6. this is lines 3,4 or MST-Linkage(S,d)
        idx = np.argsort(L[:, 2])
        Lprime = self._label_mst(L[idx, :])

        return Lprime

    @staticmethod
    def _label_mst(L: NDArray[np.float64]) -> NDArray[np.float64]:
        """Label sorted dendrogram

        Efficient algorithms return set of edges and we can provide a sorted dendrogram
        this, however, doesn't define the tree structure by grouping those sets of edges
        between sequential nearest neighbours. This requires a Union-Find on the edges in
        a similar way to Kruskal's algorithm.

        Args:
            L (NDArray[np.float64]): sorted but not stepwise dendrogram (N-1,3)

        Returns:
            NDArray[np.float64]: stepwise dendrogram (N-1,3)
        """

        # This is effectively Figure 3: NN-CHAIN-LINKAGE in reference
        labeller = UnionFind(L.shape[0] + 1)
        Lprime = np.empty_like(L)

        for k, row in enumerate(L):
            Lprime[k, :] = [
                labeller.find(int(row[0])),
                labeller.find(int(row[1])),
                row[2],
            ]
            labeller.union(int(Lprime[k, 0]), int(Lprime[k, 1]))

        return Lprime

    @staticmethod
    def rem_and_update_dissimilarity(
        dis: NDArray[np.float64], d_n: NDArray[np.float64], a: int, b: int, N: int
    ) -> NDArray[np.float64]:
        """Update condensed dissimilarities

        When we make a step in the dendrogram we remove the nodes a and b and add
        the new node n. Rather than recomupting all the dissimilarities we will instead
        update the dissimilarity matrix. Since this matrix is stored in condensed form
        that task isn't as trivial as it may at first appear.

        Args:
            dis (NDArray[np.float64]): existing condensed dissimilarities
            d_n (NDArray[np.float64]): dissimilarities from S \ {a,b} to n
            a (int): first node to remove
            b (int): second node to remove
            N (int): size of full dis matrix

        Returns:
            NDArray[np.float64]: updated condensed dissimilarity matrix
        """

        # Remove d(a,.) and d(b,.) from dis
        mask = np.ones_like(dis)
        for idx in [a, b]:
            tmask = np.ones_like(dis)
            tmask[np.concatenate(col_cond_ind(idx, N))] = False
            mask = np.logical_and(mask, tmask)

        # Get new indices
        new_points = cond_ind(np.arange(N - 2), N - 1, N - 1) - 1
        # Empty array for new dissimilarities
        new_dis = np.empty(((N - 1) * (N - 2) // 2,))
        # Which points carry from dis to new_dis
        carry_mask = np.ones_like(new_dis)
        carry_mask[new_points] = False
        # Carry old untouched dissimilarites
        new_dis[carry_mask == True] = dis[mask == True]
        # Add in new dissimilarities to n
        new_dis[carry_mask == False] = d_n

        return new_dis


class UnionFind:
    def __init__(self, N: int) -> None:
        """Union Find for non-stepwise dendrogram

        Args:
            N (int): number of leaf nodes in dendrogram
        """
        self.parent = np.arange(2 * N - 1)
        self.next_label = N

    def union(self, m: int, n: int):
        """Union nodes m and n

        Args:
            m (int): node index
            n (int): node index
        """
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.next_label += 1

    def find(self, n: int) -> int:
        """Find parent node of n

        Args:
            n (int): node index

        Returns:
            int: parent node index
        """
        while self.parent[n] != n:
            n, self.parent[n] = self.parent[n], self.parent[self.parent[n]]
        return n
