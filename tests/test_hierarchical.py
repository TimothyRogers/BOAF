import pytest

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

np.random.seed(1)


from boaf.algorithms import hierarchical
from boaf.utils import NoPredictException, binary_tree_DFS

from matplotlib import pyplot as plt


def test_no_predict():

    opts = {"nclusters": 1, "linkage": "single", "algorithm": "primitive"}

    model = hierarchical.Hierarchical(opts)
    with pytest.raises(NoPredictException):
        model.predict(None)


def test_rem_del_dis():

    X = np.random.standard_normal((20, 2))
    N = X.shape[0]

    # Pairwise distances
    dis = pdist(X)
    D = squareform(dis)

    # Points to remove
    a = 10
    b = 14

    # New point is mean for arguments sake to make distances easier
    x_new = np.mean(X[[a, b], :], axis=0)[None, :]
    mask = np.ones((N,))
    mask[[a, b]] = False
    Xprime = np.vstack((X[mask == True, :], x_new))

    # Recompute all distances as check
    dis_new = pdist(Xprime)
    Dnew = squareform(dis_new)
    # Grab new distances
    d_n = Dnew[:-1, -1]

    # Actual test
    dis_test = hierarchical.Hierarchical.rem_and_update_dissimilarity(dis, d_n, a, b, N)

    assert np.all(dis_new == dis_test)
    assert np.all(squareform(dis_new) == Dnew)


@pytest.mark.usefixtures("clustering_data")
def test_primitive(clustering_data):
    nclusters, (X, inds) = clustering_data

    opts = {"nclusters": nclusters, "linkage": "single", "algorithm": "primitive"}

    model = hierarchical.Hierarchical(opts)
    L = model.learn(X)

    N = X.shape[0]
    y = np.empty((N,))
    y[binary_tree_DFS(L, int(L[-1, 0]), N)] = 1
    y[binary_tree_DFS(L, int(L[-1, 1]), N)] = 2
    assert np.all(np.logical_or(y == 1, y == 2))


@pytest.mark.usefixtures("clustering_data")
def test_mst_single(clustering_data):
    nclusters, (X, inds) = clustering_data

    opts = {"nclusters": nclusters, "linkage": "single", "algorithm": "mst_single"}

    model = hierarchical.Hierarchical(opts)
    L = model.learn(X)

    N = X.shape[0]
    y = np.empty((N,))
    y[binary_tree_DFS(L, int(L[-1, 0]), N)] = 1
    y[binary_tree_DFS(L, int(L[-1, 1]), N)] = 2
    assert np.all(np.logical_or(y == 1, y == 2))


@pytest.mark.parametrize(
    "link_fun",
    ["single", "complete", "average", "weighted", "ward", "centroid", "median"],
)
def test_linkages(link_fun):

    nclusters = 2
    r = 100  # Move clusters comically far apart so last link always splits
    ang = 2 * np.pi / np.arange(1, nclusters + 1)[:, None]
    mean = r * np.hstack((np.cos(ang), np.sin(ang)))

    ndata = 10
    X = np.random.standard_normal((ndata * nclusters, 2)) + np.repeat(
        mean, ndata, axis=0
    )

    opts = {"nclusters": nclusters, "linkage": link_fun, "algorithm": "primitive"}

    model = hierarchical.Hierarchical(opts)
    L = model.learn(X)

    assert np.all(np.sort(binary_tree_DFS(L, 38, 20)) == np.arange(20))
    for i in range(2):
        assert len(binary_tree_DFS(L, int(L[-1, i]), 20)) == 10

    y = np.empty((X.shape[0],))
    y[binary_tree_DFS(L, int(L[-1, 0]), 20)] = 1
    y[binary_tree_DFS(L, int(L[-1, 1]), 20)] = 2
    assert np.all(np.logical_or(y == 1, y == 2))

    if link_fun != "single":
        with pytest.warns():
            opts = {
                "nclusters": nclusters,
                "linkage": link_fun,
                "algorithm": "mst_single",
            }
            model = hierarchical.Hierarchical(opts)
            L = model.learn(X)
