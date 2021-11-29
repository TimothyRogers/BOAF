import pytest

import numpy as np
np.random.seed(1)

from boaf.algorithms.kmeans import KMeans

@pytest.mark.usefixtures("clustering_data")
def test_kmeans(clustering_data):
    nclusters, (X, inds) = clustering_data

    opts = {
        'nclusters': nclusters,
        'niters': 200
    }
    model = KMeans(opts)
    model.learn(X)
    inds_pred = model.predict(X)

    assert(np.all(inds_pred < nclusters))
    assert(np.all(inds.shape == inds_pred.shape))

@pytest.mark.usefixtures("clustering_data")
def test_kmeans_pp(clustering_data):
    nclusters, (X, inds) = clustering_data

    opts = {
        'nclusters': nclusters,
        'niters': 200,
        'init': 'kpp'
    }
    model = KMeans(opts)
    model.learn(X)
    inds_pred = model.predict(X)

    assert(np.all(inds_pred < nclusters))
    assert(np.all(inds.shape == inds_pred.shape))

@pytest.mark.usefixtures("clustering_data")
def test_kmeans_rand(clustering_data):
    nclusters, (X, inds) = clustering_data

    opts = {
        'nclusters': nclusters,
        'niters': 200,
        'init': 'rand'
    }
    model = KMeans(opts)
    model.learn(X)
    inds_pred = model.predict(X)

    opts = {
        'nclusters': nclusters,
        'niters': 200,
        'init': 'random'
    }
    model = KMeans(opts)
    model.learn(X)
    inds_pred = model.predict(X)

    assert(np.all(inds_pred < nclusters))
    assert(np.all(inds.shape == inds_pred.shape))

