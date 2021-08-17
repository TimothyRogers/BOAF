import pytest

import numpy as np
np.random.seed(1)

from boaf.kmeans import KMeans

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
