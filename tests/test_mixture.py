import pytest

import numpy as np
np.random.seed(1)

from boaf.algorithms.mixture import FunctionalMixture, MixtureModel, GMM
from boaf.base_distributions.multivariate import NIW, BLR

@pytest.mark.usefixtures("clustering_data")
def test_mixture(clustering_data):

    nclusters, (X, inds) = clustering_data

    # Options as nested dict allows future JSON options
    opts = {
        'nclusters': nclusters,
        'niters': 200,
        'prior':{
            'alpha': np.ones((nclusters))/nclusters,
            'cluster':{
                'nu':2,
                'kappa':1,
                'mu':np.zeros((1,2)),
                'sigma':np.eye(2)
            }
        }
    }
    model = MixtureModel(opts, base_distribution=NIW)
    model.learn(X)
    inds_pred = model.predict(X)

    assert(np.all(inds_pred < nclusters))
    assert(np.all(inds.shape == inds_pred.shape))

    opts['init'] = 'kpp'
    model = MixtureModel(opts, base_distribution=NIW)
    model.learn(X)
    inds_pred = model.predict(X)

    assert(np.all(inds_pred < nclusters))
    assert(np.all(inds.shape == inds_pred.shape))
    
    opts['init'] = 'rand'
    model = MixtureModel(opts, base_distribution=NIW)
    model.learn(X)
    inds_pred = model.predict(X)

    assert(np.all(inds_pred < nclusters))
    assert(np.all(inds.shape == inds_pred.shape))



@pytest.mark.usefixtures("clustering_data")
def test_GMM(clustering_data):

    nclusters, (X, inds) = clustering_data

    # Options as nested dict allows future JSON options
    opts = {
        'nclusters': nclusters,
        'niters': 200,
        'prior':{
            'alpha': np.ones((nclusters))/nclusters,
            'cluster':{
                'nu':2,
                'kappa':1,
                'mu':np.zeros((1,2)),
                'sigma':np.eye(2)
            }
        }
    }
    model = GMM(opts)
    model.learn(X)
    inds_pred = model.predict(X)
    print(inds_pred)

    assert(np.all(inds_pred < nclusters))
    assert(np.all(inds.shape == inds_pred.shape))

@pytest.mark.usefixtures("regression_clustering_data")
def test_BLR(regression_clustering_data):

    nclusters, (x, y, inds) = regression_clustering_data
    X = np.power(x[:,None],np.array([0,1])[None,:]) # Mixture of linear regressions

    # Options as nested dict allows future JSON options
    opts = {
        'nclusters': nclusters,
        'niters': 200,
        'prior':{
            'alpha': np.ones((nclusters))/nclusters,
            'cluster':{
                'm0':np.array([0,0]),
                'b0':1,
                'a0':2,
                'L0':(1e-4**2)*np.eye(2)
            }
        }
    }
    model = FunctionalMixture(opts, base_distribution=BLR)
    model.learn((X,y))
    inds_pred = model.predict((X,y))
    print(inds_pred)

    assert(np.all(inds_pred < nclusters))
    assert(np.all(inds.shape == inds_pred.shape))