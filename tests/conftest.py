import pytest
import numpy as np

@pytest.fixture(params=[2,4,6])
def clustering_data(request):

    nclusters = request.param
    r = 5
    ang = 2*np.pi/np.arange(1,nclusters+1)[:,None]
    mean = r*np.hstack((np.cos(ang),np.sin(ang)))

    ndata = 10
    X = np.random.standard_normal((ndata*nclusters,2)) + np.repeat(mean,ndata, axis=0)
    # Xt = np.random.standard_normal((ndata*nclusters,2)) + np.repeat(mean,ndata, axis=0)
    inds = np.repeat(np.arange(nclusters)[:,None],ndata)
    # inds_t = np.repeat(np.arange(nclusters)[:,None],ndata)

    return nclusters, (X, inds),


@pytest.fixture(params=[2,4,6])
def regression_clustering_data(request):

    nclusters = request.param
    
    beta = 10*np.random.randn(nclusters,2)
    ndata = 10
    x = np.linspace(-1,1,ndata)
    X = np.power(x[:,None],np.array([0,1])[None,:])

    y = np.dot(X,beta.T).reshape(ndata*nclusters,) + 0.05*np.random.randn(ndata*nclusters)
    x = x.repeat(nclusters).reshape(ndata*nclusters)

    inds = np.repeat(np.arange(nclusters)[None,:],ndata,axis=0).reshape((ndata*nclusters,))

    return nclusters, (x, y, inds),
