from numpy import linalg
from numpy.core.numeric import isscalar
from numpy.linalg.linalg import cholesky
import pytest
from boaf.base_distributions.base import BaseDistribution

from boaf.base_distributions.multivariate import NIW, BLR

import numpy as np
np.random.seed(1)

def test_base():

    BaseDistribution.__abstractmethods__ = set()
    D = BaseDistribution()

    with pytest.raises(NotImplementedError):
        D.logpredpdf(None)

def test_NIW_init():

    mu = np.array([[0,0]])
    nu = 2
    kappa = 1
    S = 4*np.eye(2)

    C = NIW(mu, nu, kappa, S)

    assert( np.all(C.mu == mu ))
    assert( C.nu == nu)
    assert( C.kappa == kappa )
    assert( np.all(C.sigma == S))

def test_NIW_update_downdate():

    mu = np.array([0,0])[None,:]
    nu = 2
    kappa = 1
    S = 4*np.eye(2)

    C = NIW(mu, nu, kappa, S)

    x = np.array([1,1])[None,:]
    C2 = NIW(mu, nu, kappa, S, x)

    C.add_one(x)
    
    assert( np.allclose(C.mu,C2.mu))
    assert( C.nu == C2.nu)
    assert( C.kappa == C2.kappa )
    assert( np.allclose(C.sigma,C2.sigma))

    C.rem_one(x)

    X = np.random.standard_normal((20,2))


    C.add_data(X)
    C.rem_data(X)

    assert( np.allclose(C.mu,mu))
    assert( C.nu == nu)
    assert( C.kappa == kappa )
    assert( np.allclose(C.sigma,S))

def test_NIW_predloglik():

    mu = np.array([[0,0]])
    nu = 2
    kappa = 1
    S = 4*np.eye(2)

    C = NIW(mu, nu, kappa, S)

    ll_mu = C.logpredpdf(mu)

    x = np.array([[1,1]])
    ll = C.logpredpdf(x)

    assert(ll.shape[0] == 1)
    assert(ll < ll_mu)

    X = np.random.standard_normal((20,2))
    ll = C.logpredpdf(X)

    assert(np.all(ll.shape == (20,)))
    assert(np.all(ll < 0 ))
    assert(np.all(ll < ll_mu))


def test_BLR_init():

    m0 = np.array([0,0])
    L0 = 4*np.eye(2)
    a0 = 2
    b0 = 2

    C = BLR(m0, L0, a0, b0)

    assert( np.all(C.mu == m0 ))
    assert( np.all(C.m0 == m0 ))
    assert( C.an == a0)
    assert( C.bn == b0)
    assert( C.b0 == b0)
    assert( np.all(C.L0 == L0))
    assert( np.all(C.Lam == L0))
    

def test_BLR_update_downdate():

    m0 = np.array([0,0])
    L0 = 4*np.eye(2)
    a0 = 2
    b0 = 2

    C = BLR(m0, L0, a0, b0)


    data = (np.array([1,1]), 1)
    C2 = BLR(m0, L0, a0, b0, data)

    C.add_one(data)
    
    assert( np.allclose(C.mu,C2.mu))
    assert( C.an == C2.an)
    assert( C.bn == C2.bn)
    assert( np.allclose(C.Lam,C2.Lam))

    C.rem_one(data)

    assert( np.allclose(C.mu,m0))
    assert( C.an == a0)
    assert( C.bn == b0)
    assert( np.allclose(C.Lam,L0))

    C.add_data(data)
    C.rem_data(data)

    assert( np.allclose(C.mu,m0))
    assert( C.an == a0)
    assert( C.bn == b0)
    assert( np.allclose(C.Lam,L0))

    data = (np.random.randn(10,2), np.random.randn(10,))

    C.add_data(data)
    C.rem_data(data)

    assert( np.allclose(C.mu,m0))
    assert( np.allclose(C.an,a0))
    assert( np.allclose(C.bn,b0))
    assert( np.allclose(C.Lam,L0))

def test_BLR_predloglik():

    beta = np.array([2,-3])
    m0 = beta
    L0 = 4*np.eye(2)
    a0 = 2
    b0 = 2

    C = BLR(m0, L0, a0, b0)

    x = np.linspace(-1,1,10)
    X = np.power(x[:,None],np.array([0,1]))
    y = np.dot(X,beta) + 0.01*np.random.standard_normal((10,))

    ll_mu = C.logpredpdf((X[0,:],y[0]))

    ll = C.logpredpdf((np.random.standard_normal((1,2)),np.random.standard_normal((1,))))

    assert(ll.shape[0] == 1)
    assert(ll < ll_mu)

    ll = C.logpredpdf((X,np.random.randn(10,)))

    assert(np.all(ll.shape == (10,)))
    assert(np.all(ll < 0 ))
    assert(np.all(ll < ll_mu))

    yp, Sp, nu = C.posterior_predictive(X)
    assert(np.allclose(yp,y,rtol=0.1))
    assert(np.all(np.diag(Sp*nu/(nu-2))>0))
    assert(nu>0)
