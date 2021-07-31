from numpy import linalg
from numpy.core.numeric import isscalar
from numpy.linalg.linalg import cholesky
import pytest

from boaf.base_distributions.multivariate import NIW

import numpy as np
np.random.seed(1)

def test_NIW_init():

    mu = np.array([0,0])
    nu = 2
    kappa = 1
    S = 4*np.eye(2)

    C = NIW(mu, nu, kappa, S)

    assert( np.all(C.mu == mu ))
    assert( C.nu == nu)
    assert( C.kappa == kappa )
    assert( np.all(C.sigma == S))

def test_NIW_update_downdate():

    mu = np.array([0,0])
    nu = 2
    kappa = 1
    S = 4*np.eye(2)

    C = NIW(mu, nu, kappa, S)

    x = np.array([1,1])

    C.add_one(x)
    C.rem_one(x)

    X = np.random.standard_normal((20,2))

    C.add_data(X)
    C.rem_data(X)

    assert( np.allclose(C.mu,mu))
    assert( C.nu == nu)
    assert( C.kappa == kappa )
    assert( np.allclose(C.sigma,S))

def test_NIW_predloglik():

    mu = np.array([0,0])
    nu = 2
    kappa = 1
    S = 4*np.eye(2)

    C = NIW(mu, nu, kappa, S)

    ll_mu = C.logpredpdf(mu)

    x = np.array([1,1])
    ll = C.logpredpdf(x)

    assert(np.isscalar(ll))
    assert(ll < ll_mu)

    X = np.random.standard_normal((20,2))
    ll = C.logpredpdf(X)

    assert(np.all(ll.shape == (20,)))
    assert(np.all(ll < 0 ))
    assert(np.all(ll < ll_mu))





