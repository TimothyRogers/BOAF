from typing import NoReturn
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln

from .base import BaseDistribution
class NIW(BaseDistribution):
    '''
    A Normal Inverse Wishart Distribution
    mu, Sig ~ NIW(m0,k0,n0,S0)
    mu ~ N(m0,Sig/k0)
    Sig ~ IW(S0,n0)
    '''

    def __init__(
        self,
        mu,
        nu,
        kappa,
        sigma, 
        data=None,
        weight=None
        ) -> None:
        '''
        Initialise a NIW prior object
        '''
        
        self.mu = mu.copy()
        self.nu = nu
        self.kappa = kappa
        self.sigma = sigma.copy()
        self.N = 0 # No data yet
        self.D = mu.shape[1]

        if data is not None:
            self.add_data(data, weight=weight)

    def add_one(self, data, weight=1):
        '''
        Add a single datum to the distribution
        ''' 

        # Update scalar parameters
        self.N += weight
        self.kappa += weight
        self.nu += weight

        # Update mean
        self.mu = ((self.kappa - weight)*self.mu + data * weight)/self.kappa

        # Update Sigma (TODO change to rank 1)
        res = data - self.mu
        self.sigma += weight*(self.kappa/(self.kappa - weight)*np.outer(res,res))

    def rem_one(self, data, weight=1):
        '''
        Remove a single datum to the distribution
        ''' 

        # Downdate covariance
        res = (data-self.mu)
        self.sigma -= weight*(self.kappa/(self.kappa-weight)*np.outer(res,res))

        # Downdate scalar parameters
        self.N -= weight
        self.kappa -= weight
        self.nu -= weight

        # Downdate mean
        self.mu = ((self.kappa + weight)*self.mu - data * weight)/self.kappa

    def map_estimates(self):
        return self.mu, self.sigma/(self.nu+self.D+2)

    def logpredpdf(self, X):
        '''
        Log predictive pdf is multivariate T
        p( x* | X ) = T( m_n, (k_n+1)/(k_n*nu_prime)*S_n, nu_prime)
        nu_n' = nu_n - D + 1
        Murphy PML 2021 Eq 7.144 pp. 199  
        '''
        
        D = X.shape[-1]
        nu_p = self.nu - D + 1
        S = self.sigma * (self.kappa + 1)/(self.kappa * nu_p)      
        L = np.linalg.cholesky(S)
        res = X - self.mu # rely on broadcasting
        QU = solve_triangular(L, res.T, trans='T', lower=True)

        ll = gammaln((nu_p + D)/2) - \
             gammaln((nu_p)/2) - \
             D/2*np.log(nu_p) -\
             D/2*np.log(np.pi) -\
             np.sum(np.log(np.diag(L))) -\
             (nu_p+D)/2 * (np.log(1 + np.sum(QU.T**2,axis=-1)/nu_p))

        return ll

