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
        data=None
        ) -> None:
        '''
        Initialise a NIW prior object
        '''
        
        self.mu = mu
        self.nu = nu
        self.kappa = kappa
        self.sigma = sigma
        self.N = 0 # No data yet

        if data is not None:
            self.add_data(data)

    @property
    def sigma(self):
        # Sigma is only stored as cholesky decompositions
        return self._U.T @ self._U

    @sigma.setter
    def sigma(self, S):
        # Only store cholesky decomposition
        self._U = np.linalg.cholesky(S)

    def add_data(self, data, weight=None):
        '''
        Add data into the distribution and update params
        '''
        
        if weight is None:
            weight = np.ones((data.shape[0],1))

        # Add each point one by one
        for d, w in zip(data, weight):
            self.add_one(d, w)

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
        self.sigma += weight*(self.kappa/(self.kappa + weight))(res.T @ res)

    def rem_data(self, data, weight=None):
        '''
        Remove data from the distribution and downdate
        '''

        if weight is None:
            weight = np.ones((data.shape[0],1))

        # Remove each point one by one
        for d, w in zip(data,weight):
            self.rem_one(d,w)

    def rem_one(self, data, weight=1):
        '''
        Remove a single datum to the distribution
        ''' 

        # Downdate covariance
        res = (data-self.mn)
        self.sigma -= weight*self.kappa/(self.kappa-weight)*(res.T @ res);

        # Downdate scalar parameters
        self.N -= weight
        self.kappa -= weight
        self.nu -= weight

        # Downdate mean
        self.mu = ((self.kappa + weight)*self.mu - data * weight)/self.kappa


    def logpredpdf(self, X):
        '''
        Log predictive pdf is multivariate T
        p( x* | X ) = T( m_n, (k_n+1)/(k_n*nu_prime)*S_n, nu_prime)
        nu_n' = nu_n - D + 1
        Murphy PML 2021 Eq 7.144 pp. 199  
        '''

        _, D = X.shape
        nu_p = self.nu - D + 1
        S = self.sigma * (self.kappa + 1)/(self.kappa + nu_p)      
        U = np.linalg.cholesky(S)
        res = X - self.mu # rely on broadcasting
        QU = solve_triangular(U, res, trans='T')

        ll = gammaln((nu_p + D)/2) - \
             gammaln((nu_p)/2) - \
             D/2*np.log(nu_p) -\
             D/2*np.log(np.pi) -\
             np.sum(np.log(np.diag(U))) -\
             (nu_p+D)/2 (np.log(1 + np.sum(QU**2,axis=1)/nu_p))

        return ll

