from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular
from scipy.special import gammaln

from .base import BaseDistribution
class NIW(BaseDistribution):
    """A Normal Inverse Wishart Distribution

    mu, Sig ~ NIW(m0,k0,n0,S0)
    mu ~ N(m0,Sig/k0)
    Sig ~ IW(S0,n0)

    """

    def __init__(
        self,
        mu,
        nu,
        kappa,
        sigma, 
        data=None,
        weight=None
        ) -> None:
        """Initialise a NIW prior object
        """
        
        self.mu = mu.copy()
        self.nu = nu
        self.kappa = kappa
        self.sigma = sigma.copy()
        self.N = 0 # No data yet
        self.D = mu.shape[1]

        if data is not None:
            self.add_data(data, weight=weight)

    def add_one(self, data, weight=1):
        """Add a single datum to the distribution
        """


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
        """Remove a single datum to the distribution
        """

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
        """
        Log predictive pdf is multivariate T
        p( x* | X ) = T( m_n, (k_n+1)/(k_n*nu_prime)*S_n, nu_prime)
        nu_n' = nu_n - D + 1
        Murphy PML 2021 Eq 7.144 pp. 199  
        """
                
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

class BLR(BaseDistribution):
    """Bayes Linear Regression for Functional Clustering

    mu, s2 ~ NIG(m0,L0,a0,b0)
    mu | s2 ~ N(m0,s2*L0)
    s2 ~ IG(a0,b0)

    m0 - Prior Mean
    L0 - Prior Precision
    a0, b0 - Priors for IG

    """

    def __init__(
        self,
        m0,
        L0,
        a0,
        b0, 
        data=None,
        weight=None
        ) -> None:
        """Initialise a BLR object
        """

        self.m0 = m0.copy()
        self.L0 = L0.copy()
        self.yy = 0
        self.Xy = np.zeros_like(m0,dtype='float64')
        
        self.mu = m0.copy()
        self.Lam = L0.copy() 
        self.an = a0
        self.b0 = b0 
        self.bn = b0
        self.N = 0 # No data yet
        
        self.D = m0.shape[0]

        if data is not None:
            self.add_data(data, weight=weight)

    def add_data(
        self,
        data: Tuple[NDArray[np.float64],NDArray[np.float64]],
        weight: NDArray[np.float64] = None) -> None:
        """Add a number of data into the BLR
        
        We need to provide a class specific method because this is a regression
        mixture, i.e. the data are a tuple (X,y)

        """

        if weight is None:
            weight = np.ones_like(data[1])

        for xx, yy, ww in zip(data[0],data[1],weight):
            self.add_one((xx,yy),ww)



    def add_one(self, data, weight=1):
        """Add a single datum to the distribution
        """

        self.an += weight/2
        self.Lam += weight*np.outer(data[0], data[0])
        self.Xy += weight*np.dot(data[0],data[1])
        self.mu = np.linalg.solve(
            self.Lam,
            np.dot(self.L0,self.m0) + self.Xy
            )
        self.yy += weight*data[1]**2
        self.bn = self.b0 +  0.5*(self.yy + self.m0.T.dot(self.L0).dot(self.m0) - self.mu.T.dot(self.Lam).dot(self.mu))


    def rem_one(self, data, weight=1):
        """Remove a single datum to the distribution
        """
        
        self.an -= weight/2
        self.Lam -= weight*np.outer(data[0], data[0])
        self.Xy -= weight*np.dot(data[0],data[1])
        self.mu = np.linalg.solve(
            self.Lam,
            np.dot(self.L0,self.m0) + self.Xy
            )
        self.yy -= weight*data[1]**2
        self.bn = self.b0 +  0.5*(self.yy + self.m0.T.dot(self.L0).dot(self.m0) - self.mu.T.dot(self.Lam).dot(self.mu))

    def map_estimates(self):
        return self.mu, self.Lam*self.bn/self.an, self.an/self.bn

    def logpredpdf(self, data):
        """
        Log predictive pdf is univariate Student's T over each datum

        """

        mp = np.dot(data[0],self.mu)
        R = np.linalg.cholesky(self.Lam)
        SU = solve_triangular(R,data[0].T,'T',lower=True)
        vp = self.bn/self.an*(np.ones_like(mp) + np.sum(SU.T**2,axis=-1))
        nu = 2*self.an
        x = data[1]-mp

        ll = gammaln((nu+1)/2) - 0.5*np.log(nu*np.pi) - gammaln(nu/2) - (nu+1)/2*np.log(1+x**2/nu)
               
        return ll 

    def posterior_predictive(self,Xt):
        """Posterior Predictive Distribution for BLR

        Returns predictive mean and variance
        
        """

        # Predictive mean
        mp = np.dot(Xt, self.mu)

        # Sig - switch to diagonal computation?
        Sig = self.bn/self.an*(np.identity(Xt.shape[0]) + Xt.dot(np.linalg.inv(self.Lam)).dot(Xt.T))

        # Nu
        nu = 2*self.an

        return mp, Sig, nu


