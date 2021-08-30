import re
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from .base import Cluster
from .base_distributions.multivariate import NIW


class MixtureModel(Cluster):
    '''
    Generic density based mixture model 

    '''

    def __init__(self, opts: dict, base_distribution) -> None:
        super().__init__(opts)
        # If not initialisation specified use K++
        if 'init' not in self.opts:
            self.opts['init'] = 'kpp' 
        self.base_distribution = base_distribution
        self.mixing_proportions = opts['prior']['alpha']
        self.clusters = [self.base_distribution(**self.opts['prior']['cluster']) \
                 for _ in range(self.opts['nclusters'])]

    def learn(self, data: NDArray) -> None:
        '''
        Default learning to EM method

        Later can add Gibbs
        '''
        self._em(data)

    def predict(self, data: NDArray) -> NDArray:
        ll = self.likelihood(data)
        return np.argmax(ll, axis=1)

    def likelihood(self,data: NDArray):
        '''
        Likelihood of each datapoint in each cluster
        '''
        N, D = data.shape
        likelihood = np.empty((N,self.opts['nclusters']))
        for k, cluster in enumerate(self.clusters):
            likelihood[:,k] = cluster.logpredpdf(data)
        return likelihood

    def _init_clusters(self,data):

        N = data.shape[0]
        if self.opts['init'] == 'kpp':
            # First mean is random
            k = [np.random.choice(N)]
            self.means = data[k[0],:][None,:]
            self.clusters[0].add_one(data[k[0],:][None,:])
            # Rest of clusters add furthest data
            for i in range(self.opts['nclusters']-1):
                # Distances
                dists = np.min(cdist(data, self.means),axis=1)
                # Convert to probabilities
                pmean = dists/np.sum(dists,axis=0)
                # Assign with p = pmean 
                k.append(np.random.choice(N, p=pmean))
                self.means = np.vstack((self.means,data[k[-1],:]))
                self.clusters[i+1].add_one(data[k[-1],:][None,:])
        elif self.opts['init'][:4] == 'rand':
            k = np.random.choice(N,self.opts['nclusters'], replace=False)
            self.means = data[k,:]
        
        N = data.shape[0]
        not_used = np.ones((N,),bool)
        not_used[k] = False
        #Random initialise
        for i, x in enumerate(data[not_used,:]):
            x = x[None,:]
            ll = self.likelihood(x)
            ind = np.argmax(ll)
            self.clusters[ind].add_data(x)

    def _em(self, data: NDArray):
        '''
        Expectation Maximisation Learning of Density Mixture Models
        '''

        N, D = data.shape

        self._init_clusters(data)
        ll = self.likelihood(data)
        responsibility = np.exp(self.normalise_log_likelihood(ll.T).T)
        Q_old = -np.inf

        for it in range(self.opts['niters']):
            ll = self.likelihood(data)
            Q = np.sum(responsibility*ll)
            if Q <= Q_old and it > 3:
                return
            Q_old = Q
            responsibility = np.exp(self.normalise_log_likelihood(ll.T).T)
            self.mixing_proportions = np.sum(responsibility, axis=0)/np.sum(responsibility)
            self.clusters = [self.base_distribution(**self.opts['prior']['cluster'], data=data, weight=responsibility[:,k])\
                 for k in range(self.opts['nclusters'])]

        print('Done')

    @staticmethod
    def normalise_log_likelihood(ll: NDArray):
        '''
        Numerically stable normalise log likelihood vector
        '''
        ml = np.max(ll,axis=0)[None,:]
        ll = ll-ml
        return ll - np.log(np.sum(np.exp(ll),axis=0))[None,:]



class GMM(MixtureModel):
    '''
    Soft wrapper for GMM

    Just set base distribution
    '''

    def __init__(self, opts: dict) -> None:
        
        base_distribution = NIW
        super().__init__(opts, base_distribution)

