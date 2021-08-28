import numpy as np
from numpy.lib.index_tricks import nd_grid
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


from .base import Cluster


class KMeans(Cluster):
    '''
    Basic K-Means clustering algorithm
    '''

    def __init__(self, opts:dict) -> None:
        super().__init__(opts)
        # If not initialisation specified use K++
        if 'init' not in self.opts:
            self.opts['init'] = 'kpp' 

    def learn(self, data) -> None:
        
        self.initialise(data)
        self._fit(data)

    def initialise(self, data:NDArray) -> None:

        N = data.shape[0]
        if self.opts['init'] == 'kpp':
            # First mean is random
            k = [np.random.choice(N)]
            self.means = data[k[0],:][None,:]
            # Rest of clusters add furthest data
            for _ in range(self.opts['nclusters']-1):
                # Distances
                dists = np.min(cdist(data, self.means),axis=1)
                # Convert to probabilities
                pmean = dists/np.sum(dists,axis=0)
                # Assign with p = pmean 
                k.append(np.random.choice(N, p=pmean))
                self.means = np.vstack((self.means,data[k[-1],:]))
        elif self.opts['init'][:4] == 'rand':
            self.means = data[np.random.choice(N,self.opts['nclusters'], replace=False),:]

    def _fit(self, data:NDArray) -> None:
        
        N, D = data.shape
        inds = np.zeros((N,))

        for _ in range(self.opts['niters']):
            
            inds_old = inds.copy()
            dists = cdist(data, self.means)
            inds = np.argmin(dists, axis=1)

            if np.all(inds == inds_old):
                return

            for k in range(self.opts['nclusters']):
                self.means[k,:] = np.mean(data[inds == k,:], axis=0)

    def predict(self, data:NDArray) -> NDArray:

            dists = cdist(data, self.means)
            inds = np.argmin(dists, axis=1)
            return inds
        








    