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

    def learn(self, data) -> None:
        
        self.initialise(data)
        self._fit(data)

    def initialise(self, data:NDArray) -> None:

        D = data.shape[-1]
        self.means = data[np.random.choice(data.shape[0],self.opts['nclusters'], replace=False),:]

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
        








    