import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from .base import Cluster


class KMeans(Cluster):
    """KMeans Clustering 

    Provide methods for clustering with KMeans this includes using KMeans++ or 
    random assignment to initialise the clusters.

    """

    def __init__(self, opts:dict) -> None:
        """Initalise KMeans Clustering

        Args:
            opts: Dictionary of options with args:
                init: (Optional) Initialisation method 'kpp' or 'rand';
                niters: Number of training iterations;
                nclusters: Number of clusters
        """
        super().__init__(opts)
        # If not initialisation specified use K++
        if 'init' not in self.opts:
            self.opts['init'] = 'kpp' 

    def learn(self, data:NDArray[np.float64]) -> None:
        """Learn the KMeans model

        Learning the KMeans model has effectively two stages. First we need
        to initialise some guess of the centers for the clusters then we will
        need to iterate over the data to refine those estimates.

        Args:
            data: An array of training data size (N,D) with N observations in
                D dimensions.

        """
        
        self.initialise(data)
        self._fit(data)

    def initialise(self, data:NDArray[np.float64]) -> None:
        """Initialise the KMeans centers
        
        The success (or failure) of the KMeans process can be very sensitive 
        to the initialisation. This method provides approaches for setting said
        initial cluster centers.

        Currently two methods are supported:
        
        1. Random initialisation ('rand'), randomly choose a datum as the initial center
        2. KMeans++ ('kpp'), select points far away from each other with some probability associated with the distance

        Args:
            data: An array of training data size (N,D) with N observations in
                D dimensions.

        """

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

    def _fit(self, data:NDArray[np.float64]) -> None:
        """Fit the KMeans Centers
        
        Iterate through the data from the start point to refine the estimates 
        of the KMeans centers.

        Args:
            data: An array of training data size (N,D) with N observations in
                D dimensions.

        """
        
        N, D = data.shape
        inds = np.zeros((N,))

        for _ in range(self.opts['niters']):
            
            inds_old = inds.copy()
            dists = cdist(data, self.means)
            inds = np.argmin(dists, axis=1)
            
            # Stop iterating if KMeans has stalled
            if np.all(inds == inds_old):
                return

            for k in range(self.opts['nclusters']):
                self.means[k,:] = np.mean(data[inds == k,:], axis=0)

    def predict(self, data:NDArray[np.float64]) -> NDArray[np.int16]:
        """Predict cluster associations
        
        Determine the most likely cluster assignments for the data based 
        off the current cluster centers

        Args:
            data: An array of training data size (N,D) with N observations in
                D dimensions.
        
        Returns:
            An array of indices, size (N,)

        """

        dists = cdist(data, self.means)
        inds = np.argmin(dists, axis=1)
        return inds
        








    