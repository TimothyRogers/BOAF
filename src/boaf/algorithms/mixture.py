import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from typing import Type

from .base import Cluster
from ..base_distributions.base import BaseDistribution
from ..base_distributions.multivariate import NIW


class MixtureModel(Cluster):
    """Generic density based mixture model 

    A whole class of clustering models can be thought of as estimating a 
    mixture of different probability distributions. Mathematically, this model
    looks like:

    .. math:: 
        k \\sim Mult(\pi)\\\\
        x \\vert k, \\theta_k \\sim p_k(x \\vert \\theta_k)\\\\
        p(x \\vert \\theta) = \\prod_{k=1}^K \\pi_k p_k(x \\vert \\theta_k)

    This says that the distribution of data is a weighted mixture of K
    distributions. The likelihood of any random data point coming from the k-th
    cluster is given by a Multinomial distribution with probabilities
    :math:`\pi`. Once a cluster k is chosen, a datum x is distributed according 
    to the distribution of that k-th cluster :math:`p_k(x \\vert \\theta_k)`
    which is parameterised by :math:`\\theta_k`.

    The form of the cluster shapes is entirely described by the user's choice of
    the underlying distribution :math:`p_k(x \\vert \\theta_k)` which is often 
    referred to as the base distribution. The most common choice would be to
    select a Normal (Gaussian) base distribution.

    The model can be extended to a Bayesian heirarchical formulation where some
    priors can be placed over both the mixing proportions :math:`\\pi` and the
    paramters of the base distribution :math:`\\theta`. The full model in this 
    case is intractable so must be approximated, e.g. by MCMC or variational
    inference, or a maximum as posteriori estimate can be made via expectation
    maximisation. 

    """

    def __init__(self,
             opts: dict,
             *,
             base_distribution: Type[BaseDistribution]) -> None:
        """Initialise Mixture Model Clustering
        
        Args:
            opts: Dictionary of options with args:
                init: (Optional) Initialisation method 'kpp' or 'rand';
                niters: Number of training iterations;
                nclusters: Number of clusters;
                prior: dictionary of options for prior specification, see distribution docs;
            base_distribution: Base distribution class, see :mod:`boaf.base_distributions`
        """
        super().__init__(opts)
        # If not initialisation specified use K++
        if 'init' not in self.opts:
            self.opts['init'] = 'kpp' 
        self.base_distribution = base_distribution
        self.mixing_proportions = opts['prior']['alpha']
        self.clusters = [self.base_distribution(**self.opts['prior']['cluster']) \
                 for _ in range(self.opts['nclusters'])]

    def learn(self, data: NDArray[np.float64]) -> None:
        """Learn the Mixture Model

        Provide interface for learning the Mixture model. Currently only EM is
        supported. This function is simply a thin wrapper on :meth:`_em()`

        Todo:
            * Add Gibbs sampling 

        Args:
            data: An array of training data size (N,D) with N observations in
                D dimensions.
        """
        self._em(data)

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.int16]:
        """Predict most likelihood cluster indices

        For the data being predicted, the predicted cluster indices are given 
        by taking the maximum likelihood for the predictive given the learnt
        clusters in the model, including the mixing proportions.

        Args:
            data: An array of data size (N,D) with N observations in D dimensions
                for which to determine the clusters.
        
        Returns:
            An array of indices, size (N,)
        
        """
        ll = self.likelihood(data)
        return np.argmax(ll, axis=1)

    def likelihood(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Log Predictive Likelihood

        Compute the log predictive likelihood of a set of data points with respect
        to each of the clusters of the model. This is given by the following 
        equation for the k-th cluster:

        .. math::
            \\log p_k(x_i \\vert \\theta_k) = \\log(\\pi_k) +
            \\log(p_k(x_i \\vert \\theta_k))

        This is the sum of the log mixing proportion and the log predictive
        likelihood for the k-th cluster.

        Args:
            data: An array of size (N,D) where the likelihood is assessed for N
            points each of D dimensions.

        Returns:
            An (N,) array of log predictive likelihoods for each of the N datapoints

        """
        N, D = data.shape
        likelihood = np.empty((N,self.opts['nclusters']))
        for k, cluster in enumerate(self.clusters):
            likelihood[:,k] = (cluster.logpredpdf(data) + 
                np.log(self.mixing_proportions[k]))
        return likelihood

    def _init_clusters(self, data:NDArray[np.float64]):
        """Initalise Mixture Model Clusters

        Initialise the clusters by assigning some initial data to the clusters.
        Currently this supports two methods, either:

        1. Random initialisation ('rand')
        2. KMeans++ initialisation ('kpp') - Default

        Todo:
            * Abstract this because it's the same as KMeans...
        
        Args:
            data: An array of training data size (N,D) with N observations in
                D dimensions.

        """

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

    def _em(self, data: NDArray[np.float64]):
        """Expectation Maximisation Learning for Mixture Models

        EM allows a MAP estimate of the cluster parameters and mixing proportions
        to be made. After initialisation, points are assigned to clusters weighted
        by their *responsibility* which is the normalised predictive log likelihood
        under the estimated parameters of the model in the previous step. Then
        the mixing proportions can also be estimated by the normalised responsibility 
        in each cluster.
        
        Note:
            It is the responsibility (pun intended) of the base distibutions to
            implement the weighted updates. 
        
        Args:
            data: An array of training data size (N,D) with N observations in
                D dimensions.

        """

        N, D = data.shape

        self._init_clusters(data)
        ll = self.likelihood(data)
        responsibility = np.exp(self.normalise_log_likelihood(ll.T).T)
        Q_old = -np.inf

        for it in range(self.opts['niters']):
            ll = self.likelihood(data)
            Q = np.sum(responsibility*ll)
            # Stopping condition on the total log likelihood, may need refining
            if Q <= Q_old and it > 3:
                return
            Q_old = Q
            responsibility = np.exp(self.normalise_log_likelihood(ll.T).T)
            self.mixing_proportions = np.sum(responsibility, axis=0)/np.sum(responsibility)
            self.clusters = [self.base_distribution(
                **self.opts['prior']['cluster'],
                 data=data,
                  weight=responsibility[:,k]) for k in range(self.opts['nclusters'])]

        print('Done')

    @staticmethod
    def normalise_log_likelihood(ll: NDArray[np.float64]):
        """Numerically stable log likelihood normalisation

        Use LogSumExp trick to normalise log likelihoods. Maybe this should move 
        out of the class and into a utility module.

        Args:
            ll: An array of log likelihoods to be noramlised along axis 0
        """
        ml = np.max(ll,axis=0)[None,:]
        ll = ll-ml
        return ll - np.log(np.sum(np.exp(ll),axis=0))[None,:]



class GMM(MixtureModel):
    """Gaussian Mixture Model (Bayesian MAP)

    Provide a simple interface for learning GMMs, this is simply fixing the
    base distribution of the mixture model to be Gaussian
    
    """

    def __init__(self, opts: dict) -> None:
        """Initialise Mixture Model Clustering
        
        Args:
            opts: Dictionary of options with args:
                init: (Optional) Initialisation method 'kpp' or 'rand';
                niters: Number of training iterations;
                nclusters: Number of clusters;
                prior: dictionary of options for prior specification, see 
                :class:`NIW <boaf.base_distributions.multivariate.NIW>`
        """
        
        base_distribution = NIW
        super().__init__(opts, base_distribution)

