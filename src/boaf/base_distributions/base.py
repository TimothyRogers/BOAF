from abc import ABC, abstractmethod
import numpy as np
from numpy.lib.type_check import real_if_close
from numpy.typing import NDArray

class BaseDistribution(ABC):

    @abstractmethod
    def logpredpdf(self, X: NDArray[np.float64]) -> None:
        """Log predictive likelihood of the distribution

        All distributions must provide a function to return the log predicitive
        likelihood of the distribution to new data. That is:
        
        .. math:: 
            \\log(p(x_i \\vert \\theta_k, \\mathcal{D}))

        This method should allow you to assess the log likelihood for a set of
        N data points input at a matrix, returning a vector of log predictive 
        likelihoods given the currently seen data :math:`\\mathcal{D}`. The
        parameters of the distribution :math:`\\theta_k` should be attributes
        of the distribution instance.

        Note:
            We will work entirely in log likelihoods to (hopefully) stay 
            numerically stable. Try to avoid every calculating non-logged 
            likelihoods, i.e. doing `np.log(likelihood)` is a bad idea.

        Args:
            X: An ndarray of size (N,D) for which the log predicitive likelihood
                of each of the N data points in D dimensions should be assessed
            
        Returns:
            An (N,) array of predictive log likelihoods.

        """
        raise NotImplementedError

    
    def add_data(
            self,
            data: NDArray[np.float64],
            weight: NDArray[np.float64]=None
            ) -> None:
        """Add data into the distribution and update params
        
        Provide a method for updating the parameters of the distribution by
        incorporating data that has been seen. This is done through a set of
        weighted updates to the parameters. Somewhat inefficiently, these updates
        are done by looping over a weighted update for each datapoint individually.
        The individual updates are performed in a separate function
        :meth:`add_one(x, weight)<boaf.base_distributions.base.BaseDistribution.add_one>`.

        Args:
            data: An array of size (N,D) where N, D-dimensional datapoints are
                to be used to update the parameters of the model.
            weight: An array size(N,) which contains weights in (0,1] of each of
                the N datapoints to be included
        
        """
                
        if weight is None:
            weight = np.ones((data.shape[0],1))

        # Add each point one by one
        for d, w in zip(data, weight):
            self.add_one(d, w)
    
    
    def rem_data(self, data, weight=None):
        """Remove data from the distribution and downdate parameters

        In the same way as adding data we will often want to remove data
        from a distribuion, e.g. in Gibbs updates of a mixture model. This method
        behaves almost exactly the same as :meth:`add_data(data, weight)` but 
        removes data. The indivdual updates are implemented in a function,
        :meth:`rem_one(x, weight)<boaf.base_distributions.base.BaseDistribution.rem_one>`.

        Args:
            data: An array of size (N,D) where N, D-dimensional datapoints are
                to be used to downdate the parameters of the model.
            weight: An array size(N,) which contains weights in (0,1] of each of
                the N datapoints to be considered

        """

        if weight is None:
            weight = np.ones((data.shape[0],1))

        # Remove each point one by one
        for d, w in zip(data,weight):
            self.rem_one(d,w)

    @abstractmethod
    def add_one(
        self,
        x:NDArray[np.float64],
        weight: np.float64
        ) -> None:
        """Update based on a single weighted datum

        This method should provide the updates to the parameters of the distribution
        as attributes of the distribution object. 

        Args:
            x: a (D,) array which is a single D-dimensional data point to be
                used to update the parameters
            weight: a float in (0,1] which is the weighting of this datum
        
        """

        raise NotImplementedError

    @abstractmethod
    def rem_one(
        self,
        x:NDArray[np.float64],
        weight: np.float64
        ) -> None:
        """Downdate based on a single weighted datum

        This method provides the downdate (changes) to the parameters of the
        distribution attributes give one (weighted) datum is removed from the
        estimate.

        Args:
            x: a (D,) array which is a single D-dimensional data point to be
                used to downdate the parameters
            weight: a float in (0,1] which is the weighting of this datum
        
        """

        raise NotImplementedError
