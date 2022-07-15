from abc import ABC, abstractmethod

from ..utils import NoPredictException

import numpy as np
from numpy.typing import NDArray

"""Base Class for All Clustering Algorithms
"""


class Cluster(ABC):
    """Abstract base class for all cluster models

    All clustering algorithms run through this ABC to provide a common
    interface for clustering problems. All algorithms should accept a dict
    of options which set any hyperparameter and provide methods for learning
    from data :meth:`learn()` and prediction on (new) data :meth:`predict()`

    """

    @abstractmethod
    def __init__(self, opts: dict) -> None:
        """Initialise algorithm

        All clustering algorithms accept a dictionary or options which contains
        the hyperparameters of the method.

        Note:
            Algorithms may implement additional inputs on :meth:`__init__()`,
            however, these should only be allowed as keyword arguments not
            positional.

        Args:
            opts: Dictionary containing hyperparamters, e.g. number of clusters

        """
        self.opts = opts

    @abstractmethod
    def learn(self, data: NDArray[np.float64]) -> None:
        """Learn clusters from data

        The learn method provides the means for the algorithm to learn the shape
        of the clusters.

        Note:
            Where multiple methods for learning the model are available these should
            be selected via the hyperparameters or modifying class attributes **not**
            as additional arguments of this function

        Args:
            data: An array of training data size (N,D) with N observations in
                D dimensions.

        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: NDArray[np.float64]) -> NDArray[np.int16]:
        """Predict cluster assignments for new data

        Provide a method which returns indices for which clusters each point
        in data is assigned to.

        Args:
            data: An array of data size (N,D) with N observations in D dimensions for which to determine the clusters.

        Returns:
            An array of indices, size (N,) which indicate the cluster that each
            of the N data points are assigned to.


        """
        raise NoPredictException
