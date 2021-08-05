from abc import ABC, abstractmethod
from numpy.typing import NDArray

class Cluster(ABC):
    '''
    Abstract base class for all cluster models
    '''

    @abstractmethod
    def __init__(self, opts:dict) -> None:
        self.opts = opts

    @abstractmethod
    def learn(self, data:NDArray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, data:NDArray) -> NDArray:
        raise NotImplementedError