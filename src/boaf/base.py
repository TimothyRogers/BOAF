from abc import ABC, abstractmethod

class Cluster(ABC):
    '''
    Abstract base class for all cluster models
    '''

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

