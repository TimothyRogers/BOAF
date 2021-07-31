from abc import ABC, abstractmethod

class BaseDistribution(ABC):

    @abstractmethod
    def logpredpdf(self, X):
        '''
        Log predictive likelihood

        All distributions must provide a function to return
        the log predicitiev likelihood of the distribution 
        to new data.
        '''
        raise NotImplementedError