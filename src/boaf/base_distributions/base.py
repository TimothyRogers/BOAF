from abc import ABC, abstractmethod
import numpy as np

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

    
    def add_data(self, data, weight=None):
        '''
        Add data into the distribution and update params
        '''
        
        if weight is None:
            weight = np.ones((data.shape[0],1))

        # Add each point one by one
        for d, w in zip(data, weight):
            self.add_one(d, w)
    
    
    def rem_data(self, data, weight=None):
        '''
        Remove data from the distribution and downdate
        '''

        if weight is None:
            weight = np.ones((data.shape[0],1))

        # Remove each point one by one
        for d, w in zip(data,weight):
            self.rem_one(d,w)