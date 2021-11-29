import pytest

from boaf.algorithms.base import Cluster

def test_base_cluster():

    Cluster.__abstractmethods__ = set()
    model = Cluster(opts={})

    with pytest.raises(NotImplementedError): 
        model.learn(None)

    with pytest.raises(NotImplementedError): 
        model.predict(None)
        