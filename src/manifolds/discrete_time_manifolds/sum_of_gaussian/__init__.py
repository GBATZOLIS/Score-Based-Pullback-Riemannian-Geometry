import torch

from src.manifolds.discrete_time_manifolds import DiscreteTimeManifold

class SumOfGaussian(DiscreteTimeManifold): 
    """ Base class describing Euclidean space of dimension d under a sum of gaussian metric """

    def __init__(self, d, strongly_convexs, weights):
        super().__init__(d)
        self.psi = strongly_convexs
        self.weights = weights
        self.m = len(strongly_convexs)

        self.diagonals = torch.cat([self.psi[i].diagonal[None] for i in range(self.m)])

    def metric_tensor(self, x):
        """
        :return: N x d x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def inverse_metric_tensor(self, x):
        """
        :return: N x d x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
