import torch

from src.manifolds.discrete_time_manifolds import DiscreteTimeManifold

class SumOfGaussian(DiscreteTimeManifold): 
    """ Base class describing Euclidean space of dimension d under a sum of gaussian metric """

    def __init__(self, d, strongly_convexs, weights, L1=100, tol1=1e-2, max_iter1=20000, step_size1=1/8, L2=200, tol2=1e-4, max_iter2=200):
        super().__init__(d, 
                         L1=L1, tol1=tol1, max_iter1=max_iter1, step_size1=step_size1, 
                         L2=L2, tol2=tol2, max_iter2=max_iter2)
        
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
