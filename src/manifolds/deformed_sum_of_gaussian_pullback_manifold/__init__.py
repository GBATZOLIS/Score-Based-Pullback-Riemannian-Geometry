from src.manifolds import Manifold
from src.manifolds.euclidean import Euclidean

import torch

class DeformedGaussianPullbackManifold(Manifold): # TODO check input discrepancies between manifold mapping input and diffeomorphism input. Latter always assumes batch dimension
    """ Base class describing a R^d under a sum of Gaussian-pullback Riemannian geometry generated by a DeformedSumofGaussian multimodal distribution """

    def __init__(self, deformed_sum_of_gaussian):
        raise NotImplementedError(
            "Subclasses should implement this"
        )
        # super().__init__(deformed_sum_of_gaussian.d)
        self.dsg = deformed_sum_of_gaussian # multimodal distribution
        # self.manifold = Euclidean(self.d) # TODO

    def barycentre(self, x):
        """

        :param x: N x d
        :return: d
        """
        return self.dg.phi.inverse(self.manifold.barycentre(self.dg.phi(x))[None])[0]

    def inner(self, x, X, Y):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        _, M, _ = X.shape
        _, L, _ = Y.shape
        return self.manifold.inner(self.dg.forward(x),
                                   self.dg.differential_forward((x[:,None] * torch.ones(M)[None,:,None]).reshape(-1,self.d), X.reshape(-1,self.d)).reshape(X.shape),
                                   self.dg.differential_forward((x[:,None] * torch.ones(L)[None,:,None]).reshape(-1,self.d), Y.reshape(-1,self.d)).reshape(Y.shape)
                                   )
    
    def geodesic(self, x, y, t):
        """

        :param x: d
        :param y: d
        :param t: N
        :return: N x d
        """
        return self.dg.phi.inverse(self.manifold.geodesic(self.dg.phi.forward(x[None])[0], self.dg.phi.forward(y[None])[0], t))

    def log(self, x, y):
        """

        :param x: d
        :param y: N x d
        :return: N x d
        """
        N, _ = y.shape
        return self.dg.phi.differential_inverse(self.dg.phi.forward(x[None]) * torch.ones(N)[:,None],
                                                self.manifold.log(self.dg.phi.forward(x[None])[0], self.dg.phi.forward(y))
                                                )

    def exp(self, x, X):
        """

        :param x: d
        :param X: N x d
        :return: N x d
        """
        N, _ = X.shape
        return self.dg.phi.inverse(self.manifold.exp(self.dg.phi.forward(x[None])[0], self.dg.phi.differential_forward(x[None] * torch.ones(N)[:,None], X)))
    
    def distance(self, x, y):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L
        """
        return self.manifold.distance(self.dg.forward(x.reshape(-1,self.d)).reshape(x.shape), self.dg.forward(y.reshape(-1,self.d)).reshape(y.shape))

    def parallel_transport(self, x, X, y):
        """

        :param x: d
        :param X: N x d
        :param y: d
        :return: N x d
        """
        N, _ = X.shape
        return self.dg.phi.differential_inverse(self.dg.phi.forward(y[None]) * torch.ones(N)[:,None],
                                                self.manifold.parallel_transport(self.dg.phi.forward(x[None])[0],
                                                                                 self.dg.phi.differential_forward(x[None] * torch.ones(N)[:,None], X),
                                                                                 self.dg.phi.forward(y[None])[0]
                                                                                 )
                                                )