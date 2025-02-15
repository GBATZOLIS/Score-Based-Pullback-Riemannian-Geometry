import torch
from src.manifolds import Manifold
from src.manifolds.euclidean import Euclidean

class DeformedGaussianPullbackManifold(Manifold):
    """ Base class describing a R^d under a l^2-pullback Riemannian geometry generated by a DeformedGaussian unimodal distribution """

    def __init__(self, deformed_gaussian):
        super().__init__(deformed_gaussian.d)
        self.dg = deformed_gaussian # unimodal distribution
        self.manifold = Euclidean(self.d)

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
        device = x.device
        return self.manifold.inner(self.dg.forward(x),
                                   self.dg.differential_forward((x[:, None] * torch.ones(M, device=device)[None, :, None]).reshape(-1, self.d), X.reshape(-1, self.d)).reshape(X.shape),
                                   self.dg.differential_forward((x[:, None] * torch.ones(L, device=device)[None, :, None]).reshape(-1, self.d), Y.reshape(-1, self.d)).reshape(Y.shape)
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
        device = x.device
        return self.dg.phi.differential_inverse(self.dg.phi.forward(x[None]).to(device) * torch.ones(N, device=device)[:, None],
                                                self.manifold.log(self.dg.phi.forward(x[None])[0].to(device), self.dg.phi.forward(y).to(device))
                                                )

    def exp(self, x, X):
        """

        :param x: d
        :param X: N x d
        :return: N x d
        """
        N, _ = X.shape
        device = x.device
        return self.dg.phi.inverse(self.manifold.exp(self.dg.phi.forward(x[None])[0].to(device), self.dg.phi.differential_forward(x[None].to(device) * torch.ones(N, device=device)[:, None], X.to(device))))

    def distance(self, x, y):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L
        """
        return self.manifold.distance(self.dg.forward(x.reshape(-1, self.d)).reshape(x.shape), self.dg.forward(y.reshape(-1, self.d)).reshape(y.shape))

    def parallel_transport(self, x, X, y):
        """

        :param x: d
        :param X: N x d
        :param y: d
        :return: N x d
        """
        N, _ = X.shape
        device = x.device
        return self.dg.phi.differential_inverse(self.dg.phi.forward(y[None]).to(device) * torch.ones(N, device=device)[:, None],
                                                self.manifold.parallel_transport(self.dg.phi.forward(x[None])[0].to(device),
                                                                                 self.dg.phi.differential_forward(x[None].to(device) * torch.ones(N, device=device)[:, None], X.to(device)),
                                                                                 self.dg.phi.forward(y[None])[0].to(device)
                                                                                 )
                                                )
