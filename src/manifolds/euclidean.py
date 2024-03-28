import torch

from src.manifolds import Manifold

class Euclidean(Manifold):
    """ Base class describing Euclidean space of dimension d """

    def __init__(self, d):
        super().__init__(d)

    def barycentre(self, x):
        """

        :param x: N x d
        :return: d
        """
        return torch.mean(x, 0)

    def inner(self, x, X, Y):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        return torch.einsum("NMi,NLi->NML", X, Y)
    
    def geodesic(self, x, y, t):
        """

        :param x: d
        :param y: d
        :param t: N
        :return: N x d
        """
        return (1 - t[:,None]) * x[None] + t[:,None] * y[None]

    def log(self, x, y):
        """

        :param x: d
        :param y: N x d
        :return: N x d
        """
        return y - x[None]

    def exp(self, x, X):
        """

        :param x: d
        :param X: N x d
        :return: N x d
        """
        return x[None] + X
    
    def distance(self, x, y):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L
        """
        return torch.sqrt(torch.sum((x[:,:,None] - y[:,None,:]) ** 2, -1) + 1e-8)

    def parallel_transport(self, x, X, y):
        """

        :param x: d
        :param X: N x d
        :param y: d
        :return: N x d
        """
        return X