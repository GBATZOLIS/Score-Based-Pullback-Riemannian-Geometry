import torch

class Manifold:
    """ Base class describing a manifold of dimension d """

    def __init__(self, d):
        self.d = d

    def barycentre(self, x):
        """

        :param x: N x Mpoint
        :return: Mpoint
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def inner(self, x, X, Y):
        """

        :param x: N x Mpoint
        :param X: N x M x Mvector
        :param Y: N x L x Mvector
        :return: N x M x L
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def norm(self, x, X):
        """

        :param x: N x Mpoint
        :param X: N x M x Mvector
        :return: N x M
        """
        N, M, _ = X.shape
        return torch.sqrt(torch.abs(self.inner(x[:,None] * torch.ones((1, M, 1)), X.reshape(N * M, 1, -1), X.reshape(N * M, 1, -1)).squeeze(-1,-2).reshape(N,M,-1)))

    def geodesic(self, x, y, t):
        """

        :param x: Mpoint
        :param y: Mpoint
        :param t: N
        :return: N x Mpoint
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def log(self, x, y):
        """

        :param x: Mpoint
        :param y: N x Mpoint
        :return: N x Mvector
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def exp(self, x, X):
        """

        :param x: Mpoint
        :param X: N x Mvector
        :return: N x Mpoint
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def distance(self, x, y):
        """

        :param x: N x M x Mpoint
        :param y: N x L x Mpoint
        :return: N x M x L
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def parallel_transport(self, x, X, y):
        """

        :param x: Mpoint
        :param X: N x Mvector
        :param y: Mpoint
        :return: N x Mpoint
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def manifold_dimension(self):
        return self.d