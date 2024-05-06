import torch

from src.manifolds import Manifold

class SumOfGaussian(Manifold): # TODO get the curve classes in here and use inner or norm squared as loss function
    """ Base class describing Euclidean space of dimension d under a sum of gaussian metric """

    def __init__(self, d, strongly_convexs, weights):
        super().__init__(d)

    def barycentre(self, x):
        """

        :param x: N x d
        :return: d
        """
        # TODO this will be very hard to compute... Can use CPPA paper perhaps?
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def inner(self, x, X, Y):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        # TODO closed form
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def geodesic(self, x, y, t):
        """

        :param x: d
        :param y: d
        :param t: N
        :return: N x d
        """
        # TODO solve optimisation problem --> construct geodesic class from endpoints + order + data size, which is a neural network with just a linear layer, all the training data is just the times and we train the network
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def log(self, x, y):
        """

        :param x: d
        :param y: N x d
        :return: N x d
        """
        # TODO solve optimisation problem
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def exp(self, x, X):
        """

        :param x: d
        :param X: N x d
        :return: N x d
        """
        # TODO solve optimisation problem
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def distance(self, x, y):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L
        """
        # TODO solve optimisation problem
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def parallel_transport(self, x, X, y):
        """

        :param x: d
        :param X: N x d
        :param y: d
        :return: N x d
        """
        # TODO maybe just leave this...
        raise NotImplementedError(
            "Subclasses should implement this"
        )