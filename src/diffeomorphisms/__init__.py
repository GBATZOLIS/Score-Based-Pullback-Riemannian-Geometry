import torch.nn as nn

class Diffeomorphism(nn.Module):
    """ Base class describing a diffeomorphism phi: R^d \to R^d """

    def __init__(self, d) -> None:
        super().__init__()
        self.d = d

    def forward(self, x):
        """
        :param x: N x d
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def inverse(self, y):
        """
        :param y: N x d
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def differential_forward(self, x, X):
        """
        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def differential_inverse(self, y, Y):
        """
        :param y: N x d
        :param Y: N x d
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )