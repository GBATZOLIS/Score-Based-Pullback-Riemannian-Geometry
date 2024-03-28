import torch

from src.diffeomorphisms import Diffeomorphism

class BananaDiffeomorphism(Diffeomorphism):

    def __init__(self, shear, offset) -> None:
        super().__init__(2)

        self.a = shear # >0
        self.z = offset # float

    def forward(self, x):
        """
        :param x: N x 2
        :return: N x 2
        """
        y1 = x[:,0] - self.a * x[:,1]**2 - self.z
        y2 = x[:,1]
        return torch.cat([y1[:,None],y2[:,None]], 1)

    def inverse(self, y):
        """
        :param y: N x 2
        :return: N x 2
        """
        x1 = y[:,0] + self.a * y[:,1]**2 + self.z
        x2 = y[:,1]
        return torch.cat([x1[:,None],x2[:,None]], 1)

    def differential_forward(self, x, X):
        """
        :param x: N x 2
        :param X: N x 2
        :return: N x 2
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def differential_inverse(self, y, Y):
        """
        :param y: N x 2
        :param Y: N x 2
        :return: N x 2
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )