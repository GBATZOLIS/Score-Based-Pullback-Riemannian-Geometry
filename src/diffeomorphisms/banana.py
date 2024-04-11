import torch

from src.diffeomorphisms import Diffeomorphism

class BananaDiffeomorphism(Diffeomorphism):

    def __init__(self, shear, offset) -> None:
        super().__init__(2)

        self.a = shear # float
        self.z = offset # float

    def forward(self, x):
        """
        :param x: N x 2
        :return: N x 2
        """
        y = torch.zeros_like(x)
        y[:,0] = x[:,0] - self.a * x[:,1]**2 - self.z
        y[:,1] = x[:,1]
        return y

    def inverse(self, y):
        """
        :param y: N x 2
        :return: N x 2
        """
        x = torch.zeros_like(y)
        x[:,0] = y[:,0] + self.a * y[:,1]**2 + self.z
        x[:,1] = y[:,1]
        return x

    def differential_forward(self, x, X):
        """
        :param x: N x 2
        :param X: N x 2
        :return: N x 2
        """
        D_x = torch.zeros_like(x)
        D_x[:,0] = X[:,0] - 2 * self.a * x[:,1] * X[:,1]
        D_x[:,1] = X[:,1]
        return D_x

    def differential_inverse(self, y, Y):
        """
        :param y: N x 2
        :param Y: N x 2
        :return: N x 2
        """
        D_y = torch.zeros_like(y)
        D_y[:,0] = Y[:,0] + 2 * self.a * y[:,1] * Y[:,1]
        D_y[:,1] = Y[:,1]
        return D_y