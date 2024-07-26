import torch
import torch.nn as nn

from src.diffeomorphisms import Diffeomorphism

class RiverDiffeomorphism(Diffeomorphism):
    def __init__(self, shear, offset) -> None:
        super().__init__(2)
        self.a = shear  # float
        self.z = offset  # float

    def forward(self, x):
        """
        Computes the forward transformation of the diffeomorphism.
        :param x: Tensor of shape (N, 2), where N is the batch size.
        :return: Transformed tensor of shape (N, 2).
        """
        y = x.clone()
        y[:, 0] = x[:, 0] - torch.sin(self.a * x[:, 1]) - self.z
        return y

    def inverse(self, y):
        """
        Computes the inverse transformation of the diffeomorphism.
        :param y: Tensor of shape (N, 2), where N is the batch size.
        :return: Inverted tensor of shape (N, 2).
        """
        x = y.clone()
        x[:, 0] = y[:, 0] + torch.sin(self.a * y[:, 1]) + self.z
        return x

    def differential_forward(self, x, X):
        """
        Computes the differential of the forward transformation.
        :param x: Tensor of shape (N, 2), inputs.
        :param X: Tensor of shape (N, 2), differentials to transform.
        :return: Transformed differential tensor of shape (N, 2).
        """
        D_x = X.clone()
        D_x[:, 0] = X[:, 0] - self.a * torch.cos(self.a * x[:, 1]) * X[:, 1]
        return D_x

    def differential_inverse(self, y, Y):
        """
        Computes the differential of the inverse transformation.
        :param y: Tensor of shape (N, 2), inputs.
        :param Y: Tensor of shape (N, 2), differentials to invert.
        :return: Inverted differential tensor of shape (N, 2).
        """
        D_y = Y.clone()
        D_y[:, 0] = Y[:, 0] + self.a * torch.cos(self.a * y[:, 1]) * Y[:, 1]
        return D_y

