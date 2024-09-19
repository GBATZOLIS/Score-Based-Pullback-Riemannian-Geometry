import torch
import torch.nn as nn

from src.diffeomorphisms import Diffeomorphism

class river3d(Diffeomorphism):
    def __init__(self, shear0, shear1) -> None:
        super().__init__(3)
        """
        Initialize the diffeomorphism with two shear parameters.
        :param shear0: float, shear parameter for the influence of x2 on x0.
        :param shear1: float, shear parameter for the influence of x2 on x1.
        """
        self.a0 = shear0  # shear parameter for x0
        self.a1 = shear1  # shear parameter for x1

    def forward(self, x):
        """
        Computes the forward transformation of the diffeomorphism.
        :param x: Tensor of shape (N, 3), where N is the batch size.
        :return: Transformed tensor of shape (N, 3).
        """
        y = x.clone()
        y[:, 0] = x[:, 0] - torch.sin(self.a0 * x[:, 2])  # Transformation for x0
        y[:, 1] = x[:, 1] - torch.sin(self.a1 * x[:, 2])  # Transformation for x1
        y[:, 2] = x[:, 2]  # x2 remains unchanged
        return y

    def inverse(self, y):
        """
        Computes the inverse transformation of the diffeomorphism.
        :param y: Tensor of shape (N, 3), where N is the batch size.
        :return: Inverted tensor of shape (N, 3).
        """
        x = y.clone()
        x[:, 0] = y[:, 0] + torch.sin(self.a0 * y[:, 2])  # Inverse transformation for y0
        x[:, 1] = y[:, 1] + torch.sin(self.a1 * y[:, 2])  # Inverse transformation for y1
        x[:, 2] = y[:, 2]  # y2 remains unchanged
        return x

    def differential_forward(self, x, X):
        """
        Computes the differential of the forward transformation.
        :param x: Tensor of shape (N, 3), inputs.
        :param X: Tensor of shape (N, 3), differentials to transform.
        :return: Transformed differential tensor of shape (N, 3).
        """
        D_x = X.clone()
        D_x[:, 0] = X[:, 0] - self.a0 * torch.cos(self.a0 * x[:, 2]) * X[:, 2]  # Differential for x0
        D_x[:, 1] = X[:, 1] - self.a1 * torch.cos(self.a1 * x[:, 2]) * X[:, 2]  # Differential for x1
        D_x[:, 2] = X[:, 2]  # Differential for x2 remains unchanged
        return D_x

    def differential_inverse(self, y, Y):
        """
        Computes the differential of the inverse transformation.
        :param y: Tensor of shape (N, 3), inputs.
        :param Y: Tensor of shape (N, 3), differentials to invert.
        :return: Inverted differential tensor of shape (N, 3).
        """
        D_y = Y.clone()
        D_y[:, 0] = Y[:, 0] + self.a0 * torch.cos(self.a0 * y[:, 2]) * Y[:, 2]  # Inverse differential for y0
        D_y[:, 1] = Y[:, 1] + self.a1 * torch.cos(self.a1 * y[:, 2]) * Y[:, 2]  # Inverse differential for y1
        D_y[:, 2] = Y[:, 2]  # Inverse differential for y2 remains unchanged
        return D_y
