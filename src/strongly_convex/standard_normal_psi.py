import torch
import torch.nn as nn
from src.strongly_convex import StronglyConvex

class standard_normal_psi(StronglyConvex):
    """
    Class that implements the strongly convex function x \mapsto 1/2 x^\top A^{-1} x,
    where A is a diagonal matrix with positive entries. In this version, the diagonal elements of A are fixed to 1.
    """

    def __init__(self, d, offset=None, stds=None, use_softplus=False):
        """
        Initialize the StandardNormalPsi function with a specified dimension.

        Args:
            d (int): The dimension of the input space.
        """
        super().__init__(d)
        self.d = d
        self.register_buffer('raw_diagonal', torch.ones(d))

    @property
    def diagonal(self):
        """
        Return a tensor of ones since we are fixing the diagonal entries to 1.

        Returns:
            torch.Tensor: A tensor of ones with shape (d,).
        """
        return self.raw_diagonal  # No need to exponentiate since it's already 1.

    def forward(self, x):
        """
        Evaluate the function at x.

        Args:
            x (torch.Tensor): A tensor with shape (batchsize, d).

        Returns:
            torch.Tensor: A tensor with shape (batchsize,) of function values.
        """
        return 0.5 * torch.sum(x**2 / self.diagonal, dim=tuple(range(1, x.dim())))

    def grad_forward(self, x):
        """
        Compute the gradient of the function at x.

        Args:
            x (torch.Tensor): A tensor with shape (batchsize, d).

        Returns:
            torch.Tensor: A tensor with shape (batchsize, d) of gradients.
        """
        return x / self.diagonal

    def differential_grad_forward(self, x, X):
        """
        Compute the differential of the gradient of the function at x in the direction X.

        Args:
            x (torch.Tensor): A tensor with shape (batchsize, d).
            X (torch.Tensor): A tensor with shape (batchsize, d).

        Returns:
            torch.Tensor: A tensor with shape (batchsize, d).
        """
        return X / self.diagonal

    def fenchel_conjugate_forward(self, y):
        """
        Compute the Fenchel conjugate of the function at y.

        Args:
            y (torch.Tensor): A tensor with shape (batchsize, d).

        Returns:
            torch.Tensor: A tensor with shape (batchsize,).
        """
        return 0.5 * torch.sum(y**2 * self.diagonal, dim=tuple(range(1, y.dim())))

    def grad_fenchel_conjugate_forward(self, y):
        """
        Compute the gradient of the Fenchel conjugate at y.

        Args:
            y (torch.Tensor): A tensor with shape (batchsize, d).

        Returns:
            torch.Tensor: A tensor with shape (batchsize, d).
        """
        return y * self.diagonal

    def differential_grad_fenchel_conjugate_forward(self, y, Y):
        """
        Compute the differential of the gradient of the Fenchel conjugate at y in the direction Y.

        Args:
            y (torch.Tensor): A tensor with shape (batchsize, d).
            Y (torch.Tensor): A tensor with shape (batchsize, d).

        Returns:
            torch.Tensor: A tensor with shape (batchsize, d).
        """
        return Y * self.diagonal
