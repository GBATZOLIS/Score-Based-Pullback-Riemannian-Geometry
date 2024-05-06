import torch
import torch.nn as nn
from src.strongly_convex import StronglyConvex

class LearnablePsi(StronglyConvex):
    """
    Class that implements the strongly convex function x \mapsto 1/2 x^\top A^{-1} x,
    where A is a diagonal matrix with positive entries. The diagonal elements of A are
    learnable parameters constrained to be positive.
    """

    def __init__(self, d):
        """
        Initialize the LearnablePsi function with a specified dimension.

        Args:
            d (int): The dimension of the diagonal matrix A.
        """
        super().__init__(d)
        # Initialize the raw parameters which will be transformed to ensure positivity
        self.raw_diagonal = nn.Parameter(torch.zeros(d))  # Start with zeros, which will be exponentiated

    @property
    def diagonal(self):
        """
        Ensure the diagonal elements are always positive by taking the exponential of raw parameters.

        Returns:
            torch.Tensor: A 1-D tensor of positive diagonal entries.
        """
        return torch.exp(self.raw_diagonal)

    def forward(self, x):
        """
        Evaluate the function at x.

        Args:
            x (torch.Tensor): An N x d tensor.

        Returns:
            torch.Tensor: An N-dimensional tensor of function values.
        """
        return 0.5 * torch.sum(x**2 / self.diagonal, dim=1)
    
    def grad_forward(self, x):
        """
        Compute the gradient of the function at x.

        Args:
            x (torch.Tensor): An N x d tensor.

        Returns:
            torch.Tensor: An N x d tensor of gradients.
        """
        return x / self.diagonal
    
    def differential_grad_forward(self, x, X):
        """
        Compute the differential of the gradient of the function at x in the direction X.

        Args:
            x (torch.Tensor): An N x d tensor.
            X (torch.Tensor): An N x d tensor.

        Returns:
            torch.Tensor: An N x d tensor.
        """
        return X / self.diagonal
    
    def fenchel_conjugate_forward(self, y):
        """
        Compute the Fenchel conjugate of the function at y.

        Args:
            y (torch.Tensor): An N x d tensor.

        Returns:
            torch.Tensor: An N-dimensional tensor.
        """
        return 0.5 * torch.sum(y**2 * self.diagonal, dim=1)
    
    def grad_fenchel_conjugate_forward(self, y):
        """
        Compute the gradient of the Fenchel conjugate at y.

        Args:
            y (torch.Tensor): An N x d tensor.

        Returns:
            torch.Tensor: An N x d tensor.
        """
        return y * self.diagonal
    
    def differential_grad_fenchel_conjugate_forward(self, y, Y):
        """
        Compute the differential of the gradient of the Fenchel conjugate at y in the direction Y.

        Args:
            y (torch.Tensor): An N x d tensor.
            Y (torch.Tensor): An N x d tensor.

        Returns:
            torch.Tensor: An N x d tensor.
        """
        return Y * self.diagonal
