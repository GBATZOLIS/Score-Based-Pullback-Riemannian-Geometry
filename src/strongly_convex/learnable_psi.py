import torch
import torch.nn as nn
from src.strongly_convex import StronglyConvex

class learnable_psi(StronglyConvex):
    """
    Class that implements the strongly convex function x \mapsto 1/2 x^\top A^{-1} x,
    where A is a diagonal matrix with positive entries. The diagonal elements of A are
    learnable parameters constrained to be positive.
    """

    def __init__(self, d, offset=None, stds=None, use_softplus=False):
        """
        Initialize the LearnablePsi function with a specified shape.

        Args:
            d (int): The size of the diagonal matrix A.
            stds (torch.Tensor or None): Optional initial singular values. If None, 
                                         initialize with zeros.
            use_softplus (bool): If True, use softplus for positivity constraint. 
                                 If False, use exponential function.
        """
        super().__init__(d)
        self.d = d
        self.use_softplus = use_softplus  # Flag to determine the activation function. Default=False->exponential, but softplus is likely to work better.
        
        if stds is not None:
            # Clamp the stds to ensure they are no smaller than 1e-2
            clamped_stds = stds #torch.clamp(stds, min=1e-2)
            
            if use_softplus:
                # Initialize raw_diagonal such that softplus(raw_diagonal) = stds^2
                raw_diagonal_init = torch.log(torch.expm1(clamped_stds**2))
            else:
                # Initialize raw_diagonal such that exp(raw_diagonal) = stds^2 (original version)
                raw_diagonal_init = torch.log(clamped_stds**2)
            
            self.raw_diagonal = nn.Parameter(raw_diagonal_init)
        else:
            if self.use_softplus:
                self.raw_diagonal = nn.Parameter(torch.tensor(0.5413))  # Initialize to ensure softplus(raw_diagonal) = 1
            else:
                self.raw_diagonal = nn.Parameter(torch.zeros(d)) # Initialize to ensure exp(raw_diagonal) = 1

    @property
    def diagonal(self):
        """
        Compute the diagonal values based on the chosen method (softplus or exponential).
        
        Returns:
            torch.Tensor: A tensor of positive diagonal entries.
        """
        if self.use_softplus:
            # Use the softplus function for stable small values
            threshold = 20.0
            return torch.where(self.raw_diagonal > threshold, self.raw_diagonal, torch.log1p(torch.exp(self.raw_diagonal)))
        else:
            # Use the exponential function for the original version
            return torch.exp(self.raw_diagonal)

    def forward(self, x):
        """
        Evaluate the function at x.

        Args:
            x (torch.Tensor): A tensor with shape (batchsize, *shape).

        Returns:
            torch.Tensor: A tensor with shape (batchsize,) of function values.
        """
        return 0.5 * torch.sum(x**2 / self.diagonal, dim=tuple(range(1, x.dim())))

    def grad_forward(self, x):
        """
        Compute the gradient of the function at x.

        Args:
            x (torch.Tensor): A tensor with shape (batchsize, *shape).

        Returns:
            torch.Tensor: A tensor with shape (batchsize, *shape) of gradients.
        """
        return x / self.diagonal

    def differential_grad_forward(self, x, X):
        """
        Compute the differential of the gradient of the function at x in the direction X.

        Args:
            x (torch.Tensor): A tensor with shape (batchsize, *shape).
            X (torch.Tensor): A tensor with shape (batchsize, *shape).

        Returns:
            torch.Tensor: A tensor with shape (batchsize, *shape).
        """
        return X / self.diagonal

    def fenchel_conjugate_forward(self, y):
        """
        Compute the Fenchel conjugate of the function at y.

        Args:
            y (torch.Tensor): A tensor with shape (batchsize, *shape).

        Returns:
            torch.Tensor: A tensor with shape (batchsize,).
        """
        return 0.5 * torch.sum(y**2 * self.diagonal, dim=tuple(range(1, y.dim())))

    def grad_fenchel_conjugate_forward(self, y):
        """
        Compute the gradient of the Fenchel conjugate at y.

        Args:
            y (torch.Tensor): A tensor with shape (batchsize, *shape).

        Returns:
            torch.Tensor: A tensor with shape (batchsize, *shape).
        """
        return y * self.diagonal

    def differential_grad_fenchel_conjugate_forward(self, y, Y):
        """
        Compute the differential of the gradient of the Fenchel conjugate at y in the direction Y.

        Args:
            y (torch.Tensor): A tensor with shape (batchsize, *shape).
            Y (torch.Tensor): A tensor with shape (batchsize, *shape).

        Returns:
            torch.Tensor: A tensor with shape (batchsize, *shape).
        """
        return Y * self.diagonal
