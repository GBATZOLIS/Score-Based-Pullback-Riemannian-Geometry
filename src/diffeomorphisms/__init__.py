import torch.nn as nn
import importlib

def get_diffeomorphism(diffeomorphism_class):
    dataset_class_name = diffeomorphism_class
    module_name = f'src.diffeomorphisms.{dataset_class_name.lower()}'
    module = importlib.import_module(module_name)
    dataset_class = getattr(module, dataset_class_name)
    return dataset_class

class Diffeomorphism(nn.Module):
    """ 
    Base class describing a diffeomorphism phi: R^d -> R^d. 

    This class provides a template for defining transformations 
    that are diffeomorphisms, i.e., smooth bijections with smooth 
    inverses. It includes methods for forward and inverse transformations, 
    as well as their differentials. 

    Attributes:
        d (tuple): Dimension of one element in the batch. For Euclidean data, 
                   this is a single integer (the dimensionality of the data).
                   For image data, this should be a tuple (c, h, w) representing 
                   the number of channels, height, and width.
    """

    def __init__(self, d) -> None:
        super().__init__()
        self.d = d

    def forward(self, x):
        """
        Applies the forward transformation phi to the input data x.

        Args:
            x (torch.Tensor): Input tensor of shape (N, d) for Euclidean data 
                              or (N, c, h, w) for image data.

        Returns:
            torch.Tensor: Transformed data of the same shape as the input.
        """
        raise NotImplementedError("Subclasses should implement this")

    def inverse(self, y):
        """
        Applies the inverse transformation phi^{-1} to the input data y.

        Args:
            y (torch.Tensor): Input tensor of shape (N, d) for Euclidean data 
                              or (N, c, h, w) for image data.

        Returns:
            torch.Tensor: Inverse transformed data of the same shape as the input.
        """
        raise NotImplementedError("Subclasses should implement this")

    def differential_forward(self, x, X):
        """
        Computes the differential of the forward transformation at x 
        for a vector X.

        Args:
            x (torch.Tensor): Points in the domain, shape (N, d) or (N, c, h, w).
            X (torch.Tensor): Tangent vectors at x, shape (N, d) or (N, c, h, w).

        Returns:
            torch.Tensor: Transformed tangent vectors of the same shape.
        """
        raise NotImplementedError("Subclasses should implement this")

    def differential_inverse(self, y, Y):
        """
        Computes the differential of the inverse transformation at y 
        for a vector Y.

        Args:
            y (torch.Tensor): Points in the codomain, shape (N, d) or (N, c, h, w).
            Y (torch.Tensor): Tangent vectors at y, shape (N, d) or (N, c, h, w).

        Returns:
            torch.Tensor: Inverse transformed tangent vectors of the same shape.
        """
        raise NotImplementedError("Subclasses should implement this")
