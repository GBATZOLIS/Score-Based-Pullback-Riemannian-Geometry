import torch.nn as nn

class StronglyConvex(nn.Module):
    """ Base class describing a strongly convex function psi: R^d \to R """

    def __init__(self, d) -> None:
        super().__init__()
        self.d = d

    def forward(self, x):
        """
        :param x: N x d
        :return: N
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def grad_forward(self, x):
        """
        :param x: N x d
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def differential_grad_forward(self, x, X):
        """
        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def fenchel_conjugate_forward(self, y):
        """
        :param y: N x d
        :return: N
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def grad_fenchel_conjugate_forward(self, y):
        """
        :param y: N x d
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def differential_grad_fenchel_conjugate_forward(self, y, Y):
        """
        :param y: N x d
        :param Y: N x d
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
