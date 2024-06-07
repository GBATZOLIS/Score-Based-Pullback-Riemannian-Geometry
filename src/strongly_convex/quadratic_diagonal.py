import torch

from src.strongly_convex import StronglyConvex

class QuadraticDiagonal(StronglyConvex): # TODO
    """ Class that implements the strongly convex function x \mapsto 1/2 x^\top A^{-1} x, where A is diagonal with positive entries """
    def __init__(self, diagonal, offset=None) -> None:
        super().__init__(len(diagonal))

        self.diagonal = diagonal # torch tensor of size d
        if offset is None:
            self.offset = torch.zeros_like(diagonal)
        else:
            self.offset = offset

    def forward(self, x):
        """
        :param x: N x d
        :return: N
        """
        result = 0.5 * torch.sum(1 / self.diagonal[None] * (x - self.offset[None]) **2, 1)
        #print("QuadraticDiagonal.forward - result.requires_grad:", result.requires_grad)
        return result
    
    def grad_forward(self, x):
        """
        :param x: N x d
        :return: N x d
        """
        return 1 / self.diagonal[None] * (x - self.offset[None])
    
    def differential_grad_forward(self, x, X):
        """
        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        return 1 / self.diagonal[None] * X
    
    def fenchel_conjugate_forward(self, y):
        """
        :param y: N x d
        :return: N
        """
        return 1/2 * torch.sum(self.diagonal[None] * y **2 + y * self.offset[None], 1)
    
    def grad_fenchel_conjugate_forward(self, y):
        """
        :param y: N x d
        :return: N x d
        """
        return self.diagonal[None] * y + self.offset[None]
    
    def differential_grad_fenchel_conjugate_forward(self, y, Y):
        """
        :param y: N x d
        :param Y: N x d
        :return: N x d
        """
        return self.diagonal[None] * Y