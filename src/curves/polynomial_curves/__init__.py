import torch
import torch.nn as nn

from src.curves import Curve

class PolynomialCurve(Curve): # TODO evaluation function without grad
    def __init__(self, d, p):
        super().__init__(d, p)
        assert p >= 2

        self.d = d  # dimension
        self.p = p  # order of the polynomial
        
        # Coefficients of the polynomial as learnable parameters
        self.coefficients = nn.Parameter(torch.zeros(p-1, d))
        
    def forward(self, t):
        """Evaluate the curve at parameter values t
        :param t: N 
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def differential_forward(self, t):
        """Evaluate the speed of the curve at parameter values t
        :param t: N 
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    