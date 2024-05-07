import torch
import torch.nn as nn

from src.curves import PolynomialCurve

class BoundaryPolynomialCurve(PolynomialCurve):
    def __init__(self, d, p, start_point, end_point):
        """
        
        :d:
        :p:
        :start_point: d
        :end_point: d
        """
        super().__init__(d, p)
        self.x = start_point
        self.y = end_point

    def forward(self, t): 
        powers = torch.pow(t[:,None], torch.arange(2, self.p+1, dtype=torch.float32)[None])
        
        fixed_contribution = (self.y[None] - self.x[None]) * t[:,None] + self.x[None]
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, powers) - torch.sum(self.coefficients,0)[None] * t[:,None]
        
        result = fixed_contribution + coefficient_contribution
        
        return result
    
    def differential_forward(self, t):
        powers = torch.arange(2, self.p+1, dtype=torch.float32)[None] * torch.pow(t[:,None], torch.arange(1, self.p, dtype=torch.float32)[None])
        
        fixed_contribution = self.y[None] - self.x[None]
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, powers) - torch.sum(self.coefficients,0)[None]
        
        result = fixed_contribution + coefficient_contribution
        
        return result