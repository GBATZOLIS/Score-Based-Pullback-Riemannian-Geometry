import torch

from src.curves.polynomial_curves import PolynomialCurve

class InitialPolynomialCurve(PolynomialCurve):
    def __init__(self, d, p, start_point, start_velocity):
        """
        
        :d:
        :p:
        :start_point: d
        :end_point: d
        """
        super().__init__(d, p)
        self.x = start_point
        self.X = start_velocity

    def forward(self, t): 
        powers = torch.pow(t[:,None], torch.arange(2, self.p+1, dtype=torch.float32)[None])
        
        fixed_contribution = self.X[None] * t[:,None] + self.x[None]
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, powers)
        
        result = fixed_contribution + coefficient_contribution
        
        return result
    
    def differential_forward(self, t):
        powers = torch.arange(2, self.p+1, dtype=torch.float32)[None] * torch.pow(t[:,None], torch.arange(1, self.p, dtype=torch.float32)[None])
        
        fixed_contribution = self.X[None]
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, powers)
        
        result = fixed_contribution + coefficient_contribution
        
        return result