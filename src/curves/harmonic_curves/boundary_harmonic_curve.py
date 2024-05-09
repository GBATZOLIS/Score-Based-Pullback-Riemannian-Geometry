import torch

from src.curves.harmonic_curves import HarmonicCurve

class BoundaryHarmonicCurve(HarmonicCurve): # TODO
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
        sines = torch.sin(torch.arange(1, self.p+1)[None] * torch.pi * t[:,None]) 
        
        fixed_contribution = (self.y[None] - self.x[None]) * t[:,None] + self.x[None]
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, sines)
        
        result = fixed_contribution + coefficient_contribution
        
        return result
    
    def differential_forward(self, t): # TODO
        cosines = torch.arange(1, self.p+1)[None] * torch.pi * torch.cos(torch.arange(1, self.p+1)[None] * torch.pi * t[:,None]) 
        
        fixed_contribution = self.y[None] - self.x[None]
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, cosines)
        
        result = fixed_contribution + coefficient_contribution
        
        return result