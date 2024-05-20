import torch

from src.curves.harmonic_curves import HarmonicCurve

class InitialHarmonicCurve(HarmonicCurve):
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
        sines = torch.sin(torch.arange(1, self.p+1)[None] * torch.pi * t[:,None])
        
        fixed_contribution = self.X[None] * t[:,None] + self.x[None]
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, sines) -  torch.sum(torch.arange(1, self.p+1)[:,None] * torch.pi * self.coefficients, 0)[None] * t[:,None]
        
        result = fixed_contribution + coefficient_contribution
        
        return result
    
    def differential_forward(self, t):
        cosines = torch.arange(1, self.p+1)[None] * torch.pi * torch.cos(torch.arange(1, self.p+1)[None] * torch.pi * t[:,None]) 
        
        fixed_contribution = self.X[None]
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, cosines) -  torch.sum(torch.arange(1, self.p+1)[:,None] * torch.pi * self.coefficients, 0)[None]
        
        result = fixed_contribution + coefficient_contribution
        
        return result
    
    def double_differential_forward(self, t): 
        sines = - (torch.arange(1, self.p+1)[None] * torch.pi) ** 2 * torch.sin(torch.arange(1, self.p+1)[None] * torch.pi * t[:,None]) 
        
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, sines)
        
        result = coefficient_contribution
        
        return result