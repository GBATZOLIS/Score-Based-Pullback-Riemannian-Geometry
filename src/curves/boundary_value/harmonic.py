import torch
import torch.nn as nn

from src.curves.boundary_value import BoundaryValueCurve

class HarmonicBoundaryValueCurve(BoundaryValueCurve):

    def __init__(self, start_point, end_point, num_sines):
        super().__init__(start_point, end_point)
        self.num_sines = num_sines # Int

        # Coefficients of the sines as learnable parameters
        self.coefficients = nn.Parameter(torch.zeros(self.num_sines, self.d))

    def forward(self, t): 
        sines = torch.sin(torch.arange(1, self.num_sines+1)[None] * torch.pi * t[:,None]) 
        
        fixed_contribution = (self.y[None] - self.x[None]) * t[:,None] + self.x[None]
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, sines)
        
        result = fixed_contribution + coefficient_contribution
        
        return result
    
    def differential_forward(self, t):
        cosines = torch.arange(1, self.num_sines+1)[None] * torch.pi * torch.cos(torch.arange(1, self.num_sines+1)[None] * torch.pi * t[:,None]) 
        
        fixed_contribution = self.y[None] - self.x[None]
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, cosines)
        
        result = fixed_contribution + coefficient_contribution
        
        return result
    
    def double_differential_forward(self, t): 
        sines = - (torch.arange(1, self.num_sines+1)[None] * torch.pi) ** 2 * torch.sin(torch.arange(1, self.num_sines+1)[None] * torch.pi * t[:,None]) 
        
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, sines)
        
        result = coefficient_contribution
        
        return result
