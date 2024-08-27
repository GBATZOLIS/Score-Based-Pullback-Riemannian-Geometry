import torch
import torch.nn as nn

from src.curves.initial_value import InitialValueCurve

class HarmonicInitialValueCurve(InitialValueCurve):
    def __init__(self, start_point, start_velocity, K):
        super().__init__(start_point, start_velocity)
        self.K = K # Int

        # Coefficients of the sines as learnable parameters
        self.coefficients = nn.Parameter(torch.zeros(self.K, self.d))

    def forward(self, t): 
        sines = torch.sin(torch.arange(1, self.K+1)[None] * torch.pi * t[:,None])
        
        fixed_contribution = self.X[None] * t[:,None] + self.x[None]
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, sines) -  torch.sum(torch.arange(1, self.K+1)[:,None] * torch.pi * self.coefficients, 0)[None] * t[:,None]
        
        result = fixed_contribution + coefficient_contribution
        
        return result
    
    def differential_forward(self, t):
        cosines = torch.arange(1, self.K+1)[None] * torch.pi * torch.cos(torch.arange(1, self.K+1)[None] * torch.pi * t[:,None]) 
        
        fixed_contribution = self.X[None]
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, cosines) -  torch.sum(torch.arange(1, self.K+1)[:,None] * torch.pi * self.coefficients, 0)[None]
        
        result = fixed_contribution + coefficient_contribution
        
        return result
    
    def double_differential_forward(self, t): 
        sines = - (torch.arange(1, self.K+1)[None] * torch.pi) ** 2 * torch.sin(torch.arange(1, self.K+1)[None] * torch.pi * t[:,None]) 
        
        coefficient_contribution = torch.einsum("pd,Np->Nd", self.coefficients, sines)
        
        result = coefficient_contribution
        
        return result