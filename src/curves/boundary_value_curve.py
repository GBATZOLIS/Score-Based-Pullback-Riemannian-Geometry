import torch
import torch.nn as nn

class BoundaryPolynomialCurve(nn.Module):
    def __init__(self, d, p, start_point, end_point):
        super(self).__init__(d, p)
        self.x = start_point
        self.y = end_point

    def polynomial_evaluation(self, t): # TODO
        # Create powers of t for polynomial evaluation
        powers = torch.pow(t, torch.arange(self.p+1, dtype=torch.float32)[1:].unsqueeze(0))
        
        # Expand coefficients to match the shape of powers
        coefficients_expanded = self.coefficients.unsqueeze(1)
        
        # Multiply coefficients with powers and sum along polynomial order dimension
        result = torch.sum(coefficients_expanded * powers, dim=2)
        
        return result