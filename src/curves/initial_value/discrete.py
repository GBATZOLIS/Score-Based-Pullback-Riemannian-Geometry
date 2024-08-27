import torch
import torch.nn as nn

from src.curves.initial_value import InitialValueCurve

class DiscreteInitialValueCurve(InitialValueCurve):

    def __init__(self, start_point, start_velocity, num_intervals):
        super().__init__(start_point, start_velocity)
        self.num_intervals = num_intervals # Number of intervals

        # Intermediate interpolants as learnable parameters
        self.coefficients = torch.zeros(self.num_intervals-1, self.d) # TODO probably make them linear interpolants of self.x and self.x + self.X

    def forward(self, t): 
        z = torch.cat([self.x[None], (self.x + 1 / self.num_intervals * self.X)[None], self.coefficients], dim=0)

        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError("All t values must be between 0 and 1.")

        # Determine which interval i each t value falls into
        i_values = (t * self.num_intervals).long()
        i_values = torch.clamp(i_values, 0, self.num_intervals - 1)  # Ensure i_values are within bounds

        # Calculate local interpolation factors within the intervals
        local_t_values = (t - i_values.float() / self.num_intervals) * self.num_intervals

        # Interpolate between vectors i and i+1
        v_i = z[i_values]            
        v_i_plus_1 = z[i_values + 1] 

        result = (1 - local_t_values)[:,None] * v_i + local_t_values[:,None] * v_i_plus_1
        return result
    
    def differential_forward(self, t):
        z = torch.cat([self.x[None], self.x + 1 / self.num_intervals * self.X[None], self.coefficients], dim=0)

        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError("All t values must be between 0 and 1.")

        # Determine which interval i each t value falls into
        i_values = (t * self.num_intervals).long()
        i_values = torch.clamp(i_values, 0, self.num_intervals - 1)  # Ensure i_values are within bounds

        # Interpolate between vectors i and i+1
        v_i = z[i_values]            
        v_i_plus_1 = z[i_values + 1] 

        result = v_i_plus_1 - v_i 
        return result
    
    def double_differential_forward(self, t): # TODO probably construct an effective acceleration vector
        return torch.zeros(t.shape[0], self.d)
