import torch
import torch.nn as nn
import torch.optim as optim

class PolynomialCurve(nn.Module): # TODO
    def __init__(self, d, p):
        super(PolynomialCurve, self).__init__()
        assert p >= 2

        self.d = d  # dimension
        self.p = p  # order of the polynomial
        
        # Coefficients of the polynomial as learnable parameters
        self.coefficients = nn.Parameter(torch.zeros(p-1, d)) # we assume we always know a begin and end point or a begin point and speed
        
    def forward(self, t):
        """Evaluate the curve at parameter values t"""
        # t: a tensor of parameter values
        
        if len(t.shape) == 1:  # if t is 1-dimensional
            t = t.unsqueeze(1)  # convert to 2-dimensional tensor

        # Evaluate polynomial at parameter values t
        return self.polynomial_evaluation(t)
    
    def differential_forward(self, t):
        """Evaluate the speed of the curve at parameter values t"""
        return None
    
    def fit(self, loss_function, num_time_points=10, num_epochs=100, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            gamma_t = self.forward(self.time_points)
            gamma_dot_t = 9.
            loss = loss_function(gamma_t, gamma_dot_t)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    def polynomial_evaluation(self, t):
        """Evaluate the polynomial at parameter values t"""
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def differential_polynomial_evaluation(self, t):
        """Evaluate the derivative of the polynomial at parameter values t"""
        raise NotImplementedError(
            "Subclasses should implement this"
        )