import torch
import torch.nn as nn
import torch.optim as optim

class PolynomialCurve(nn.Module): # TODO evaluation function without grad
    def __init__(self, d, p):
        super().__init__()
        assert p >= 2

        self.d = d  # dimension
        self.p = p  # order of the polynomial
        
        # Coefficients of the polynomial as learnable parameters
        self.coefficients = nn.Parameter(torch.zeros(p-1, d)) # we assume we always know a begin and end point or a begin point and speed
        
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
    
    def fit(self, loss_function, num_time_points=200, num_epochs=500, lr=1.):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        t = torch.linspace(0.,1.,num_time_points)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            gamma_t = self.forward(t)
            dot_gamma_t = self.differential_forward(t)
            loss = torch.mean(loss_function(gamma_t, dot_gamma_t))
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                print(self.coefficients)