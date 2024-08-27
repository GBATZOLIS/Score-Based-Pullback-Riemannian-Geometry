import torch
import torch.optim as optim

from src.curve_solvers import CurveSolver

class InitialValueCurveSolver(CurveSolver):
    """ Base class describing a boundary value curve solver """
    
    def __init__(self, curve, loss_function, num_time_points=200, num_epochs=100, lr=0.01, weight_decay=0.) -> None:
        super().__init__(curve)

        self.num_time_points = num_time_points
        self.loss_function = loss_function
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay

    def solve(self):
        optimizer = optim.Adam(self.curve.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        t = torch.linspace(0.,1.,self.num_time_points)
        
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            gamma_t = self.curve.forward(t)
            dot_gamma_t = self.curve.differential_forward(t)
            dot_dot_gamma_t = self.curve.double_differential_forward(t)

            segment_lengths = self.loss_function(gamma_t[0:-1], gamma_t[1:] -  gamma_t[0:-1])
            length = torch.mean(segment_lengths) 

            regularizer = (torch.einsum("Nd,Nd->N", dot_gamma_t, dot_dot_gamma_t)).pow(2).mean()

            loss = length #ß+ regularizer

            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss {loss.item()}, Length {length.item()}, Regularizer {regularizer.item()}")