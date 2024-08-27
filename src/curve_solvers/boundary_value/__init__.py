import torch
import torch.optim as optim

from src.curve_solvers import CurveSolver

class BoundaryValueCurveSolver(CurveSolver): # TODO add termination conditions
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

            segment_lengths = self.loss_function(gamma_t[0:-1], gamma_t[1:] -  gamma_t[0:-1])
            length = torch.sum(segment_lengths)
            energy = (self.num_time_points - 1) * torch.sum(segment_lengths ** 2) 

            loss = length

            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss {loss.item()}, Validation {(energy - length**2).item()}")