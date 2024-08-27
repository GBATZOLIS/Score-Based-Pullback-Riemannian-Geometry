import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from src.time_flow_solvers import TimeFlowSolver
from src.time_flows.rq_spline import RationalQuadraticSplineFlow

class RationalQuadraticSplineFlowSolver(TimeFlowSolver):
    """ Base class describing a time flow solver """

    def __init__(self, num_bins, loss_function, num_time_points=100, num_epochs=100, lr=0.1, weight_decay=0.) -> None:
        super().__init__(RationalQuadraticSplineFlow(num_bins),
                         loss_function, num_time_points=num_time_points, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay)

    def solve(self):
        optimizer = optim.Adam(self.time_flow.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        t = torch.linspace(0.,1.,self.num_time_points)
        
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            s_t = self.time_flow.forward(t)

            segment_energies = self.loss_function(s_t)
            length = torch.sum(torch.sqrt(segment_energies))
            energy = (self.num_time_points - 1) * torch.sum(segment_energies) 
            
            loss = energy

            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss {loss.item()}, Validation {(energy - length**2).item()}")
                