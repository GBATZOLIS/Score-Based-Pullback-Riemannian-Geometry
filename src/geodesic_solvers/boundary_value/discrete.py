import torch
import torch.nn as nn
import torch.optim as optim

from src.curves.boundary_value.discrete import DiscreteBoundaryValueCurve
from src.geodesic_solvers.boundary_value import BoundaryValueGeodesicSolver
from src.geodesic_solvers.boundary_value.two_step_solvers.harmonic_rq_spline import HarmonicRationalQuadraticSplineFlowBoundaryValueGeodesicSolver

import matplotlib.pyplot as plt

class DiscreteBoundaryValueGeodesicSolver(BoundaryValueGeodesicSolver): # TODO add termination conditions

    def __init__(self, x, y, norm, num_intervals=200, num_time_points=None, num_epochs=1000, lr=1e-4, weight_decay=0., initialize=True, num_sines=1, num_bins=None) -> None:
        if num_time_points is None:
            num_time_points = num_intervals + 1
        assert num_time_points > num_intervals
        super().__init__(DiscreteBoundaryValueCurve(x, y, num_intervals), norm)

        self.num_intervals = num_intervals
        
        self.num_time_points = num_time_points
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.initialize = initialize
        self.num_sines = num_sines
        self.num_bins = num_bins

    def initialize_geodesic(self):
        init_geo_solver = HarmonicRationalQuadraticSplineFlowBoundaryValueGeodesicSolver(self.x, self.y, self.norm, num_sines=self.num_sines, num_bins=self.num_bins)
        init_geo_solver.solve()
        init_geo = init_geo_solver.geodesic
        self.geodesic.coefficients = nn.Parameter(init_geo(torch.linspace(0.,1.,self.num_intervals+1))[1:self.num_intervals])

    def solve(self):
        if self.initialize:
            self.initialize_geodesic()

        optimizer = optim.SGD(self.geodesic.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        t = torch.linspace(0.,1.,self.num_time_points)
        
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            gamma_t = self.geodesic(t)

            segment_lengths = self.norm(gamma_t[0:self.num_time_points-1], gamma_t[1:,None] - gamma_t[0:self.num_time_points-1,None])
            length = torch.sum(segment_lengths)
            energy = (self.num_time_points-1) * torch.sum(segment_lengths ** 2) 

            loss = energy

            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss {loss.item()}, Validation {(energy - length**2).item()}")