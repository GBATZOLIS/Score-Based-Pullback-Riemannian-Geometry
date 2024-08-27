import torch
import torch.nn as nn
import torch.optim as optim

from src.curves.initial_value.discrete import DiscreteInitialValueCurve
from src.geodesic_solvers.initial_value import InitialValueGeodesicSolver

import matplotlib.pyplot as plt

class DiscreteInitialValueGeodesicSolver(InitialValueGeodesicSolver): # TODO add termination conditions

    def __init__(self, x, X, norm, metric_tensor, gradient_metric_tensor, num_intervals=200, num_epochs=1000, tol=1e-3) -> None:
        super().__init__(DiscreteInitialValueCurve(x, X, num_intervals), norm)

        self.metric_tensor = metric_tensor
        self.gradient_metric_tensor = gradient_metric_tensor

        self.num_intervals = num_intervals
        
        self.num_epochs = num_epochs
        self.tol = tol

    def solve(self):
        y0 = self.x
        y1 = self.x + 1/self.num_intervals * self.X
        for l in range(self.num_intervals-1):
            y2 = y1.clone()

            y0 = y0.detach()
            y1.requires_grad_()
            y2.requires_grad_()

            k = 0
            error_0 = 0.
            while k < self.num_epochs: 
                # compute gradient and jacobian components
                metric_tensor_y0 = self.metric_tensor(y0[None])[0]
                metric_tensor_y1 = self.metric_tensor(y1[None])[0]
                gradient_metric_tensor_y1 = self.gradient_metric_tensor(y1[None])[0]

                # compute gradient terms
                term_1 = torch.einsum("ab,b->a", metric_tensor_y1, y1 - y2)
                term_2 = 1/2 * torch.einsum("cba,c,b->a", gradient_metric_tensor_y1, y1 - y2, y1 - y2)
                term_3 = torch.einsum("ab,b->a", metric_tensor_y0, y1 - y0)

                # compute full gradients
                Fy =  term_1 + term_2 + term_3
                if k == 0:
                    error_0 = torch.norm(Fy.clone())

                error = torch.norm(Fy)
                if error / error_0 < self.tol:
                    break
                
                # compute jacobian terms
                term_1_gradient_y2 = - metric_tensor_y1
                term_2_gradient_y2 = torch.einsum("cba,b->ab", gradient_metric_tensor_y1, y2 - y1)

                #compute full jacobian
                J = term_1_gradient_y2 + term_2_gradient_y2

                # solve linear systems
                s = torch.linalg.solve(J, -Fy)

                # update y2
                y2 = y2 + s

                k += 1

            y0 = y1
            y1 = y2
            self.geodesic.coefficients[l] = y2.detach()

            # DEBUG
            if l % 10 == 0:
                print(f"updating entry {l}")
            #     plt.scatter(self.geodesic.coefficients[0:l+1,0], self.geodesic.coefficients[0:l+1,1])
            #     plt.show()
        