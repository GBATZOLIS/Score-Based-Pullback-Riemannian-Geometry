import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from src.curves.harmonic_curves.boundary_harmonic_curve import BoundaryHarmonicCurve
from src.curves.harmonic_curves.initial_harmonic_curve import InitialHarmonicCurve
from src.manifolds import Manifold

class DiscreteTimeManifold(Manifold):
    """ Base class describing Euclidean space of dimension d under a metric with discrete time manifold mappings """

    def __init__(self, d):
        super().__init__(d)

    def metric_tensor(self, x):
        """
        :return: N x d x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def inverse_metric_tensor(self, x):
        """
        :return: N x d x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def barycentre(self, x, tol=1e-3, max_iter=50, step_size=1.):
        """

        :param x: N x d
        :return: d
        """
        k = 0
        y = torch.mean(x,0)
        
        gradient_0 = torch.mean(self.log(y, x),0)
        error = self.norm(y[None], gradient_0[None,None])
        rel_error = 1.
        while k <= max_iter and rel_error >= tol:
            gradient = torch.mean(self.log(y, x),0)
            y = y + step_size * gradient
            k+=1
            rel_error = self.norm(y[None], gradient[None,None]) / error
            print(f"iteration {k} | rel_error = {rel_error.item()}")

        print(f"gradient descent was terminated after reaching a relative error {rel_error.item()} in {k} iterations")

        return y
    
    def norm(self, x, X):
        """

        :param x: N x d
        :param X: N x M x d
        :return: N x M
        """
        metric_tensor_x = self.metric_tensor(x)
        return torch.sqrt(torch.einsum("Nab,NMa,NMb->NM", metric_tensor_x, X, X) + 1e-8)
    
    def inner(self, x, X, Y):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        metric_tensor_x = self.metric_tensor(x)
        return torch.einsum("Nab,NMa,NLb->NML", metric_tensor_x, X, Y)

    
    def geodesic(self, x, y, t, L=100, tol=1e-5, max_iter=5000, step_size=1/4):
        """

        :param x: d
        :param y: d
        :param t: N
        :return: N x d
        """
        N = t.shape # TODO actually we want to have L and later interpolate
        tau = torch.linspace(0.,1.,L)
        Z = ((1 - tau[:,None]) * x[None] + tau[:,None] * y[None])[1:-1].requires_grad_()
        
        k = 0
        while k < max_iter: 
            loss = 1/2 * (torch.sum(self.norm(Z[:-1], Z[1:,None] - Z[:-1,None])[:,0]**2) + self.norm(x[None], Z[0,None,None] - x[None,None])**2 + self.norm(Z[-1,None], y[None,None] - Z[-1,None,None])**2)

            loss.backward()

            gradient = Z.grad
            inverse_metric_tensor_Z = self.inverse_metric_tensor(Z)
            Riemannian_gradient_Z = torch.einsum("Nab,Nb->Na", inverse_metric_tensor_Z, gradient)

            with torch.no_grad():  # Disable gradient tracking for parameter updates
                Z -= step_size * Riemannian_gradient_Z
            Z.grad.zero_()  # Zero gradients manually
            
            if k % 500 == 0: # TODO geodesic discrepancy loss
                print(f"Epoch {k}, Loss {loss.item()}")

            k += 1

        return Z.detach() # TODO fix output

    def log(self, x, y, L=50):
        """

        :param x: d
        :param y: N x d
        :return: N x d
        """
        # if L is None:
        #     L = 100 # TODO
        # raise NotImplementedError(
        #     "Subclasses should implement this"
        # )

        # DEBUG ONLY
        N, _ = y.shape
        logs = torch.zeros_like(y)
        for i in range(N):
            gamma = BoundaryHarmonicCurve(self.d, 3, x, y[i])
            gamma.fit(self.geodesic_loss_function)
            # gamma.fit(self.exponential_loss_function)
            logs[i] = gamma.differential_forward(torch.zeros(1))
        return logs.detach()

    def exp(self, x, X, L=50, tol=1e-5, max_iter=50):
        """
        Use Newton Raphson for inner iteration
        :param x: d
        :param X: N x d
        :return: N x d
        """
        N, _ = X.shape
        y0 = x[None] * torch.ones(N)[:,None]
        y1 = x[None] + 1/L * X

        Z = torch.zeros(N,L+1,self.d)
        Z[:,0] = y0
        Z[:,1] = y1

        for l in range(L-1):
            # print(f"iteration {l+2}")
            y2 = y1.clone()

            y0 = y0.detach()
            y1.requires_grad_()
            y2.requires_grad_()

            k = 0
            while k < max_iter: 
                # compute gradient and jacobian components
                metric_tensor_y0 = self.metric_tensor(y0)
                metric_tensor_y1 = self.metric_tensor(y1)
                metric_tensor_gradient_y1 = torch.cat([torch.autograd.functional.jacobian(self.metric_tensor, y1[i][None]).squeeze()[None] for i in range(N)])

                # compute gradient terms
                term_1 = torch.einsum("Nab,Nb->Na", metric_tensor_y1, y1 - y2)
                term_2 = 1/2 * torch.einsum("Ncba,Nc,Nb->Na", metric_tensor_gradient_y1, y1 - y2, y1 - y2)
                term_3 = torch.einsum("Nab,Nb->Na", metric_tensor_y0, y1 - y0)

                # compute full gradients
                Fy =  term_1 + term_2 + term_3
                if torch.norm(Fy,2,-1).max() < tol:
                    break
                
                # compute jacobian terms
                term_1_gradient_y2 = - metric_tensor_y1
                term_2_gradient_y2 = torch.einsum("Ncba,Nb->Nab", metric_tensor_gradient_y1, y2 - y1)

                #compute full jacobian
                J = term_1_gradient_y2 + term_2_gradient_y2

                # solve linear systems
                s = torch.linalg.solve(J, -Fy)

                # update y2
                y2 = y2 + s

                k += 1
            y0 = y1
            y1 = y2
            Z[:,l+2] = y2.detach()

        # DEBUG
        plt.scatter(Z[0,:,0], Z[0,:,1])
        plt.show()
        
        return y2

    
    def distance(self, x, y, L=50):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def parallel_transport(self, x, X, y, L=50):
        """

        :param x: d
        :param X: N x d
        :param y: d
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def geodesic_loss_function(self, x, X, Y):
        """
        :x: N x d
        :X: N x d
        return: N
        """
        return self.norm(x, X[:,None])[:,0]**2
    
    