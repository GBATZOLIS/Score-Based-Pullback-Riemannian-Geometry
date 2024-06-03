import torch

import matplotlib.pyplot as plt

from src.curves.harmonic_curves.boundary_harmonic_curve import BoundaryHarmonicCurve
from src.curves.harmonic_curves.initial_harmonic_curve import InitialHarmonicCurve
from src.manifolds import Manifold

class SumOfGaussian(Manifold): # TODO get the curve classes in here and use inner or norm squared as loss function
    """ Base class describing Euclidean space of dimension d under a sum of gaussian metric """

    def __init__(self, d, strongly_convexs, weights, p=3):
        super().__init__(d)
        self.psi = strongly_convexs
        self.weights = weights
        self.m = len(strongly_convexs)
        self.p = p

        self.diagonals = torch.cat([self.psi[i].diagonal[None] for i in range(self.m)])

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
        return torch.sqrt(torch.einsum("Na,NMa,NMa->NM", metric_tensor_x, X, X))
    
    def inner(self, x, X, Y):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        metric_tensor_x = self.metric_tensor(x)
        return torch.einsum("Na,NMa,NLa->NML", metric_tensor_x, X, Y)

    # def inner(self, x, X, Y, return_inners=False):
    #     """

    #     :param x: N x d
    #     :param X: N x M x d
    #     :param Y: N x L x d
    #     :return: N x M x L
    #     """
    #     N, M, _ = X.shape
    #     _, L, _ = Y.shape
    #     psi_x = torch.zeros(N,self.m)
    #     inners_x = torch.zeros(N,M,L,self.m,self.m)
        
    #     for i in range(self.m):
    #         psi_x[:,i] = self.psi[i].forward(x)

    #     for i in range(self.m):
    #         for j in range(self.m):
    #             inners_x[:,:,:,i,j] = self.inner_i(x, X, Y, i, j)
    #     prefactors_i_x = (- psi_x + torch.log(self.weights[None] + 1e-8)).softmax(1)
    #     if return_inners:
    #         return torch.sum(prefactors_i_x[:,None,None,:] * inners_x, -1), inners_x
    #     else:
    #         return torch.sum(prefactors_i_x[:,None,None,:] * inners_x, -1)
    
    # def inner_ij(self, x, X, Y, i, j):
    #     """

    #     :param x: N x d
    #     :param X: N x M x d
    #     :param Y: N x L x d
    #     :return: N x M x L
    #     """
    #     _, M, _ = X.shape
    #     _, L, _ = Y.shape
        
    #     return torch.einsum("NMi,NLi->NML",
    #                                 self.psi[i].differential_grad_forward((x[:,None] * torch.ones(M)[None,:,None]).reshape(-1,self.d), X.reshape(-1,self.d)).reshape(X.shape),
    #                                 self.psi[i].differential_grad_forward((x[:,None] * torch.ones(L)[None,:,None]).reshape(-1,self.d), Y.reshape(-1,self.d)).reshape(Y.shape)
    #                                 )
    
    def metric_tensor(self, x):
        """
        :return: N x d
        """
        N, _ = x.shape
        psi_x = torch.zeros(N,self.m)
        for i in range(self.m):
            psi_x[:,i] = self.psi[i].forward(x)
        softmax_psi_x = (- psi_x + torch.log(self.weights[None] + 1e-8)).softmax(1)
        return torch.sum((softmax_psi_x[:,:,None] / self.diagonals[None]**2), 1)
        # return torch.sum((softmax_psi_x[:,:,None] / self.diagonals[None])[:,:,None,:] * (softmax_psi_x[:,:,None] / self.diagonals[None])[:,None,:,:],[1,2])

    
    def geodesic(self, x, y, t, p=None):
        """

        :param x: d
        :param y: d
        :param t: N
        :return: N x d
        """
        if p is None:
            p = self.p
        gamma = BoundaryHarmonicCurve(self.d, p, x, y)
        gamma.fit(self.geodesic_loss_function)
        return gamma.forward(t).detach()

    def log(self, x, y, p=None):
        """

        :param x: d
        :param y: N x d
        :return: N x d
        """
        if p is None:
            p = self.p
        N, _ = y.shape
        logs = torch.zeros_like(y)
        for i in range(N):
            gamma = BoundaryHarmonicCurve(self.d, p, x, y[i])
            gamma.fit(self.geodesic_loss_function)
            # gamma.fit(self.exponential_loss_function)
            logs[i] = gamma.differential_forward(torch.zeros(1))
        return logs.detach()

    def exp(self, x, X, L=None):
        """

        :param x: d
        :param X: N x d
        :return: N x d
        """
        if L is None:
            L = 100 # TODO
        N, _ = X.shape
        y = x[None] * torch.ones(N)[:,None]
        z = x[None] + 1/L * X

        Z = torch.zeros(N,L+1,self.d)
        Z[:,0] = y
        Z[:,1] = z

        metric_tensor_y = self.metric_tensor(y)
        metric_tensor_z = self.metric_tensor(z)
        for l in range(L-1):
            print(f"iteration {l+2}")
            tmp = (2 * metric_tensor_z * z - metric_tensor_y * y) / (2 * metric_tensor_z - metric_tensor_y)
            metric_tensor_tmp = self.metric_tensor(tmp)

            y = z
            z = tmp
            metric_tensor_y = metric_tensor_z
            metric_tensor_z = metric_tensor_tmp
            Z[:,l+2] = z

        plt.scatter(Z[0,:,0], Z[0,:,1])
        plt.show()
        
        return z

    
    def distance(self, x, y, p=None):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L
        """
        if p is None:
            p = self.p
        # TODO solve optimisation problem
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def parallel_transport(self, x, X, y, p=None): # TODO now that we have christoffels, we can also solve this (first geo than pt)
        """

        :param x: d
        :param X: N x d
        :param y: d
        :return: N x d
        """
        if p is None:
            p = self.p
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
    