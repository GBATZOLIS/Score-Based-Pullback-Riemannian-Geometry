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

    def inner(self, x, X, Y, return_inners=False):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        N, M, _ = X.shape
        _, L, _ = Y.shape
        psi_x = torch.zeros(N,self.m)
        inners_x = torch.zeros(N,M,L,self.m)
        
        for i in range(self.m):
            psi_x[:,i] = self.psi[i].forward(x)
            inners_x[:,:,:,i] = self.inner_i(x, X, Y, i)
        prefactors_i_x = (- psi_x + torch.log(self.weights[None] + 1e-8)).softmax(1)
        if return_inners:
            return torch.sum(prefactors_i_x[:,None,None,:] * inners_x, -1), inners_x
        else:
            return torch.sum(prefactors_i_x[:,None,None,:] * inners_x, -1)
    
    def inner_i(self, x, X, Y, i):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        _, M, _ = X.shape
        _, L, _ = Y.shape
        
        return torch.einsum("NMi,NLi->NML",
                                    self.psi[i].differential_grad_forward((x[:,None] * torch.ones(M)[None,:,None]).reshape(-1,self.d), X.reshape(-1,self.d)).reshape(X.shape),
                                    self.psi[i].differential_grad_forward((x[:,None] * torch.ones(L)[None,:,None]).reshape(-1,self.d), Y.reshape(-1,self.d)).reshape(Y.shape)
                                    )
    
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

    def exp(self, x, X, T=100):
        """

        :param x: d
        :param X: N x d
        :return: N x d
        """
        # if p is None:
        #     p = self.p
        N, _ = X.shape
        exps = x[None] * torch.ones(N)[:,None]
        iterates = [x[None]]
        dot_exps = X
        for t in range(T):
            tmps = exps
            exps += 1/T * dot_exps
            
            dot_exps -= 1/T + self.christoffel_operator(tmps,dot_exps,dot_exps)
            if t % 10 == 0:
                print(f"t = {t}")
                print(exps)
                print(dot_exps)
                iterates.append(exps)

        # for i in range(N):
        #     gamma = InitialHarmonicCurve(self.d, p, x, X[i])
        #     gamma.fit(self.exponential_loss_function)
        #     exps[i] = gamma.forward(torch.ones(1))
        # return exps.detach()
        iter_tensor = torch.cat(iterates)
        print(iter_tensor)
        plt.scatter(iter_tensor[:,0], iter_tensor[:,1])
        plt.show()
        return exps
    
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
    
    def christoffel_operator(self, x, X, Y):
        """
        
        :param x: N x d
        :param X: N x d
        :param Y: N x d
        return: N x d
        """
        N = x.shape[0]
        psi_i_x = torch.zeros((N,self.m))
        grad_psi_i_x = torch.zeros((N,self.m,self.d))
        for i in range(self.m):
            psi_i_x[:,i] = self.psi[i].forward(x)
            grad_psi_i_x[:,i] = self.psi[i].grad_forward(x)
        
        prefactors_i_x = (- psi_i_x + torch.log(self.weights[None] + 1e-8)).softmax(1)
        flat_i = 1 / self.diagonals[None] ** 2 
        sharp_m_x =  1 / torch.sum(prefactors_i_x[:,:,None] / self.diagonals[None] **2,1)[:,None]
        inner_x, inner_i_x = self.inner(x, X[:,None], Y[:,None], return_inners=True)

        term_1 = torch.einsum("Nmi,Ni->Nm",grad_psi_i_x, Y)[:,:,None] * (X[:,None] - sharp_m_x * flat_i * X[:,None])
        # print(f"term 1 = {term_1}")
        term_2 = torch.einsum("Nmi,Ni->Nm",grad_psi_i_x, X)[:,:,None] * (Y[:,None] - sharp_m_x * flat_i * Y[:,None])
        # print(f"term 2 = {term_2}")
        term_3 = (inner_x[:,0,0,None] - inner_i_x[:,0,0,:])[:,:,None] * sharp_m_x * grad_psi_i_x
        # print(f"term 3 = {term_3}")

        return 1/2 * torch.sum(prefactors_i_x[:,:,None] * (term_1 + term_2 - term_3),1)
    
    def geodesic_loss_function(self, x, X, Y):
        """
        :x: N x d
        :X: N x d
        return: N
        """
        return self.norm(x, X[:,None])[:,0]**2
    
    def exponential_loss_function(self, x, X, Y):
        """
        :param x: N x d
        :param X: N x d
        :param Y: N x d
        return: N
        """
        return torch.norm(Y + self.christoffel_operator(x, X, X), 2, -1)**2
    