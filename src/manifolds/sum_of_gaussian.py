import torch

from src.curves.boundary_value_curve import BoundaryPolynomialCurve
from src.manifolds import Manifold

class SumOfGaussian(Manifold): # TODO get the curve classes in here and use inner or norm squared as loss function
    """ Base class describing Euclidean space of dimension d under a sum of gaussian metric """

    def __init__(self, d, strongly_convexs, weights, p=3):
        super().__init__(d)
        self.psi = strongly_convexs
        self.weights = weights
        self.m = len(strongly_convexs)
        self.p = p

    def barycentre(self, x): # TODO use here standard definition for barycentre through solving minimisation problem
        """

        :param x: N x d
        :return: d
        """
        # TODO this will be very hard to compute... Can use CPPA paper perhaps?
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def inner(self, x, X, Y):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        N, M, _ = X.shape
        _, L, _ = Y.shape
        psi_x = torch.zeros(self.m,N)
        inners_x = torch.zeros(self.m,N,M,L)
        
        for i in range(self.m):
            psi_x[i] = self.psi[i].forward(x)
            inners_x[i] = torch.einsum("NMi,NLi->NML",
                                     self.psi[i].differential_grad_forward((x[:,None] * torch.ones(M)[None,:,None]).reshape(-1,self.d), X.reshape(-1,self.d)).reshape(X.shape),
                                     self.psi[i].differential_grad_forward((x[:,None] * torch.ones(L)[None,:,None]).reshape(-1,self.d), Y.reshape(-1,self.d)).reshape(Y.shape)
                                     )
        prefactors = (- psi_x + torch.log(self.weights[:,None] + 1e-8)).softmax(0)
        return torch.sum(prefactors[:,:,None,None] * inners_x, 0)
    
    def geodesic(self, x, y, t, p=None):
        """

        :param x: d
        :param y: d
        :param t: N
        :return: N x d
        """
        if p is None:
            p = self.p
        gamma = BoundaryPolynomialCurve(self.d, p, x, y)
        gamma.fit(self.geodesic_loss_function)
        return gamma.forward(t)

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
            gamma = BoundaryPolynomialCurve(self.d, p, x, y[i])
            gamma.fit(self.geodesic_loss_function)
            logs[i] = gamma.differential_forward(torch.zeros(1))
        return logs

    def exp(self, x, X, p=None):
        """

        :param x: d
        :param X: N x d
        :return: N x d
        """
        if p is None:
            p = self.p
        # TODO solve optimisation problem
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
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

    def parallel_transport(self, x, X, y, p=None): # TODO make this into pole ladder or Schild's ladder
        """

        :param x: d
        :param X: N x d
        :param y: d
        :return: N x d
        """
        if p is None:
            p = self.p
        # TODO maybe just leave this...
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def geodesic_loss_function(self, x, X):
            """
            :x: N x d
            :X: N x d
            return: N
            """
            return self.norm(x, X[:,None])**2