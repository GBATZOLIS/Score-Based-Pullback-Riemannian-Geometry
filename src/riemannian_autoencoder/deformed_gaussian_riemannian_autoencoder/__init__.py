import torch

class DeformedGaussianRiemannianAutoencoder:

    def __init__(self, deformed_gaussian_pullback_manifold, epsilon):
        self.dgpm = deformed_gaussian_pullback_manifold
        self.d = self.dgpm.d

        # construct basis from the diagonal matrix A
        diagonal = self.dgpm.dg.psi.diagonal
        sorted_diagonal, sorted_indices = diagonal.sort()
        sorted_inv_diagonal = 1 / sorted_diagonal # largest value is first

        if sorted_inv_diagonal[-1] <= epsilon * sorted_inv_diagonal.sum():
            tmp = [sorted_inv_diagonal[i+1:].sum() <= epsilon * sorted_inv_diagonal.sum() for i in range(self.d-1)]
            self.d_eps = torch.arange(0,self.d-1)[tmp].min() + 1
            self.eps = sorted_inv_diagonal[self.d_eps:].sum()/sorted_inv_diagonal.sum()
        else:
            self.d_eps = self.d
            self.eps = 0.

        self.idx_eps = sorted_indices[:self.d_eps]
        self.basis_eps = torch.eye(self.d)[self.idx_eps] # d_eps x d

        print(f"constructed a Riemannian autoencoder with d_eps = {self.d_eps} and eps = {self.eps}")

    def encode(self, x):
        """
        :param x: N x d tensor
        :return : N x d_eps tensor
        """
        return self.dgpm.dg.phi.forward(x)[:,self.idx_eps]

    def decode(self, p):
        """
        :param a: N x d_eps tensor
        :return : N x d tensor
        """
        return self.dgpm.dg.phi.inverse(torch.einsum("Nk,kd->Nd", p, self.basis_eps))

    def project_on_manifold(self, x):
        """
        :param x: N x d tensor
        :return : N x d tensor
        """
        return self.decode(self.encode(x))