import torch

class DeformedGaussianRiemannianAutoencoder:

    def __init__(self, deformed_gaussian_pullback_manifold, epsilon):
        self.dgpm = deformed_gaussian_pullback_manifold
        self.eps = epsilon
        self.d = self.dgpm.d

        # construct basis from the diagonal matrix A
        diagonal = self.dgpm.dg.psi.diagonal
        sorted_diagonal, sorted_indices = diagonal.sort()
        sorted_inv_diagonal = 1 / sorted_diagonal # largest value is first

        if sorted_inv_diagonal[-1] <= self.eps * sorted_inv_diagonal.sum():
            tmp = [sorted_inv_diagonal[i+1:].sum() <= self.eps * sorted_inv_diagonal.sum() for i in range(self.d-1)]
            print(tmp)
            self.d_eps = torch.arange(0,self.d-1)[tmp].min()
            print(self.d_eps)
        else:
            self.d_eps = self.d

        self.idx_eps = sorted_indices[:self.d_eps]
        self.basis_eps = torch.eye(self.d)[self.idx_eps] # d_eps x d

        print("constructed a Riemannian autoencoder with d_eps = {self.d_eps}")

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