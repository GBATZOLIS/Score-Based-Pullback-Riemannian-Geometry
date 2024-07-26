import torch

class DeformedSumOfIdenticalGaussianRiemannianAutoencoder:

    def __init__(self, deformed_sum_of_identical_gaussian_pullback_manifold, epsilon):
        self.dsigpm = deformed_sum_of_identical_gaussian_pullback_manifold
        self.d = self.dsigpm.d
        self.m = self.dsigpm.dsg.m

        # construct the matrix B from A and all the offsets
        diagonal = self.dsigpm.dsg.psi[0].diagonal # TODO check whether we need the inverse of this
        offsets = torch.cat([self.dsigpm.dsg.psi[i].offset[None] for i in range(self.m)])
        weights = self.dsigpm.dsg.weights
        A = torch.diag_embed(diagonal)

        u = torch.sum(weights[:,None] * offsets, 0) / torch.sum(weights)
        B = A + 1/torch.sum(weights) * torch.einsum("i,ia,ib->ab", weights, offsets - u[None], offsets - u[None])

        Lambda, Q = torch.linalg.eigh(B)

        # construct basis from the diagonal matrix A 
        
        sorted_inv_Lambda, sorted_indices = (1/Lambda).sort()
        sorted_Lambda = 1 / sorted_inv_Lambda # largest value is first

        if sorted_Lambda[-1] <= epsilon * sorted_Lambda.sum():
            tmp = [sorted_Lambda[i+1:].sum() <= epsilon * sorted_Lambda.sum() for i in range(self.d-1)]
            self.d_eps = torch.arange(0,self.d-1)[tmp].min() + 1
            self.eps = sorted_Lambda[self.d_eps:].sum()/sorted_Lambda.sum()
        else:
            self.d_eps = self.d
            self.eps = 0.

        self.idx_eps = sorted_indices[:self.d_eps]
        self.basis_eps = torch.einsum("da,ka->kd", Q, torch.eye(self.d)[self.idx_eps]) # d_eps x d
        self.offset = u

        print(f"constructed a Riemannian autoencoder with d_eps = {self.d_eps} and eps = {self.eps}")

    def encode(self, x):
        """
        :param x: N x d tensor
        :return : N x d_eps tensor
        """
        return torch.einsum("Nd,kd->Nk",self.dsigpm.dg.phi.forward(x) - self.offset[None], self.basis_eps)

    def decode(self, p):
        """
        :param a: N x d_eps tensor
        :return : N x d tensor
        """
        return self.dsigpm.dsg.phi.inverse(self.offset[None] + torch.einsum("Nk,kd->Nd", p, self.basis_eps))

    def project_on_manifold(self, x):
        """
        :param x: N x d tensor
        :return : N x d tensor
        """
        return self.decode(self.encode(x))