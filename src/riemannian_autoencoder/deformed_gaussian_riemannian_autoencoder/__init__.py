import torch

class DeformedGaussianRiemannianAutoencoder:
    def __init__(self, deformed_gaussian_pullback_manifold, epsilon):
        self.dgpm = deformed_gaussian_pullback_manifold
        self.d = self.dgpm.d
        device = self.dgpm.dg.psi.diagonal.device

        # construct basis from the diagonal matrix A
        diagonal = self.dgpm.dg.psi.diagonal
        sorted_inv_diagonal, sorted_indices = (1/diagonal).sort()
        sorted_diagonal = 1 / sorted_inv_diagonal  # largest value is first

        if sorted_diagonal[-1] <= epsilon * sorted_diagonal.sum():
            tmp = [sorted_diagonal[i+1:].sum() <= epsilon * sorted_diagonal.sum() for i in range(self.d-1)]
            tmp_indices = torch.arange(0, self.d-1, device=device)[tmp]
            try:
                self.d_eps = tmp_indices.min() + 1
            except RuntimeError as e:
                print(f"Error: {e}")
                print(f"tmp_indices: {tmp_indices}")
                self.d_eps = self.d  # default value in case of error
            self.eps = sorted_diagonal[self.d_eps:].sum()/sorted_diagonal.sum()
        else:
            self.d_eps = self.d
            self.eps = 0.

        self.idx_eps = sorted_indices[:self.d_eps]
        self.basis_eps = torch.eye(self.d, device=device)[self.idx_eps]  # d_eps x d

        print(f"constructed a Riemannian autoencoder with d_eps = {self.d_eps} and eps = {self.eps}")

    def encode(self, x):
        """
        :param x: N x d tensor
        :return : N x d_eps tensor
        """
        return self.dgpm.dg.phi.forward(x)[:, self.idx_eps]

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
