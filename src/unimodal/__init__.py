import torch

class Unimodal:
    def __init__(self, diffeomorphism, strongly_convex) -> None:
        assert diffeomorphism.d == strongly_convex.d
        self.d = diffeomorphism.d
        self.phi = diffeomorphism # Diffeomorphism
        self.psi = strongly_convex # StronglyConvex
    
    def log_density(self, x):
        phi_x = self.phi.forward(x)
        psi_phi_x = self.psi.forward(phi_x)
        result = -psi_phi_x
        return result


    def score(self, x):
        """ Compute the score function using PyTorch's autograd """
        original_requires_grad = x.requires_grad if isinstance(x, torch.Tensor) else False

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        elif not x.requires_grad:
            x.requires_grad_(True)

        # Compute the log density
        y = self.log_density(x)

        # Compute the sum of log density to make it a scalar
        y_sum = y.sum()

        # Compute gradients of y_sum with respect to x using torch.autograd.grad
        gradients = torch.autograd.grad(y_sum, x, create_graph=True)

        # Ensure that gradients are computed
        if gradients[0] is None:
            raise ValueError("Gradient not computed, check the computational graph and inputs.")

        # Clone the gradients
        cloned_gradients = gradients[0].clone()

        # Reset requires_grad to its original state if necessary
        if not original_requires_grad:
            x.requires_grad_(False)

        return cloned_gradients

    
    def forward(self, x): 
        """ evaluate pseudo score """
        return self.psi.grad_forward(self.phi.forward(x))
    
    def differential_forward(self, x, X):
        """ evaluate differential of pseudo score """
        return self.psi.differential_grad_forward(self.phi.forward(x), self.phi.differential_forward(x, X))