import torch

class Multimodal:
    def __init__(self, diffeomorphisms, strongly_convexs, weights) -> None:
        self.d = diffeomorphisms[0].d
        if len(diffeomorphisms) == 1:
            self.single_diffeomorphism = True
            self.phi = diffeomorphisms[0] # Diffeomorphism
        else:
            self.single_diffeomorphism = False
            self.phi = diffeomorphisms # [Diffeomorphism]
        self.psi = strongly_convexs # [StronlyConvex]
        self.weights = weights # [Float32]
        self.m = len(strongly_convexs)
    

    def log_density(self, x):
        N = x.shape[0]
        
        # Ensure psi_phi_x is on the same device as x
        psi_phi_x = torch.zeros((N, self.m), device=x.device)
        
        if self.single_diffeomorphism:
            phi_x = self.phi.forward(x)
            for i in range(self.m):
                psi_phi_x[:, i] = self.psi[i].forward(phi_x)
        else:
            for i in range(self.m):
                phi_x = self.phi[i].forward(x)
                psi_phi_x[:, i] = self.psi[i].forward(phi_x)

        # Ensure weights are on the same device as x
        weights = self.weights.to(x.device)
        
        # Optimize the computation using torch.logsumexp for numerical stability
        result = torch.logsumexp(-psi_phi_x + torch.log(weights)[None, :], dim=-1)

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