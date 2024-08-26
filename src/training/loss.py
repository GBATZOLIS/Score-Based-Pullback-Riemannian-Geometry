import torch
import numpy as np
from src.training.train_utils import get_score_fn, get_log_density_fn
import torch.autograd as autograd

def get_loss_function(config):
    return LossFunctionWrapper(config)

class LossFunctionWrapper:
    def __init__(self, config):
        self.loss_type = config.get('loss', 'denoising score matching')
        self.args_std = config.get('std', 0.1)
        self.use_cv = config.get('use_cv', False)
        self.use_reg = config.get('use_reg', False)
        self.reg_factor = config.get('reg_factor', 1)
        self.reg_type = config.get('reg_type', 'volume')
        self.mcmc_steps = config.get('mcmc_steps', 20)
        self.epsilon = config.get('epsilon', 0.1)

        if self.loss_type == 'loglikelihood':
            self.loss_fn = loglikelihood_maximisation
            self.loss_name = "Loss/Log Likelihood"
        elif self.loss_type == 'denoising score matching':
            self.loss_fn = compute_loss_variance_reduction
            self.loss_name = "Loss/Score"
        elif self.loss_type == 'normalizing flow':
            self.loss_fn = normalizing_flow_loss
            self.loss_name = "Loss/Normalizing Flow"
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        self.reg_loss_name = "Loss/Regularization"

    def __call__(self, phi, psi, x, train, device):
        if self.loss_type == 'loglikelihood':
            return self.loss_fn(phi, psi, x, self.args_std, train, self.use_reg, self.reg_factor, self.mcmc_steps, self.epsilon, self.reg_type, device)
        elif self.loss_type == 'denoising score matching':
            return self.loss_fn(phi, psi, x, self.args_std, train, self.use_cv, self.use_reg, self.reg_factor, self.reg_type, device)
        elif self.loss_type == 'normalizing flow':
            return self.loss_fn(phi, psi, x, train, self.use_reg, self.reg_factor, self.reg_type, device)



def volume_regularisation(logabsdetjac):
    """
    Computes the volume preservation regularization term.
    
    :param logabsdetjac: The log absolute determinant of the Jacobian.
    :return: The volume preservation regularization loss.
    """
    return torch.mean(torch.abs(logabsdetjac))


def regularisation_term(reg_type, phi, x, train, device):
    if reg_type=='isometry':
        #promotes isometry regularisation

        batch_size, *dims = x.shape
        v = torch.randn(batch_size, *dims, device=device, requires_grad=True)
        v_norm = v.view(batch_size, -1).norm(dim=1, keepdim=True).view(batch_size, *[1]*len(dims))
        v = v / v_norm

        if not train:
            torch.set_grad_enabled(True)

        _, Jv = torch.autograd.functional.jvp(lambda x: phi(x), (x,), (v,), create_graph=True)

        if not train:
            torch.set_grad_enabled(False)
        
        norms = Jv.view(batch_size, -1).norm(dim=1)
        return torch.mean((norms - 1) ** 2)
    elif reg_type == 'volume':
        _, logabsdetjac = phi._transform(x, context=None)
        return torch.mean(torch.abs(logabsdetjac))

def get_energy_fn(phi, psi):
    """
    Defines the energy function E_theta(x) = psi(phi(x)).
    """
    def energy_fn(x):
        phi_x = phi.forward(x)
        return psi.forward(phi_x)
    return energy_fn

def get_fast_energy_fn(phi, psi):
    def fast_energy_fn(phi_x):
        return psi.forward(phi_x)
    return fast_energy_fn

def langevin_mcmc(energy_fn, x0, steps=20, epsilon=0.1, train=True):
    """
    Langevin MCMC to draw samples from p_theta(x).
    """
    x = x0.clone().detach().requires_grad_(True)

    if not train:
        torch.set_grad_enabled(True)
    
    for _ in range(steps):
        energy = energy_fn(x)
        grad = autograd.grad(energy.sum(), x, retain_graph=False, create_graph=False)[0]
        
        with torch.no_grad():
            x = x - 0.5 * epsilon**2 * grad + epsilon * torch.randn_like(x)
        
        # Detach x to prevent accumulation of gradients and re-enable requires_grad
        x = x.detach().requires_grad_(True)
    
    if not train:
        torch.set_grad_enabled(False)
    
    return x.detach()

def loglikelihood_maximisation(phi, psi, x, args_std, train=True, use_reg=False, reg_factor=1, mcmc_steps=20, epsilon=0.1, reg_type='volume', device='cuda:0'):
    """
    Maximizes the log-likelihood with respect to the energy model. Maximum Likelihood Training with MCMC
    """
    # Add noise to input
    x = x + args_std * torch.randn_like(x, device=device)

    # Define the energy function E_theta(x)
    energy_fn = get_energy_fn(phi, psi)

    # Run Langevin MCMC to get samples from the model distribution
    x_sampled = langevin_mcmc(energy_fn, x, mcmc_steps, epsilon, train=train)

    # Compute the gradient term for the normalizing constant Z_theta
    energy_x_sampled = energy_fn(x_sampled)
    grad_log_z_theta = energy_x_sampled.mean()

    # Compute the energy of the original inputs
    energy_x = energy_fn(x)

    # Compute density learning loss (negative log likelihood)
    density_learning_loss = energy_x.mean() - grad_log_z_theta

    # Regularization term if applicable
    reg_loss = regularisation_term(reg_type, phi, x, train, device) if use_reg else torch.tensor(0.0, device=device)

    # Total loss
    loss = density_learning_loss + reg_factor * reg_loss
    
    return loss, density_learning_loss, reg_loss

def normalizing_flow_loss(phi, psi, x, train=True, use_reg=False, reg_factor=1, reg_type='volume', device='cuda:0'):
    """
    Maximizes the log-likelihood with respect to the normalizing flow model.
    """
    # Derive the diagonal covariance matrix from psi
    diagonal = psi.diagonal
    base_dist_cov = torch.diag(diagonal)
    
    # Apply the transformation
    z, logabsdetjacobian = phi._transform(x, context=None)

    # Base distribution is N(0, D)
    base_dist = torch.distributions.MultivariateNormal(
        loc=torch.zeros_like(z, device=device),
        covariance_matrix=base_dist_cov
    )

    # Compute the negative log-likelihood
    log_p_z = base_dist.log_prob(z)
    log_p_x = log_p_z + logabsdetjacobian
    nll_loss = -torch.mean(log_p_x)

    # Regularization term if applicable
    reg_loss = regularisation_term(reg_type, phi, x, train, device) if use_reg else torch.tensor(0.0, device=device)

    # Total loss
    loss = nll_loss + reg_factor * reg_loss

    return loss, nll_loss, reg_loss


def compute_loss_variance_reduction(phi, psi, x, args_std, train=True, use_cv=False, use_reg=False, reg_factor=1, reg_type='volume', device='cuda:0'):
    def score_matching_loss():
        x.requires_grad_(True)  # Ensure x requires gradients
        batch_size = x.size(0)
        std = args_std * torch.ones_like(x, device=device)
        score_fn = get_score_fn(phi, psi, train=train)
        z = torch.randn_like(x, device=device)
        x_pert = x + std * z
        score = score_fn(x_pert)
        term_inside_norm = (z / std + score)
        loss = torch.mean(torch.norm(term_inside_norm.view(batch_size, -1), dim=1)**2) / 2

        if use_cv:
            grad_log_p = score_fn(x)  # ∇_x log p_theta(x)
            control_variate = (2 / args_std) * (z * grad_log_p).view(batch_size, -1).sum(dim=1) + \
                              (z.view(batch_size, -1).norm(dim=1) ** 2 / args_std ** 2) - \
                              torch.prod(torch.tensor(x.shape[1:], device=device)) / args_std ** 2
            loss -= torch.mean(control_variate)

        return loss

    density_learning_loss = score_matching_loss()
    reg_loss = regularisation_term(reg_type, phi, x, train, device) if use_reg else torch.tensor(0.0, device=device) #use_reg or not train 

    loss = density_learning_loss + (reg_factor * reg_loss if use_reg else torch.tensor(0.0, device=device))
    
    return loss, density_learning_loss, reg_loss