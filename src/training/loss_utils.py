import torch
import random
import functorch
import torch.func as func

def regularisation_term(reg_type, phi, x, train, device, reg_iso_type='length', logabsdetjac=None, phi_x=None, psi=None):
    if reg_type == 'isometry':
        iso_reg = isometry_regularisation(reg_iso_type, phi, x, train, device)
        return iso_reg, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    elif reg_type == 'volume':
        volume_reg = volume_regularisation(phi, x, logabsdetjac)
        return torch.tensor(0.0, device=device), volume_reg, torch.tensor(0.0, device=device)

    elif reg_type in ['volume+isometry', 'isometry+volume']:
        # Isometry regularization
        iso_reg = isometry_regularisation(reg_iso_type, phi, x, train, device)

        # Volume regularization
        volume_reg = volume_regularisation(phi, x, logabsdetjac)

        return iso_reg, volume_reg, torch.tensor(0.0, device=device)

    elif reg_type == 'isometry+volume+hessian':
        # Isometry regularization
        iso_reg = isometry_regularisation(reg_iso_type, phi, x, train, device)

        # Volume regularization
        volume_reg = volume_regularisation(phi, x, logabsdetjac)

        # Hessian regularization (v is generated inside the hessian regularisation)
        hessian_reg = hessian_regularisation(phi, x, phi_x, psi, train)

        return iso_reg, volume_reg, hessian_reg

    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")


def isometry_regularisation(reg_iso_type, phi, x, train, device):
    batch_size, *dims = x.shape

    if reg_iso_type == 'length':
        # Generate v for length regularisation
        v = torch.randn(batch_size, *dims, device=device, requires_grad=True)
        return length_regularisation(phi, x, v, train, device)
    
    elif reg_iso_type == 'angle':
        # Generate two random perturbation vectors v1 and v2 for angle regularisation
        v1 = torch.randn(batch_size, *dims, device=device, requires_grad=True)
        v2 = torch.randn(batch_size, *dims, device=device, requires_grad=True)
        return angle_regularisation(phi, x, v1, v2, train, device)
    
    elif reg_iso_type in ['length+angle', 'angle+length']:
        # Generate two random perturbation vectors v1 and v2 for length+angle regularisation
        v1 = torch.randn(batch_size, *dims, device=device, requires_grad=True)
        v2 = torch.randn(batch_size, *dims, device=device, requires_grad=True)
        return length_and_angle_regularisation(phi, x, v1, v2, train, device)

    elif reg_iso_type == 'orthogonal-jacobian':
        # Enforce orthogonality of the Jacobian
        return orthogonal_jacobian_regularisation(phi, x, device)

    else:
        raise ValueError(f"Unknown isometry regularization type: {reg_iso_type}")


def orthogonal_jacobian_regularisation(phi, x, device):
    # Compute the Jacobian using vmap and functorch for efficiency
    def flow_fn(x_single):
        z = phi(x_single.unsqueeze(0), detach_logdet=True).squeeze(0)  # Ensure logabsdet is detached during Jacobian calculation
        return z

    # Vectorize the Jacobian computation for the batch
    jacobian = torch.vmap(func.jacrev(flow_fn))(x)

    # Flatten the input across all dimensions except batch size
    batch_size = x.shape[0]
    input_dim = x[0].numel()  # Calculate the total number of input dimensions after the batch dimension

    # Reshape the Jacobian to (batch_size, input_dim, input_dim)
    jacobian = jacobian.view(batch_size, input_dim, input_dim)

    # Enforce orthogonality: J^T J should be close to identity matrix I
    identity = torch.eye(input_dim, device=device)
    JTJ = torch.matmul(jacobian.transpose(1, 2), jacobian)  # J^T * J
    ortho_reg = torch.mean((JTJ - identity).norm(dim=(1, 2)) ** 2)

    return ortho_reg


def length_regularisation(phi, x, v, train, device):
    batch_size, *dims = x.shape
    v_norm = v.view(batch_size, -1).norm(dim=1, keepdim=True).view(batch_size, *[1]*len(dims))
    v = v / v_norm

    if not train:
        torch.set_grad_enabled(True)

    #print(f'x.requires_grad:{x.requires_grad}')
    #print(f'v.requires_grad:{x.requires_grad}')

    _, Jv = torch.autograd.functional.jvp(lambda x: phi(x), (x,), (v,), create_graph=True)

    #print(f'Jv.requires_grad:{Jv.requires_grad}')

    if not train:
        torch.set_grad_enabled(False)

    norms = Jv.view(batch_size, -1).norm(dim=1)
    length_reg = torch.mean((norms - 1) ** 2)
    return length_reg


def angle_regularisation(phi, x, v1, v2, train, device):
    batch_size, *dims = x.shape
    v1_norm = v1.view(batch_size, -1).norm(dim=1, keepdim=True).view(batch_size, *[1]*len(dims))
    v2_norm = v2.view(batch_size, -1).norm(dim=1, keepdim=True).view(batch_size, *[1]*len(dims))
    v1, v2 = v1 / v1_norm, v2 / v2_norm

    if not train:
        torch.set_grad_enabled(True)

    _, Jv1 = torch.autograd.functional.jvp(lambda x: phi(x), (x,), (v1,), create_graph=True)
    _, Jv2 = torch.autograd.functional.jvp(lambda x: phi(x), (x,), (v2,), create_graph=True)

    if not train:
        torch.set_grad_enabled(False)

    # Compute the cosine similarity between Jv1 and Jv2
    dot_product = (Jv1.view(batch_size, -1) * Jv2.view(batch_size, -1)).sum(dim=1)
    norm_Jv1 = Jv1.view(batch_size, -1).norm(dim=1)
    norm_Jv2 = Jv2.view(batch_size, -1).norm(dim=1)
    cos_theta_prime = dot_product / (norm_Jv1 * norm_Jv2)

    # Regularize based on angle preservation
    angle_reg = torch.mean((cos_theta_prime - (v1.view(batch_size, -1) * v2.view(batch_size, -1)).sum(dim=1)) ** 2)
    return angle_reg


def length_and_angle_regularisation(phi, x, v1, v2, train, device):
    batch_size, *dims = x.shape
    v1_norm = v1.view(batch_size, -1).norm(dim=1, keepdim=True).view(batch_size, *[1]*len(dims))
    v2_norm = v2.view(batch_size, -1).norm(dim=1, keepdim=True).view(batch_size, *[1]*len(dims))
    v1, v2 = v1 / v1_norm, v2 / v2_norm

    if not train:
        torch.set_grad_enabled(True)

    _, Jv1 = torch.autograd.functional.jvp(lambda x: phi(x), (x,), (v1,), create_graph=True)
    _, Jv2 = torch.autograd.functional.jvp(lambda x: phi(x), (x,), (v2,), create_graph=True)

    if not train:
        torch.set_grad_enabled(False)

    # Length preservation for Jv1 and Jv2
    norm_Jv1 = Jv1.view(batch_size, -1).norm(dim=1)
    norm_Jv2 = Jv2.view(batch_size, -1).norm(dim=1)
    length_reg = 0.5*(torch.mean((norm_Jv1 - 1) ** 2) + torch.mean((norm_Jv2 - 1) ** 2))

    # Angle preservation
    dot_product = (Jv1.view(batch_size, -1) * Jv2.view(batch_size, -1)).sum(dim=1)
    cos_theta_prime = dot_product / (norm_Jv1 * norm_Jv2)
    cos_theta = (v1.view(batch_size, -1) * v2.view(batch_size, -1)).sum(dim=1)

    angle_reg = torch.mean((cos_theta_prime - cos_theta) ** 2)

    # Combine both regularization terms
    return length_reg + angle_reg


def volume_regularisation(phi, x, logabsdetjac=None):
    if logabsdetjac is None:
        _, logabsdetjac = phi._transform(x, context=None)
    volume_reg = torch.mean(torch.abs(logabsdetjac))
    return volume_reg


def hessian_regularisation(phi, x, phi_x, psi, train):
    """
    Computes the Hessian-vector product for a randomly sampled dimension and returns
    the L2-norm of the result as the regularization term.
    
    Args:
        phi (callable): The flow transformation function.
        x (torch.Tensor): Input tensor.
        phi_x (torch.Tensor): Precomputed transformation φ(x).
        psi (torch.Tensor): The diagonal covariance matrix parameters.

    Returns:
        torch.Tensor: L2 norm of the Hessian-vector product for the sampled dimension.
    """
    
    # Dimensionality of the input
    d = x.shape[-1]  
    
    # Randomly sample one dimension
    j = random.randint(0, d-1)

    # Inverse of the diagonal covariance matrix Σ^{-1}
    sigma_inv = 1.0 / psi.diagonal.detach()  # Assuming psi.diagonal provides the diagonal elements
    
    # Reshape sigma_inv to be broadcastable with phi_x
    v = phi_x * sigma_inv.unsqueeze(0)  # Shape: (batch_size, d)

    # Compute the Hessian-vector product for the sampled dimension
    def scalar_phi_component_batch(i, x):
        return phi(x)[:, i].sum()  # Return the i-th component for the entire batch

    #check gradient requirements
    #print(f'x.requires_grad:{x.requires_grad}')
    #print(f'v.requires_grad:{v.requires_grad}')

    if not train:
        torch.set_grad_enabled(True)

    _, hvp_j_batch = torch.autograd.functional.hvp(lambda x: scalar_phi_component_batch(j, x), x, v, create_graph=True)
    
    #print("hvp_j_batch.requires_grad:", hvp_j_batch.requires_grad)

    if not train:
        torch.set_grad_enabled(False)
    

    # Return the L2 norm (2-norm) of the Hessian-vector product
    hessian_reg = torch.norm(hvp_j_batch, p=2)
    
    return hessian_reg

