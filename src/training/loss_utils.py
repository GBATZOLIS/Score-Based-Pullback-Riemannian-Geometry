import torch
import random
import functorch
import torch.func as func

def regularisation_term(reg_type, phi, x, train, device, reg_iso_type='length', logabsdetjac=None, phi_x=None, psi=None, num_v=8):
    if reg_type == 'isometry':
        iso_reg = isometry_regularisation(reg_iso_type, phi, x, train, device, num_v)
        return iso_reg, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    elif reg_type == 'volume':
        volume_reg = volume_regularisation(phi, x, logabsdetjac)
        return torch.tensor(0.0, device=device), volume_reg, torch.tensor(0.0, device=device)

    elif reg_type in ['volume+isometry', 'isometry+volume']:
        # Isometry regularization
        iso_reg = isometry_regularisation(reg_iso_type, phi, x, train, device, num_v)

        # Volume regularization
        volume_reg = volume_regularisation(phi, x, logabsdetjac)

        return iso_reg, volume_reg, torch.tensor(0.0, device=device)

    elif reg_type == 'isometry+volume+hessian':
        # Isometry regularization
        iso_reg = isometry_regularisation(reg_iso_type, phi, x, train, device, num_v)

        # Volume regularization
        volume_reg = volume_regularisation(phi, x, logabsdetjac)

        # Hessian regularization (v is generated inside the hessian regularisation)
        hessian_reg = hessian_regularisation(phi, x, phi_x, psi, train)

        return iso_reg, volume_reg, hessian_reg

    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")


def isometry_regularisation(reg_iso_type, phi, x, train, device, num_v=8):
    batch_size, *dims = x.shape

    if reg_iso_type == 'length':
        # Generate v for length regularisation
        #v = torch.randn(batch_size, *dims, device=device, requires_grad=True)
        return length_regularisation(phi, x, num_v, train, device)
    
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
    
    elif reg_iso_type == 'approximate-orthogonal-jacobian':
        # Enforce orthogonality of the Jacobian using the approximate method
        return approximate_orthogonal_jacobian_regularisation(phi, x, num_v, train, device)
    
    else:
        raise ValueError(f"Unknown isometry regularization type: {reg_iso_type}")


def approximate_orthogonal_jacobian_regularisation(phi, x, num_v, train, device):
    batch_size, *dims = x.shape
    input_dim = x[0].numel()  # Total number of input dimensions after batch dimension

    # Flatten x to (batch_size, input_dim)
    #x_flat = x.view(batch_size, -1).requires_grad_(True)

    # Generate orthonormal vectors per batch sample using batched QR decomposition
    # Create a random matrix of shape (batch_size, input_dim, num_v)
    random_matrix = torch.randn(batch_size, input_dim, num_v, device=device)

    # Perform batched QR decomposition to obtain orthonormal vectors
    q, _ = torch.linalg.qr(random_matrix)  # q has shape (batch_size, input_dim, num_v)

    # Transpose q to get v_flat of shape (batch_size, num_v, input_dim)
    v_flat = q.permute(0, 2, 1)  # Shape: (batch_size, num_v, input_dim)

    # Reshape v_flat to match the input dimensions
    v = v_flat.view(batch_size, num_v, *dims)  # Shape: (batch_size, num_v, *dims)

    # Permute v to (num_v, batch_size, *dims) for compatibility with vmap
    v = v.permute(1, 0, *range(2, v.ndim))  # Shape: (num_v, batch_size, *dims)

    # Ensure that x has requires_grad=True
    x = x.requires_grad_(True)

    if not train:
        torch.set_grad_enabled(True)

    # Use functorch.jvp to compute the JVP
    def jvp_fn(v_single):
        return func.jvp(phi, (x,), (v_single,))[1]

    # Apply vmap over the num_v perturbations to compute the JVP for all v in parallel
    Jv = func.vmap(jvp_fn)(v)  # Shape: (num_v, batch_size, *output_dims)

    if not train:
        torch.set_grad_enabled(False)

    # Flatten Jv to (num_v, batch_size, output_dim)
    Jv = Jv.view(num_v, batch_size, -1)  # Shape: (num_v, batch_size, output_dim)

    # Rearrange Jv to (batch_size, num_v, output_dim)
    Jv = Jv.permute(1, 0, 2)  # Shape: (batch_size, num_v, output_dim)

    # Compute the Gram matrix G = Jv @ Jv^T for each sample in the batch
    G = torch.bmm(Jv, Jv.transpose(1, 2))  # Shape: (batch_size, num_v, num_v)

    # Create identity matrix I of size (batch_size, num_v, num_v)
    I = torch.eye(num_v, device=device).unsqueeze(0).expand(batch_size, num_v, num_v)

    # Compute G - I
    G_minus_I = G - I

    # Compute the Frobenius norm squared of G_minus_I for each sample
    frob_norm_squared = (G_minus_I ** 2).sum(dim=(1, 2))  # Shape: (batch_size,)

    # Compute the mean over the batch
    ortho_reg = torch.mean(frob_norm_squared)

    # Scale the loss to compensate for computing only a subset of entries
    #scaling_factor = (input_dim / num_v) ** 2
    #ortho_reg = scaling_factor * ortho_reg

    return ortho_reg


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



def length_regularisation(phi, x, num_v, train, device):
    batch_size, *dims = x.shape
    flat_dims = batch_size, -1
    
    # Generate num_v random perturbation vectors v
    v = torch.randn(num_v, batch_size, *dims, device=device)
    
    # Normalize v
    v_norm = v.view(num_v, *flat_dims).norm(dim=2, keepdim=True).view(num_v, batch_size, *[1]*len(dims))
    v = v / v_norm

    # Ensure that x has requires_grad=True outside of functorch transformations
    x = x.requires_grad_(True)

    if not train:
        torch.set_grad_enabled(True)

    # Use functorch.jvp to compute the JVP
    def jvp_fn(v_single):
        return func.jvp(phi, (x,), (v_single,))[1]

    # Apply vmap over the num_v perturbations to compute the JVP for all v in parallel
    Jv = func.vmap(jvp_fn)(v)

    if not train:
        torch.set_grad_enabled(False)

    # Compute the norms of Jv
    norms = Jv.view(num_v, *flat_dims).norm(dim=2)

    # Compute the average length regularization across all perturbations
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
        x (torch.Tensor): Input tensor with image shape (batch_size, channels, height, width).
        phi_x (torch.Tensor): Flattened output of φ(x) with shape (batch_size, flattened_dim).
        psi (torch.Tensor): The diagonal covariance matrix parameters.
        train (bool): Whether the model is in training mode.

    Returns:
        torch.Tensor: L2 norm of the Hessian-vector product for the sampled dimension.
    """
    # Flatten x to match the shape of phi_x
    batch_size = x.size(0)
    x_flat = x.view(batch_size, -1)  # Shape: (batch_size, flattened_dim)

    # Dimensionality of the flattened input
    d_flat = x_flat.size(1)

    # Randomly sample one dimension
    j = random.randint(0, d_flat - 1)

    # Inverse of the diagonal covariance matrix Σ^{-1}
    sigma_inv = 1.0 / psi.diagonal.detach()  # Assuming psi.diagonal provides the diagonal elements
    sigma_inv = sigma_inv.unsqueeze(0)  # Shape: (1, flattened_dim) for broadcasting

    # Create v aligned with x_flat
    v = phi_x * sigma_inv  # Shape: (batch_size, flattened_dim)

    # Define scalar_phi_component_batch for the j-th dimension
    def scalar_phi_component_batch(i, x_flat):
        # Reshape x_flat back to image shape for φ if needed
        x_image = x_flat.view_as(x)
        return phi(x_image)[:, i].sum()  # Use φ(x)[:, i] for the batch

    if not train:
        torch.set_grad_enabled(True)

    # Compute Hessian-vector product
    _, hvp_j_batch = torch.autograd.functional.hvp(
        lambda x_flat: scalar_phi_component_batch(j, x_flat), x_flat, v, create_graph=True
    )

    if not train:
        torch.set_grad_enabled(False)

    # Return the L2 norm (2-norm) of the Hessian-vector product
    hessian_reg = torch.norm(hvp_j_batch, p=2)

    return hessian_reg


