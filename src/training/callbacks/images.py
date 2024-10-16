import os
import torch
import torchvision.utils as vutils
from math import sqrt, ceil
import random
import imageio
import numpy as np
from src.manifolds.deformed_gaussian_pullback_manifold import DeformedGaussianPullbackManifold
from src.riemannian_autoencoder.deformed_gaussian_riemannian_autoencoder import DeformedGaussianRiemannianAutoencoder
from src.unimodal import Unimodal
from src.training.callbacks.utils import check_orthogonality, deviation_from_volume_preservation
import matplotlib.pyplot as plt
import time 
from tqdm import tqdm

def generate_and_plot_samples_images(phi, psi, num_samples, device, writer, epoch):
    c, h, w = phi.args.c, phi.args.h, phi.args.w
    base_samples = torch.randn(num_samples, c*h*w, device=device) * psi.diagonal.sqrt()
    transformed_samples = phi.inverse(base_samples)
    grid_images = vutils.make_grid(transformed_samples, nrow=int(sqrt(num_samples)), padding=2, normalize=True, scale_each=True)
    writer.add_image("Generated Samples", grid_images, epoch)

def create_interpolation_pairs(images, num_pairs, num_interpolations, device, manifold, method='random'):
    num_images = images.size(0)
    pairs = []

    if method == 'random':
        while len(pairs) < num_pairs:
            i, j = random.sample(range(num_images), 2)
            if j != i and (i, j) not in pairs and (j, i) not in pairs:
                pairs.append((i, j))
    elif method == 'max_distance':
        # Flatten the images for distance calculation
        flattened_images = images.view(num_images, -1)

        # Compute all pairwise distances
        distances = []
        for i in range(num_images):
            for j in range(i + 1, num_images):
                dist = torch.norm(flattened_images[i] - flattened_images[j], p=2).item()
                distances.append((dist, i, j))

        # Sort distances in decreasing order
        distances.sort(reverse=True, key=lambda x: x[0])

        # Select the pairs with the highest distances
        pairs = [(i, j) for _, i, j in distances[:num_pairs]]
    else:
        raise ValueError(f"Unsupported method: {method}")

    t = torch.linspace(0., 1., num_interpolations, device=device)
    all_interpolations = []
    for (i, j) in pairs:
        x0, x1 = images[i], images[j]
        geodesic = manifold.geodesic(x0, x1, t).detach().cpu()
        all_interpolations.append(geodesic)

    return all_interpolations

def log_interpolations(all_interpolations, num_interpolations, writer, epoch):
    grid_images = []
    for geodesic in all_interpolations:
        grid_images.extend([geodesic[0], *geodesic[1:-1], geodesic[-1]])

    interpolation_grid = vutils.make_grid(grid_images, nrow=num_interpolations, padding=2, normalize=True, scale_each=True)
    writer.add_image("Geodesic Interpolation", interpolation_grid, epoch)

def create_interpolation_gif(all_interpolations, epoch, writer, gif_path):
    # Transpose the list of geodesics so that we can iterate through each "step" of all geodesics
    steps = list(zip(*all_interpolations))
    mirrored_steps = steps + steps[::-1]  # Create a back-and-forth effect
    frames = []

    # Compute the first and last grids
    first_images = [geodesic[0] for geodesic in all_interpolations]
    last_images = [geodesic[-1] for geodesic in all_interpolations]

    num_images = len(first_images)
    nrow = int(sqrt(num_images))  # Calculate the number of rows for a square-like grid
    if nrow * nrow < num_images:  # If it's not a perfect square, round up the number of rows
        nrow += 1

    first_grid = vutils.make_grid(first_images, nrow=nrow, padding=2, normalize=True, scale_each=True)
    last_grid = vutils.make_grid(last_images, nrow=nrow, padding=2, normalize=True, scale_each=True)
    first_grid_np = (first_grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    last_grid_np = (last_grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Create a white line separator
    separator_height = first_grid_np.shape[0]
    separator = np.ones((separator_height, 5, 3), dtype=np.uint8) * 255  # 5-pixel wide white line

    for step_images in mirrored_steps:
        step_images = list(step_images)
        
        # Create grid for the current step
        grid = vutils.make_grid(step_images, nrow=nrow, padding=2, normalize=True, scale_each=True)
        frame = grid.permute(1, 2, 0).numpy()
        frame = (frame * 255).astype(np.uint8)  # Convert frame to range [0, 255]
        
        # Combine first grid, separator, current step grid, separator, and last grid
        combined_frame = np.concatenate((first_grid_np, separator, frame, separator, last_grid_np), axis=1)
        frames.append(combined_frame)

    imageio.mimsave(gif_path, frames, fps=5)
    writer.add_video("Geodesic Interpolation GIF", torch.tensor(np.array(frames)).permute(0, 3, 1, 2).unsqueeze(0), epoch, fps=5)

def encode_decode_images(manifold, images, epsilon, writer, epoch):
    rae = DeformedGaussianRiemannianAutoencoder(manifold, epsilon)

    writer.add_scalar("rae/d_eps", rae.d_eps, epoch)
    writer.add_scalar("rae/eps", rae.eps, epoch)

    encoded_images = rae.encode(images)
    decoded_images = rae.decode(encoded_images).detach().cpu()

    num_images = images.size(0)
    num_rows = int(sqrt(num_images))
    num_cols = ceil(num_images / num_rows)

    original_grid = vutils.make_grid(images.cpu(), nrow=num_cols, padding=2, normalize=True, scale_each=True)
    writer.add_image("Original Images", original_grid, epoch)

    decoded_grid = vutils.make_grid(decoded_images, nrow=num_cols, padding=2, normalize=True, scale_each=True)
    writer.add_image("Projected on Manifold Images", decoded_grid, epoch)

def log_diagonal_values(psi, writer, epoch):
    # Access the diagonal values of psi
    diagonal_values = psi.diagonal.detach().cpu().numpy()
    diagonal_values = np.sort(diagonal_values)[::-1]

    # Create normal scale plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.arange(len(diagonal_values)), diagonal_values, marker='o', linestyle='-', color='b')
    ax.set_title('Diagonal Values of psi (Normal Scale)')
    ax.set_xlabel('Index')
    ax.set_ylabel('Diagonal Value')
    ax.grid(True)

    # Save the plot to a temporary buffer and log to TensorBoard
    writer.add_figure("Diagonal Values (Normal Scale)", fig, epoch)
    plt.close(fig)

    # Create log scale plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.arange(len(diagonal_values)), diagonal_values, marker='o', linestyle='-', color='b')
    ax.set_yscale('log')
    ax.set_title('Diagonal Values of psi (Log Scale)')
    ax.set_xlabel('Index')
    ax.set_ylabel('Log Diagonal Value')
    ax.grid(True)

    # Save the plot to a temporary buffer and log to TensorBoard
    writer.add_figure("Diagonal Values (Log Scale)", fig, epoch)
    plt.close(fig)


def compute_jacobian_batch(flow, x):
    def flow_fn(x_single):
        z = flow(x_single.unsqueeze(0), detach_logdet=True).squeeze(0)  # Ensure logabsdet is detached during Jacobian calculation
        return z
    jacobian = torch.vmap(torch.func.jacrev(flow_fn))(x)
    batch_size = x.shape[0]
    input_dim = x[0].numel()
    jacobian = jacobian.view(batch_size, input_dim, input_dim) # Shape: (batchsize, c*w*h, c*w*h)
    return jacobian


def compute_batched_hessian_vmap(x_batch, v_batch_reshaped, phi, selected_columns):
    """
    Compute the batched Hessian-vector product for multiple selected columns using vmap.

    Args:
        x_batch (torch.Tensor): Input batch with shape (num_samples, c, w, h).
        v_batch_reshaped (torch.Tensor): Reshaped vector for Hessian-vector product computation.
        phi (callable): Flow transformation function.
        selected_columns (torch.Tensor): Indices of the selected columns.

    Returns:
        torch.Tensor: Hessian-vector product matrix of shape (num_samples, K, c*w*h).
    """
    # Step 1: Define a function to compute the Hessian-vector product using grad and vmap
    def hessian_vector_product(fun, x, v):
        """
        Compute the Hessian-vector product using torch.func.

        Args:
            fun (callable): Scalar function whose Hessian-vector product we are computing.
            x (torch.Tensor): Input tensor with respect to which we compute the Hessian.
            v (torch.Tensor): Vector used for the Hessian-vector product.

        Returns:
            torch.Tensor: The Hessian-vector product (same shape as x).
        """
        # Compute the gradient of the function with respect to x
        grad_fun = torch.func.grad(fun)

        # Compute the Jacobian-vector product using vmap over the gradient function
        Jv = torch.func.vmap(grad_fun, in_dims=(0))(x) * v  # Vector-Jacobian product

        return Jv

    # Step 2: Define a function that extracts the scalar component of phi(x)
    def scalar_phi_component_fun(x, idx):
        """
        Extract the idx-th component of the output of phi for the entire batch.

        Args:
            x (torch.Tensor): Input batch.
            idx (int): Index of the component to extract.

        Returns:
            torch.Tensor: The sum of the idx-th component of phi(x) for the batch.
        """
        return phi(x)[:, idx].sum()

    # Step 3: Create a function that computes the Hessian-vector product for each selected column
    def hvp_for_column(x, v, idx):
        """
        Compute the Hessian-vector product for the idx-th column.

        Args:
            x (torch.Tensor): Input batch.
            v (torch.Tensor): Reshaped vector for the Hessian-vector product computation.
            idx (int): Index of the column.

        Returns:
            torch.Tensor: The Hessian-vector product for the idx-th column.
        """
        # Define the scalar function for the idx-th component of phi(x)
        scalar_fun = lambda x: scalar_phi_component_fun(x, idx)

        # Compute the Hessian-vector product
        return hessian_vector_product(scalar_fun, x, v)

    # Step 4: Use vmap to vectorize the Hessian-vector product computation over the selected columns
    vmap_hvp_fun = torch.func.vmap(hvp_for_column, in_dims=(None, None, 0))

    # Apply the vmap function to compute the Hessian-vector products in parallel
    H_v_matrix_batch = vmap_hvp_fun(x_batch, v_batch_reshaped, selected_columns)

    # Step 5: Reshape the result to the desired shape
    H_v_matrix_batch = H_v_matrix_batch.view(x_batch.size(0), len(selected_columns), -1)  # (num_samples, K, c*w*h)

    return H_v_matrix_batch


def scalar_phi_component_batch(i, x, phi):
    """
    Extract the i-th component of the output of phi for the entire batch.
    
    Args:
        i (int): Index of the component to extract.
        x (torch.Tensor): Input batch with shape (batch_size, c, w, h).
        phi (callable): Flow transformation function.

    Returns:
        torch.Tensor: Sum of the i-th component of phi(x) over the batch.
    """
    # Compute phi(x) and sum over the i-th component for all batch samples
    return phi(x)[:, i].sum()  # Return the i-th component for the entire batch

def compute_flow_metrics(phi, psi, val_loader, device, num_batches=1, num_samples_per_batch=1, K=10):
    """
    Compute flow metrics, including the Hessian-vector product for K randomly selected columns.
    
    Args:
        phi (callable): Flow transformation function.
        psi (torch.Tensor): Diagonal covariance matrix parameters.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to perform computations on (CPU/GPU).
        num_batches (int, optional): Number of batches to process. Defaults to 1.
        num_samples_per_batch (int, optional): Number of samples per batch. Defaults to 1.
        K (int, optional): Number of randomly selected columns for Hessian-vector product computation. Defaults to 10.

    Returns:
        metrics (dict): Dictionary containing computed metrics.
    """
    
    # Initialize accumulators for metrics
    iso_scores_list = []
    volume_regs_list = []
    jacobian_norms_list = []
    hessian_norms_list = []
    singular_values_list = []

    # Inverse of the diagonal covariance matrix Σ^{-1}
    sigma_diagonal = psi.diagonal.detach()  # Shape: (c*w*h,)
    sigma_inv = 1.0 / sigma_diagonal  # Shape: (c*w*h,)
    sigma_inv_matrix = torch.diag(sigma_inv).to(device)  # Shape: (c*w*h, c*w*h)

    batch_counter = 0

    with torch.no_grad():  # Disable gradient tracking
        for batch in val_loader:
            if batch_counter >= num_batches:
                break  # Process only the specified number of batches

            # Handle cases where batch is a tuple or list
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            batch_size, c, w, h = x.size()  # Image input dimensions

            # Limit the number of samples per batch
            num_samples = min(num_samples_per_batch, batch_size)
            x_batch = x[:num_samples]  # Shape: (num_samples, c, w, h)

            # Compute phi(x_batch), which flattens the image
            phi_x_batch = phi(x_batch)  # phi_x_batch.shape: (num_samples, c*w*h)

            # Compute v_batch = Σ^{-1} φ(x_batch)
            v_batch = phi_x_batch * sigma_inv.unsqueeze(0)  # Shape: (num_samples, c*w*h)

            # Reshape v_batch to match the shape of x_batch: (num_samples, c, w, h)
            v_batch_reshaped = v_batch.view(num_samples, c, w, h)

            # Compute Jacobian J for the entire batch without tracking gradients
            J_batch = compute_jacobian_batch(phi, x_batch)  # Shape: (num_samples, c*w*h, c*w*h)
            print(J_batch.size())

            # Compute deviation from identity: ||J^T J - I||_F for each sample
            I = torch.eye(c*w*h, device=device)
            J_T_J_batch = torch.matmul(J_batch.transpose(1, 2), J_batch)  # Shape: (num_samples, c*w*h, c*w*h)
            deviation_batch = J_T_J_batch - I
            iso_scores = torch.norm(deviation_batch, p='fro', dim=(1, 2))  # Shape: (num_samples,)
            iso_scores_list.extend(iso_scores.cpu().numpy())

            # Compute volume regularization term for the entire batch
            _, logabsdetjacobian_batch = phi._transform(x_batch, context=None)
            volume_reg_batch = torch.mean(logabsdetjacobian_batch ** 2)  # Scalar for each batch
            volume_regs_list.append(volume_reg_batch.item())

            # Compute M2 = J^T Σ^{-1} J for the entire batch
            sigma_inv_J_batch = torch.matmul(sigma_inv_matrix, J_batch)  # Shape: (num_samples, c*w*h, c*w*h)
            M2_batch = torch.matmul(J_batch.transpose(1, 2), sigma_inv_J_batch)  # Shape: (num_samples, c*w*h, c*w*h)
            jacobian_norm_batch = torch.norm(M2_batch, p='fro', dim=(1, 2))  # Shape: (num_samples,)
            jacobian_norms_list.extend(jacobian_norm_batch.cpu().numpy())

            # Randomly select K distinct columns to compute the Hessian-vector product
            selected_columns = random.sample(range(c*w*h), K)

            # Initialize the reduced Hessian-vector product matrix (rectangular shape)
            H_v_matrix_batch = torch.zeros(num_samples, K, c*w*h, device=device)

            for i, j in enumerate(tqdm(selected_columns, desc="Computing HVP")):
                # Compute the Hessian-vector product for each selected component of phi(x)
                _, hvp_j_batch = torch.autograd.functional.hvp(lambda x: scalar_phi_component_batch(j, x, phi), x_batch, v_batch_reshaped)
                
                # Flatten hvp_j_batch to fit in H_v_matrix_batch
                H_v_matrix_batch[:, i] = hvp_j_batch.view(num_samples, c*w*h)  # Reshape to (num_samples, c*w*h)

            # Compute the Frobenius norm of the reduced Hessian matrix
            hessian_norm_batch = torch.norm(H_v_matrix_batch, p='fro', dim=(1, 2))  # Shape: (num_samples,)
            hessian_norms_list.extend(hessian_norm_batch.cpu().numpy())

            # Compute total Hessian M_total = H_v + M2 for the selected columns
            M_total_batch = H_v_matrix_batch + M2_batch[:, selected_columns]  # Shape: (num_samples, K, c*w*h)

            # Perform SVD on M_total for the selected columns in the batch
            for i in range(num_samples):
                singular_values = torch.linalg.svdvals(M_total_batch[i])  # Shape: (K,)
                singular_values_list.append(singular_values.cpu().numpy())

            batch_counter += 1

    # Compute overall averages
    iso_score = np.mean(iso_scores_list)
    volume_reg = np.mean(volume_regs_list)
    jacobian_norm = np.mean(jacobian_norms_list)
    hessian_norm = np.mean(hessian_norms_list)
    singular_values_avg = np.mean(np.stack(singular_values_list), axis=0)

    # Return metrics
    metrics = {
        'iso_score': iso_score,
        'volume_reg': volume_reg,
        'jacobian_norm': jacobian_norm,
        'hessian_norm': hessian_norm,
        'singular_values_avg': singular_values_avg
    }
    return metrics


def log_flow_metrics(writer, epoch, phi, psi, val_loader, device, num_batches=3, num_samples_per_batch=6, K=10):
    """
    Compute flow metrics and log results to TensorBoard.
    
    Args:
        writer (SummaryWriter): TensorBoard writer instance.
        epoch (int): Current epoch number.
        phi (callable): Flow transformation function.
        psi (torch.Tensor): Diagonal covariance matrix parameters.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to perform computations on (CPU/GPU).
        num_batches (int, optional): Number of batches to process from the validation loader. Defaults to 3.
        num_samples_per_batch (int, optional): Number of samples per batch. Defaults to 6.
    """
    # Start time for metrics computation
    start_time = time.time()
    
    # Compute flow metrics
    metrics = compute_flow_metrics(phi, psi, val_loader, device, num_batches=num_batches, num_samples_per_batch=num_samples_per_batch, K=K)
    
    # End time for metrics computation
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Computation_Time for flow metrics: {elapsed_time:.2f} seconds')

    # Log metrics to TensorBoard
    writer.add_scalar("Metrics/Isometry_Deviation", metrics['iso_score'], epoch)
    writer.add_scalar("Metrics/Volume_Deviation", metrics['volume_reg'], epoch)
    writer.add_scalar("Metrics/Hessian_Norm", metrics['hessian_norm'], epoch)
    writer.add_scalar("Metrics/Jacobian_Norm", metrics['jacobian_norm'], epoch)
    writer.add_scalar("Metrics/Computation_Time", elapsed_time, epoch)

    # Log singular values of the total Hessian
    singular_values_avg = metrics['singular_values_avg']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.arange(len(singular_values_avg)), singular_values_avg, marker='o', linestyle='-', color='b')
    ax.set_title('Singular Values of Total Hessian')
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.set_yscale('log')  # Set y-axis to log scale
    ax.grid(True)
    
    # Log the figure to TensorBoard
    writer.add_figure("Metrics/Singular_Values_Hessian", fig, epoch)
    
    # Close the figure to free up memory
    plt.close(fig)


def check_manifold_properties_images(phi, psi, writer, epoch, device, val_loader, create_gif=False):
    num_samples = 64
    generate_and_plot_samples_images(phi, psi, num_samples, device, writer, epoch)

    K = 30 if epoch==0 else phi.d
    log_flow_metrics(writer, epoch, phi, psi, val_loader, device, num_batches=1, num_samples_per_batch=10, K=K)

    distribution = Unimodal(diffeomorphism=phi, strongly_convex=psi)
    manifold = DeformedGaussianPullbackManifold(distribution)

    val_batch = next(iter(val_loader))
    images = val_batch[0].to(device)
    num_pairs = 25
    num_interpolations = 25

    all_interpolations = create_interpolation_pairs(images, num_pairs, num_interpolations, device, manifold, method='max_distance')
    log_interpolations(all_interpolations, num_interpolations, writer, epoch)

    if create_gif:
        tensorboard_log_dir = writer.log_dir
        gif_path = os.path.join(tensorboard_log_dir, f"interpolation_epoch_{epoch}.gif")
        create_interpolation_gif(all_interpolations, epoch, writer, gif_path)

    epsilon = 0.1
    encode_decode_images(manifold, images, epsilon, writer, epoch)

    # Log diagonal values of psi
    log_diagonal_values(psi, writer, epoch)
