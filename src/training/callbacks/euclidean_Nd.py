import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian
from torch.utils.tensorboard import SummaryWriter
import time 
import torch.func as func

# For batch Jacobian and Hessian computations
try:
    from torch.func import vmap, jacfwd
except ImportError:
    # If torch.func is not available, import from functorch
    from functorch import vmap, jacfwd

# Import specific functions from torch.linalg
from torch.linalg import norm, svdvals

# For plotting and visualization (Axes3D is not used, so we can omit it)
# from mpl_toolkits.mplot3d import Axes3D  # Not used in the current code

# Import your custom modules (ensure these paths are correct)
from src.manifolds.deformed_gaussian_pullback_manifold import DeformedGaussianPullbackManifold
from src.riemannian_autoencoder.deformed_gaussian_riemannian_autoencoder import DeformedGaussianRiemannianAutoencoder
from src.training.callbacks.utils import check_orthogonality  # If still needed elsewhere
from src.unimodal import Unimodal

from functorch import jacrev, vmap

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
    jacobian = torch.vmap(func.jacrev(flow_fn))(x)
    return jacobian

def compute_flow_metrics_fast(phi, psi, val_loader, device, num_batches=3, num_samples_per_batch=5):
    # Initialize accumulators for metrics
    iso_scores_list = []
    volume_regs_list = []
    jacobian_norms_list = []
    hessian_norms_list = []
    singular_values_list = []

    # Inverse of the diagonal covariance matrix Σ^{-1}
    sigma_diagonal = psi.diagonal.detach()  # Shape: (d,)
    sigma_inv = 1.0 / sigma_diagonal  # Shape: (d,)
    sigma_inv_matrix = torch.diag(sigma_inv).to(device)  # Shape: (d, d)

    batch_counter = 0

    with torch.no_grad():  # Disable gradient tracking
        for batch in val_loader:
            if batch_counter >= num_batches:
                break  # Process only the specified number of batches

            # Handle cases where batch is a tuple or list
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            batch_size = x.size(0)
            d = x.size(1)  # Input dimension

            # Limit the number of samples per batch
            num_samples = min(num_samples_per_batch, batch_size)
            x_batch = x[:num_samples]  # Shape: (num_samples, d)

            # Compute phi(x_batch) for the entire batch
            phi_x_batch = phi(x_batch)  # Shape: (num_samples, d)

            # Compute v_batch = Σ^{-1} φ(x_batch)
            v_batch = phi_x_batch * sigma_inv  # Shape: (num_samples, d)

            # Compute Jacobian J for the entire batch without tracking gradients
            J_batch = compute_jacobian_batch(phi, x_batch) # Shape will be [batch_size, d, d]
            print(f'J_batch.size():{J_batch.size()}')

            # Compute deviation from identity: ||J^T J - I||_F for each sample
            I = torch.eye(d, device=device)
            J_T_J_batch = torch.matmul(J_batch.transpose(1, 2), J_batch)  # Shape: (num_samples, d, d)
            deviation_batch = J_T_J_batch - I
            iso_scores = torch.norm(deviation_batch, p='fro', dim=(1, 2))  # Shape: (num_samples,)
            iso_scores_list.extend(iso_scores.cpu().numpy())

            # Compute volume regularization term for the entire batch
            _, logabsdetjacobian_batch = phi._transform(x_batch, context=None)
            volume_reg_batch = torch.mean(logabsdetjacobian_batch ** 2)  # Scalar for each batch
            volume_regs_list.append(volume_reg_batch.item())

            # Compute M2 = J^T Σ^{-1} J for the entire batch
            sigma_inv_J_batch = torch.matmul(sigma_inv_matrix, J_batch)  # Shape: (num_samples, d, d)
            M2_batch = torch.matmul(J_batch.transpose(1, 2), sigma_inv_J_batch)  # Shape: (num_samples, d, d)
            jacobian_norm_batch = torch.norm(M2_batch, p='fro', dim=(1, 2))  # Shape: (num_samples,)
            jacobian_norms_list.extend(jacobian_norm_batch.cpu().numpy())

            # Compute the Hessian-vector product for the entire batch
            def scalar_phi_component_batch(i, x):
                return phi(x)[:, i].sum()  # Return the i-th component for the entire batch

            H_v_matrix_batch = torch.zeros(num_samples, d, d, device=device)
            for j in range(d):
                # Compute the Hessian-vector product for each component of phi(x) in the entire batch
                _, hvp_j_batch = torch.autograd.functional.hvp(lambda x: scalar_phi_component_batch(j, x), x_batch, v_batch)
                H_v_matrix_batch[:, j] = hvp_j_batch  # Accumulate the result into the matrix

            hessian_norm_batch = torch.norm(H_v_matrix_batch, p='fro', dim=(1, 2))  # Shape: (num_samples,)
            hessian_norms_list.extend(hessian_norm_batch.cpu().numpy())

            # Compute total Hessian M_total = H_v + M2 for the entire batch
            M_total_batch = H_v_matrix_batch + M2_batch  # Shape: (num_samples, d, d)

            # Perform SVD on M_total for the entire batch
            for i in range(num_samples):
                singular_values = torch.linalg.svdvals(M_total_batch[i])  # Shape: (d,)
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


def compute_flow_metrics(phi, psi, val_loader, device, num_batches=3, num_samples_per_batch=5):
    # Initialize accumulators for metrics
    iso_scores_list = []
    volume_regs_list = []
    jacobian_norms_list = []
    hessian_norms_list = []
    singular_values_list = []

    # Inverse of the diagonal covariance matrix Σ^{-1}
    sigma_diagonal = psi.diagonal.detach()  # Shape: (d,)
    sigma_inv = 1.0 / sigma_diagonal  # Shape: (d,)
    sigma_inv_matrix = torch.diag(sigma_inv).to(device)  # Shape: (d, d)

    batch_counter = 0

    for batch in val_loader:
        if batch_counter >= num_batches:
            break  # Process only the specified number of batches

        # Handle cases where batch is a tuple or list
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        batch_size = x.size(0)
        d = x.size(1)  # Input dimension

        # Limit the number of samples per batch
        num_samples = min(num_samples_per_batch, batch_size)
        x_batch = x[:num_samples]  # Shape: (num_samples, d)

        # Process each sample individually
        for i in range(num_samples):
            # Prepare x_sample
            x_sample = x_batch[i]  # Shape: (d,)
            x_sample = x_sample.clone().detach().requires_grad_(True)

            # Compute phi(x_sample)
            phi_x_sample = phi(x_sample.unsqueeze(0)).squeeze(0)  # Shape: (d,)

            # Compute v_sample = Σ^{-1} φ(x_sample)
            v_sample = phi_x_sample * sigma_inv  # Shape: (d,)
            v_sample = v_sample.detach()  # Detach to treat v_sample as constant during differentiation

            # Compute Jacobian J at x_sample
            def phi_single_sample(x):
                # x has shape (d,)
                x = x.unsqueeze(0)  # Shape: (1, d)
                phi_x = phi(x)  # Shape: (1, d)
                return phi_x.squeeze(0)  # Shape: (d,)

            J = torch.autograd.functional.jacobian(phi_single_sample, x_sample, create_graph=True)  # Shape: (d, d)

            # Compute deviation from identity: ||J^T J - I||_F
            I = torch.eye(d, device=device)
            J_T_J = torch.matmul(J.transpose(0, 1), J)  # Shape: (d, d)
            deviation = J_T_J - I
            iso_score = torch.norm(deviation, p='fro')  # Scalar
            iso_scores_list.append(iso_score.item())

            # Compute volume regularization term for x_sample
            _, logabsdetjacobian = phi._transform(x_sample.unsqueeze(0), context=None)
            volume_reg = torch.mean(logabsdetjacobian ** 2)  # Scalar
            volume_regs_list.append(volume_reg.item())

            # Compute M2 = J^T Σ^{-1} J
            sigma_inv_J = torch.matmul(sigma_inv_matrix, J)  # Shape: (d, d)
            M2 = torch.matmul(J.transpose(0, 1), sigma_inv_J)  # Shape: (d, d)
            jacobian_norm = torch.norm(M2, p='fro')  # Scalar
            jacobian_norms_list.append(jacobian_norm.item())

            # Use hvp to compute Hessian-vector product for each dimension
            def scalar_phi_component(i, x):
                return phi(x.unsqueeze(0)).squeeze(0)[i]

            H_v_matrix = torch.zeros(d, d).to(device)
            for j in range(d):
                # Compute the Hessian-vector product for each component of phi(x)
                _, hvp_j = torch.autograd.functional.hvp(lambda x: scalar_phi_component(j, x), x_sample, v_sample)
                H_v_matrix[j] = hvp_j  # Accumulate the result into the matrix

            hessian_norm = torch.norm(H_v_matrix, p='fro')  # Scalar
            hessian_norms_list.append(hessian_norm.item())

            # Compute total Hessian M_total = H_v + M2
            M_total = H_v_matrix + M2  # Shape: (d, d)

            # Perform SVD on M_total
            singular_values = torch.linalg.svdvals(M_total)  # Shape: (d,)
            singular_values_list.append(singular_values.detach().cpu().numpy())

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



def generate_and_plot_ND_samples(phi, psi, num_samples, device, writer, epoch):
    d = phi.args.d
    # Generate base samples from the base distribution
    base_samples = torch.randn(num_samples, d, device=device) * psi.diagonal.sqrt()
    # Transform samples to data space
    transformed_samples = phi.inverse(base_samples)
    transformed_samples_np = transformed_samples.detach().cpu().numpy()
    
    # Number of dimensions
    num_dimensions = transformed_samples_np.shape[1]
    
    # The last dimension is the 'z' axis (dimension N-1)
    z = transformed_samples_np[:, -1]
    
    # Decide how many dimensions to plot (excluding the last one)
    # For high dimensions, limit the number of plots
    max_plots = min(10, num_dimensions - 1)  # Adjust as needed
    dimensions_to_plot = range(num_dimensions - 1)  # Plot all except last dimension
    if num_dimensions - 1 > max_plots:
        # Select evenly spaced dimensions to plot
        dimensions_to_plot = np.linspace(0, num_dimensions - 2, max_plots, dtype=int)
    
    # Create a figure with subplots
    num_subplots = len(dimensions_to_plot)
    cols = 2
    rows = (num_subplots + 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(12, rows * 4))
    axs = axs.flatten()
    
    for idx, i in enumerate(dimensions_to_plot):
        x = transformed_samples_np[:, i]
        ax = axs[idx]
        ax.scatter(z, x, alpha=0.5, s=10)
        ax.set_xlabel('Dimension {}'.format(num_dimensions - 1))
        ax.set_ylabel('Dimension {}'.format(i))
        ax.set_title('Dimension {} vs Dimension {}'.format(i, num_dimensions - 1))
        ax.grid(True)
    
    # Remove any unused subplots
    for j in range(idx + 1, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    # Log the figure to TensorBoard
    writer.add_figure('Samples/Dimensions_vs_LastDimension', fig, epoch)
    plt.close(fig)


def check_manifold_properties_ND_distributions(phi, psi, writer, epoch, device, val_loader, range_vals=[[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]], special_points=[[1., 0., 0.], [0., 0.33, 0.]]):
    # Generate and plot samples
    num_samples = 512
    generate_and_plot_ND_samples(phi, psi, num_samples, device, writer, epoch)

    # Compute flow metrics
    start_time = time.time()
    metrics = compute_flow_metrics_fast(phi, psi, val_loader, device, num_batches=3, num_samples_per_batch=6)
    end_time = time.time()
    elapsed_time = end_time - start_time

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
    writer.add_figure("Metrics/Singular_Values_Hessian", fig, epoch)
    plt.close(fig)
    

    # Log diagonal values of psi for comparison
    log_diagonal_values(psi, writer, epoch)

    # ---- Include the RAE functionality ----
    # Initialize the distribution and manifold
    distribution = Unimodal(diffeomorphism=phi, strongly_convex=psi)
    manifold = DeformedGaussianPullbackManifold(distribution)

    # Initialize the Riemannian Autoencoder
    epsilon = 0.1
    rae = DeformedGaussianRiemannianAutoencoder(manifold, epsilon)

    # Log RAE parameters
    writer.add_scalar("RAE/d_eps", rae.d_eps, epoch)
    writer.add_scalar("RAE/epsilon", rae.eps, epoch)

    # Compute reconstruction errors
    rec_errors = []
    for batch in iter(val_loader):
        x = batch[0] if isinstance(batch, list) else batch
        x = x.to(device)
        encoding = rae.encode(x)
        x_rec = rae.decode(encoding)
        rec_error = torch.linalg.norm(x_rec - x, dim=1)  # Compute reconstruction error per sample
        rec_errors.append(rec_error)  # Append the error to the list

    # Compute the average reconstruction error
    rec_errors = torch.cat(rec_errors)  # Combine all batch errors
    avg_rec_error = torch.mean(rec_errors)  # Compute mean
    writer.add_scalar("RAE/reconstruction_error", avg_rec_error.item(), epoch)

    
