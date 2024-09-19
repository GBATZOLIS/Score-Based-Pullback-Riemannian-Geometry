import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from torch.autograd.functional import jacobian
from mpl_toolkits.mplot3d import Axes3D
from src.manifolds.deformed_gaussian_pullback_manifold import DeformedGaussianPullbackManifold
from src.riemannian_autoencoder.deformed_gaussian_riemannian_autoencoder import DeformedGaussianRiemannianAutoencoder
from src.training.callbacks.utils import check_orthogonality
from src.unimodal import Unimodal
from torch.utils.tensorboard import SummaryWriter

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

            #print(H_v_matrix)
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





def generate_and_plot_3D_samples(phi, psi, num_samples, device, writer, epoch):
    d = phi.args.d
    base_samples = torch.randn(num_samples, d, device=device) * psi.diagonal.sqrt()
    transformed_samples = phi.inverse(base_samples)
    transformed_samples_np = transformed_samples.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(transformed_samples_np[:, 0], transformed_samples_np[:, 1], transformed_samples_np[:, 2], alpha=0.5)
    ax.set_title('Generated Samples')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.grid(True)
    
    writer.add_figure('Generated Samples', fig, global_step=epoch)
    plt.close(fig)

def pad_frame(frame, shape):
    padded_frame = np.zeros(shape, dtype=frame.dtype)
    padded_frame[:frame.shape[0], :frame.shape[1], :frame.shape[2]] = frame
    return padded_frame

def check_manifold_properties_3D_distributions(phi, psi, writer, epoch, device, val_loader, range_vals=[[-1.5, 1.5],[-1.5, 1.5],[-1.5, 1.5]], special_points=[[1., 0., 0.], [0., 0.33, 0.]]):
    plots_dir = os.path.join(writer.log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    num_samples = 512
    generate_and_plot_3D_samples(phi, psi, num_samples, device, writer, epoch)

    orthogonality_deviation = check_orthogonality(phi, val_loader, device)
    writer.add_scalar("Orthogonality Deviation", orthogonality_deviation, epoch)

    distribution = Unimodal(diffeomorphism=phi, strongly_convex=psi)
    manifold = DeformedGaussianPullbackManifold(distribution)
    
    if special_points is None:
        data = next(iter(val_loader))
        val_batch = data[0] if isinstance(data, list) else data
        special_points = val_batch[:2]
    
    x0 = torch.tensor(special_points[0], device=device)
    x1 = torch.tensor(special_points[1], device=device)
    x = torch.zeros((2, 3), device=device)
    x[0] = x0
    x[1] = x1
    barycentre = manifold.barycentre(x).detach().cpu().numpy()

    x0_cpu = x0.cpu().numpy()
    x1_cpu = x1.cpu().numpy()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([x0_cpu[0], x1_cpu[0]], [x0_cpu[1], x1_cpu[1]], [x0_cpu[2], x1_cpu[2]])
    ax.scatter(barycentre[0], barycentre[1], barycentre[2], color="orange")
    writer.add_figure("Barycentre", fig, epoch)
    fig.savefig(os.path.join(plots_dir, f'barycentre_epoch_{epoch}.png'))
    plt.close(fig)

    t = torch.linspace(0., 1., 100, device=device)
    geodesic = manifold.geodesic(x0, x1, t).detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(geodesic[:, 0], geodesic[:, 1], geodesic[:, 2], color="orange")
    ax.scatter([x0_cpu[0], x1_cpu[0]], [x0_cpu[1], x1_cpu[1]], [x0_cpu[2], x1_cpu[2]])
    writer.add_figure("Geodesic", fig, epoch)
    fig.savefig(os.path.join(plots_dir, f'geodesic_epoch_{epoch}.png'))
    plt.close(fig)

    logarithmic = manifold.log(x0, x1[None])[0].detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x0_cpu[0], x0_cpu[1], x0_cpu[2], logarithmic[0], logarithmic[1], logarithmic[2], color="orange")
    ax.scatter([x0_cpu[0], x1_cpu[0]], [x0_cpu[1], x1_cpu[1]], [x0_cpu[2], x1_cpu[2]])
    writer.add_figure("Logarithmic Mapping", fig, epoch)
    fig.savefig(os.path.join(plots_dir, f'logarithmic_epoch_{epoch}.png'))
    plt.close(fig)

    exponential = manifold.exp(x0, torch.tensor(logarithmic, device=device)[None])[0].detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x0_cpu[0], x0_cpu[1], x0_cpu[2])
    ax.quiver(x0_cpu[0], x0_cpu[1], x0_cpu[2], logarithmic[0], logarithmic[1], logarithmic[2], color="orange")
    ax.scatter(exponential[0], exponential[1], exponential[2], color="orange")
    writer.add_figure("Exponential Mapping", fig, epoch)
    fig.savefig(os.path.join(plots_dir, f'exponential_epoch_{epoch}.png'))
    plt.close(fig)

    error = torch.norm(torch.tensor(exponential, device=device) - x1).item()
    writer.add_scalar("Error exp(log) vs x1", error, epoch)

    l2_distance = torch.norm(x0 - x1).item()
    distance = manifold.distance(x0[None, None], x1[None, None])[0, 0, 0].item()
    writer.add_text("Distance", f"L2: {l2_distance}, Manifold: {distance}", epoch)

    parallel_transport = manifold.parallel_transport(x0, torch.tensor(logarithmic, device=device)[None], x1)[0].detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([x0_cpu[0], x1_cpu[0]], [x0_cpu[1], x1_cpu[1]], [x0_cpu[2], x1_cpu[2]])
    ax.quiver(x0_cpu[0], x0_cpu[1], x0_cpu[2], logarithmic[0], logarithmic[1], logarithmic[2])
    ax.quiver(x1_cpu[0], x1_cpu[1], x1_cpu[2], parallel_transport[0], parallel_transport[1], parallel_transport[2], color="orange")
    writer.add_figure("Parallel Transport", fig, epoch)
    fig.savefig(os.path.join(plots_dir, f'parallel_transport_epoch_{epoch}.png'))
    plt.close(fig)

    epsilon = 0.1
    rae = DeformedGaussianRiemannianAutoencoder(manifold, epsilon)
    max_std = torch.sqrt(torch.max(psi.diagonal)).item()
    p = torch.linspace(-3*max_std, 3*max_std, 100, device=device)[:, None]
    rae_decode_p = rae.decode(p).detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(rae_decode_p[:, 0], rae_decode_p[:, 1], rae_decode_p[:, 2], color="orange")
    writer.add_figure("Riemannian Autoencoder", fig, epoch)
    fig.savefig(os.path.join(plots_dir, f'riemannian_autoencoder_epoch_{epoch}.png'))
    plt.close(fig)

    # Compute flow metrics
    metrics = compute_flow_metrics(phi, psi, val_loader, device)

    # Log metrics to TensorBoard
    writer.add_scalar("Metrics/Isometry_Deviation", metrics['iso_score'], epoch)
    writer.add_scalar("Metrics/Volume_Deviation", metrics['volume_reg'], epoch)
    writer.add_scalar("Metrics/Hessian_Norm", metrics['hessian_norm'], epoch)
    writer.add_scalar("Metrics/Jacobian_Norm", metrics['jacobian_norm'], epoch)

    # Log singular values of the total Hessian
    singular_values_avg = metrics['singular_values_avg']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.arange(len(singular_values_avg)), singular_values_avg, marker='o', linestyle='-', color='b')
    ax.set_title('Singular Values of Total Hessian')
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.grid(True)
    writer.add_figure("Metrics/Singular_Values_Hessian", fig, epoch)
    plt.close(fig)

    # Log diagonal values of psi
    log_diagonal_values(psi, writer, epoch)
    
