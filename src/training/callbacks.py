import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian
from torchvision.utils import make_grid
import numpy as np
from src.manifolds.deformed_gaussian_pullback_manifold import DeformedGaussianPullbackManifold
from src.riemannian_autoencoder.deformed_gaussian_riemannian_autoencoder import DeformedGaussianRiemannianAutoencoder
from src.unimodal import Unimodal

def check_orthogonality(phi, val_loader, device, num_samples=100):
    orthogonality_deviation = []

    for i, data in enumerate(val_loader):
        if i >= num_samples:
            break
        
        # Use the validation sample
        if isinstance(data, list):
            x = data[0]
        else:
            x = data

        x = x.to(device)
        
        # Compute the Jacobian matrix
        J = jacobian(lambda x: phi(x), x)

        # Reshape the Jacobian to a 2D matrix for each sample in the batch
        batch_size = J.shape[0]
        J = J.view(batch_size, -1, J.shape[-1])

        # Compute J^T J for all samples in the batch using batched matrix multiplication
        J_T_J = torch.matmul(J.transpose(1, 2), J)

        # Compute the Frobenius norm of (J^T J - I) for each sample in the batch
        identity_matrix = torch.eye(J_T_J.shape[-1], device=device).unsqueeze(0)
        deviations = torch.norm(J_T_J - identity_matrix, p='fro', dim=(1, 2)).cpu().numpy()
        
        orthogonality_deviation.extend(deviations)

    avg_deviation = np.mean(orthogonality_deviation)
    return avg_deviation

def check_manifold_properties(phi, psi, writer, epoch, device, val_loader):
    # Test and log orthogonality
    orthogonality_deviation = check_orthogonality(phi, val_loader, device)
    writer.add_scalar("Orthogonality Deviation", orthogonality_deviation, epoch)

    distribution = Unimodal(diffeomorphism=phi, strongly_convex=psi)
    manifold = DeformedGaussianPullbackManifold(distribution)

    # Check the density
    xx = torch.linspace(-6.0, 6.0, 500, device=device)
    yy = torch.linspace(-6.0, 6.0, 500, device=device)
    x_grid, y_grid = torch.meshgrid(xx, yy, indexing='ij')

    xy_grid = torch.zeros((*x_grid.shape, 2), device=device)
    xy_grid[:, :, 0] = x_grid
    xy_grid[:, :, 1] = y_grid

    density = torch.exp(manifold.dg.log_density(xy_grid.reshape(-1, 2)).reshape(x_grid.shape)).detach().cpu().numpy()
    x_grid_cpu = x_grid.cpu().numpy()
    y_grid_cpu = y_grid.cpu().numpy()

    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    writer.add_figure("Density", fig, epoch)
    plt.close(fig)

    # Check barycenter

    # Special points
    x0 = torch.tensor([2., 4.], device=device)
    x1 = torch.tensor([2., -4.], device=device)
    x = torch.zeros((2, 2), device=device)
    x[0] = x0
    x[1] = x1
    barycentre = manifold.barycentre(x).detach().cpu().numpy()

    x0_cpu = x0.cpu().numpy()
    x1_cpu = x1.cpu().numpy()

    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    ax.scatter([x0_cpu[0], x1_cpu[0]], [x0_cpu[1], x1_cpu[1]])
    ax.scatter(barycentre[0], barycentre[1], color="orange")
    writer.add_figure("Barycentre", fig, epoch)
    plt.close(fig)

    # Test interpolation
    t = torch.linspace(0., 1., 100, device=device)
    geodesic = manifold.geodesic(x0, x1, t).detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    ax.plot(geodesic[:, 0], geodesic[:, 1], color="orange")
    ax.scatter([x0_cpu[0], x1_cpu[0]], [x0_cpu[1], x1_cpu[1]])
    writer.add_figure("Geodesic", fig, epoch)
    plt.close(fig)

    # Test logarithmic mapping
    logarithmic = manifold.log(x0, x1[None])[0].detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    ax.arrow(x0_cpu[0], x0_cpu[1], logarithmic[0], logarithmic[1], head_width=0.2, color="orange")
    ax.scatter([x0_cpu[0], x1_cpu[0]], [x0_cpu[1], x1_cpu[1]])
    writer.add_figure("Logarithmic Mapping", fig, epoch)
    plt.close(fig)

    # Test exponential mapping
    exponential = manifold.exp(x0, torch.tensor(logarithmic, device=device)[None])[0].detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    ax.scatter(x0_cpu[0], x0_cpu[1])
    ax.arrow(x0_cpu[0], x0_cpu[1], logarithmic[0], logarithmic[1], head_width=0.2)
    ax.scatter(exponential[0], exponential[1], color="orange")
    writer.add_figure("Exponential Mapping", fig, epoch)
    plt.close(fig)

    # Log the error between exp_x0(log_x0 (x1)) and x1 to TensorBoard
    error = torch.norm(torch.tensor(exponential, device=device) - x1).item()
    writer.add_scalar("Error exp(log) vs x1", error, epoch)

    # Test distance
    l2_distance = torch.norm(x0 - x1).item()
    distance = manifold.distance(x0[None, None], x1[None, None])[0, 0, 0].item()
    writer.add_text("Distance", f"L2: {l2_distance}, Manifold: {distance}", epoch)

    # Test parallel transport
    parallel_transport = manifold.parallel_transport(x0, torch.tensor(logarithmic, device=device)[None], x1)[0].detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    ax.scatter([x0_cpu[0], x1_cpu[0]], [x0_cpu[1], x1_cpu[1]])
    ax.arrow(x0_cpu[0], x0_cpu[1], logarithmic[0], logarithmic[1], head_width=0.2)
    ax.arrow(x1_cpu[0], x1_cpu[1], parallel_transport[0], parallel_transport[1], head_width=0.2, color="orange")
    writer.add_figure("Parallel Transport", fig, epoch)
    plt.close(fig)

    # Riemannian autoencoder
    epsilon = 0.1
    banana_rae = DeformedGaussianRiemannianAutoencoder(manifold, epsilon)
    p = torch.linspace(-6, 6, 100, device=device)[:, None]
    rae_decode_p = banana_rae.decode(p).detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    ax.plot(rae_decode_p[:, 0], rae_decode_p[:, 1], color="orange")
    writer.add_figure("Riemannian Autoencoder", fig, epoch)
    plt.close(fig)



