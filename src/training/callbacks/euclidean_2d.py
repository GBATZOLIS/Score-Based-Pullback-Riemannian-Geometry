import torch
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, ceil
from src.manifolds.deformed_gaussian_pullback_manifold import DeformedGaussianPullbackManifold
from src.riemannian_autoencoder.deformed_gaussian_riemannian_autoencoder import DeformedGaussianRiemannianAutoencoder
from src.training.callbacks.utils import check_orthogonality
from src.unimodal import Unimodal

def generate_and_plot_2Dsamples(phi, psi, num_samples, device, writer, epoch):
    d = phi.args.d
    base_samples = torch.randn(num_samples, d, device=device) * psi.diagonal.sqrt()
    transformed_samples = phi.inverse(base_samples)
    transformed_samples_np = transformed_samples.detach().cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(transformed_samples_np[:, 0], transformed_samples_np[:, 1], alpha=0.5)
    plt.title('Generated Samples')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    writer.add_figure('Generated Samples', plt.gcf(), global_step=epoch)
    plt.close()

def check_manifold_properties_2D_distributions(phi, psi, writer, epoch, device, val_loader, range=[-6,6], special_points=[[2., 4.], [2., -4.]]):
    num_samples = 512
    generate_and_plot_2Dsamples(phi, psi, num_samples, device, writer, epoch)
    orthogonality_deviation = check_orthogonality(phi, val_loader, device)
    writer.add_scalar("Orthogonality Deviation", orthogonality_deviation, epoch)

    distribution = Unimodal(diffeomorphism=phi, strongly_convex=psi)
    manifold = DeformedGaussianPullbackManifold(distribution)

    xx = torch.linspace(range[0], range[1], 500, device=device)
    yy = torch.linspace(range[0], range[1], 500, device=device)
    x_grid, y_grid = torch.meshgrid(xx, yy, indexing='ij')
    xy_grid = torch.zeros((*x_grid.shape, 2), device=device)
    xy_grid[:, :, 0] = x_grid
    xy_grid[:, :, 1] = y_grid

    density = torch.exp(manifold.dg.log_density(xy_grid.reshape(-1, 2)).reshape(x_grid.shape)).detach().cpu().numpy()
    x_grid_cpu, y_grid_cpu = x_grid.cpu().numpy(), y_grid.cpu().numpy()
    contour_levels = np.linspace(density.min(), density.max(), 10)

    fig, ax = plt.subplots()
    contour = ax.contour(x_grid_cpu, y_grid_cpu, density, levels=contour_levels)
    plt.colorbar(contour, ax=ax)
    writer.add_figure("Density", fig, epoch)
    plt.close(fig)

    x0 = torch.tensor(special_points[0], device=device)
    x1 = torch.tensor(special_points[1], device=device)
    x = torch.zeros((2, 2), device=device)
    x[0] = x0
    x[1] = x1
    barycentre = manifold.barycentre(x).detach().cpu().numpy()
    x0_cpu, x1_cpu = x0.cpu().numpy(), x1.cpu().numpy()

    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    ax.scatter([x0_cpu[0], x1_cpu[0]], [x0_cpu[1], x1_cpu[1]])
    ax.scatter(barycentre[0], barycentre[1], color="orange")
    writer.add_figure("Barycentre", fig, epoch)
    plt.close(fig)

    t = torch.linspace(0., 1., 100, device=device)
    geodesic = manifold.geodesic(x0, x1, t).detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    ax.plot(geodesic[:, 0], geodesic[:, 1], color="orange")
    ax.scatter([x0_cpu[0], x1_cpu[0]], [x0_cpu[1], x1_cpu[1]])
    writer.add_figure("Geodesic", fig, epoch)
    plt.close(fig)

    logarithmic = manifold.log(x0, x1[None])[0].detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    ax.arrow(x0_cpu[0], x0_cpu[1], logarithmic[0], logarithmic[1], head_width=0.2, color="orange")
    ax.scatter([x0_cpu[0], x1_cpu[0]], [x0_cpu[1], x1_cpu[1]])
    writer.add_figure("Logarithmic Mapping", fig, epoch)
    plt.close(fig)

    exponential = manifold.exp(x0, torch.tensor(logarithmic, device=device)[None])[0].detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    ax.scatter(x0_cpu[0], x0_cpu[1])
    ax.arrow(x0_cpu[0], x0_cpu[1], logarithmic[0], logarithmic[1], head_width=0.2)
    ax.scatter(exponential[0], exponential[1], color="orange")
    writer.add_figure("Exponential Mapping", fig, epoch)
    plt.close(fig)

    error = torch.norm(torch.tensor(exponential, device=device) - x1).item()
    writer.add_scalar("Error exp(log) vs x1", error, epoch)

    l2_distance = torch.norm(x0 - x1).item()
    distance = manifold.distance(x0[None, None], x1[None, None])[0, 0, 0].item()
    writer.add_text("Distance", f"L2: {l2_distance}, Manifold: {distance}", epoch)

    parallel_transport = manifold.parallel_transport(x0, torch.tensor(logarithmic, device=device)[None], x1)[0].detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    ax.scatter([x0_cpu[0], x1_cpu[0]], [x0_cpu[1], x1_cpu[1]])
    ax.arrow(x0_cpu[0], x0_cpu[1], logarithmic[0], logarithmic[1], head_width=0.2)
    ax.arrow(x1_cpu[0], x1_cpu[1], parallel_transport[0], parallel_transport[1], head_width=0.2, color="orange")
    writer.add_figure("Parallel Transport", fig, epoch)
    plt.close(fig)

    epsilon = 0.1
    banana_rae = DeformedGaussianRiemannianAutoencoder(manifold, epsilon)
    max_std = torch.sqrt(torch.max(psi.diagonal)).item()
    p = torch.linspace(-3*max_std, 3*max_std, 100, device=device)[:, None]
    rae_decode_p = banana_rae.decode(p).detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    ax.plot(rae_decode_p[:, 0], rae_decode_p[:, 1], color="orange")
    writer.add_figure("Riemannian Autoencoder", fig, epoch)
    plt.close(fig)
