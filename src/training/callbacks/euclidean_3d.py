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

def check_manifold_properties_3D_distributions(phi, psi, writer, epoch, device, val_loader, range_vals=[-1.5, 1.5], special_points=[[1., 0., 0.], [0., 0.33, 0.]]):
    if isinstance(range_vals, list):
        range_vals = tuple(range_vals)

    plots_dir = os.path.join(writer.log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    num_samples = 512
    generate_and_plot_3D_samples(phi, psi, num_samples, device, writer, epoch)

    orthogonality_deviation = check_orthogonality(phi, val_loader, device)
    writer.add_scalar("Orthogonality Deviation", orthogonality_deviation, epoch)

    distribution = Unimodal(diffeomorphism=phi, strongly_convex=psi)
    manifold = DeformedGaussianPullbackManifold(distribution)

    xx = torch.linspace(range_vals[0], range_vals[1], 20, device=device)
    yy = torch.linspace(range_vals[0], range_vals[1], 20, device=device)
    zz = torch.linspace(range_vals[0], range_vals[1], 20, device=device)
    x_grid, y_grid, z_grid = torch.meshgrid(xx, yy, zz, indexing='ij')

    xyz_grid = torch.zeros((*x_grid.shape, 3), device=device)
    xyz_grid[:, :, :, 0] = x_grid
    xyz_grid[:, :, :, 1] = y_grid
    xyz_grid[:, :, :, 2] = z_grid

    density = torch.exp(manifold.dg.log_density(xyz_grid.reshape(-1, 3)).reshape(x_grid.shape)).detach().cpu().numpy()
    x_grid_cpu = x_grid.cpu().numpy()
    y_grid_cpu = y_grid.cpu().numpy()
    z_grid_cpu = z_grid.cpu().numpy()

    slices_dir = os.path.join(plots_dir, 'slices')
    os.makedirs(slices_dir, exist_ok=True)

    images = []
    for z_index in range(z_grid.shape[2]):
        density_2d = density[:, :, z_index]
        x_grid_2d = x_grid_cpu[:, :, z_index]
        y_grid_2d = y_grid_cpu[:, :, z_index]

        contour_levels = np.linspace(density_2d.min(), density_2d.max(), 10)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        contour = ax.contourf(x_grid_2d, y_grid_2d, density_2d, levels=contour_levels, cmap='viridis')
        plt.colorbar(contour, ax=ax)

        image_path = os.path.join(slices_dir, f'density_slice_{z_index}.png')
        fig.savefig(image_path)
        plt.close(fig)

        images.append(imageio.imread(image_path))

    gif_path = os.path.join(plots_dir, f'density_slices_{epoch}.gif')
    imageio.mimsave(gif_path, images, duration=0.5)

    # Convert GIF to a format compatible with TensorBoard
    gif_frames = imageio.mimread(gif_path)
    max_shape = tuple(max(s) for s in zip(*(im.shape for im in gif_frames)))
    padded_frames = [pad_frame(im, max_shape) for im in gif_frames]
    gif_data = np.stack(padded_frames, axis=0)
    gif_data = torch.tensor(gif_data).permute(0, 3, 1, 2)  # Convert to (T, C, H, W)

    # Add a batch dimension
    gif_data = gif_data.unsqueeze(0)  # Now (1, T, C, H, W)

    # Log the GIF to TensorBoard
    writer.add_video("Density Slices GIF", gif_data, epoch, fps=2)

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
    banana_rae = DeformedGaussianRiemannianAutoencoder(manifold, epsilon)
    p = torch.linspace(range_vals[0], range_vals[1], 100, device=device)[:, None]
    rae_decode_p = banana_rae.decode(p).detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(rae_decode_p[:, 0], rae_decode_p[:, 1], rae_decode_p[:, 2], color="orange")
    writer.add_figure("Riemannian Autoencoder", fig, epoch)
    fig.savefig(os.path.join(plots_dir, f'riemannian_autoencoder_epoch_{epoch}.png'))
    plt.close(fig)
