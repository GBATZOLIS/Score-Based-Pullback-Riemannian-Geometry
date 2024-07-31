import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian
from torchvision.utils import make_grid
import numpy as np
from src.manifolds.deformed_gaussian_pullback_manifold import DeformedGaussianPullbackManifold
from src.riemannian_autoencoder.deformed_gaussian_riemannian_autoencoder import DeformedGaussianRiemannianAutoencoder
from src.unimodal import Unimodal
import torchvision.utils as vutils
from math import sqrt, ceil
import random 

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

def generate_and_plot_samples_images(phi, psi, num_samples, device, writer, epoch):
    """
    Generate image samples from the learned distribution and plot them.
    
    :param phi: The learned diffeomorphism.
    :param psi: The strongly convex function.
    :param num_samples: Number of samples to generate.
    :param device: The device on which tensors are located.
    :param writer: The TensorBoard logger.
    :param epoch: The current training epoch.
    """
    # Determine the shape of the images
    c, h, w = phi.args.c, phi.args.h, phi.args.w
    
    # Sample from the base distribution N(0, I) and reshape to image dimensions
    base_samples = torch.randn(num_samples, c, h, w, device=device)
    
    # Transform samples through the inverse of phi
    transformed_samples = phi.inverse(base_samples)
    
    # Ensure the samples are in the same shape as images
    transformed_samples = transformed_samples.view(num_samples, c, h, w)
    
    # Create a grid of the generated samples
    grid_images = vutils.make_grid(transformed_samples, nrow=int(sqrt(num_samples)), padding=2, normalize=True)
    
    # Save the grid to TensorBoard
    writer.add_image("Generated Samples", grid_images, epoch)


def check_manifold_properties_images(phi, psi, writer, epoch, device, val_loader):
    # Generate and plot samples
    num_samples = 64
    generate_and_plot_samples_images(phi, psi, num_samples, device, writer, epoch)

    distribution = Unimodal(diffeomorphism=phi, strongly_convex=psi)
    manifold = DeformedGaussianPullbackManifold(distribution)

    # Get the first batch of validation images
    num_pairs=20 
    num_interpolations=10

    val_batch = next(iter(val_loader))
    images = val_batch[0].to(device)
    num_images = images.size(0)
    
    # Generate the specified number of unique pairs of images
    pairs = []
    while len(pairs) < num_pairs:
        i, j = random.sample(range(num_images), 2)
        if j != i and (i, j) not in pairs and (j, i) not in pairs:
            pairs.append((i, j))

    # Test interpolation for each pair
    t = torch.linspace(0., 1., num_interpolations, device=device)
    all_interpolations = []
    for (i, j) in pairs:
        x0, x1 = images[i], images[j]
        geodesic = manifold.geodesic(x0, x1, t).detach().cpu()
        all_interpolations.append(geodesic)

    # Create a grid of interpolations
    grid_images = []
    for geodesic in all_interpolations:
        grid_images.extend([geodesic[0], *geodesic[1:-1], geodesic[-1]])

    interpolation_grid = vutils.make_grid(grid_images, nrow=num_interpolations, padding=2, normalize=True)
    writer.add_image("Geodesic Interpolation", interpolation_grid, epoch)

    # Riemannian autoencoder
    epsilon = 0.1
    banana_rae = DeformedGaussianRiemannianAutoencoder(manifold, epsilon)

    # Encode and decode the first batch
    encoded_images = banana_rae.encode(images)
    decoded_images = banana_rae.decode(encoded_images).detach().cpu()

    num_rows = int(sqrt(num_images))
    num_cols = ceil(num_images / num_rows)

    original_grid = vutils.make_grid(images.cpu(), nrow=num_cols, padding=2, normalize=True)
    writer.add_image("Original Images", original_grid, epoch)

    decoded_grid = vutils.make_grid(decoded_images, nrow=num_cols, padding=2, normalize=True)
    writer.add_image("Projected on Manifold Images", decoded_grid, epoch)

def generate_and_plot_samples(phi, psi, num_samples, device, writer, epoch):
    """
    Generate samples from the learned distribution and plot them.
    
    :param phi: The learned diffeomorphism.
    :param psi: The strongly convex function.
    :param num_samples: Number of samples to generate.
    :param device: The device on which tensors are located.
    :param writer: The TensorBoard logger.
    :param epoch: The current training epoch.
    """
    d = phi.args.d
    # Sample from the base distribution N(0, D)
    base_samples = torch.randn(num_samples, d, device=device) * psi.diagonal.sqrt()
    
    # Transform samples through the inverse of phi
    transformed_samples = phi.inverse(base_samples)
    
    # Convert samples to numpy for plotting
    transformed_samples_np = transformed_samples.detach().cpu().numpy()
    
    # Plot the samples
    plt.figure(figsize=(8, 8))
    plt.scatter(transformed_samples_np[:, 0], transformed_samples_np[:, 1], alpha=0.5)
    plt.title('Generated Samples')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    
    # Save the plot to TensorBoard
    writer.add_figure('Generated Samples', plt.gcf(), global_step=epoch)
    plt.close()

def check_manifold_properties_2D_distributions(phi, psi, writer, epoch, device, val_loader, range=[-6,6], special_points=[[2., 4.], [2., -4.]]):
    # Generate and plot samples
    num_samples = 512
    generate_and_plot_samples(phi, psi, num_samples, device, writer, epoch)

    # Test and log orthogonality
    orthogonality_deviation = check_orthogonality(phi, val_loader, device)
    writer.add_scalar("Orthogonality Deviation", orthogonality_deviation, epoch)

    distribution = Unimodal(diffeomorphism=phi, strongly_convex=psi)
    manifold = DeformedGaussianPullbackManifold(distribution)

    # Check the density
    xx = torch.linspace(range[0], range[1], 500, device=device)
    yy = torch.linspace(range[0], range[1], 500, device=device)
    x_grid, y_grid = torch.meshgrid(xx, yy, indexing='ij')

    xy_grid = torch.zeros((*x_grid.shape, 2), device=device)
    xy_grid[:, :, 0] = x_grid
    xy_grid[:, :, 1] = y_grid

    density = torch.exp(manifold.dg.log_density(xy_grid.reshape(-1, 2)).reshape(x_grid.shape)).detach().cpu().numpy()
    x_grid_cpu = x_grid.cpu().numpy()
    y_grid_cpu = y_grid.cpu().numpy()

    # Set contour levels focusing on high-density region
    contour_levels = np.linspace(density.min(), density.max(), 10)

    fig, ax = plt.subplots()
    contour = ax.contour(x_grid_cpu, y_grid_cpu, density, levels=contour_levels)
    plt.colorbar(contour, ax=ax)  # Add color bar for contour values
    writer.add_figure("Density", fig, epoch)
    plt.close(fig)

    # Check barycenter

    # Special points
    x0 = torch.tensor(special_points[0], device=device)
    x1 = torch.tensor(special_points[1], device=device)
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
    p = torch.linspace(range[0], range[1], 100, device=device)[:, None]
    rae_decode_p = banana_rae.decode(p).detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.contour(x_grid_cpu, y_grid_cpu, density)
    ax.plot(rae_decode_p[:, 0], rae_decode_p[:, 1], color="orange")
    writer.add_figure("Riemannian Autoencoder", fig, epoch)
    plt.close(fig)


def check_manifold_properties(dataset, phi, psi, writer, epoch, device, val_loader):
    if dataset == 'mnist':
        check_manifold_properties_images(phi, psi, writer, epoch, device, val_loader)
    elif dataset in 'single_banana':
        range=[-6.,6.]
        special_points=[[2., 4.], [2., -4.]]
        check_manifold_properties_2D_distributions(phi, psi, writer, epoch, 
                                                   device, val_loader, range, special_points)
    elif dataset == 'combined_elongated_gaussians':
        range=[-3.,3.]
        special_points=[[0., 1.], [-1., 0.]]
        check_manifold_properties_2D_distributions(phi, psi, writer, epoch, 
                                                   device, val_loader, range, special_points)

