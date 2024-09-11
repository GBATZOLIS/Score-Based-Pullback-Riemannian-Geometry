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

def check_manifold_properties_images(phi, psi, writer, epoch, device, val_loader, create_gif=False):
    num_samples = 64
    generate_and_plot_samples_images(phi, psi, num_samples, device, writer, epoch)

    #orthogonality_deviation = check_orthogonality(phi, val_loader, device)
    #writer.add_scalar("Orthogonality Deviation", orthogonality_deviation, epoch)

    volume_pres_dev = deviation_from_volume_preservation(phi, val_loader, device)
    writer.add_scalar("Volume Preservation Deviation", volume_pres_dev, epoch)

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
