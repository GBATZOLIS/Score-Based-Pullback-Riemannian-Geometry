import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

def evaluate_manifold_maps(learned_pl_manifold, gt_pl_manifold, test_loader, device, writer, epoch=1000, num_pairs=5, value_range = [-8., 8.]):
    print(f'learned_pl_manifold.dg.psi.diagonal.device:{learned_pl_manifold.dg.psi.diagonal.device}')
    print(f'gt_pl_manifold.dg.psi.diagonal.device:{gt_pl_manifold.dg.psi.diagonal.device}')

    # Fetch the first batch from the test_loader and ensure it is on the correct device
    test_data = iter(test_loader)
    data = next(test_data)
    if isinstance(data, (list, tuple)):
        first_batch = data[0].to(device)  # If data is a tuple/list, take the first element
    else:
        first_batch = data.to(device)

    first_batch = first_batch[5:30]

    # Compute pairwise distances and select pairs with highest L2 distances
    num_points = first_batch.shape[0]
    with torch.no_grad():
        # Compute pairwise differences and distances
        diff = first_batch.unsqueeze(1) - first_batch.unsqueeze(0)  # shape (N, N, D)
        dist_matrix = torch.norm(diff, dim=2)  # shape (N, N)

    # Get upper triangular indices (excluding diagonal) to avoid duplicate pairs
    i_indices, j_indices = torch.triu_indices(num_points, num_points, offset=1)

    # Get distances and pair indices
    distances = dist_matrix[i_indices, j_indices]  # shape (num_pairs,)
    pairs_indices = list(zip(i_indices.tolist(), j_indices.tolist()))

    # Sort distances in descending order
    sorted_indices = torch.argsort(distances, descending=True)

    # Select top num_pairs based on the highest distances
    top_indices = sorted_indices[:num_pairs]

    # Get the top pairs
    top_pairs = [pairs_indices[idx] for idx in top_indices]

    # Get the actual pairs of data points
    pairs = [(first_batch[i], first_batch[j]) for i, j in top_pairs]

    # Compute the density grid for the ground truth manifold once
    x_grid_cpu, y_grid_cpu, grid_density = compute_density_grid(gt_pl_manifold, device, value_range)

    for idx, (x0, x1) in enumerate(pairs):
        print(f'Processing pair {idx}: x0={x0}, x1={x1}')
        # Ensure both x0 and x1 are on the same device
        x0, x1 = x0.to(device), x1.to(device)
        
        # Plot exponential maps
        plot_exponential_map(gt_pl_manifold, x0, x1, x_grid_cpu, y_grid_cpu, grid_density, writer, epoch, idx, "Ground Truth")
        plot_exponential_map(learned_pl_manifold, x0, x1, x_grid_cpu, y_grid_cpu, grid_density, writer, epoch, idx, "Learned")

        # Plot geodesics
        plot_geodesic(gt_pl_manifold, x0, x1, x_grid_cpu, y_grid_cpu, grid_density, writer, epoch, idx, "Ground Truth")
        plot_geodesic(learned_pl_manifold, x0, x1, x_grid_cpu, y_grid_cpu, grid_density, writer, epoch, idx, "Learned")


def compute_density_grid(manifold, device, value_range):
    """Generate a grid and compute the density for the ground truth manifold."""
    xx = torch.linspace(value_range[0], value_range[1], 500, device=device)
    yy = torch.linspace(value_range[0], value_range[1], 500, device=device)
    x_grid, y_grid = torch.meshgrid(xx, yy, indexing='ij')
    xy_grid = torch.stack((x_grid, y_grid), dim=2).reshape(-1, 2)
    
    # Compute the density using the manifold (ensure tensors are on the same device)
    density = torch.exp(manifold.dg.log_density(xy_grid)).reshape(500, 500)
    density = density.detach().cpu().numpy()
    x_grid_cpu = x_grid.detach().cpu().numpy()
    y_grid_cpu = y_grid.detach().cpu().numpy()
    return x_grid_cpu, y_grid_cpu, density

def plot_geodesic(manifold, x0, x1, x_grid_cpu, y_grid_cpu, grid_density, writer, epoch, pair_idx, manifold_type):
    """Plot and log the geodesic for a given manifold with improved visibility and range adjustments."""
    t = torch.linspace(0., 1., 100, device=x0.device)
    geodesic = manifold.geodesic(x0, x1, t).detach().cpu().numpy()

    x0_cpu, x1_cpu = x0.cpu().numpy(), x1.cpu().numpy()

    fig, ax = plt.subplots()
    # Plot the density with transparency
    levels = np.linspace(np.min(grid_density), np.max(grid_density), 25)
    ax.contourf(x_grid_cpu, y_grid_cpu, grid_density, levels=levels, cmap='Blues', alpha=0.5)
    
    # Plot geodesic with a thicker line and more visible color (e.g., dark blue)
    ax.plot(geodesic[:, 0], geodesic[:, 1], color='orange', linewidth=3, zorder=3)
    ax.scatter(x0_cpu[0], x0_cpu[1], color='red', zorder=3)
    ax.scatter(x1_cpu[0], x1_cpu[1], color='green', zorder=3)

    # Remove unnecessary elements
    ax.set_xlim([-2, 2])  # Limit x range as requested
    ax.set_ylim([-3, 3])  # Adjusted for better visibility
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')  # Remove axis
    ax.grid(False)
    
    # Save the figure to a buffer as an image (in high DPI) and add it to TensorBoard
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')  # Save high-resolution figure to buffer
    buf.seek(0)
    image = Image.open(buf)
    
    # Convert the image to a format suitable for TensorBoard
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)  # Convert to tensor, permute to (C, H, W)
    writer.add_image(f"{manifold_type} Geodesic/Pair {pair_idx}", image_tensor, epoch)  # Log the image to TensorBoard
    
    # Close the figure to free up memory
    plt.close(fig)


def plot_exponential_map(manifold, x0, x1, x_grid_cpu, y_grid_cpu, grid_density, writer, epoch, pair_idx, manifold_type):
    """Plot and log the exponential map for a given manifold with improved visibility and range adjustments."""
    # Compute logarithmic map and exponential map
    logarithmic = manifold.log(x0, x1[None])[0].detach().cpu().numpy()
    exponential = manifold.exp(x0, torch.tensor(logarithmic, device=x0.device)[None])[0].detach().cpu().numpy()

    x0_cpu, x1_cpu = x0.cpu().numpy(), x1.cpu().numpy()

    fig, ax = plt.subplots()
    # Plot the density with transparency
    levels = np.linspace(np.min(grid_density), np.max(grid_density), 25)
    ax.contourf(x_grid_cpu, y_grid_cpu, grid_density, levels=levels, cmap='Blues', alpha=0.5)
    
    # Plot points and geodesic with improved thickness and color
    ax.scatter(x0_cpu[0], x0_cpu[1], color='red', zorder=3)
    ax.arrow(x0_cpu[0], x0_cpu[1], logarithmic[0], logarithmic[1], head_width=0.2, color='darkblue', length_includes_head=True, linewidth=3, zorder=3)
    ax.scatter(exponential[0], exponential[1], color='darkblue', zorder=3)
    ax.scatter(x1_cpu[0], x1_cpu[1], color='green', zorder=3)

    # Remove unnecessary elements
    ax.set_xlim([-2, 2])  # Adjust the range to focus on the shape
    ax.set_ylim([-3, 3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')  # Remove axis
    ax.grid(False)

    # Save the figure to a buffer as an image (in high DPI) and add it to TensorBoard
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')  # Save high-resolution figure to buffer
    buf.seek(0)
    image = Image.open(buf)
    
    # Convert the image to a format suitable for TensorBoard
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)  # Convert to tensor, permute to (C, H, W)
    writer.add_image(f"{manifold_type} Exponential Map/Pair {pair_idx}", image_tensor, epoch)  # Log the image to TensorBoard
    
    # Close the figure to free up memory
    plt.close(fig)
