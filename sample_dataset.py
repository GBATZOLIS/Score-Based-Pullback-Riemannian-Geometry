from src.unimodal.deformed_gaussian.quadratic_banana import QuadraticBanana
from src.unimodal.deformed_gaussian.quadratic_river import QuadraticRiver
from src.multimodal.deformed_sum_of_identical_gaussian.multi_identical_quadratic_river import MultiIdenticalQuadraticRiver
from src.diffeomorphisms.spherical_diffeomorphism import spherical_diffeomorphism
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import os
import argparse
from src.unimodal.deformed_gaussian import DeformedGaussian
from mpl_toolkits.mplot3d import Axes3D
    
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def langevin_mcmc(model, num_samples, step_size, initial_value):
    # Ensure the initial value has requires_grad = True
    if not initial_value.requires_grad:
        initial_value.requires_grad_(True)
    
    batch_size = initial_value.shape[0]
    dim = initial_value.shape[1]  # Generalizing for different dimensions (e.g., 2D or 3D)
    
    samples = torch.zeros(num_samples, batch_size, dim)  # Initialize tensor to store trajectories
    current_x = initial_value.clone()  # Clone to avoid modifying the input

    # Initialize the rejection counter
    num_rejections = 0

    for i in tqdm(range(num_samples)):
        current_grad = model.score(current_x)

        # Create a proposed_x based on current_x modifications
        proposed_x = current_x + 0.5 * step_size**2 * current_grad + step_size * torch.randn_like(current_x)
        proposed_x = proposed_x.detach().requires_grad_()  # Detach and require grad

        proposed_grad = model.score(proposed_x)
        
        # Compute Metropolis-Hastings acceptance probability
        log_acceptance_ratio = model.log_density(proposed_x) - model.log_density(current_x) \
                               + 0.5 * ((current_x - proposed_x + 0.5 * step_size**2 * proposed_grad).pow(2).sum(dim=1) / step_size**2) \
                               - 0.5 * ((proposed_x - current_x + 0.5 * step_size**2 * current_grad).pow(2).sum(dim=1) / step_size**2)

        # Accept or reject
        accept = torch.log(torch.rand(batch_size)) < log_acceptance_ratio

        # Count the number of rejections (when accept is False)
        num_rejections += (~accept).sum().item()  # Count the number of rejections

        # Use torch.where to handle tensor updates without in-place operations
        current_x = torch.where(accept.unsqueeze(1), proposed_x, current_x)

        # Store the current_x in samples at index i
        samples[i] = current_x.detach()  # Store a detached version of the current_x

    # Calculate the acceptance rate
    total_proposals = num_samples * batch_size
    acceptance_rate = (total_proposals - num_rejections) / total_proposals

    print(f"Acceptance Rate: {acceptance_rate * 100:.2f}%")
    print(f"Total Rejections: {num_rejections}")

    return samples

def plot_samples(samples, log_density_values=None, x_grid=None, y_grid=None, step=None):
    dim = samples.shape[1]  # Determine the dimensionality of the samples

    if dim == 2:
        # 2D plotting
        fig = plt.figure(figsize=(10, 8))  # Create figure object
        if log_density_values is not None and x_grid is not None and y_grid is not None:
            plt.contourf(x_grid.numpy(), y_grid.numpy(), np.exp(log_density_values), levels=50, cmap='viridis')
        plt.scatter(samples[:, 0], samples[:, 1], color='red', s=10, label=f'Samples at step {step}')
        plt.title('Langevin MCMC Samples on Log Density Contour' if log_density_values is not None else 'Langevin MCMC Samples')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        if log_density_values is not None:
            plt.colorbar(label='Probability Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'step_{step}.png' if step is not None else 'samples.png')
        plt.close(fig)  # Close the figure after saving it

    elif dim == 3:
        # 3D plotting
        fig = plt.figure(figsize=(10, 8))  # Create figure object
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], color='blue', s=10, label=f'Samples at step {step}')

        # Setting the title and labels
        ax.set_title('Langevin MCMC Samples in 3D', fontsize=14)
        ax.set_xlabel('X axis', fontsize=12)
        ax.set_ylabel('Y axis', fontsize=12)
        ax.set_zlabel('Z axis', fontsize=12)
        ax.legend()

        plt.savefig(f'step_{step}.png' if step is not None else 'samples.png')
        plt.close(fig)  # Close the figure after saving it
    else:
        raise ValueError("Samples must be 2D or 3D for plotting.")

def plot_density_3d_slices(log_density_values, x_grid, y_grid, z_grid, slice_indices=[0, 5, 9], save_dir='./plots'):
    """
    Plot and save 3D density slices for specific z-values.
    :param log_density_values: 3D array of log density values (reshaped to 10x10x10).
    :param x_grid: Grid values for x.
    :param y_grid: Grid values for y.
    :param z_grid: Grid values for z.
    :param slice_indices: List of indices along the z-axis for which to plot slices.
    :param save_dir: Directory to save the plot images.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

    for i, z_idx in enumerate(slice_indices):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface for the fixed z slice
        surf = ax.plot_surface(x_grid[:, :, z_idx].numpy(), 
                               y_grid[:, :, z_idx].numpy(), 
                               np.exp(log_density_values[:, :, z_idx]),  # Plot the exponential of log-density
                               cmap='viridis')
        
        ax.set_title(f'Log Density Slice at z={z_grid[0, 0, z_idx].item():.2f}')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Density')

        # Add color bar to the plot
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # Save the figure
        save_path = os.path.join(save_dir, f'density_slice_z_{z_grid[0, 0, z_idx].item():.2f}.png')
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure after saving

    print(f'Saved density slice plots in {save_dir}')


def main():
    parser = argparse.ArgumentParser(description='Run Langevin MCMC to generate samples and optional outputs.')
    parser.add_argument('--dataset', type=str, choices=['spherical', 'single_banana', 'squeezed_single_banana', 'combined_elongated_gaussians', 'spiral', 'river'], required=True, help='Choose the dataset.')
    parser.add_argument('--create_gif', action='store_true', help='Create a GIF of the sampling process.')
    parser.add_argument('--save_data', action='store_false', help='Save the last sample in the MCMC chain.')
    parser.add_argument('--save_dir', type=str, default='./data/', help='Directory to save the data.')

    args = parser.parse_args()

    # Run the Langevin MCMC
    num_samples = 1000 #5000
    num_mcmc_samples = 1200
    step_size = 0.11

    if args.dataset == 'single_banana':
        shear, offset, a1, a2 = 1/9, 0., 1/4, 4
        model = QuadraticBanana(shear, offset, torch.tensor([a1, a2]))
        initial_value = torch.zeros((num_samples, 2), requires_grad=True)
    elif args.dataset == 'squeezed_single_banana':
        shear, offset, a1, a2 = 1/9, 0., 1/81, 4
        model = QuadraticBanana(shear, offset, torch.tensor([a1, a2]))
        initial_value = torch.zeros((num_samples, 2), requires_grad=True)
    elif args.dataset == 'river':
        shear, offset, a1, a2 = 2, 0, 1/25, 3
        model = QuadraticRiver(shear, offset, torch.tensor([a1, a2]))
        initial_value = torch.zeros((num_samples, 2), requires_grad=True)
    elif args.dataset == 'spherical':
        variances = torch.tensor([0.02, torch.pi/8, torch.pi/8])
        model = DeformedGaussian(spherical_diffeomorphism(), variances)
        initial_value = torch.tensor([0.5, 0.5, 0.7071], requires_grad=True).unsqueeze(0).repeat(num_samples, 1).clone().detach().requires_grad_(True)



    # Define the grid for visualization based on dimensionality
    if initial_value.shape[1] == 2:
        xx = torch.linspace(-12.0, 12.0, 500)
        yy = torch.linspace(-12.0, 12.0, 500)
        x_grid, y_grid = torch.meshgrid(xx, yy, indexing='ij')
        xy_grid = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
        log_density_values = model.log_density(xy_grid).reshape(500, 500).detach().numpy()
        z_grid = None
    else:
        # 3D grid creation for log density
        xx = torch.linspace(-1., 1., 10)
        yy = torch.linspace(-1., 1., 10)
        zz = torch.linspace(-1., 1., 10)
        x_grid, y_grid, z_grid = torch.meshgrid(xx, yy, zz, indexing='ij')
        xyz_grid = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)
        log_density_values = model.log_density(xyz_grid).reshape(10, 10, 10).detach().numpy()

        # Plot 3D density slices for fixed z values
        plot_density_3d_slices(log_density_values, x_grid, y_grid, z_grid, slice_indices=[0, 5, 9])


        
    interval_samples = langevin_mcmc(model, num_mcmc_samples, step_size, initial_value)
    samples = interval_samples[-1].numpy()
    plot_samples(samples, log_density_values, x_grid, y_grid, -1)

    if args.create_gif:
        # Generate plots and GIF
        filenames = []
        indices = np.linspace(0, num_mcmc_samples-1, 20, dtype=int)
        for idx in indices:
            samples = interval_samples[idx].numpy()
            plot_samples(samples, log_density_values, x_grid, y_grid, idx)
            filenames.append(f'step_{idx}.png')

        with imageio.get_writer('langevin_mcmc.gif', mode='I', duration=0.5) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in filenames:
            os.remove(filename)  # Clean up files

    if args.save_data:
        # Save the last sample in the MCMC chain
        final_samples = interval_samples[-1].numpy()

        # Calculate the sizes for train, validation, and test sets
        num_samples = len(final_samples)
        train_size = int(0.8 * num_samples)
        val_size = int(0.1 * num_samples)

        # Split the data
        train_data = final_samples[:train_size]
        val_data = final_samples[train_size:train_size + val_size]
        test_data = final_samples[train_size + val_size:]

        # Ensure the directory exists
        save_path = os.path.join(args.save_dir, args.dataset)
        os.makedirs(save_path, exist_ok=True)

        # Save the datasets as .npy files
        np.save(os.path.join(save_path, 'train.npy'), train_data)
        np.save(os.path.join(save_path, 'val.npy'), val_data)
        np.save(os.path.join(save_path, 'test.npy'), test_data)

if __name__ == "__main__":
    main()
