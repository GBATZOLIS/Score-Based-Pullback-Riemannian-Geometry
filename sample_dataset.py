from src.unimodal.deformed_gaussian.quadratic_banana import QuadraticBanana
from src.unimodal.deformed_gaussian.quadratic_river import QuadraticRiver
from src.multimodal.deformed_sum_of_identical_gaussian.multi_identical_quadratic_river import MultiIdenticalQuadraticRiver
from src.diffeomorphisms.spherical_diffeomorphism import spherical_diffeomorphism
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import imageio
import os
import argparse
from src.unimodal.deformed_gaussian import DeformedGaussian
from mpl_toolkits.mplot3d import Axes3D

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

#Function to return the appropriate generation function based on dataset
def get_generation_fn(args):
    if args.dataset == 'single_banana':
        num_samples, num_mcmc_samples, step_size = 2500, 1000, 0.1
        shear, offset, a1, a2 = 1/9, 0., 1/4, 4
        model = QuadraticBanana(shear, offset, torch.tensor([a1, a2]))
        initial_value = torch.zeros((num_samples, 2), requires_grad=True)
        return lambda: langevin_mcmc(model, num_mcmc_samples, step_size, initial_value)
    
    elif args.dataset == 'squeezed_single_banana':
        num_samples, num_mcmc_samples, step_size = 5000, 2000, 0.1
        shear, offset, a1, a2 = 1/9, 0., 1/81, 4
        model = QuadraticBanana(shear, offset, torch.tensor([a1, a2]))
        initial_value = torch.zeros((num_samples, 2), requires_grad=True)
        return lambda: langevin_mcmc(model, num_mcmc_samples, step_size, initial_value)
    
    elif args.dataset == 'river':
        num_samples, num_mcmc_samples, step_size = 5000, 2000, 0.1
        shear, offset, a1, a2 = 2, 0, 1/25, 3
        model = QuadraticRiver(shear, offset, torch.tensor([a1, a2]))
        initial_value = torch.zeros((num_samples, 2), requires_grad=True)
        return lambda: langevin_mcmc(model, num_mcmc_samples, step_size, initial_value)
    
    elif args.dataset == 'river3d':
        num_samples = 10000
        shear0, shear1 = 1, 2
        variances=torch.tensor([1/1000, 1/1000, 3])
        return lambda: generate_generalised_river_samples(num_samples, shear0=shear0, shear1=shear1, variances=variances)
    elif args.dataset == 'spiral':
        num_samples = 15000
        num_revolutions = 2  # Use number of revolutions from arguments
        return lambda: generate_spiral_samples_2d(num_samples, num_revolutions=num_revolutions)
    
    elif args.dataset == 'spherical':
        num_samples, num_mcmc_samples, step_size = 5000, 2000, 0.1
        variances = torch.tensor([0.02, torch.pi/16, torch.pi/16])
        model = DeformedGaussian(spherical_diffeomorphism(), variances)
        initial_value = torch.tensor([0.5, 0.5, 0.7071], requires_grad=True).unsqueeze(0).repeat(num_samples, 1).clone().detach().requires_grad_(True)
        return lambda: langevin_mcmc(model, num_mcmc_samples, step_size, initial_value)
    elif args.dataset.startswith('sinusoid'):
        parts = args.dataset.split('_')
        K, N = int(parts[1]), int(parts[2])
        num_samples = 100000 #5000 * K * int(np.maximum(np.sqrt(N/5), 1)) #for N=100, we used 1e5 for K=1, 1.5e-5 for K=5 and 2e-5 for K=10. Otherwise we used the formula.
        return lambda: generate_generalised_sinusoid_samples(num_samples, K, N)
    elif args.dataset.startswith('hemisphere'):
        parts = args.dataset.split('_')
        K, N = int(parts[1]), int(parts[2])
        num_samples = 5000 * K * int(np.maximum(np.sqrt(N/5), 1))
        return lambda: generate_hemisphere_samples(num_samples, K, N)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

# Save function to handle saving the generated data
def save_data(samples, save_dir, dataset):
    """
    Save the samples to train, validation, and test sets.
    
    Args:
    - samples (torch.Tensor): Samples to save.
    - save_dir (str): Directory to save the data.
    - dataset (str): Name of the dataset.
    """
    final_samples = samples

    # Calculate the sizes for train, validation, and test sets
    num_samples = len(final_samples)
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)

    # Split the data
    train_data = final_samples[:train_size]
    val_data = final_samples[train_size:train_size + val_size]
    test_data = final_samples[train_size + val_size:]

    # Ensure the directory exists
    save_path = os.path.join(save_dir, dataset)
    os.makedirs(save_path, exist_ok=True)

    # Save the datasets as .npy files
    np.save(os.path.join(save_path, 'train.npy'), train_data)
    np.save(os.path.join(save_path, 'val.npy'), val_data)
    np.save(os.path.join(save_path, 'test.npy'), test_data)

    print(f"Data saved to {save_path}.")


def generate_spiral_samples_2d(num_samples, num_revolutions=3):
    """
    Generates samples of a spiral dataset in 2D.

    Args:
    - num_samples (int): Number of samples to generate.
    - num_revolutions (int): Number of revolutions of the spiral.
        
    Returns:
    - samples (torch.Tensor): The generated spiral samples.
    """
    theta = torch.linspace(0, num_revolutions * 2 * np.pi, num_samples)  # Angular coordinate
    r = theta  # Radial coordinate equals the angular coordinate

    # 2D spiral
    x = r * torch.cos(theta)  # X-coordinates
    y = r * torch.sin(theta)  # Y-coordinates
    samples = torch.stack([x, y], dim=1)  # Shape (num_samples, 2)

    return samples


def generate_generalised_sinusoid_samples(num_samples, K, N):
    #K the dimension of the manifold
    #N the ambient dimension
    #shearing in the sinusoid formation
    #variances dictate the manifold concentration
    
    # Extract variances for the latent space (manifold) and ambient space
    manifold_variances = 3 * torch.ones(K)
    ambient_variances = 1e-3 * torch.ones(N-K)

    # Sample from the K-dimensional latent space (manifold)
    z_samples = torch.randn(num_samples, K) * torch.sqrt(manifold_variances)

    # Initialize random shears for the off-manifold (N-K) dimensions
    off_manifold_shears = torch.rand(N - K, K) * (2 - 1) + 1  # Random values between [1, 2]

    # Generate the off-manifold samples using the shear vectors and z_samples
    off_manifold_means = []
    for j in range(N - K):
        shear_j = off_manifold_shears[j]  # Get the shear vector for the j-th off-manifold dimension
        off_manifold_mean = torch.sin(z_samples @ shear_j.unsqueeze(1))  # Sinusoidal transformation for off-manifold
        off_manifold_means.append(off_manifold_mean.squeeze(1))  # Squeeze to get correct shape

    # Stack the off-manifold dimensions together
    off_manifold_means = torch.stack(off_manifold_means, dim=1)

    # Generate ambient samples for the remaining (N-K) dimensions
    off_manifold_samples = off_manifold_means + torch.randn(num_samples, N - K) * torch.sqrt(ambient_variances)

    # Combine the manifold and ambient samples
    samples = torch.cat([off_manifold_samples, z_samples], dim=1)
    
    return samples

def generate_hemisphere_samples(num_samples, K, N, alpha=5, beta=5):
    """
    Generates samples from the upper hemisphere of S^K and embeds them in N dimensions using a random isometry.

    Args:
    - num_samples (int): Number of samples to generate.
    - K (int): Dimension of the manifold (S^K).
    - N (int): Ambient dimension to embed into.
    - alpha (float): Shape parameter for the Beta distribution (controls emphasis on central values).
    - beta (float): Shape parameter for the Beta distribution (controls emphasis on central values).
        
    Returns:
    - samples (torch.Tensor): The generated samples in N dimensions.
    """
    # Use a Beta distribution to sample theta_1, emphasizing central values
    beta_dist = torch.distributions.Beta(alpha, beta)
    theta_1 = beta_dist.sample((num_samples,)) * (np.pi / 2)  # Scale to [0, pi/2]

    # Uniformly sample the other angles from [0, pi]
    other_angles = torch.rand(num_samples, K) * np.pi

    # Generate Cartesian coordinates from spherical angles
    manifold_samples = []
    for i in range(num_samples):
        angles = torch.cat((theta_1[i:i+1], other_angles[i]), dim=0)
        sin_prod = 1.0  # Initialize as scalar
        cartesian_coords = []
        for theta in angles[:-1]:
            cartesian_coords.append(sin_prod * torch.cos(theta))
            sin_prod *= torch.sin(theta)
        cartesian_coords.append(sin_prod)  # Final coordinate
        cartesian_coords = torch.tensor(cartesian_coords)  # Convert list of scalars to tensor
        manifold_samples.append(cartesian_coords)

    manifold_samples = torch.stack(manifold_samples)  # Shape: (num_samples, K+1)

    # Random isometric embedding into N dimensions (N >= K+1)
    randomness_generator = torch.Generator().manual_seed(0)
    embedding_matrix = torch.randn((N, K + 1), generator=randomness_generator)
    q, _ = torch.linalg.qr(embedding_matrix, mode='reduced')  # Create orthogonal embedding matrix
    embedded_samples = manifold_samples @ q.T  # Apply isometric embedding

    return embedded_samples


# Function to generate samples for generalised river
def generate_generalised_river_samples(num_samples, shear0=1, shear1=2, variances=torch.tensor([1/25, 1/25, 3])):
    var_x0, var_x1, var_z = variances[0], variances[1], variances[2]

    z_samples = torch.randn(num_samples) * torch.sqrt(var_z)
    x0_means = torch.sin(shear0 * z_samples)
    x1_means = torch.sin(shear1 * z_samples)

    x0_samples = x0_means + torch.randn(num_samples) * torch.sqrt(var_x0)
    x1_samples = x1_means + torch.randn(num_samples) * torch.sqrt(var_x1)

    samples = torch.stack([x0_samples, x1_samples, z_samples], dim=1)
    return samples

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

    return samples[-1]

def plot_samples(samples, step=None):
    dim = samples.shape[1]  # Determine the dimensionality of the samples

    if dim == 2:
        # 2D plotting (similar to 3D scatter but in 2D)
        fig = plt.figure(figsize=(10, 8))  # Create figure object
        ax = fig.add_subplot(111)
        ax.scatter(samples[:, 0], samples[:, 1], color='blue', s=10, label=f'Samples at step {step}')

        # Setting the title and labels
        ax.set_title('Langevin MCMC Samples in 2D', fontsize=14)
        ax.set_xlabel('X axis', fontsize=12)
        ax.set_ylabel('Y axis', fontsize=12)
        ax.legend()

        plt.savefig(f'step_{step}.png' if step is not None else 'samples.png')
        plt.show()  # Display the figure
        plt.close(fig)  # Close the figure after showing it

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
        plt.show()  # Display the figure
        plt.close(fig)  # Close the figure after showing it

    else:
        raise ValueError("Samples must be 2D or 3D for plotting.")

def plot_and_save_samples(samples, step=None, save_path=None):
    """
    Plots and saves the samples in the current directory with high quality.

    Args:
    - samples (torch.Tensor): The samples to plot.
    - step (int): Optional step for naming the saved file.
    - save_path (str): Path to save the plot.
    """
    dim = samples.shape[1]  # Determine the dimensionality of the samples

    if dim == 2:
        # 2D plotting
        fig = plt.figure(figsize=(10, 8), dpi=300)  # Create figure object with high DPI for quality
        ax = fig.add_subplot(111)
        ax.scatter(samples[:, 0], samples[:, 1], color='blue', s=10)

        # Remove legends and titles
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Save with high quality
        plot_filename = save_path if save_path else f'samples_{step}.png'
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.show()  # Display the figure
        plt.close(fig)  # Close the figure after showing it

    elif dim == 3:
        # 3D plotting
        fig = plt.figure(figsize=(10, 8), dpi=300)  # Create figure object with high DPI for quality
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], color='blue', s=10)

        # Remove legends and titles
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

        # Save with high quality
        plot_filename = save_path if save_path else f'samples_{step}.png'
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.show()  # Display the figure
        plt.close(fig)  # Close the figure after showing it

    else:
        raise ValueError("Samples must be 2D or 3D for plotting.")

def main():
    parser = argparse.ArgumentParser(description='Run Langevin MCMC to generate samples and optional outputs.')
    parser.add_argument('--dataset', type=str, required=True, help='Choose the dataset.')
    parser.add_argument('--create_gif', action='store_true', help='Create a GIF of the sampling process.')
    parser.add_argument('--save_data', action='store_false', help='Save the last sample in the MCMC chain.')
    parser.add_argument('--save_dir', type=str, default='./data/', help='Directory to save the data.')

    args = parser.parse_args()

    generation_fn = get_generation_fn(args)
    samples = generation_fn().numpy()

    # Save the data if required
    if args.save_data:
        save_data(samples, args.save_dir, args.dataset)

    # Plot and save samples with high quality
    #plot_and_save_samples(samples, save_path=f'{args.dataset}.png')

if __name__ == "__main__":
    main()


