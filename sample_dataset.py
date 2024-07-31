from src.unimodal.deformed_gaussian.quadratic_banana import QuadraticBanana
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import imageio
import os
import argparse

class combined_elongated_gaussians:
    def __init__(self):
        self.std_x = torch.tensor([1.0, 0.1])  # Standard deviations for the first Gaussian
        self.std_y = torch.tensor([0.1, 1.0])  # Standard deviations for the second Gaussian

    def log_density(self, x):
        term1 = 0.5 * torch.sum((x / self.std_x) ** 2, dim=1)
        term2 = 0.5 * torch.sum((x / self.std_y) ** 2, dim=1)
        log_density_x = -term1 - torch.log(2 * np.pi * self.std_x[0] * self.std_x[1])
        log_density_y = -term2 - torch.log(2 * np.pi * self.std_y[0] * self.std_y[1])
        combined_log_density = torch.logaddexp(log_density_x, log_density_y) - torch.log(torch.tensor(2.0))
        return combined_log_density

    def score(self, x):
        """ Compute the score function using PyTorch's autograd """
        original_requires_grad = x.requires_grad if isinstance(x, torch.Tensor) else False

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        elif not x.requires_grad:
            x.requires_grad_(True)

        # Compute the log density
        y = self.log_density(x)

        # Compute the sum of log density to make it a scalar
        y_sum = y.sum()

        # Compute gradients of y_sum with respect to x using torch.autograd.grad
        gradients = torch.autograd.grad(y_sum, x, create_graph=True)

        # Ensure that gradients are computed
        if gradients[0] is None:
            raise ValueError("Gradient not computed, check the computational graph and inputs.")

        # Clone the gradients
        cloned_gradients = gradients[0].clone()

        # Reset requires_grad to its original state if necessary
        if not original_requires_grad:
            x.requires_grad_(False)

        return cloned_gradients

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def langevin_mcmc(model, num_samples, step_size, initial_value):
    if not initial_value.requires_grad:
        initial_value.requires_grad_(True)
    
    batch_size = initial_value.shape[0]
    samples = torch.zeros(num_samples, batch_size, 2)  # Initialize tensor to store trajectories
    current_x = initial_value.clone()  # Ensure it is a leaf tensor

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

        # Use torch.where to handle tensor updates without in-place operations
        current_x = torch.where(accept.unsqueeze(1), proposed_x, current_x)

        # Store the current_x in samples at index i
        samples[i] = current_x.detach()  # Store a detached version of the current_x

    return samples

def plot_samples(samples, log_density_values, x_grid, y_grid, step):
    plt.figure(figsize=(10, 8))
    plt.contourf(x_grid.numpy(), y_grid.numpy(), np.exp(log_density_values), levels=50, cmap='viridis')
    plt.scatter(samples[:, 0], samples[:, 1], color='red', s=10, label=f'Samples at step {step}')
    plt.title('Langevin MCMC Samples on Log Density Contour')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.colorbar(label='Probability Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'step_{step}.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run Langevin MCMC to generate samples and optional outputs.')
    parser.add_argument('--dataset', type=str, choices=['single_banana', 'combined_elongated_gaussians'], required=True, help='Choose the dataset.')
    parser.add_argument('--create_gif', action='store_true', help='Create a GIF of the sampling process.')
    parser.add_argument('--save_data', action='store_false', help='Save the last sample in the MCMC chain.')
    parser.add_argument('--save_dir', type=str, default='./data/', help='Directory to save the data.')

    args = parser.parse_args()

    if args.dataset == 'single_banana':
        shear, offset, a1, a2 = 1/9, 0., 1/4, 4
        model = QuadraticBanana(shear, offset, torch.tensor([a1, a2]))
    elif args.dataset == 'combined_elongated_gaussians':
        model = combined_elongated_gaussians()

    # Define the grid for visualization
    xx = torch.linspace(-6.0, 6.0, 500)
    yy = torch.linspace(-6.0, 6.0, 500)
    x_grid, y_grid = torch.meshgrid(xx, yy, indexing='ij')

    # Prepare the grid as input to the model for log density evaluation
    xy_grid = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)

    # Compute log density over the grid
    log_density_values = model.log_density(xy_grid).reshape(500, 500).detach().numpy()
    
    # Run the Langevin MCMC
    num_samples = 2500
    num_mcmc_samples = 1000
    step_size = 0.1
    initial_value = torch.zeros((num_samples, 2), requires_grad=True)
    interval_samples = langevin_mcmc(model, num_mcmc_samples, step_size, initial_value)

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
