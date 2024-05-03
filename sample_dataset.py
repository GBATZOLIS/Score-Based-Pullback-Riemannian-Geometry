from src.unimodal.deformed_gaussian.quadratic_banana import QuadraticBanana
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import imageio
import os 
import argparse

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def langevin_mcmc(model, num_samples, step_size, initial_value):
    if not initial_value.requires_grad:
        initial_value.requires_grad_(True)
    
    batch_size = initial_value.shape[0]
    samples = torch.zeros(num_samples, batch_size, model.d)  # Initialize tensor to store trajectories
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
    parser.add_argument('--create_gif', action='store_true', help='Create a GIF of the sampling process.')
    parser.add_argument('--save_data', action='store_false', help='Save the last sample in the MCMC chain.')
    parser.add_argument('--save_dir', type=str, default='./data/', help='Directory to save the data.')

    args = parser.parse_args()

    #parameters
    shear, offset, a1, a2 = 1/9, 0., 1/4, 4
    unimodal = QuadraticBanana(shear, offset, torch.tensor([a1, a2]))

    # Define the grid for visualization
    xx = torch.linspace(-6.0, 6.0, 500)
    yy = torch.linspace(-6.0, 6.0, 500)
    x_grid, y_grid = torch.meshgrid(xx, yy, indexing='ij')

    # Prepare the grid as input to the model for log density evaluation
    xy_grid = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)

    # Compute log density over the grid
    log_density_values = unimodal.log_density(xy_grid).reshape(500, 500).detach().numpy()
    
    # Run the Langevin MCMC
    num_samples = 2500
    num_mcmc_samples = 1000
    step_size = 0.1
    initial_value = torch.zeros((num_samples, 2), requires_grad=True)  # Batch of 1
    interval_samples = langevin_mcmc(unimodal, num_mcmc_samples, step_size, initial_value)

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
        #test_size = num_samples - train_size - val_size

        # Split the data
        train_data = final_samples[:train_size]
        val_data = final_samples[train_size:train_size + val_size]
        test_data = final_samples[train_size + val_size:]

        # Ensure the directory exists
        os.makedirs(args.save_dir, exist_ok=True)

        # Save the datasets as .npy files
        np.save(os.path.join(args.save_dir, 'train.npy'), train_data)
        np.save(os.path.join(args.save_dir, 'val.npy'), val_data)
        np.save(os.path.join(args.save_dir, 'test.npy'), test_data)

if __name__ == "__main__":
    main()
