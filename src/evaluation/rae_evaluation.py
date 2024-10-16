import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

def log_diagonal_values(psi, plots_dir):
    """
    Log and save the diagonal values of psi in both normal and log scale.
    """
    # Access the diagonal values of psi
    diagonal_values = psi.diagonal.detach().cpu().numpy()
    diagonal_values = np.sort(diagonal_values)[::-1]

    # Change x-axis indices to integers starting from 1
    x_values = np.arange(1, len(diagonal_values) + 1)

    # Create and save normal scale plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, diagonal_values, marker='o', linestyle='-', color='b')  # Use x_values for x-axis
    ax.set_xlabel('Index', fontsize=16)  # Increased font size for label
    ax.set_ylabel('Diagonal Value', fontsize=16)  # Increased font size for label
    ax.grid(True)

    # Set x-axis ticks to integers (e.g., 1, 5, 10, 15, etc.)
    ax.set_xticks(np.arange(1, len(diagonal_values) + 1, step=4))
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increased font size for ticks

    normal_diag_plot_path = os.path.join(plots_dir, 'diagonal_values_normal_scale.png')
    fig.savefig(normal_diag_plot_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Create and save log scale plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, diagonal_values, marker='o', linestyle='-', color='b')  # Use x_values for x-axis
    ax.set_yscale('log')
    ax.set_xlabel('Index', fontsize=16)  # Increased font size for label
    ax.set_ylabel('Log Diagonal Value', fontsize=16)  # Increased font size for label
    ax.grid(True)

    # Set x-axis ticks to integers (e.g., 1, 5, 10, 15, etc.)
    ax.set_xticks(np.arange(1, len(diagonal_values) + 1, step=4))
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increased font size for ticks

    log_diag_plot_path = os.path.join(plots_dir, 'diagonal_values_log_scale.png')
    fig.savefig(log_diag_plot_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"Diagonal value plots saved at: {normal_diag_plot_path} and {log_diag_plot_path}")

def compute_reconstruction_errors(psi, phi, test_loader, device, ordered_indices):
    """
    Compute the average reconstruction errors for a given order of indices.
    """
    reconstruction_errors = []
    
    for i in tqdm(range(1, len(ordered_indices) + 1), desc="Evaluating dimensions"):
        current_indices = ordered_indices[:i]
        total_reconstruction_error = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch.to(device)
                encoded = phi.forward(x)[:, current_indices]
                reconstructed_x = phi.inverse(torch.einsum("Nk,kd->Nd", encoded, torch.eye(psi.diagonal.size(0), device=device)[current_indices]))
                reconstruction_error = torch.mean((x - reconstructed_x) ** 2).item()
                total_reconstruction_error += reconstruction_error * x.size(0)
                total_samples += x.size(0)
        
        avg_reconstruction_error = total_reconstruction_error / total_samples
        reconstruction_errors.append(avg_reconstruction_error)
    
    return reconstruction_errors

def evaluate_reconstruction_errors(psi, phi, test_loader, device, plots_dir):
    """
    Fetch and sort the diagonal elements of psi, compute reconstruction errors
    for max-to-min, min-to-max, and random variance orders, and save the plots.
    """
    # Fetch and sort the diagonal elements from psi
    diagonal = psi.diagonal
    sorted_inv_diagonal, sorted_indices = (1 / diagonal).sort()
    sorted_diagonal = 1 / sorted_inv_diagonal
    
    # 1. Maximum-to-Minimum order (already sorted)
    max_to_min_order = sorted_indices
    
    # 2. Minimum-to-Maximum order (reverse of max_to_min_order)
    min_to_max_order = max_to_min_order.flip(dims=[0])
    
    # 3. Random order
    random_order = torch.randperm(len(sorted_indices), device=device)
    
    # Compute reconstruction errors for each order
    max_to_min_errors = compute_reconstruction_errors(psi, phi, test_loader, device, max_to_min_order)
    min_to_max_errors = compute_reconstruction_errors(psi, phi, test_loader, device, min_to_max_order)
    random_errors = compute_reconstruction_errors(psi, phi, test_loader, device, random_order)
    
    # X-axis values (1-based index)
    x_values = np.arange(1, len(sorted_indices) + 1)
    
    # Create high-quality plots
    plt.figure(figsize=(10, 6), dpi=300)  # Widen the figure to provide more space
    
    # Plot the reconstruction errors for each order
    plt.plot(x_values, max_to_min_errors, marker='o', linestyle='-', color='b', label='Max to Min Variance')
    plt.plot(x_values, min_to_max_errors, marker='s', linestyle='--', color='g', label='Min to Max Variance')
    plt.plot(x_values, random_errors, marker='d', linestyle='-.', color='r', label='Random Order')
    
    plt.xlabel('Number of Dimensions Used', fontsize=16)  # Increased font size for label
    plt.ylabel('Average Reconstruction Error', fontsize=16)  # Increased font size for label
    plt.grid(True, which="both", ls="--")
    
    # Increase font size for ticks
    plt.xticks(np.arange(1, len(x_values) + 1, step=4), fontsize=14)  # Increased font size for ticks
    plt.yticks(fontsize=14)

    # Save the normal scale plot
    normal_plot_path = os.path.join(plots_dir, 'reconstruction_error_plot_normal_scale.png')
    plt.savefig(normal_plot_path, bbox_inches='tight', dpi=300)
    
    # Set y-axis to log scale and save the log scale plot
    plt.yscale('log')
    log_plot_path = os.path.join(plots_dir, 'reconstruction_error_plot_log_scale.png')
    plt.savefig(log_plot_path, bbox_inches='tight', dpi=300)
    
    plt.close()

    print(f"Reconstruction error plots saved at: {normal_plot_path} and {log_plot_path}")

def rae_evaluation(psi, phi, test_loader, tensorboard_dir, device):
    """
    Main function that logs diagonal values and evaluates reconstruction errors.
    """
    # Create the 'plots' directory if it doesn't exist
    plots_dir = os.path.join(tensorboard_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Log the diagonal values
    log_diagonal_values(psi, plots_dir)
    
    # Evaluate and plot reconstruction errors for different variance orders
    evaluate_reconstruction_errors(psi, phi, test_loader, device, plots_dir)
