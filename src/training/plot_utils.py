import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

def plot_data(writer, dataloader, std, num_points=-1):
    # Collect data points
    data_points = []
    collected_points = 0
    for data in dataloader:
        if isinstance(data, list):
            x = data[0]
        else:
            x = data
        
        remaining_points = num_points - collected_points if num_points != -1 else x.shape[0]
        if remaining_points <= 0:
            break
        
        if x.shape[0] > remaining_points:
            x = x[:remaining_points]
        
        data_points.append(x)
        collected_points += x.shape[0]
        
        if num_points != -1 and collected_points >= num_points:
            break
    
    data_points = torch.cat(data_points, dim=0)
    if num_points == -1:
        num_points = len(data_points)

    # Perturb data points using the provided std
    perturbed_data_points = data_points + std * torch.randn_like(data_points)

    # Check the shape of the data points
    shape = perturbed_data_points.shape[1:]
    data_dim = len(shape)
    
    if data_dim == 1 and shape[0] in [1, 2, 3]:
        # Plot based on the dimension
        if shape[0] == 1:
            # 1D plot
            fig, ax = plt.subplots()
            ax.plot(perturbed_data_points.cpu().numpy())
            ax.set_title('Perturbed Distribution')
            writer.add_figure('1D Plot', fig)
            plt.close(fig)
        elif shape[0] == 2:
            # 2D plot
            fig, ax = plt.subplots()
            ax.scatter(perturbed_data_points[:, 0].cpu().numpy(), perturbed_data_points[:, 1].cpu().numpy())
            ax.set_title('Perturbed Distribution')
            writer.add_figure('2D Plot', fig)
            plt.close(fig)
        elif shape[0] == 3:
            # 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(perturbed_data_points[:, 0].cpu().numpy(), perturbed_data_points[:, 1].cpu().numpy(), perturbed_data_points[:, 2].cpu().numpy())
            ax.set_title('Perturbed Distribution')
            writer.add_figure('3D Plot', fig)
            plt.close(fig)
        else:
            raise ValueError("Current functionality only supports 1D, 2D, and 3D data.")
    
    elif data_dim == 3 and shape[0] in [1, 3]:
        # Image data
        nrow = int(np.ceil(np.sqrt(num_points)))
        img_grid = make_grid(perturbed_data_points, nrow=nrow, normalize=True, scale_each=True)
        writer.add_image('Perturbed Distribution', img_grid)
    
    else:
        raise ValueError("Current functionality only supports data with dimensions 1, 2, 3, or image shaped data with 1 or 3 channels.")
    


def plot_variances(writer, stds, epoch, epsilon=0.1):
    # Access the diagonal values of psi (which is stds**2)
    diagonal_values = stds**2  # Make sure to move to CPU for numpy operations if necessary
    device = stds.device
    d = len(stds)

    # Sort the diagonal values in descending order
    sorted_inv_diagonal, sorted_indices = (1 / diagonal_values).sort()
    sorted_diagonal = 1 / sorted_inv_diagonal  # Largest value is first

    # Calculate d_eps based on epsilon threshold
    if sorted_diagonal[-1] <= epsilon * sorted_diagonal.sum():
        tmp = [sorted_diagonal[i+1:].sum() <= epsilon * sorted_diagonal.sum() for i in range(d-1)]
        tmp_indices = torch.arange(0, d-1, device=device)[tmp]
        try:
            d_eps = tmp_indices.min().item() + 1
        except RuntimeError as e:
            print(f"Error: {e}")
            print(f"tmp_indices: {tmp_indices}")
            d_eps = d  # default value in case of error
        eps = sorted_diagonal[d_eps:].sum() / sorted_diagonal.sum()
    else:
        d_eps = d
        eps = 0.

    # Log d_eps and eps to TensorBoard
    writer.add_scalar("d_eps", d_eps, epoch)
    writer.add_scalar("epsilon", eps, epoch)

    print(f"Calculated d_eps = {d_eps} and eps = {eps}")

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