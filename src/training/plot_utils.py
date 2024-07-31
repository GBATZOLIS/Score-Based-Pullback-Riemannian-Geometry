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