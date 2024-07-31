import torch
import numpy as np

def compute_mean_distance_and_sigma(train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_data = []

    # Collect all data points from the train_loader
    for data in train_loader:
        if isinstance(data, list):
            x = data[0]
        else:
            x = data
        
        all_data.append(x.to(device))
    
    all_data = torch.cat(all_data, dim=0)
    total_points = all_data.size(0)
    dims = all_data.shape[1:]
    
    # Reshape the data to (total_points, c*w*h) for pairwise distance computation
    all_data_flat = all_data.view(total_points, -1)
    
    # Compute pairwise distances using broadcasting and vectorized operations
    distances_matrix = torch.cdist(all_data_flat, all_data_flat, p=2)
    
    # Fill the diagonal with a large value to exclude self-distance
    distances_matrix.fill_diagonal_(float('inf'))
    
    # Find the smallest distance for each data point
    min_distances, _ = torch.min(distances_matrix, dim=1)
    
    distances = min_distances.cpu().numpy()
    sigma_values = distances / (2 * np.sqrt(np.prod(dims)))
    
    mean_distance = np.mean(distances) if distances.size > 0 else 0
    std_distance = np.std(distances) if distances.size > 0 else 0
    mean_sigma = np.mean(sigma_values) if sigma_values.size > 0 else 0
    std_sigma = np.std(sigma_values) if sigma_values.size > 0 else 0

    print(f"Mean Distance: {mean_distance}")
    print(f"Standard Deviation of Distance: {std_distance}")
    print(f"Mean Sigma Value: {mean_sigma}")
    print(f"Standard Deviation of Sigma Value: {std_sigma}")

    return mean_distance, std_distance, mean_sigma, std_sigma