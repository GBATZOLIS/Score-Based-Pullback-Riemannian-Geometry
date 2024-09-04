import numpy as np
import torch
from .images import check_manifold_properties_images
from .euclidean_2d import check_manifold_properties_2D_distributions
from .euclidean_3d import check_manifold_properties_3D_distributions

def check_manifold_properties(dataset, phi, psi, writer, epoch, device, val_loader, d=2, create_gif=False):
    if dataset in ['mnist', 'blobs']:
        check_manifold_properties_images(phi, psi, writer, epoch, device, val_loader, create_gif=create_gif)
    elif dataset == 'single_banana':
        range = [-6., 6.]
        special_points = [[2., 4.], [2., -4.]]
        check_manifold_properties_2D_distributions(phi, psi, writer, epoch, device, val_loader, range, special_points)
    elif dataset == 'combined_elongated_gaussians':
        range = [-3., 3.]
        special_points = [[0., 1.], [-1., 0.]]
        check_manifold_properties_2D_distributions(phi, psi, writer, epoch, device, val_loader, range, special_points)
    elif dataset == 'spiral':
        range = [-8,8]
        special_points = [[0., 0.], [2*np.pi, 0.]] #theta=pi/4, theta=11pi/4
        check_manifold_properties_2D_distributions(phi, psi, writer, epoch, device, val_loader, range, special_points)
    elif dataset in ['sphere', 'ellipsoid']:
        if d == 3:
            range = [-1.5, 1.5]
            if dataset == 'sphere':
                special_points = [
                    [1 / np.sqrt(2), 0, 1 / np.sqrt(2)], 
                    [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]
                ]
            else:  # ellipsoid
                special_points = [
                    [1 / np.sqrt(2), 0, 1 / np.sqrt(2)], 
                    [0, 1 / (3 * np.sqrt(2)), 1 / np.sqrt(2)]
                ]
            # Convert special points to torch tensors with float32 type
            special_points = [torch.tensor(point, dtype=torch.float32) for point in special_points]
            check_manifold_properties_3D_distributions(
                phi, psi, writer, epoch, device, val_loader, range, special_points
            )
        elif d == 2:
            range = [-1.5, 1.5]
            special_points = [
                    [1., 0.], 
                    [0., 1.]
                ]
            special_points = [torch.tensor(point, dtype=torch.float32) for point in special_points]
            check_manifold_properties_2D_distributions(phi, psi, writer, epoch, device, val_loader, range, special_points)
