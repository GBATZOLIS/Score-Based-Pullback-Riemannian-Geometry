import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class euclidean_datasets_3d(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.dimension = config.d
        self.samples = self.generate_data(config.data_samples)

        # Split the data
        total_samples = len(self.samples)
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)
        test_size = total_samples - train_size - val_size

        train_data, val_data, test_data = random_split(self.samples, [train_size, val_size, test_size])

        # Assign the appropriate split
        if split == 'train':
            self.samples = train_data
        elif split == 'val':
            self.samples = val_data
        elif split == 'test':
            self.samples = test_data
        else:
            raise ValueError(f"Unknown split: {split}")

    def generate_data(self, num_samples):
        if self.config.dataset == 'sphere':
            return self._generate_sphere_data(num_samples)
        elif self.config.dataset == 'ellipsoid':
            return self._generate_ellipsoid_data(num_samples)
        else:
            raise ValueError("Unknown dataset type")

    def _generate_sphere_data(self, num_samples):
        data = []
        while len(data) < num_samples:
            point = np.random.randn(self.dimension)
            point /= np.linalg.norm(point)  # Normalize to lie on the unit sphere
            if all(coord >= 0 for coord in point):  # Keep only the points in the positive orthant
                data.append(point)
        return np.array(data)

    def _generate_ellipsoid_data(self, num_samples):
        data = []
        while len(data) < num_samples:
            point = np.random.randn(self.dimension)
            if self.dimension == 3:
                point[1] *= 3  # Scale the second dimension to create the ellipsoid
                point /= np.linalg.norm(point)  # Normalize to lie on the ellipsoid surface
                point[1] *= 1 / 3  # Scale back to get the ellipsoid shape
            elif self.dimension == 2:
                point[0] *= 2  # Scale the first dimension to create the ellipse
                point /= np.linalg.norm(point)  # Normalize to lie on the ellipse
                point[0] *= 1 / 2  # Scale back to get the ellipse shape
            if point[-1] >= 0:  # Keep only the points in the upper hemisphere (z >= 0) or upper half (y >= 0)
                data.append(point)
        return np.array(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = []  # Empty label as specified
        return torch.tensor(sample, dtype=torch.float32), label
