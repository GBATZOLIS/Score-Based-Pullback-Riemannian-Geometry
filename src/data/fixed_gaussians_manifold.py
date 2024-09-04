import torch
import numpy as np
from torch.utils.data import Dataset, random_split
import random 

class fixed_gaussians_manifold(Dataset):
    def __init__(self, config, split):
        super(fixed_gaussians_manifold, self).__init__()
        self.config = config
        self.split = split

        # Set the random seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

        # Create the full dataset
        full_data, full_labels = self.create_dataset(config)

        # Perform the train/val/test split
        train_size = int(config.split[0] * len(full_data))
        val_size = int(config.split[1] * len(full_data))
        test_size = len(full_data) - train_size - val_size

        train_data, val_data, test_data = random_split(full_data, [train_size, val_size, test_size])
        train_labels, val_labels, test_labels = random_split(full_labels, [train_size, val_size, test_size])

        # Select the appropriate split
        if split == 'train':
            self.data = train_data
            self.labels = train_labels
        elif split == 'val':
            self.data = val_data
            self.labels = val_labels
        elif split == 'test':
            self.data = test_data
            self.labels = test_labels
        else:
            raise ValueError(f"Unknown split: {split}")

    def get_the_gaussian_centers(self, seed, num_gaussians, img_size):
        random.seed(seed)
        pairs = [(i, j) for i in range(img_size) for j in range(img_size)]

        # Select the centers without replacement
        centers_info = random.sample(pairs, k=num_gaussians)

        return centers_info

    def create_dataset(self, config):
        num_samples = config.data_samples
        num_gaussians = config.num_gaussians
        std_range = config.std_range
        img_size = config.image_size

        centers_info = self.get_the_gaussian_centers(config.seed, num_gaussians, img_size)

        data = []
        labels = []  # Labels are empty or unused, but we'll return them anyway
        for _ in range(num_samples):
            img = torch.zeros(size=(img_size, img_size))
            for center in centers_info:
                x, y = center

                # Paint the gaussians efficiently
                img = self.paint_the_gaussian(img, x, y, std_range)

            # Scale the image to [0, 1] range
            img -= img.min()
            img /= img.max()

            data.append(img.to(torch.float32).unsqueeze(0))
            labels.append(0)  # Append a placeholder label

        data = torch.stack(data)
        labels = torch.tensor(labels)
        return data, labels

    def paint_the_gaussian(self, img, center_x, center_y, std_range):
        std = random.uniform(std_range[0], std_range[1])
        c = 1 / (np.sqrt(2 * np.pi) * std)
        new_img = torch.zeros_like(img)

        x = torch.arange(img.size(0))
        y = torch.arange(img.size(1))
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        d = -1 / (2 * std ** 2)
        new_img = torch.exp(d * ((xx - center_x) ** 2 + (yy - center_y) ** 2))
        new_img *= c
        img += new_img
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]  # Retrieve the corresponding label (even if it's a placeholder)
        return image, label