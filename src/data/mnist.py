import torch
import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision import datasets

class mnist(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split

        # Load the dataset
        if split in ['train', 'val']:
            full_dataset = datasets.MNIST(
                root=config.data_path,
                train=True,
                download=True,
                transform=None  # Do not apply any transformation
            )
            # Split into training and validation sets (90% train, 10% val)
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

            # Select the appropriate split
            self.dataset = train_dataset if split == 'train' else val_dataset

        elif split == 'test':
            self.dataset = datasets.MNIST(
                root=config.data_path,
                train=False,
                download=True,
                transform=None  # Do not apply any transformation
            )
        else:
            raise ValueError(f"Unknown split: {split}")

        # Filter dataset based on the specified digit
        if config.digit != 'all':
            self.dataset = [(image, label) for image, label in self.dataset if label == config.digit]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Convert the image from PIL to a numpy array and keep pixel values in the range [0, 255]
        image = np.array(image, dtype=np.float32)
        # Add padding to make the image 32x32
        image = np.pad(image, ((2, 2), (2, 2)), mode='constant', constant_values=0)
        # Add a channel dimension (1, h, w)
        image = image[None, :, :]
        return torch.tensor(image), label
