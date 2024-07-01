import torch
import numpy as np
from torch.utils.data import Dataset
import os

class numpy_dataset(Dataset):
    def __init__(self, args, split):
        """
        Args:
            file_path (str): Path to the .npy file containing the data.
        """
        # Load the data from the .npy file
        self.data = np.load(os.path.join(args.data_path, args.dataset, split + '.npy'))

        # Convert the data to a torch tensor
        self.data = torch.from_numpy(self.data).float()  # Convert to float for consistency in most PyTorch models

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        Args:
            idx (int): The index of the sample to return.
        Returns:
            Tensor: The sample corresponding to the given index.
        """
        return self.data[idx]