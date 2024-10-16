from .torchutils import (
    create_alternating_binary_mask,
    create_mid_split_binary_mask,
    create_random_binary_mask,
    get_num_parameters,
    logabsdet,
    random_orthogonal,
    sum_except_batch,
    split_leading_dim,
    merge_leading_dims,
    repeat_rows,
    tensor2numpy,
    tile,
    searchsorted,
    cbrt,
    get_temperature
)

from .typechecks import is_bool
from .typechecks import is_int
from .typechecks import is_positive_int
from .typechecks import is_nonnegative_int
from .typechecks import is_power_of_two

from .io import get_data_root
from .io import NoDataRootError

import torch
from sklearn.decomposition import PCA

def get_principal_components(train_loader, std, num_times_to_add_noise=3):
    data = []
    
    # Collect all data into a single tensor
    for batch in train_loader:
        x = batch[0] if isinstance(batch, list) else batch
        x_flat = x.view(x.size(0), -1)  # Flatten the images
        data.append(x_flat)
    
    data = torch.cat(data, dim=0)  # Concatenate along the batch dimension

    # If std > 0, add noise multiple times and concatenate the noisy versions
    if std > 0:
        noisy_data_list = []
        for _ in range(num_times_to_add_noise):
            noisy_data = data + std * torch.randn_like(data)  # Add noise
            noisy_data_list.append(noisy_data)
        concatenated_data = torch.cat(noisy_data_list, dim=0)  # Concatenate noisy versions
    else:
        concatenated_data = data  # No noise added if std == 0
    
    # Number of samples (N)
    N = concatenated_data.size(0)
    
    # Compute the mean of the concatenated data
    mean = torch.mean(concatenated_data, dim=0, keepdim=True)
    data_centered = concatenated_data - mean  # Center the data by subtracting the mean
    
    # Compute SVD on the centered data
    U, S, Vh = torch.linalg.svd(data_centered, full_matrices=False)
    
    # Keep only the positive singular values
    positive_singular_values = S[S > 0]

    # Assert that the number of positive singular values equals the dimension of the data
    assert positive_singular_values.size(0) == concatenated_data.size(1), (
        f"Expected {concatenated_data.size(1)} positive singular values, but got {positive_singular_values.size(0)}."
    )
    
    # Compute the corrected standard deviations
    corrected_stds = positive_singular_values / torch.sqrt(torch.tensor(N - 1, dtype=torch.float32))

    # Return the principal components (Vh.T), the mean, and the corrected standard deviations
    return Vh.T, mean.squeeze(0), corrected_stds