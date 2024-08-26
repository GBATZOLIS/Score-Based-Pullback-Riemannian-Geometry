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

def get_principal_components(train_loader):
    data = []
    
    # Collect all data into a single tensor
    for batch in train_loader:
        x = batch[0] if isinstance(batch, list) else batch
        x_flat = x.view(x.size(0), -1)  # Flatten the images
        data.append(x_flat)
    
    data = torch.cat(data, dim=0)  # Concatenate along the batch dimension
    
    # Compute the mean of the data
    mean = torch.mean(data, dim=0, keepdim=True)
    data_centered = data - mean  # Center the data by subtracting the mean
    
    # Compute SVD on the centered data
    U, S, Vh = torch.linalg.svd(data_centered, full_matrices=False)
    
    # Return the principal components and the mean
    return Vh.T, mean.squeeze(0)  # Return U and the mean