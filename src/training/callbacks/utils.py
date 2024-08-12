import torch
import numpy as np
from torch.autograd.functional import jacobian
from tqdm import tqdm 

def deviation_from_volume_preservation(phi, val_loader, device, num_samples=256):
    volume_deviations = []
    sample_count = 0

    for i, data in enumerate(val_loader):
        if sample_count >= num_samples:
            break
        
        # Use the validation sample
        if isinstance(data, list) or isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        batch_size = x.size(0)
        if sample_count + batch_size > num_samples:
            x = x[:num_samples - sample_count]

        x = x.to(device)

        # Ensure that phi._transform returns the required output
        with torch.no_grad():
            _, logabsdetjac = phi._transform(x, context=None)
    
        volume_deviations.extend(torch.abs(logabsdetjac).cpu().numpy())
        sample_count += batch_size

    avg_volume_deviation = np.mean(volume_deviations)
    return avg_volume_deviation

def check_orthogonality(phi, val_loader, device, num_samples=256):
    orthogonality_deviation = []
    sample_count = 0

    for i, data in tqdm(enumerate(val_loader)):
        if sample_count >= num_samples:
            break
        
        if isinstance(data, list) or isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        batch_size = x.size(0)
        if sample_count + batch_size > num_samples:
            x = x[:num_samples - sample_count]

        x = x.to(device)
        
        J = jacobian(lambda x: phi(x), x)
        batch_size = J.shape[0]
        num_elements = np.prod(x.shape[1:])  # Product of all dimensions except the batch size
        J = J.view(batch_size, -1, num_elements)
        J_T_J = torch.matmul(J.transpose(1, 2), J)
        identity_matrix = torch.eye(J_T_J.shape[-1], device=device).unsqueeze(0)
        deviations = torch.norm(J_T_J - identity_matrix, p='fro', dim=(1, 2)).cpu().numpy()
        
        orthogonality_deviation.extend(deviations)
        sample_count += batch_size

    avg_deviation = np.mean(orthogonality_deviation)
    return avg_deviation