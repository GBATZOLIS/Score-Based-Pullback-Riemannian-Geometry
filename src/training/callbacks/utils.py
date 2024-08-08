import torch
import numpy as np
from torch.autograd.functional import jacobian

def check_orthogonality(phi, val_loader, device, num_samples=100):
    orthogonality_deviation = []

    for i, data in enumerate(val_loader):
        if i >= num_samples:
            break
        
        # Use the validation sample
        if isinstance(data, list):
            x = data[0]
        else:
            x = data

        x = x.to(device)
        
        # Compute the Jacobian matrix
        J = jacobian(lambda x: phi(x), x)

        # Reshape the Jacobian to a 2D matrix for each sample in the batch
        batch_size = J.shape[0]
        J = J.view(batch_size, -1, J.shape[-1])

        # Compute J^T J for all samples in the batch using batched matrix multiplication
        J_T_J = torch.matmul(J.transpose(1, 2), J)

        # Compute the Frobenius norm of (J^T J - I) for each sample in the batch
        identity_matrix = torch.eye(J_T_J.shape[-1], device=device).unsqueeze(0)
        deviations = torch.norm(J_T_J - identity_matrix, p='fro', dim=(1, 2)).cpu().numpy()
        
        orthogonality_deviation.extend(deviations)

    avg_deviation = np.mean(orthogonality_deviation)
    return avg_deviation