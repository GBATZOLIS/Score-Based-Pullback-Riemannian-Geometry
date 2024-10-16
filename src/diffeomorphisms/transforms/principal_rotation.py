import torch
from src.diffeomorphisms import transforms

class PrincipalRotationTransform(transforms.Transform):
    def __init__(self, U=None, mean=None):
        super().__init__()
        # If U is provided, convert it to a buffer with requires_grad=False to freeze it
        if U is not None:
            self.register_buffer('U', U)  # Register U as a buffer to keep it unlearnable
        else:
            self.U = None
        
        # Register mean as a buffer if provided, otherwise initialize to None
        if mean is not None:
            self.register_buffer('mean', mean)
        else:
            self.mean = None

    def forward(self, inputs, context=None, detach_logdet=False):
        #print(inputs.size())
        # Flatten the input tensor if necessary
        original_shape = inputs.shape
        B = original_shape[0]
        if len(original_shape) > 2:
            inputs = inputs.view(B, -1)  # Flatten to (B, D)
        
        # Center the data by subtracting the mean if available
        if self.mean is not None:
            inputs = inputs - self.mean
        
        # Apply the linear transformation U^T * x
        if self.U is not None:
            outputs = torch.matmul(inputs, self.U.T)
        else:
            outputs = inputs

        # Reshape back to original shape if needed
        if len(original_shape) > 2:
            outputs = outputs.view(B, *original_shape[1:])
        
        # Since U is orthogonal, logabsdet is 0
        logabsdet = torch.zeros(inputs.size(0), device=inputs.device)

        if detach_logdet:
            logabsdet = logabsdet.detach()
        
        #print(outputs.size())
        return outputs, logabsdet

    def inverse(self, inputs, context=None, detach_logdet=False):
        # Flatten the input tensor if necessary
        original_shape = inputs.shape
        B = original_shape[0]
        if len(original_shape) > 2:
            inputs = inputs.view(B, -1)  # Flatten to (B, D)
        
        # Apply the inverse linear transformation U * y
        if self.U is not None:
            outputs = torch.matmul(inputs, self.U)
        else:
            outputs = inputs
        
        # Re-center the data by adding the mean back if available
        if self.mean is not None:
            outputs = outputs + self.mean
        
        # Reshape back to original shape if needed
        if len(original_shape) > 2:
            outputs = outputs.view(B, *original_shape[1:])
        
        # Since U is orthogonal, logabsdet is 0
        logabsdet = torch.zeros(inputs.size(0), device=inputs.device)
        
        return outputs, logabsdet
