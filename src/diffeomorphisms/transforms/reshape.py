import torch
from src.diffeomorphisms import utils
from src.diffeomorphisms import transforms

class SqueezeTransform(transforms.Transform):
    def __init__(self, factor=2):
        super(SqueezeTransform, self).__init__()
        if not utils.is_int(factor) or factor <= 1:
            raise ValueError('Factor must be an integer > 1.')
        self.factor = factor

    def get_output_shape(self, c, h, w):
        return (c * self.factor * self.factor, h // self.factor, w // self.factor)

    def forward(self, inputs, context=None, detach_logdet=False):
        if inputs.dim() != 4:
            raise ValueError('Expecting inputs with 4 dimensions')
        batch_size, c, h, w = inputs.size()
        device = inputs.device

        if h % self.factor != 0 or w % self.factor != 0:
            raise ValueError('Input image size not compatible with the factor.')

        inputs = inputs.view(batch_size, c, h // self.factor, self.factor, w // self.factor, self.factor)
        inputs = inputs.permute(0, 1, 3, 5, 2, 4).contiguous()
        inputs = inputs.view(batch_size, c * self.factor * self.factor, h // self.factor, w // self.factor)

        logabsdetjacobian = torch.zeros(batch_size, device=device)
        if detach_logdet:
            logabsdetjacobian = logabsdetjacobian.detach()
        return inputs, logabsdetjacobian

    def inverse(self, inputs, context=None, detach_logdet=False):
        if inputs.dim() != 4:
            raise ValueError('Expecting inputs with 4 dimensions')

        batch_size, c, h, w = inputs.size()
        device = inputs.device

        if c < 4 or c % 4 != 0:
            raise ValueError('Invalid number of channel dimensions.')

        inputs = inputs.view(batch_size, c // self.factor ** 2, self.factor, self.factor, h, w)
        inputs = inputs.permute(0, 1, 4, 2, 5, 3).contiguous()
        inputs = inputs.view(batch_size, c // self.factor ** 2, h * self.factor, w * self.factor)

        logabsdetjacobian = torch.zeros(batch_size, device=device)
        if detach_logdet:
            logabsdetjacobian = logabsdetjacobian.detach()
        return inputs, logabsdetjacobian


class ReshapeTransform(transforms.Transform):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs, context=None, detach_logdet=False):
        device = inputs.device
        if tuple(inputs.shape[1:]) != self.input_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.input_shape))
        logabsdetjacobian = torch.zeros(inputs.shape[0], device=device)
        if detach_logdet:
            logabsdetjacobian = logabsdetjacobian.detach()
        return inputs.reshape(-1, *self.output_shape), logabsdetjacobian

    def inverse(self, inputs, context=None, detach_logdet=False):
        device = inputs.device
        if tuple(inputs.shape[1:]) != self.output_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.output_shape))
        logabsdetjacobian = torch.zeros(inputs.shape[0], device=device)
        if detach_logdet:
            logabsdetjacobian = logabsdetjacobian.detach()
        return inputs.reshape(-1, *self.input_shape), logabsdetjacobian
