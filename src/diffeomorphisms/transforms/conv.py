from src.diffeomorphisms import utils
from src.diffeomorphisms import transforms


class OneByOneConvolution(transforms.LULinear):
    def __init__(self, num_channels, using_cache=False, identity_init=True):
        super().__init__(num_channels, using_cache, identity_init)
        self.permutation = transforms.RandomPermutation(num_channels, dim=1)

    def _lu_forward_inverse(self, inputs, inverse=False, detach_logdet=False):
        b, c, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).reshape(b * h * w, c)

        if inverse:
            outputs, logabsdet = super().inverse(inputs)
        else:
            outputs, logabsdet = super().forward(inputs)

        outputs = outputs.reshape(b, h, w, c).permute(0, 3, 1, 2)
        logabsdet = logabsdet.reshape(b, h, w)
        if detach_logdet:
            logabsdet = logabsdet.detach()

        return outputs, utils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None, detach_logdet=False):
        if inputs.dim() != 4:
            raise ValueError('Inputs must be a 4D tensor.')

        inputs, _ = self.permutation(inputs)
        return self._lu_forward_inverse(inputs, inverse=False, detach_logdet=detach_logdet)

    def inverse(self, inputs, context=None, detach_logdet=False):
        if inputs.dim() != 4:
            raise ValueError('Inputs must be a 4D tensor.')

        outputs, logabsdet = self._lu_forward_inverse(inputs, inverse=True, detach_logdet=detach_logdet)
        outputs, _ = self.permutation.inverse(outputs)

        return outputs, logabsdet
