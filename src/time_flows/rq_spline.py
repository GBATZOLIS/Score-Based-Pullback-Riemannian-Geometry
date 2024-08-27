import torch
import torch.nn as nn

from nflows import transforms
from nflows.nn import nets
from nflows import utils

from src.time_flows import TimeFlow

class RationalQuadraticSplineFlow(TimeFlow): 
    def __init__(self, num_bins):
        super().__init__()
        self.num_bins = num_bins

        class SingleLayerNet(nn.Module):
            def __init__(self, in_features, out_features):
                super(SingleLayerNet, self).__init__()
                self.linear = nn.Linear(in_features, out_features)

            def forward(self, x, context=None):
                return self.linear(x)
        
        self._transform = transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=utils.create_alternating_binary_mask(
                    features=1,
                    even=True
                ),
                transform_net_create_fn=lambda in_features, out_features:
                    SingleLayerNet(in_features=in_features, out_features=out_features),
                num_bins=num_bins,
                tails = 'linear',
                tail_bound= 0.5,
                apply_unconditional_transform=False,
            )
        
    def forward(self, t):
        output, _ = self._transform(t[:,None] - 0.5, context=None)
        return output[:,0] + 0.5
    
    def differential_forward(self, t):
        def forwards(t):
            return torch.sum(self.forward(t))
        output = torch.autograd.functional.jacobian(forwards, t)
        return output