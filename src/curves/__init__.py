import torch.nn as nn

class Curve(nn.Module):
    """ Base class describing a curves as mappings c:[0,1]-> R^d """
    def __init__(self, d):
        super().__init__()

        self.d = d  # dimension
        
    def forward(self, t):
        """Evaluate the curve at parameter values t
        :param t: N tensor
        :return: N x d tensor
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def differential_forward(self, t):
        """Evaluate the speed of the curve at parameter values t
        :param t: N tensor
        :return: N x d tensor
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def double_differential_forward(self, t):
        """Evaluate the acceleration of the curve at parameter values t
        :param t: N tensor
        :return: N x d tensor
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )