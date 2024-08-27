import torch.nn as nn

class TimeFlow(nn.Module):
    """ Base class describing a diffeomorphism phi:[0,1]-> [0,1] with phi(0)=0 and phi(1)=1 """

    def forward(self, t):
        """ Evaluate the time at parameter values t
        :param t: N tensor
        :return: N tensor
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def differential_forward(self, t):
        """ Evaluate the time derivative at parameter values t
        :param t: N tensor
        :return: N  tensor
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )