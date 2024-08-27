import torch

from src.time_flows import TimeFlow

class LinearTimeFlow(TimeFlow):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, t):
        return t
    
    def differential_forward(self, t):
        return torch.ones_like(t)