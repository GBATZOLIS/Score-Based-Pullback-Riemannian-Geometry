import torch
import torch.optim as optim

class CurveSolver:
    """ Base class describing a curve solver """

    def __init__(self, curve) -> None:

        self.curve = curve # Curve
        self.d = self.curve.d  # dimension

    def solve(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )