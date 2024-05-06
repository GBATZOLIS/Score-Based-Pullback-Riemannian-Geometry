import torch

from src.multimodal.deformed_sum_of_gaussian import DeformedSumOfGaussian
from src.diffeomorphisms.banana import BananaDiffeomorphism

class QuadraticDoubleBanana(DeformedSumOfGaussian):
    def __init__(self, shear, offset, diagonal_1, diagonal_2) -> None:
        super().__init__([BananaDiffeomorphism(shear, offset)], [diagonal_1, diagonal_2], torch.tensor([1.,1.]))