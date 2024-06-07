import torch

from src.multimodal.deformed_sum_of_gaussian import DeformedSumOfGaussian
from src.diffeomorphisms.banana import BananaDiffeomorphism

class MultiQuadraticBanana(DeformedSumOfGaussian):
    def __init__(self, banana_shear, banana_offset, quadratic_diagonals, quadratic_offsets, quadratic_weights) -> None:
        super().__init__([BananaDiffeomorphism(banana_shear, banana_offset)], 
                         quadratic_diagonals, quadratic_offsets, quadratic_weights)