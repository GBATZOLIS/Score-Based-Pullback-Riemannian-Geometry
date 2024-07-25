import torch

from src.multimodal.deformed_sum_of_identical_gaussian import DeformedSumOfIdenticalGaussian
from src.diffeomorphisms.banana import BananaDiffeomorphism

class MultiIdenticalQuadraticBanana(DeformedSumOfIdenticalGaussian):
    def __init__(self, banana_shear, banana_offset, quadratic_diagonal, quadratic_offsets, quadratic_weights) -> None:
        super().__init__([BananaDiffeomorphism(banana_shear, banana_offset)], 
                         quadratic_diagonal, quadratic_offsets, quadratic_weights)