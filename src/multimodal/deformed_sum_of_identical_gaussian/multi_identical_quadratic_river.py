import torch

from src.multimodal.deformed_sum_of_identical_gaussian import DeformedSumOfIdenticalGaussian
from src.diffeomorphisms.river import RiverDiffeomorphism

class MultiIdenticalQuadraticRiver(DeformedSumOfIdenticalGaussian):
    def __init__(self, river_shear, river_offset, quadratic_diagonal, quadratic_offsets, quadratic_weights) -> None:
        super().__init__([RiverDiffeomorphism(river_shear, river_offset)], 
                         quadratic_diagonal, quadratic_offsets, quadratic_weights)