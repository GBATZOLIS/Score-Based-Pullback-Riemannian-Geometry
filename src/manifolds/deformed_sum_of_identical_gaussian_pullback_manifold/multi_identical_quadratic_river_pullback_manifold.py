import torch

from src.manifolds.deformed_sum_of_identical_gaussian_pullback_manifold import DeformedSumOfIdenticalGaussianPullbackManifold
from src.multimodal.deformed_sum_of_identical_gaussian.multi_identical_quadratic_river import MultiIdenticalQuadraticRiver

class MultiIdenticalQuadraticRiverPullbackManifold(DeformedSumOfIdenticalGaussianPullbackManifold):

    def __init__(self, river_shear=1/2, river_offset=0., quadratic_diagonal=torch.tensor([1/4, 4.]), quadratic_offsets=torch.tensor([[0., -6.], [0.,0.], [0., 6.]]), quadratic_weights=torch.ones(3)):
        super().__init__(MultiIdenticalQuadraticRiver(river_shear, river_offset, 
                                              quadratic_diagonal, quadratic_offsets, quadratic_weights))