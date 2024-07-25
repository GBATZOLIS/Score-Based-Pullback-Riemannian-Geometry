import torch

from src.manifolds.deformed_sum_of_gaussian_pullback_manifold.deformed_sum_of_identical_gaussian_pullback_manifold import DeformedSumOfIdenticalGaussianPullbackManifold
from src.multimodal.deformed_sum_of_identical_gaussian.multi_identical_quadratic_banana import MultiIdenticalQuadraticBanana

class MultiIdenticalQuadraticBananaPullbackManifold(DeformedSumOfIdenticalGaussianPullbackManifold):

    def __init__(self, banana_shear=1/9, banana_offset=0., quadratic_diagonal=torch.tensor([1/4, 4.]), quadratic_offsets=torch.tensor([[0., -6.], [0.,0.], [0., 6.]]), quadratic_weights=torch.ones(3)):
        super().__init__(MultiIdenticalQuadraticBanana(banana_shear, banana_offset, 
                                              quadratic_diagonal, quadratic_offsets, quadratic_weights))