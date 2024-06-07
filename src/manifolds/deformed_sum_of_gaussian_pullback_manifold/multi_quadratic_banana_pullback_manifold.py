import torch

from src.manifolds.deformed_sum_of_gaussian_pullback_manifold import DeformedSumOfGaussianPullbackManifold
from src.multimodal.deformed_sum_of_gaussian.multi_quadratic_banana import MultiQuadraticBanana

class MultiQuadraticBananaPullbackManifold(DeformedSumOfGaussianPullbackManifold):

    def __init__(self, banana_shear=1/9, banana_offset=0., quadratic_diagonals=torch.tensor([[1/4, 4.], [4., 1/4], [1/4, 4.]]), quadratic_offsets=torch.tensor([[-5., -5.], [0.,0.], [5., 5.]]), quadratic_weights=torch.ones(3)):
        super().__init__(MultiQuadraticBanana(banana_shear, banana_offset, 
                                              quadratic_diagonals, quadratic_offsets, quadratic_weights))