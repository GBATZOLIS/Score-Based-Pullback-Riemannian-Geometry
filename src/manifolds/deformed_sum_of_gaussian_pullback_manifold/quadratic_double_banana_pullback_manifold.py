import torch

from src.manifolds.deformed_sum_of_gaussian_pullback_manifold import DeformedSumOfGaussianPullbackManifold
from src.multimodal.deformed_sum_of_gaussian.quadratic_double_banana import QuadraticDoubleBanana

class QuadraticDoubleBananaPullbackManifold(DeformedSumOfGaussianPullbackManifold):

    def __init__(self, shear=1/9, offset=0., a1=1/4, a2=4):
        super().__init__(QuadraticDoubleBanana(shear, offset, torch.tensor([a1, a2]), torch.tensor([a2/2, a1])))