import torch

from src.manifolds.deformed_sum_of_gaussian_pullback_manifold import DeformedSumOfGaussianPullbackManifold
from src.multimodal.deformed_sum_of_gaussian.double_quadratic_banana import DoubleQuadraticBanana

class DoubleQuadraticBananaPullbackManifold(DeformedSumOfGaussianPullbackManifold):

    def __init__(self, shear=1/9, offset=0., a1=1/4, a2=4, L=100, tol=1e-2, max_iter=20000, step_size=1/8):
        super().__init__(DoubleQuadraticBanana(shear, offset, torch.tensor([a1, a2]), torch.tensor([a2/2, a1])), L=L, tol=tol, max_iter=max_iter, step_size=step_size)