import torch

from src.manifolds.deformed_sum_of_gaussian_pullback_manifold import DeformedSumOfGaussianPullbackManifold
from src.multimodal.deformed_sum_of_gaussian.double_quadratic_banana import DoubleQuadraticBanana

class DoubleQuadraticBananaPullbackManifold(DeformedSumOfGaussianPullbackManifold):

    def __init__(self, shear=1/9, offset=0., a1=1/4, a2=4, L1=100, tol1=1e-2, max_iter1=20000, step_size1=1/8, L2=200, tol2=1e-4, max_iter2=100):
        super().__init__(DoubleQuadraticBanana(shear, offset, torch.tensor([a1, a2]), torch.tensor([a2/2, a1])), 
                         L1=L1, tol1=tol1, max_iter1=max_iter1, step_size1=step_size1,
                         L2=L2, tol2=tol2, max_iter2=max_iter2)