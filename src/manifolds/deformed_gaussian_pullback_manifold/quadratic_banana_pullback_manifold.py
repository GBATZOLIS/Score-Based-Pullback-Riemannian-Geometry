import torch

from src.manifolds.deformed_gaussian_pullback_manifold import DeformedGaussianPullbackManifold
from unimodal.deformed_gaussian.quadratic_banana import QuadraticBanana

class QuadraticBananaPullbackManifold(DeformedGaussianPullbackManifold):

    def __init__(self, shear=1/9, offset=0., a1=1/4, a2=4):
        super().__init__(QuadraticBanana(shear, offset, torch.tensor([a1, a2])))