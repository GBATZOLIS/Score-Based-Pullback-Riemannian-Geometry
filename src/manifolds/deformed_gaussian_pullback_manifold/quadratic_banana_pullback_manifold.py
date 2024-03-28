from src.manifolds.deformed_gaussian_pullback_manifold import DeformedGaussianPullbackManifold
from src.unimodal.deformed_gaussian.quadratic_banana import QuadraticBanana

class QuadraticBananaPullbackManifold(DeformedGaussianPullbackManifold):

    def __init__(self, shear, offset, diagonal):
        super().__init__(QuadraticBanana(shear, offset, diagonal))