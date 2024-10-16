import torch
from src.manifolds.deformed_gaussian_pullback_manifold.quadratic_banana_pullback_manifold import QuadraticBananaPullbackManifold
from src.unimodal import Unimodal
from src.unimodal.deformed_gaussian.quadratic_river import QuadraticRiver
from src.manifolds.deformed_gaussian_pullback_manifold import DeformedGaussianPullbackManifold

def get_ground_truth_pullback_manifold(config):
    if config.dataset == 'single_banana':
        pullback_manifold = QuadraticBananaPullbackManifold()
    elif config.dataset == 'squeezed_single_banana':
        pullback_manifold = QuadraticBananaPullbackManifold(shear=1/9, offset=0., a1=1/81, a2=4)
    elif config.dataset == 'river':
        shear, offset, a1, a2 = 2, 0, 1/25, 3
        river_unimodal_distribution = QuadraticRiver(shear, offset, torch.tensor([a1, a2]))
        pullback_manifold = DeformedGaussianPullbackManifold(river_unimodal_distribution)

    return pullback_manifold

def get_learned_pullback_manifold(phi, psi):
    distribution = Unimodal(diffeomorphism=phi, strongly_convex=psi)
    manifold = DeformedGaussianPullbackManifold(distribution)
    return manifold
