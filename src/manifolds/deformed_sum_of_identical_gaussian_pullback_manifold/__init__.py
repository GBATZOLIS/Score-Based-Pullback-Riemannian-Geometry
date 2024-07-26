from src.manifolds import Manifold
from src.manifolds.deformed_sum_of_gaussian_pullback_manifold import DeformedSumOfGaussianPullbackManifold

import torch

class DeformedSumOfIdenticalGaussianPullbackManifold(DeformedSumOfGaussianPullbackManifold):
    """ Base class describing a R^d under a sum of identical Gaussian-pullback Riemannian geometry generated by a DeformedSumofIdenticalGaussian multimodal distribution """

    def __init__(self, deformed_sum_of_gaussian): 
        super().__init__(deformed_sum_of_gaussian, L1=1, L2=1)
