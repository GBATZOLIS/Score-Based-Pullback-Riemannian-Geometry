from src.multimodal import Multimodal
from src.strongly_convex.quadratic_diagonal import QuadraticDiagonal

class DeformedSumOfIdenticalGaussian(Multimodal):
    def __init__(self, diffeomorphism, diagonal, offsets, weights) -> None:
        """
        
        :param diffeomorphism
        :param diagonal: d
        :param offsets: m x d or [None, ..., None]
        :param weights: m
        """
        super().__init__(diffeomorphism, [QuadraticDiagonal(diagonal, offset=offsets[i]) for i in range(len(offsets))], weights) 