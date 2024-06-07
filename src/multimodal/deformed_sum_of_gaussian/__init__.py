from src.multimodal import Multimodal
from src.strongly_convex.quadratic_diagonal import QuadraticDiagonal

class DeformedSumOfGaussian(Multimodal):
    def __init__(self, diffeomorphism, diagonals, offsets, weights) -> None:
        """
        
        :param diffeomorphism
        :param diagonals: m x d
        :param offsets: m x d or [None, ..., None]
        :param weights: m
        """
        super().__init__(diffeomorphism, [QuadraticDiagonal(diagonals[i], offset=offsets[i]) for i in range(len(diagonals))], weights) 