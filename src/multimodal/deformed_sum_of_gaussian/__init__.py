from src.multimodal import Multimodal
from src.strongly_convex.quadratic_diagonal import QuadraticDiagonal

class DeformedSumOfGaussian(Multimodal):
    def __init__(self, diffeomorphism, diagonals, weights) -> None:
        super().__init__(diffeomorphism, [QuadraticDiagonal(diagonal) for diagonal in diagonals], weights) 