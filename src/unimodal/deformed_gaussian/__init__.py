from src.unimodal import Unimodal
from src.strongly_convex.quadratic_diagonal import QuadraticDiagonal

class DeformedGaussian(Unimodal):
    def __init__(self, diffeomorphism, diagonal) -> None:
        super().__init__(diffeomorphism, QuadraticDiagonal(diagonal)) # TODO construct strongly convex function in here

