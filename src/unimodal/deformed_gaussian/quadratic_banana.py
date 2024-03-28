from src.unimodal.deformed_gaussian import DeformedGaussian
from src.diffeomorphisms.banana import BananaDiffeomorphism

class QuadraticBanana(DeformedGaussian):
    def __init__(self, shear, offset, diagonal) -> None:
        super().__init__(BananaDiffeomorphism(shear, offset), diagonal)

