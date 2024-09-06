from src.diffeomorphisms.river import RiverDiffeomorphism
from src.unimodal.deformed_gaussian import DeformedGaussian

class QuadraticRiver(DeformedGaussian):
    def __init__(self, shear, offset, diagonal) -> None:
        super().__init__(RiverDiffeomorphism(shear, offset), diagonal)
