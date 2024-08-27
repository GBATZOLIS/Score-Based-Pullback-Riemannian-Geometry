from src.geodesic_solvers import GeodesicSolver
from src.curves import Curve

class InitialValueGeodesicSolver(GeodesicSolver):

    def __init__(self, initial_value_curve, norm) -> None:
        super().__init__(initial_value_curve, norm)

        self.x = self.geodesic.x
        self.X = self.geodesic.X

    def solve(self, x, X) -> Curve:
        raise NotImplementedError(
            "Subclasses should implement this"
        )