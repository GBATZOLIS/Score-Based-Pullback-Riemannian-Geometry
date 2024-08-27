from src.geodesic_solvers import GeodesicSolver
from src.curves import Curve

class BoundaryValueGeodesicSolver(GeodesicSolver):

    def __init__(self, boundary_value_curve, norm) -> None:
        super().__init__(boundary_value_curve, norm)
        
        self.x = self.geodesic.x
        self.y = self.geodesic.y

    def solve(self) -> Curve:
        raise NotImplementedError(
            "Subclasses should implement this"
        )