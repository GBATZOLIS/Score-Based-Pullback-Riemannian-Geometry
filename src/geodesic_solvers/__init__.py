class GeodesicSolver:
    """ Base class describing a geodesic solver """

    def __init__(self, curve, norm) -> None:

        self.geodesic = curve # Curve
        self.d = self.geodesic.d
        self.norm = norm
        
    def solve(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )