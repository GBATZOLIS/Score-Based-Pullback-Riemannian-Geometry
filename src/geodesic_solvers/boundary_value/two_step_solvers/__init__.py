from src.curves import Curve
from src.curves.boundary_value.time_changed import TimeChangedBoundaryValueCurve
from src.geodesic_solvers.boundary_value import BoundaryValueGeodesicSolver

class TwoStepBoundaryValueGeodesicSolver(BoundaryValueGeodesicSolver):
    """ Base class describing a two step boundary value geodesic solver """

    def __init__(self, curve_solver, time_flow_solver, norm) -> None:
        super().__init__(TimeChangedBoundaryValueCurve(curve_solver.curve, time_flow_solver.time_flow), norm)
        self.curve_solver = curve_solver
        self.time_flow_solver = time_flow_solver

    def solve(self) -> Curve:
        self.curve_solver.solve()
        self.time_flow_solver.solve()
        # return self.geodesic
    
    def curve(self):
        return self.curve_solver.curve
    
    def time_flow(self):
        return self.time_flow_solver.time_flow
    
    # def geodesic(self):
    #     return TimeChangedCurve(self.curve(), self.time_flow())
    