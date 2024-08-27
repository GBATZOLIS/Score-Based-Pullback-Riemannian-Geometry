from src.curve_solvers.boundary_value.harmonic import HarmonicBoundaryValueCurveSolver
from src.geodesic_solvers.boundary_value.two_step_solvers import TwoStepBoundaryValueGeodesicSolver
from src.time_flow_solvers.rq_spline import RationalQuadraticSplineFlowSolver

class HarmonicRationalQuadraticSplineFlowBoundaryValueGeodesicSolver(TwoStepBoundaryValueGeodesicSolver):
    """ Base class describing a two step boundary value geodesic solver """

    def __init__(self, x, y, norm, num_sines=1, num_bins=None) -> None:
        def curve_loss_function(x, X):
            return self.norm(x, X[:,None])
    
        def time_flow_loss_fcn(s):
            return self.norm(self.curve().forward(s)[0:-1], (self.curve().forward(s)[1:] - self.curve().forward(s)[0:-1])[:,None])**2
        
        self.num_sines = num_sines
        if num_bins is None:
            self.num_bins = 2 * self.num_sines + 1
        else:
            self.num_bins = num_bins

        super().__init__(HarmonicBoundaryValueCurveSolver(x, y, self.num_sines, curve_loss_function), RationalQuadraticSplineFlowSolver(self.num_bins, time_flow_loss_fcn), norm)
        