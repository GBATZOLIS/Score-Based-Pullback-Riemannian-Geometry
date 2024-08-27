from src.curve_solvers.boundary_value import BoundaryValueCurveSolver
from src.curves.boundary_value.harmonic import HarmonicBoundaryValueCurve

class HarmonicBoundaryValueCurveSolver(BoundaryValueCurveSolver):
    """ Class describing a boundary value curve solver for a harmonic curve """

    def __init__(self, start_point, end_point, num_sines, loss_function, num_time_points=200, num_epochs=100, lr=0.1, weight_decay=0.01) -> None:
        super().__init__(HarmonicBoundaryValueCurve(start_point, end_point, num_sines), 
                        loss_function, num_time_points=num_time_points, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay)