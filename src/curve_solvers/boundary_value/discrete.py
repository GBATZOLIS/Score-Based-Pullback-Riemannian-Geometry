from src.curve_solvers.boundary_value import BoundaryValueCurveSolver
from src.curves.boundary_value.discrete import DiscreteBoundaryValueCurve

class DiscreteBoundaryValueCurveSolver(BoundaryValueCurveSolver):
    """ Class describing a boundary value curve solver for a harmonic curve """

    def __init__(self, start_point, end_point, K, loss_function, num_time_points=None, num_epochs=1000, lr=0.001, weight_decay=0.) -> None:
        if num_time_points is None:
            num_time_points = K+1
        assert num_time_points > K
        super().__init__(DiscreteBoundaryValueCurve(start_point, end_point, K), 
                         loss_function, num_time_points=num_time_points, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay)