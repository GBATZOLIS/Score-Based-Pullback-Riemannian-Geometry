from src.curve_solvers.initial_value import InitialValueCurveSolver
from src.curves.initial_value.harmonic import HarmonicInitialValueCurve

class HarmonicInitialValueCurveSolver(InitialValueCurveSolver):
    """ Class describing a boundary value curve solver for a harmonic curve """

    def __init__(self, start_point, start_velocity, K, loss_function, num_time_points=200, num_epochs=100, lr=0.1, weight_decay=0.01) -> None:
        super().__init__(HarmonicInitialValueCurve(start_point, start_velocity, K), 
                        loss_function, num_time_points=num_time_points, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay)