from src.curves import Curve

class InitialValueCurve(Curve):

    def __init__(self, start_point, start_velicity):
        super().__init__(start_point.shape[0])

        self.x = start_point # d tensor
        self.X = start_velicity # d tensor