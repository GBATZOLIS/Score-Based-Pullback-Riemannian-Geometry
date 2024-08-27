from src.curves import Curve

class BoundaryValueCurve(Curve):

    def __init__(self, start_point, end_point):
        super().__init__(start_point.shape[0])

        self.x = start_point # d tensor
        self.y = end_point # d tensor