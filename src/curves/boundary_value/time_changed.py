from src.curves.boundary_value import BoundaryValueCurve

class TimeChangedBoundaryValueCurve(BoundaryValueCurve):
    """ Base class describing a curves as mappings c:[0,1]-> R^d """
    def __init__(self, base_curve, time_flow):
        super().__init__(base_curve.x, base_curve.y)
        self.base_curve = base_curve
        self.time_flow = time_flow
        
    def forward(self, t):
        """Evaluate the curve at parameter values t
        :param t: N tensor
        :return: N x d tensor
        """
        return self.base_curve(self.time_flow(t))
    
    def differential_forward(self, t):
        """Evaluate the speed of the curve at parameter values t
        :param t: N tensor
        :return: N x d tensor
        """
        return self.base_curve.differential_forward(self.time_flow(t)) * self.time_flow.differential_forward(t)[:,None]
    
    def double_differential_forward(self, t):
        """Evaluate the acceleration of the curve at parameter values t
        :param t: N tensor
        :return: N x d tensor
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )