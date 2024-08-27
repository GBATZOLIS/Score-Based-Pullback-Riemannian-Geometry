class TimeFlowSolver: # TODO add termination conditions
    """ Base class describing a curve solver """

    def __init__(self, time_flow, loss_function, num_time_points=100, num_epochs=100, lr=0.1, weight_decay=0.) -> None:

        self.time_flow = time_flow # TimeFlow

        self.num_time_points = num_time_points
        self.loss_function = loss_function
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay

    def solve(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )