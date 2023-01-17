import numpy as np
from esbox.problems.functions.function_base import FunctionBase

__all__ = ['Rosenbrock']


class Rosenbrock(FunctionBase):
    """Rosenbrock objective function.
    Has a global minimum at `f(1,1,...,1)=0` 
    with a search domain of `[-2.048, 2.048]` (or `[-inf, inf]`).
    """

    def __init__(self, dim=2, scale=False, scaled_x_low=-1, scaled_x_up=1):
        FunctionBase.__init__(self, dim, x_low=-2.048, x_up=2.048, scale=scale, scaled_x_low=-1, scaled_x_up=1)

        self.pf_x = [1] * dim
        self.pf = 0

    def evaluate(self, x):
        out = np.sum(100 * (x.T[1:] - x.T[:-1]**2.0)**2 + (1 - x.T[:-1])**2.0, axis=0)
        return out
