import numpy as np
from esbox.problems.functions.function_base import FunctionBase

__all__ = ['Rastrigin']


class Rastrigin(FunctionBase):
    """Rastrigin objective function.
    Has a global minimum at `f(0,0,...,0)=0` 
    with a search domain of `[-5.12, 5.12]`.
    """

    def __init__(self, dim=2, scale=False, scaled_x_low=-1, scaled_x_up=1, A=10):
        FunctionBase.__init__(self, dim, x_low=-5.12, x_up=5.12, scale=scale, scaled_x_low=-1, scaled_x_up=1)

        self.A = 10

        self.pf_x = [0] * dim
        self.pf = 0

    def evaluate(self, x):
        out = self.A * self.dim + (x**2.0 - self.A * np.cos(2.0 * np.pi * x)).sum(axis=1)
        return out
