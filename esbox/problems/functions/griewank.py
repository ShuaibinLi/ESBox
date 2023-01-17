import numpy as np
from esbox.problems.functions.function_base import FunctionBase

__all__ = ['Griewank']


class Griewank(FunctionBase):
    """Griewank objective function.
    Has a global minimum at `f(0,0,...,0)=0` 
    with a search domain of `[-600, 600]`.
    """

    def __init__(self, dim=2, scale=False, scaled_x_low=-1, scaled_x_up=1):
        FunctionBase.__init__(self, dim, x_low=-600, x_up=600, scale=scale, scaled_x_low=-1, scaled_x_up=1)

        self.pf_x = [0] * dim
        self.pf = 0

    def evaluate(self, x):
        part2 = 1 / 4000 * np.power(x, 2).sum(axis=1)
        part3 = -np.prod(np.cos(x / np.sqrt(np.arange(1, self.dim + 1))), axis=1)
        out = 1 + part2 + part3
        return out
