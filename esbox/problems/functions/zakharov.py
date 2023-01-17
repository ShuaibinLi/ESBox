import numpy as np
from esbox.problems.functions.function_base import FunctionBase

__all__ = ['Zakharov']


class Zakharov(FunctionBase):
    """Zakharov objective function.
    Has a global minimum at `f(0,0,...,0)=0` 
    with a search domain of `[-10, 10]`.
    """

    def __init__(self, dim=2, scale=False, scaled_x_low=-1, scaled_x_up=1, a=20.0, b=1 / 5, c=2 * np.pi):
        FunctionBase.__init__(self, dim, x_low=-10, x_up=10, scale=scale, scaled_x_low=-1, scaled_x_up=1)

        self.pf_x = [0] * dim
        self.pf = 0

    def evaluate(self, x):
        a = (0.5 * np.arange(1, self.dim + 1) * x).sum(axis=1)
        out = np.square(x).sum(axis=1) + np.square(a) + np.power(a, 4)
        return out
