import numpy as np
from esbox.problems.functions.function_base import FunctionBase

__all__ = ['Ackley']


class Ackley(FunctionBase):
    """Ackley objective function.
    Has a global minimum at `f(0,0,...,0)=0` 
    with a search domain of `[-32.768, 32.768]`.
    """

    def __init__(self, dim=2, scale=False, scaled_x_low=-1, scaled_x_up=1, a=20.0, b=1 / 5, c=2 * np.pi):
        FunctionBase.__init__(self, dim, x_low=-32.768, x_up=32.768, scale=scale, scaled_x_low=-1, scaled_x_up=1)

        self.a = a
        self.b = b
        self.c = c

        self.pf_x = [0] * dim
        self.pf = 0

    def evaluate(self, x):
        """
        Args:
            x : int, float, list, or numpy.ndarray(n, dim)
        Returns:
            numpy.ndarray
                computed cost of size `(n, )`
        """
        part1 = -self.a * np.exp(-self.b * np.sqrt((1 / self.dim) * (x**2).sum(axis=1)))
        part2 = -np.exp((1 / float(self.dim)) * np.cos(self.c * x).sum(axis=1))
        out = part1 + part2 + self.a + np.exp(1)
        return out
