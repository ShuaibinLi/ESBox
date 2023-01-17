import numpy as np
from esbox.problems.functions.function_base import FunctionBase

__all__ = ['StyblinskiTang']


class StyblinskiTang(FunctionBase):
    """StyblinskiTang objective function.
    Has a global minimum at `f(-2.903534,...,-2.903534) = -39.16617*dim` 
    with a search domain of `[-5, 5]`, 
    """

    def __init__(self, dim=2, scale=False, scaled_x_low=-1, scaled_x_up=1):
        FunctionBase.__init__(self, dim, x_low=-5, x_up=5, scale=scale, scaled_x_low=-1, scaled_x_up=1)

        self.pf_x = [-2.903534] * dim
        self.pf = -39.16617 * dim

    def evaluate(self, x):
        out = 0.5 * np.sum(np.power(x, 4) - 16 * np.power(x, 2) + 5 * x, axis=1)
        return out


if __name__ == "__main__":
    tang = StyblinskiTang(dim=11, scale=False)

    x = [-0.58] * 11
    # x = [[-0.58]*11, [-0.58]*11]
    print(tang(x))
