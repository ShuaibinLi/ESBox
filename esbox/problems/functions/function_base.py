from abc import abstractmethod
import numpy as np

__all__ = ['FunctionBase']


class FunctionBase(object):
    """`FunctionBase` is the base class of math functions.
    This base class mainly do the following things:
        1. Defines common APIs that function problems should implement.
    """

    def __init__(self, dim=2, x_low=None, x_up=None, scale=True, scaled_x_low=-1, scaled_x_up=1, *args, **kwargs):
        """
        Parameters:
            dim : int
                Dimension of Functions
            x_low : float, int
                Lower bounds for the variables.
            x_up : float, int
                Upper bounds for the variable.
            pf_x: float, int
                Optimal solutions
            pf: float, int
                Optimal value
        """

        self.dim = dim
        self.x_low = x_low
        self.x_up = x_up

        self.pf_x = [0] * dim
        self.pf = 0

        # scaled
        self.scale = scale
        self.scaled_x_low = scaled_x_low
        self.scaled_x_up = scaled_x_up

    def __call__(self, x, *args, **kwargs):
        """
        Args:
            x : int, float, list, or
                expected type: numpy.ndarray, set of inputs of shape `(n, dim)`
        Returns:
            numpy.ndarray
                computed cost of size `(n, )`
        Raises:
            ValueError
            When the input is out of bounds with respect to the function domain
        """
        if x is not None:
            if not isinstance(x, np.ndarray):
                x = np.ones(self.dim) * x
            x = x.astype(float)
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)
        # if not np.logical_and(x >= self.x_low, x <= self.x_up).all():
        #     raise ValueError("Input for function must be within [{}, {}].".format(self.x_low, self.x_up))

        scaled_x = self.rescale(x) if self.scale else x

        out = self.evaluate(scaled_x, *args, **kwargs)
        return out

    @abstractmethod
    def evaluate(self, x, *args, **kwargs):
        """Evaluate symbolic expression
        """
        raise NotImplementedError

    def rescale(self, x):
        """
        Implements linear scaling from [scaled_x_low, scaled_x_up] to [x_low, x_up]
        """
        if self.scale:
            scaled_x_low = self.scaled_x_low
            scaled_x_up = self.scaled_x_up
            return self.x_low + ((x - scaled_x_low) * (self.x_up - self.x_low)) / (scaled_x_up - scaled_x_low)
        else:
            return x
