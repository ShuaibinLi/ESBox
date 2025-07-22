import numpy as np

EPS = 1e-8

__all__ = ['BoundedSampler']


class BoundedSampler(object):
    """Base Samplers with boundary-bounded limits.
    Args:
        weights_size (int): number of sampling parameters
        bounds (list): lower and upper domain boundaries for each parameter, e.g. [-5, 5]
        n_max_resampling (int): A maximum number of resampling parameters (default: 100).
                                If all sampled parameters are infeasible, the last sampled one
                                will be clipped with lower and upper bounds.
    """

    def __init__(self, weights_size, bounds=None, n_max_resampling=100):

        self.weights_size = weights_size

        # bounds contains low and high of each parameter.
        self._set_bounds(bounds)
        self._n_max_resampling = n_max_resampling

    def sample(self, *args, **kwargs):
        """Sample `sample_batch` parameter"""
        raise NotImplementedError

    def _is_feasible(self, param):
        if self.bounds is None:
            return True
        return np.all(param >= self.bounds[:, 0]) and np.all(param <= self.bounds[:, 1])

    def _repair_infeasible_params(self, param):
        if self.bounds is None:
            return param

        # clip with lower and upper bound.
        param = np.where(param < self.bounds[:, 0], self.bounds[:, 0], param)
        param = np.where(param > self.bounds[:, 1], self.bounds[:, 1], param)
        return param

    def _set_bounds(self, bounds):
        """Update boundary constraints"""
        self.bounds = bounds
        if bounds is not None:
            self.bounds = np.array(bounds)
            if len(self.bounds.shape) == 1 and self.bounds.shape[0] == 2:
                self.bounds = np.array([self.bounds] * self.weights_size)
            if (self.weights_size, 2) != self.bounds.shape:
                raise ValueError("invalid bounds")
