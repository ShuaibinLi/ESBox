import numpy as np
from esbox.utils.noises import SharedNoiseTable
from esbox.sampler.base_sampler import BoundedSampler

EPS = 1e-8

__all__ = ['CMASampler']


class CMASampler(BoundedSampler):
    """Class implementing the CMA sampler.
    Args:
        weights_size (int): number of sampling parameters
        bounds (list): lower and upper domain boundaries for each parameter, e.g. [-5, 5]
        n_max_resampling (int): A maximum number of resampling parameters (default: 100).
                                If all sampled parameters are infeasible, the last sampled one
                                will be clipped with lower and upper bounds.
        seed (int): random seed of sampling
        sigma (float): standard deviation of covariance matrix
        cov (numpy.ndarray): covariance matrix. Uses default if not provided
    """

    def __init__(self, weights_size, bounds=None, n_max_resampling=100, seed=1234, sigma=None, cov=None):
        BoundedSampler.__init__(self, weights_size=weights_size, bounds=bounds, n_max_resampling=n_max_resampling)

        # create shared table for storing noise
        self.noise_table = SharedNoiseTable(seed=seed)

        if cov is None:
            self._C = np.eye(self.weights_size)
        else:
            assert cov.shape == (self.weights_size, self.weights_size), "Invalid shape of covariance matrix"
            self._C = cov
        self._sigma = sigma
        print("Initialization of CMA sampler complete.")

    def __call__(self, weights, sample_batch=1, other_info=None):
        sample_info = {}
        covariance_matrix = other_info.get('covariance_matrix', None)
        sigma = other_info.get('sigma', None)
        noise_index, batch_flatten_weights = self.sample(weights, sample_batch, covariance_matrix, sigma)
        sample_info['noise_index'] = noise_index
        sample_info['batch_flatten_weights'] = batch_flatten_weights
        return sample_info

    def sample(self, weights, sample_batch=1, covariance_matrix=None, sigma=None):
        """Sample `sample_batch` parameter"""
        self.weights = weights
        if covariance_matrix is not None:
            self._C = covariance_matrix
        if sigma is not None:
            self._sigma = sigma
        B, D = self._eigen_decomposition(self._C)

        sample_noise_index = []
        sampled_x = []
        for _ in range(sample_batch):
            for i in range(self._n_max_resampling):
                noise_idx, x = self._sample_solution(B, D)
                if self._is_feasible(x):
                    x = x
                    break
                x = None
            if x is None:
                noise_idx, x = self._sample_solution(B, D)
                x = self._repair_infeasible_params(x)
            sample_noise_index.append(noise_idx)
            sampled_x.append([x])
        return np.array(sample_noise_index), np.array(sampled_x)

    def _eigen_decomposition(self, C):
        _C = (C + C.T) / 2
        D2, B = np.linalg.eigh(_C)
        D = np.sqrt(np.where(D2 < 0, EPS, D2))
        return B, D

    def _sample_solution(self, B, D):
        noise_idx, z = self.noise_table.get_delta(self.weights_size, batch_size=1)
        y = B.dot(np.diag(D)).dot(z[0])  # ~ N(0, C)
        x = self.weights + self._sigma * y  # ~ N(m, σ^2 C)
        return noise_idx[0], x
