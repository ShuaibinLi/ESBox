# An implementation of "The CMA Evolution Strategy: A Tutorial".
# (https://arxiv.org/abs/1604.00772).
# Specially, this code is written based on page 28~32,
# you can find the equations mentioned below in these pages.

import math
import numpy as np

_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32

__all__ = ['CMAES']


class CMAES(object):
    """ CMAES-ES algorithm
    Args:
        weights_size (int): number of parameters that need to be optimized
        sigma (float): initial standard deviation of covariance matrix
        population_size (int, None): population size. If None, it will automatically calculate according to weights_size
        mu (int, None): top mu directions used to update cov
        cov (numpy.ndarray): covariance matrix.
        init_weights (list): initial weights
    """

    def __init__(self, weights_size, sigma=1.3, population_size=None, mu=None, cov=None, init_weights=None):

        self.weights_size = weights_size
        self._sigma = sigma
        self._popsize = population_size
        self._mu = mu

        assert isinstance(self.weights_size, int)
        assert isinstance(self._sigma, float)
        assert isinstance(self._popsize, (int, type(None)))
        assert isinstance(self._mu, (int, type(None)))
        assert init_weights is not None
        assert len(init_weights) == self.weights_size

        self._mean = init_weights

        if self._popsize is None:
            # eq.48
            self._popsize = 4 + math.floor(3 * math.log(self.weights_size))
        if self._mu is None:
            # table.1
            self._mu = self._popsize // 2

        # initialize default parameters, noise
        self._init_default_parameters()

        # Termination criteria
        self._set_termination_criteria()
        # evolution path
        self._g = 0
        self._p_sigma = np.zeros(self.weights_size)
        self._pc = np.zeros(self.weights_size)

        if cov is None:
            self._C = np.eye(self.weights_size)
        else:
            assert cov.shape == (self.weights_size, self.weights_size), "Invalid shape of covariance matrix"
            self._C = cov

        self._D = None
        self._B = None
        print("Initialization of CMAES learner complete.")

    @property
    def weights(self):
        return self._mean

    def _init_default_parameters(self):
        """Init default strategy parameters according to equations in Table1 in page.31.
        """
        # eq.49
        weights_prime = np.array([math.log((self._popsize + 1) / 2) - math.log(i + 1) for i in range(self._popsize)])
        self._mu_eff = (np.sum(weights_prime[:self._mu])**2) / np.sum(weights_prime[:self._mu]**2)
        mu_eff_minus = (np.sum(weights_prime[self._mu:])**2) / np.sum(weights_prime[self._mu:]**2)

        # learning rate for the rank-one update, eq.57
        alpha_cov = 2
        self._c1 = alpha_cov / ((self.weights_size + 1.3)**2 + self._mu_eff)
        # learning rate for the rank-μ update, eq.58
        self._cmu = min(
            1 - self._c1 - 1e-8,  # 1e-8 is for large popsize.
            alpha_cov * (self._mu_eff - 2 + 1 / self._mu_eff) /
            ((self.weights_size + 2)**2 + alpha_cov * self._mu_eff / 2),
        )
        assert self._c1 <= 1 - self._cmu, "invalid learning rate for the rank-one update"
        assert self._cmu <= 1 - self._c1, "invalid learning rate for the rank-μ update"

        min_alpha = min(
            1 + self._c1 / self._cmu,  # eq.50
            1 + (2 * mu_eff_minus) / (self._mu_eff + 2),  # eq.51
            (1 - self._c1 - self._cmu) / (self.weights_size * self._cmu),  # eq.52
        )
        # eq.53
        positive_sum = np.sum(weights_prime[weights_prime > 0])
        negative_sum = np.sum(np.abs(weights_prime[weights_prime < 0]))
        self.w_weights = np.where(weights_prime >= 0, 1 / positive_sum * weights_prime,
                                  min_alpha / negative_sum * weights_prime)
        # eq.54
        self._cm = 1

        # learning rate for the cumulation for the step-size control eq.55
        self._c_sigma = (self._mu_eff + 2) / (self.weights_size + self._mu_eff + 5)
        self._d_sigma = 1 + 2 * max(0, math.sqrt((self._mu_eff - 1) / (self.weights_size + 1)) - 1) + self._c_sigma
        assert (self._c_sigma < 1), "invalid learning rate for cumulation for the step-size control"

        # learning rate for cumulation for the rank-one update eq.56
        self._cc = (4 + self._mu_eff / self.weights_size) / (self.weights_size + 4 +
                                                             2 * self._mu_eff / self.weights_size)
        assert self._cc <= 1, "invalid learning rate for cumulation for the rank-one update"

        # E||N(0, I)|| (p.28)
        self._chi_n = math.sqrt(self.weights_size) * (1.0 - (1.0 / (4.0 * self.weights_size)) + 1.0 /
                                                      (21.0 * (self.weights_size**2)))

    def _set_termination_criteria(self):
        """Init default strategy parameters according to Termination Criteria in page.33.
        """
        self._tolx = 1e-12 * self._sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

        self._funhist_term = 10 + math.ceil(30 * self.weights_size / self._popsize)
        self._funhist_values = np.empty(self._funhist_term * 2)

    def _eigen_decomposition(self):
        if self._B is not None and self._D is not None:
            return self._B, self._D

        self._C = (self._C + self._C.T) / 2
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        self._C = np.dot(np.dot(B, np.diag(D**2)), B.T)

        self._B, self._D = B, D
        return B, D

    def learn(self, v_batch, sampled_info):
        """solutions
        Args:
            v_batch: list or numpy array of data to use
            sampled_info (dict): stats of the sampled data
        
        Returns:
            learned_info (dict): info after learning (e.g., the latest weights, )
        """
        assert 'batch_flatten_weights' in sampled_info.keys()

        x_batch = sampled_info['batch_flatten_weights'].squeeze()
        assert len(x_batch) == self._popsize, "Values must popsize-length."
        assert np.all(np.array(x_batch) <
                      _MEAN_MAX), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors"

        self._g += 1
        v_batch = v_batch.squeeze()
        sort_idx = np.argsort(v_batch)[::-1]

        # Stores 'best' and 'worst' values of the
        # last 'self._funhist_term' generations.
        funhist_idx = 2 * (self._g % self._funhist_term)
        self._funhist_values[funhist_idx] = v_batch[sort_idx[0]]
        self._funhist_values[funhist_idx + 1] = v_batch[sort_idx[-1]]

        # Sample new population of search_points, for k=1, ..., popsize
        B, D = self._eigen_decomposition()
        self._B, self._D = None, None

        x_k = np.array([x_batch[i] for i in sort_idx])  # ~ N(m, σ^2 C)
        y_k = (x_k - self._mean) / self._sigma  # ~ N(0, C)

        # Selection and recombination
        y_w = np.sum(y_k[:self._mu].T * self.w_weights[:self._mu], axis=1)  # eq.41
        self._mean += self._cm * self._sigma * y_w

        # Step-size control
        C_2 = B.dot(np.diag(1 / D)).dot(B.T)  # C^(-1/2) = B D^(-1) B^T
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff) * C_2.dot(y_w)

        norm_p_sigma = np.linalg.norm(self._p_sigma)
        self._sigma *= np.exp((self._c_sigma / self._d_sigma) * (norm_p_sigma / self._chi_n - 1))
        self._sigma = min(self._sigma, _SIGMA_MAX)

        # Covariance matrix adaption
        h_sigma_cond_left = norm_p_sigma / math.sqrt(1 - (1 - self._c_sigma)**(2 * (self._g + 1)))
        h_sigma_cond_right = (1.4 + 2 / (self.weights_size + 1)) * self._chi_n
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0  # (p.28)

        # eq.45
        self._pc = (1 - self._cc) * self._pc + h_sigma * math.sqrt(self._cc * (2 - self._cc) * self._mu_eff) * y_w

        # eq.46
        w_io = self.w_weights * np.where(
            self.w_weights >= 0,
            1,
            self.weights_size / (np.linalg.norm(C_2.dot(y_k.T), axis=0)**2 + _EPS),
        )

        delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)  # (p.28)
        assert delta_h_sigma <= 1

        # eq.47
        rank_one = np.outer(self._pc, self._pc)
        rank_mu = np.sum(np.array([w * np.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0)
        self._C = ((1 + self._c1 * delta_h_sigma - self._c1 - self._cmu * np.sum(self.w_weights)) * self._C +
                   self._c1 * rank_one + self._cmu * rank_mu)
        self.learned_info = {'weights': self.weights, 'covariance_matrix': self._C, 'sigma': self._sigma}
        return self.learned_info

    def should_stop(self):
        B, D = self._eigen_decomposition()
        dC = np.diag(self._C)

        # Stop if the range of function values of the recent generation is below tolfun.
        if (self._g > self._funhist_term
                and np.max(self._funhist_values) - np.min(self._funhist_values) < self._tolfun):
            return True

        # Stop if the std of the normal distribution is smaller than tolx
        # in all coordinates and pc is smaller than tolx in all components.
        if np.all(self._sigma * dC < self._tolx) and np.all(self._sigma * self._pc < self._tolx):
            return True

        # Stop if detecting divergent behavior.
        if self._sigma * np.max(D) > self._tolxup:
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if np.any(self._mean == self._mean + (0.2 * self._sigma * np.sqrt(dC))):
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. "pycma" check
        # axis one by one at each generation.
        i = self._g % self.weights_size
        if np.all(self._mean == self._mean + (0.1 * self._sigma * D[i] * B[:, i])):
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        condition_cov = np.max(D) / np.min(D)
        if condition_cov > self._tolconditioncov:
            return True

        return False
