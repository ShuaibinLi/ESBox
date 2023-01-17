import numpy as np
from scipy import spatial
from esbox.optimizers import *
from esbox.utils.noises import SharedNoiseTable
from esbox.utils.rewards_utils import compute_centered_ranks
from esbox.utils import utils

__all__ = ['NSRAES']


class NSRAES(object):
    """ NSRAES algorithm
    Args:
        weights_size (int): number of parameters that need to be optimized
        step_size (float): step size (learning rate) of optimizer
        k (int): number of nearest neigbhours used in the calculation of the novelty
        sigma (float): standart deviation of noise for sampler to use while sampling.
        pop_size int: population size
        l2_coeff (float): l2 regularization coefficient of optimizer
        init_weights (list): initial weights
    """

    def __init__(self, weights_size, step_size=0.02, k=10, pop_size=256, sigma=0.01, l2_coeff=0.005, init_weights=None):

        self.weights_size = weights_size
        self.step_size = step_size
        self.l2_coeff = l2_coeff

        self.pop_size = pop_size
        self.sigma = sigma

        assert isinstance(self.weights_size, int)
        assert isinstance(self.l2_coeff, float)
        assert isinstance(self.step_size, float)
        assert isinstance(self.sigma, float)
        assert isinstance(self.pop_size, int)
        assert init_weights is not None
        assert len(init_weights) == self.weights_size

        self.weights = init_weights

        # for bc
        # self.meta_population_size = meta_population_size
        self.k = k
        self._archives = []
        self.latest_r = None

        # for w decay
        self.best_w_r = -float('inf')
        self.w = 1.0
        self.min_w = 0.0
        self.w_t = 10
        self.w_delta = 0.05
        self.t = 0

        self.noise_table = SharedNoiseTable()

        self.optimizer = Adam(self.weights_size, self.step_size)
        print("Initialization of NSRA-ES learner complete.")

    def learn(self, noisy_rewards, sampled_info):
        """learn weights
        Args:
            rollout_rewards (np.array): rewards vector
            sampled_info (dict): stats of the sampled data
        
        Returns:
            learned_info (dict): info after learning (e.g., the latest weights, )
        """
        assert 'noise_index' in sampled_info.keys()
        assert 'bcs' in sampled_info.keys()

        noise_idx = sampled_info['noise_index']
        bcs = sampled_info['bcs']

        assert noisy_rewards.shape[0] == noise_idx.shape[0] == bcs.shape[0]
        assert noisy_rewards.shape[1] == bcs.shape[1]

        if noisy_rewards.shape[1] == 2:
            # use mirror sampling: evaluate pairs of perturbations \epsilon, −\epsilon
            noisy_rewards = np.concatenate([noisy_rewards[:, 0], noisy_rewards[:, 1]])
            noise_idx = np.concatenate([noise_idx, -noise_idx])
            bcs = np.concatenate([bcs[:, 0], bcs[:, 1]])
        elif noisy_rewards.shape[1] == 1:
            # do not use mirror sampling: evaluate perturbation \epsilon
            noisy_rewards = np.squeeze(noisy_rewards, axis=1)
            bcs = np.squeeze(bcs, axis=1)
        else:
            raise ValueError("Noise rewards shape {} error.".format(noisy_rewards.shape))

        rewards = self._rewrite_rewards(noisy_rewards, bcs)

        g = self._calculate_grad(rewards, noise_idx)

        # Compute the new weights.
        self.weights, ratio = self.optimizer.update(self.weights, -g + self.l2_coeff * self.weights)

        # after learning
        if self.latest_r > self.best_w_r:
            self.best_w_r = self.latest_r
            self.w = min(self.w + self.w_delta, 1.0)
            self.t = 0
        else:
            self.t += 1
            if self.t > self.w_t:
                self.w = max(self.w - self.w_delta, self.min_w)
                self.t = 0

        self.learned_info = {
            'weights': self.weights,
        }
        return self.learned_info

    def _calculate_novelty(self, bc):
        kd = spatial.cKDTree(self._archives)
        distances, idxs = kd.query(bc, k=self.k)
        distances = distances[distances < float('inf')]
        novelty = np.sum(distances) / (np.linalg.norm(self._archives) + 1e-8)
        return novelty

    def _rewrite_rewards(self, rewards, bcs):
        returns = []
        for r, bc in zip(rewards, bcs):
            novelty = self._calculate_novelty(bc)
            returns.append((r, novelty))
        return np.array(returns, dtype=np.float64)

    def _calculate_grad(self, noisy_rewards, noise_idx, batch_size=500):
        # normalize rewards to (-0.5, 0.5), shape:[batch_size, 2]
        proc_noisy_rewards = compute_centered_ranks(noisy_rewards)
        proc_noisy_rewards = np.array(proc_noisy_rewards)

        grad, count = utils.batched_weighted_sum(
            # mirrored sampling: evaluate pairs of perturbations \epsilon, −\epsilon
            self.w * proc_noisy_rewards[:, 0] + (1.0 - self.w) * proc_noisy_rewards[:, 1],
            (self.noise_table.get(idx, self.weights_size) for idx in noise_idx),
            batch_size=batch_size)

        grad /= self.pop_size * self.sigma
        grad = np.clip(grad, -1., 1.)
        return grad
