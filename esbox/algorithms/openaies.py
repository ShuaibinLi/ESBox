import numpy as np
from esbox.optimizers import *
from esbox.utils.noises import SharedNoiseTable
from esbox.utils.rewards_utils import compute_centered_ranks
from esbox.utils import utils

__all__ = ['OpenAIES']


class OpenAIES(object):
    """ OpenAI-ES algorithm
    Args:
        weights_size (int): number of parameters that need to be optimized
        step_size (float): step size (learning rate) of optimizer
        l2_coeff (float): l2 regularization coefficient of optimizer
        init_weights (list): initial weights
    """

    def __init__(self, weights_size, step_size=0.02, l2_coeff=0.005, init_weights=None):
        self.weights_size = weights_size if weights_size is not None else None
        self.step_size = step_size if step_size is not None else 0.02
        self.l2_coeff = l2_coeff if l2_coeff is not None else 0.005

        assert isinstance(self.weights_size, int)
        assert isinstance(self.step_size, float)
        assert isinstance(self.l2_coeff, float)
        assert init_weights is not None
        assert len(init_weights) == self.weights_size

        self.weights = init_weights

        self.noise_table = SharedNoiseTable()

        self.optimizer = Adam(self.weights_size, self.step_size)
        print("Initialization of OpenAI-ES learner complete.")

    def learn(self, noisy_rewards, sampled_info):
        """learn weights
        Args:
            rollout_rewards (np.array): rewards vector
            sampled_info (dict): stats of the sampled data
        
        Returns:
            learned_info (dict): info after learning (e.g., the latest weights, )
        """
        assert 'noise_index' in sampled_info.keys()

        noise_idx = sampled_info['noise_index']
        # normalize rewards to (-0.5, 0.5), shape: [batch_size, 2] or [batch_size, 1]
        proc_noisy_rewards = compute_centered_ranks(noisy_rewards)
        proc_noisy_rewards = np.array(proc_noisy_rewards)

        if proc_noisy_rewards.shape[1] == 2:
            g, count = utils.batched_weighted_sum(
                # use mirror sampling: evaluate pairs of perturbations \epsilon, −\epsilon
                proc_noisy_rewards[:, 0] - proc_noisy_rewards[:, 1],
                (self.noise_table.get(idx, self.weights_size) for idx in noise_idx),
                batch_size=500)
        elif proc_noisy_rewards.shape[1] == 1:
            g, count = utils.batched_weighted_sum(
                # do not use mirror sampling: evaluate perturbation \epsilon
                proc_noisy_rewards[:, 0],
                (self.noise_table.get(idx, self.weights_size) for idx in noise_idx),
                batch_size=500)
        else:
            raise ValueError("Noise rewards shape {} error.".format(proc_noisy_rewards.shape))
        g /= proc_noisy_rewards.size

        # Compute the new weights.
        self.weights, ratio = self.optimizer.update(self.weights, -g + self.l2_coeff * self.weights)

        self.learned_info = {
            'weights': self.weights,
        }
        return self.learned_info
