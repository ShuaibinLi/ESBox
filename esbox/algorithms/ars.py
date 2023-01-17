import numpy as np
from esbox.optimizers import *
from esbox.utils.noises import SharedNoiseTable
from esbox.utils import utils

__all__ = ['ARS']


class ARS(object):
    """ OpenAI-ES algorithm
    Args:
        weights_size (int): number of parameters that need to be optimized
        step_size (float): step size (learning rate) of optimizer
        top_k (int, None): pick from amongst the top performing weights
        init_weights (list): initial weights
    """

    def __init__(self, weights_size, step_size=0.02, top_k=None, init_weights=None):
        self.step_size = step_size if step_size is not None else 0.02
        self.weights_size = weights_size if weights_size is not None else None
        self.top_k = top_k

        assert isinstance(self.weights_size, int)
        assert isinstance(self.top_k, (int, type(None)))
        assert isinstance(self.step_size, float)
        assert init_weights is not None
        assert len(init_weights) == self.weights_size

        self.weights = init_weights

        # create shared table for learning
        self.noise_table = SharedNoiseTable()

        # initialize optimization algorithm
        self.optimizer = SGD(self.weights_size, self.step_size, momentum=0.0)
        print("Initialization of ARS learner complete.")

    def learn(self, rollout_rewards, sampled_info):
        """learn weights
        Args:
            rollout_rewards (np.array): rewards vector
            sampled_info (dict): stats of the sampled data
        
        Returns:
            learned_info (dict): info after learning (e.g., the latest weights, )
        """
        assert 'noise_index' in sampled_info.keys()

        noise_idx = sampled_info['noise_index']
        # select top performing directions if top_k < len(rollout_rewards)
        if self.top_k is not None:
            max_rewards = np.max(rollout_rewards, axis=1)
            if self.top_k > rollout_rewards.shape[0]:
                self.top_k = rollout_rewards.shape[0]

            idx = np.arange(
                max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100 *
                                                               (1 - (self.top_k / rollout_rewards.shape[0])))]
            noise_idx = noise_idx[idx]
            rollout_rewards = rollout_rewards[idx, :]

        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        if rollout_rewards.shape[1] == 2:
            g_hat, count = utils.batched_weighted_sum(
                # use mirror sampling: evaluate pairs of perturbations \epsilon, −\epsilon
                rollout_rewards[:, 0] - rollout_rewards[:, 1],
                (self.noise_table.get(idx, self.weights_size) for idx in noise_idx),
                batch_size=500)
        elif rollout_rewards.shape[1] == 1:
            g_hat, count = utils.batched_weighted_sum(
                # do not use mirror sampling: evaluate perturbation \epsilon
                rollout_rewards[:, 0],
                (self.noise_table.get(idx, self.weights_size) for idx in noise_idx),
                batch_size=500)
        else:
            raise ValueError("Noise rewards shape {} error.".format(rollout_rewards.shape))
        g_hat /= noise_idx.size

        # Perform one update step of the policy weights.
        self.weights, ratio = self.optimizer.update(self.weights, -g_hat)

        self.learned_info = {
            'weights': self.weights,
        }
        return self.learned_info
