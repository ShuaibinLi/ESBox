import collections
import numpy as np
from copy import deepcopy
from abc import abstractmethod
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from esbox.utils.gym_wrappers import wrap_mujoco, wrap_atari
from esbox.problems.problem_lists import MUJOCO_PROBLEM, ATARI_PROBLEM
from esbox.utils.utils import _HAS_PADDLE, _HAS_TORCH

__all__ = ['RLProblem']


class RLProblem(object):

    def __init__(self, env_name=None, seed=123, reward_shift=0., n=None):

        if env_name in MUJOCO_PROBLEM:
            env = gym.make(env_name)
            self.env = wrap_mujoco(env, norm_obs=True)
            self.test_env = wrap_mujoco(env, norm_obs=True)
            self.continuous = True
            self.obs_rms = None
        elif env_name in ATARI_PROBLEM:
            env = gym.make(env_name)
            self.env = wrap_atari(env, reward_func=np.sign, noop_max=30, frame_skip=4, screen_size=84, stack_size=4)
            self.test_env = wrap_atari(env,
                                       reward_func=np.sign,
                                       noop_max=30,
                                       frame_skip=4,
                                       screen_size=84,
                                       stack_size=4)
            self.continuous = False
        else:
            try:
                self.env = RecordEpisodeStatistics(gym.make(env_name))
                self.test_env = RecordEpisodeStatistics(gym.make(env_name))
            except:
                raise Exception("Environment {} not found in gym.".format(env_name))

        self.reward_shift = reward_shift
        self.n = n
        self.add_noise = False
        self.seed = seed
        np.random.seed(seed)

    def get_dim(self):
        assert hasattr(self, 'model'), "Please use model to optimize gym problem."
        return self.model.weights_total_size

    def evaluate_batch(self, batch_flatten_weights, *args, **kwargs):
        """
        Generate multiple rollouts with a policy parametrized by w_policy.
        """
        rollout_rewards = []
        bcs = []
        steps = 0
        if len(batch_flatten_weights.shape) == 2:
            batch_flatten_weights = np.expand_dims(batch_flatten_weights, axis=0)
        assert len(batch_flatten_weights.shape) == 3

        for mini_batch in batch_flatten_weights:
            batch_rewards = []
            batch_bcs = []
            for weight in mini_batch:
                ret = self.evaluate(weight)
                batch_rewards.append(ret['value'])
                batch_bcs.append(ret['info'].get('bc', None))
                steps += ret['info']['step']
            rollout_rewards.append(batch_rewards)
            bcs.append(batch_bcs)
        return {'values': rollout_rewards, 'info': {"steps": steps, 'bcs': bcs}}

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """"Evaluate"""
        raise NotImplementedError

    def test_episode(self, weight):
        """Evaluates model weight while not training to get better evaluation of model qualities.
        (Without adding noise to action, using original rewards (NO reward_shift), etc.)
        """
        bc = None
        if self.n is not None:
            assert isinstance(self.n, int)
            start_actions = []
            last_actions = collections.deque(maxlen=self.n)

        self.model.set_flat_weights(weight)
        model = deepcopy(self.model)
        if self.continuous and self.obs_rms:
            self.test_env.obs_rms = self.obs_rms
        steps = 0
        obs, info = self.test_env.reset()
        while True:
            if _HAS_TORCH:
                import torch
                obs = torch.FloatTensor(obs.reshape(1, -1))
                action = model(obs)
                action = action.cpu().detach().numpy().flatten()
            elif _HAS_PADDLE:
                import paddle
                obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
                action = model(obs)
                action = action.cpu().numpy().flatten()
            else:
                raise NotImplementedError("Unable to find torch or paddle!")
            obs, reward, terminated, truncated, info = self.test_env.step(action)
            if self.n is not None:
                last_actions.append(action)
                if steps < self.n:
                    start_actions.append(action)
            steps += 1
            if terminated or truncated:
                break
        if self.n is not None:
            bc = np.concatenate([start_actions, last_actions]).flatten()
        return {'value': info['episode']['r'], 'info': {"step": info['episode']['l'], 'bc': bc}}

    def run_episode(self, weight, add_noise=False):
        """ 
        Performs one rollout while not done. 
        At each time-step it substracts reward_shift from the reward.
        """
        bc = None
        if self.n is not None:
            assert isinstance(self.n, int)
            start_actions = []
            last_actions = collections.deque(maxlen=self.n)

        self.model.set_flat_weights(weight)
        model = deepcopy(self.model)

        total_reward = 0.
        steps = 0
        obs, info = self.env.reset()
        while True:
            if _HAS_TORCH:
                import torch
                obs = torch.FloatTensor(obs.reshape(1, -1))
                action = model(obs)
                action = action.cpu().detach().numpy().flatten()
            elif _HAS_PADDLE:
                import paddle
                obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
                action = model(obs)
                action = action.cpu().numpy().flatten()
            else:
                raise NotImplementedError("Unable to find torch or paddle!")
            if add_noise:
                if self.continuous:
                    action += np.random.randn(*action.shape) * 0.01  # self.config['action_noise_std']
                else:
                    pass
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.n is not None:
                last_actions.append(action)
                if steps < self.n:
                    start_actions.append(action)
            total_reward += (reward - self.reward_shift)
            steps += 1
            if terminated or truncated:
                if self.continuous:
                    self.obs_rms = self.env.obs_rms
                break
        if self.n is not None:
            bc = np.concatenate([start_actions, last_actions]).flatten()
        return {
            'value': total_reward,
            'info': {
                "step": info['episode']['l'],
                "original_reward": info['episode']['r'],
                'bc': bc,
                "reward_shift": self.reward_shift,
                "add_noise": add_noise
            }
        }
