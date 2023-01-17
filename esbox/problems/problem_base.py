import numpy as np
from abc import abstractmethod

__all__ = ['ProblemBase']


class ProblemBase(object):
    def __init__(self, *args, **kwargs):
        pass

    def evaluate_batch(self, batch_flatten_weights, *args, **kwargs):
        rollout_rewards = []
        steps = 0
        if len(batch_flatten_weights.shape) == 2:
            batch_flatten_weights = np.expand_dims(batch_flatten_weights, axis=0)
        assert len(batch_flatten_weights.shape) == 3

        for mini_batch in batch_flatten_weights:
            batch_rewards = []
            for weight in mini_batch:
                ret = self.evaluate(weight)
                batch_rewards.append(ret['value'])
                steps += ret['info'].get('step', 1)
            rollout_rewards.append(batch_rewards)
        return {'values': rollout_rewards, 'info': {"steps": steps}}

    def get_dim(self):
        if hasattr(self, 'model'):
            self.dim = self.model.weights_total_size
        elif hasattr(self, 'dim'):
            self.dim = self.dim
        else:
            raise AttributeError("Please set model or dim for your problem.")
        return self.dim

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """"Evaluate"""
        raise NotImplementedError
