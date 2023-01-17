import numpy as np
from abc import abstractmethod

from esbox.problems.functions.ackley import *
from esbox.problems.functions.griewank import *
from esbox.problems.functions.zakharov import *
from esbox.problems.functions.rastrigin import *
from esbox.problems.functions.rosenbrock import *
from esbox.problems.functions.styblinskiTang import *

SO_FUCNTION = {
    'ackley': Ackley,
    'rosenbrock': Rosenbrock,
    'zakharov': Zakharov,
    'griewank': Griewank,
    'styblinskitang': StyblinskiTang,
    'rastrigin': Rastrigin
}
""" single-objective functions
ackley, griewank, zakharov, rastrigin, rosenbrock, styblinskitang
"""


class FuncProblem(object):
    def __init__(self, func_name="ackley", dim=2, scale=False, *args, **kwargs):
        self.problem = SO_FUCNTION[func_name](dim=dim, scale=scale, *args, **kwargs)

    def evaluate_batch(self, batch_flatten_weights, *args, **kwargs):
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
                batch_bcs.append(ret['info'].get('bc', weight))
                steps += ret['info'].get('step', 1)
            rollout_rewards.append(batch_rewards)
            bcs.append(batch_bcs)
        rollout_rewards = np.array(rollout_rewards).squeeze(-1)
        bcs = np.array(bcs)
        return {'values': rollout_rewards, 'info': {"steps": steps, 'bcs': bcs}}

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
