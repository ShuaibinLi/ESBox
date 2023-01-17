# The following code are copied or modified from:
# https://github.com/PaddlePaddle/PARL/blob/develop/benchmark/torch/ES/noise.py

import numpy as np

__all__ = ['SharedNoiseTable']


class SharedNoiseTable(object):
    """Shared noise table used by learner and actor.
    Learner and actor will create a same noise table by passing the same seed.
    With the same noise table, learner and actor can communicate the noises by
    index of noise table instead of numpy array of noises.
    """

    def __init__(self, seed=110):
        self.noise = self._create_shared_noise()
        assert self.noise.dtype == np.float64

        self.rg = np.random.RandomState(seed)

    def _create_shared_noise(self):
        """
        Create a large array of noise.
        """
        seed = 12345
        count = 250000000
        noise = np.random.RandomState(seed).randn(count).astype(np.float64)
        return noise

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, dim, batch_size=1):
        idx = []
        for _ in range(batch_size):
            idx.append(self.rg.randint(0, len(self.noise) - dim + 1))
        return idx

    def get_delta(self, dim, batch_size=1):
        idx = self.sample_index(dim, batch_size)
        noises = []
        for id in idx:
            noises.append(self.get(id, dim))
        return idx, noises
