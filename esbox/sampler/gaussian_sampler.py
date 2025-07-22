import numpy as np
from esbox.utils.noises import SharedNoiseTable
from esbox.sampler.base_sampler import BoundedSampler

__all__ = ['GaussianSampler']


class GaussianSampler(BoundedSampler):
    """Class implementing the Gaussian sampler.
    Args:
        weights_size (int): number of sampling parameters
        bounds (list): lower and upper domain boundaries for each parameter, e.g. [-5, 5]
        noise_stdev (float): noise standard deviation
        seed (int): random seed of sampling
        mirror_sampling (bool):
        n_max_resampling (int): A maximum number of resampling parameters (default: 100).
                                If all sampled parameters are infeasible, the last sampled one
                                will be clipped with lower and upper bounds.
    """

    def __init__(self,
                 weights_size,
                 noise_stdev=0.01,
                 seed=123,
                 mirro_sampling=True,
                 bounds=None,
                 n_max_resampling=100):
        BoundedSampler.__init__(self, weights_size=weights_size, bounds=bounds, n_max_resampling=n_max_resampling)

        self.noise_stdev = noise_stdev
        self.seed = seed
        self.mirro_sampling = mirro_sampling

        # create shared table for storing noise
        self.noise_table = SharedNoiseTable(seed=self.seed)
        if self.mirro_sampling:
            print("Initialization of Mirror Gaussian sampler complete.")
        else:
            print("Initialization of Gaussian sampler complete.")

    def __call__(self, flat_weights, sample_batch=1, *args, **kwargs):
        sample_info = {}
        noise_index, batch_flatten_weights = self.sample(flat_weights, sample_batch)
        sample_info['noise_index'] = noise_index
        sample_info['batch_flatten_weights'] = batch_flatten_weights
        return sample_info

    def sample(self, flat_weights, sample_batch=1):
        assert len(flat_weights) == self.weights_size

        batch_flatten_weights = []
        noise_index = []
        # while len(batch_flatten_weights) != sample_batch:
        done_flag = False
        for _ in range(self._n_max_resampling):
            noise_idx, noises = self.noise_table.get_delta(self.weights_size, batch_size=sample_batch)
            for idx, noise in zip(noise_idx, noises):
                # parameter perturbations.
                perturbation = self.noise_stdev * noise
                pos_weights = flat_weights + perturbation
                if self._is_feasible(pos_weights):
                    if self.mirro_sampling:
                        # mirrored sampling: pairs of perturbations \epsilon, −\epsilon
                        neg_weights = flat_weights - perturbation
                        if self._is_feasible(neg_weights):
                            batch_flatten_weights.append([pos_weights, neg_weights])
                            noise_index.append(idx)
                    else:
                        batch_flatten_weights.append([pos_weights])
                        noise_index.append(idx)
                if len(batch_flatten_weights) == sample_batch:
                    done_flag = True
                    break
            if done_flag:
                break

        if len(batch_flatten_weights) != sample_batch:
            extra_batch = sample_batch - len(batch_flatten_weights)
            noise_idx, noises = self.noise_table.get_delta(self.weights_size, batch_size=extra_batch)
            for idx, noise in zip(noise_idx, noises):
                # parameter perturbations.
                perturbation = self.noise_stdev * noise
                pos_weights = flat_weights + perturbation
                pos_weights = self._repair_infeasible_params(pos_weights)
                if self.mirro_sampling:
                    # mirrored sampling: pairs of perturbations \epsilon, −\epsilon
                    neg_weights = flat_weights - perturbation
                    neg_weights = self._repair_infeasible_params(neg_weights)
                    batch_flatten_weights.append([pos_weights, neg_weights])
                else:
                    batch_flatten_weights.append([pos_weights])
                noise_index.append(idx)

        noise_index = np.array(noise_index)
        batch_flatten_weights = np.array(batch_flatten_weights)
        return noise_index, batch_flatten_weights


if __name__ == "__main__":

    sampler = GaussianSampler(
        2,
        1,
        mirro_sampling=True,
        bounds=[-1.1, 1.1],
    )

    sample_info = sampler([0] * 2, 3)
    print(sample_info)
    # print(sample_info == sample_info1)
