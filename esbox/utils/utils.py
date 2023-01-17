import numpy as np

__all__ = ['_HAS_PADDLE', '_HAS_TORCH', 'batched_weighted_sum']

try:
    import paddle
    _HAS_PADDLE = True
except ImportError:
    _HAS_PADDLE = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def itergroups(items, group_size):
    """An iterator that iterates a list with batch data."""
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    """Compute the gradients for updating the parameters.
        Args:
            weights(np.array): the nomalized rewards computed by the function `compute_centered_ranks`.
            vecs(np.array): the noise added to the parameters.
            batch_size(int): the batch_size for speeding up the computation.
        Return:
            total(np.array): aggregated gradient. 
            num_items_summed(int): the number of weights used during computing the total gradient.
    """
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float64), np.asarray(batch_vecs, dtype=np.float64))
        num_items_summed += len(batch_weights)
    return total, num_items_summed
