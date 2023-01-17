import numpy as np

__all__ = ['compute_centered_ranks']


def compute_ranks(x):
    """Returns ranks in [0, len(x))

    Note: This is different from scipy.stats.rankdata, which returns ranks in
    [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """Return ranks that is normliazed to [-0.5, 0.5] with the rewards as input.
    Args:
        x(np.array): an array of rewards.
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= 0.5
    return y
