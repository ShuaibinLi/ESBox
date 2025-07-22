import numpy as np
import collections
import paddle
import paddle.nn as nn

__all__ = ['PaddleModel']


class PaddleModel(nn.Layer):
    """ Class for models that use the paddlepaddle interface 

    Example:
        .. code-block:: python
            import paddle.nn as nn
            from esbox.models import PaddleModel

            class MyModel(PaddleModel):
                def __init__(self, obs_dim, act_dim):
                    super(MyModel, self).__init__()

                    self.fc = nn.Linear(obs_dim, act_dim)
                    self.initialization()

                def forward(self, obs):
                    out = self.fc(obs)
                    return out

            model = MyModel(obs_dim=17, act_dim=6)
    """

    def __init__(self):
        super(PaddleModel, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def initialization(self):
        weights = self.get_weights()
        self.weights_name = list(weights.keys())
        weights = list(weights.values())
        self.weights_shapes = [x.shape for x in weights]
        self.weights_total_size = int(np.sum([np.prod(x) for x in self.weights_shapes]))

    def get_flat_weights(self):
        weights = list(self.get_weights().values())
        flat_weights = np.concatenate([x.flatten() for x in weights])
        return flat_weights

    def set_flat_weights(self, flat_weights):
        weights = self._unflatten(flat_weights, self.weights_shapes)
        weights_dcit = {}
        assert len(weights) == len(self.weights_name)
        for name, values in zip(self.weights_name, weights):
            weights_dcit[name] = values
        self.set_weights(weights_dcit)

    def _unflatten(self, flat_array, array_shapes):
        i = 0
        arrays = []
        for shape in array_shapes:
            size = np.prod(shape, dtype=np.int32)
            array = flat_array[i:(i + size)].reshape(shape)
            arrays.append(array)
            i += size
        assert len(flat_array) == i
        return arrays

    def get_weights(self):
        """get weight of model.
        
        Returns: 
            weights (dict): a Python dict containing the parameters of current model.
        """
        weights = self.state_dict()
        for key in weights.keys():
            weights[key] = weights[key].numpy()
        return weights

    def set_weights(self, weights):
        """set weights for model.
        
        Args:
            weights (dict): a Python dict containing the parameters.
        """
        old_weights = self.state_dict()
        assert len(old_weights) == len(weights), '{} params are expected, but got {}'.format(
            len(old_weights), len(weights))
        new_weights = collections.OrderedDict()
        for key in old_weights.keys():
            assert key in weights, 'key: {} is expected to be in weights.'.format(key)
            assert old_weights[key].shape == list(
                weights[key].shape), 'key \'{}\' expects the data with shape {}, but gets {}'.format(
                    key, old_weights[key].shape, list(weights[key].shape))
            new_weights[key] = paddle.to_tensor(weights[key], dtype='float32')
        self.set_state_dict(new_weights)
