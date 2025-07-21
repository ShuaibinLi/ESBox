import torch
import torch.nn as nn
import numpy as np

__all__ = ['TorchModel']


class TorchModel(nn.Module):
    """ Class for models that use the torch interface 

    Example:
        .. code-block:: python
            import torch.nn as nn
            from esbox.models import TorchModel

            class MyModel(TorchModel):
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
        super(TorchModel, self).__init__()

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
            weights[key] = weights[key].cpu().numpy()
        return weights

    def set_weights(self, weights):
        """set weights for model.
        
        Args:
            weights (dict): a Python dict containing the parameters.
        """
        new_weights = dict()
        for key in weights.keys():
            new_weights[key] = torch.from_numpy(weights[key])
        self.load_state_dict(new_weights)
