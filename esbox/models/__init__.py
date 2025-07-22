from esbox.utils import _HAS_PADDLE, _HAS_TORCH

if _HAS_PADDLE:
    from esbox.models.paddle_model import PaddleModel
elif _HAS_TORCH:
    from esbox.models.torch_model import TorchModel
