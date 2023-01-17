import os
import yaml
from parl.utils import logger
from esbox.utils import _HAS_PADDLE, _HAS_TORCH

__all__ = ['Config']


class Config(object):
    """`Config` is the base class for loading configuration from a yaml file.
    """

    def __init__(self, config_file, model_cls=None):
        """
        Args:
            config_file (file_path): Path to yaml configuration file.
            model_cls (model): Model class created by user. Defaults to None.
        """

        logger.info("Loading config ......")

        self.model_cls = model_cls
        self.config = self.get_yaml_data(config_file)
        self.alg_name = list(self.config.keys())[0]
        self.hyparams = self.config[self.alg_name]

        if self.model_cls is not None:
            # self.hyparams['param_num'] = model_cls.weights_total_size
            if _HAS_PADDLE:
                from esbox.models.paddle_model import PaddleModel
                assert issubclass(self.model_cls,
                                  PaddleModel), "Please define your model by inheriting from 'PaddleModel'."
            elif _HAS_TORCH:
                from esbox.models.torch_model import TorchModel
                assert issubclass(self.model_cls,
                                  TorchModel), "Please define your model by inheriting from 'TorchModel'."
            else:
                raise AttributeError(
                    "Please install torch or paddlepaddle, and define your model by inheriting from `TorchModel` or `PaddleModel` correspondingly."
                )

        logger.info("data type: {}, contents: \n{}".format(type(self.config), self.config))
        logger.info("Load config done !!!")
        logger.info("Adjusting the configuration ...")

    def get_yaml_data(self, yaml_file):
        """Get config data from yaml file.

        Args:
            yaml_file: configuration file name.
        
        Returns:
            data: config dictionary.
        
        """
        logger.info("get yaml file from \"{}\" ...".format(yaml_file))
        file = open(yaml_file, 'r', encoding="utf-8")
        file_data = file.read()
        file.close()
        logger.info("config file data type: {}, contents: \n{}".format(type(file_data), file_data))

        logger.info("convert yaml to dict data ...")
        data = yaml.safe_load(file_data)
        return data
