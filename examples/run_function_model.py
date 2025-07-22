import os
import argparse
from loguru import logger
import numpy as np

import torch
import torch.nn as nn

from esbox.models import TorchModel
from esbox.core import Config, Task
from esbox.problems.functions import FuncProblem


class MyModel(TorchModel):

    def __init__(self, obs_dim, act_dim):
        super(MyModel, self).__init__()

        self.fc = nn.Linear(obs_dim, act_dim)
        self.initialization()

    def forward(self, obs):
        out = torch.tanh(self.fc(obs))
        # out = self.fc(obs)
        return out


class MyProblem(FuncProblem):

    def __init__(self, func_name='ackley', dim=11, scale=False):
        FuncProblem.__init__(self, func_name=func_name, dim=dim, scale=scale)
        self.dim = dim

        self.obs_dim = 6
        self.model = MyModel(obs_dim=self.obs_dim, act_dim=self.dim)

    def evaluate(self, weight):
        self.model.set_flat_weights(weight)
        # fixed inputs to model
        inputs = np.array([1] * self.obs_dim)
        inputs = torch.FloatTensor(inputs.reshape(1, -1))
        outputs = self.model(inputs)
        outputs = outputs.cpu().detach().numpy()

        result = {'value': 0, 'info': {}}
        value = self.problem(outputs)
        # print(outputs, value)

        result['value'] = -value
        return result


def main():
    cfg = Config(config_file=args.config_file, model_cls=MyModel)
    cfg.hyparams['seed'] = args.seed
    if args.work_dir:
        cfg.hyparams['work_dir'] = args.work_dir
    else:
        cfg.hyparams['work_dir'] = './esbox_train_log/{}/{}_{}_model'.format(cfg.alg_name, cfg.hyparams['func_name'],
                                                                             args.seed)
    tk = Task(config=cfg, eval_func=MyProblem)
    result = tk.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='', help='config file')
    parser.add_argument('--work_dir', type=str, default='', help='work dir path')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    args.config_file = os.path.join(os.path.abspath("."), args.config_file)
    main()
