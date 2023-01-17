import os
import argparse
import parl
from parl.utils import logger
import torch.nn as nn

from esbox.models import TorchModel
from esbox.core import Config, ParallelTask
from esbox.problems.gym_problems import RLProblem


class MyModel(TorchModel):
    def __init__(self, obs_dim, act_dim):
        super(MyModel, self).__init__()

        self.fc = nn.Linear(obs_dim, act_dim)
        self.initialization()

    def forward(self, obs):
        out = self.fc(obs)
        return out


@parl.remote_class(wait=False)
class MujocoEnv(RLProblem):
    def __init__(self, env_name, seed=123, shift=0, n=None):
        RLProblem.__init__(self, env_name, seed=seed, shift=shift, n=n)
        self.seed = seed

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.model = MyModel(obs_dim, act_dim)

    def evaluate(self, weight):
        ret = self.run_episode(weight, add_noise=False)
        # ret = {'value': , 'info': {"step": , "shift": , "add_noise": }}
        return ret


def main():
    cfg = Config(config_file=args.config_file, model_cls=MyModel)
    cfg.hyparams['seed'] = args.seed
    logger.set_dir('./train_log/{}/{}_{}'.format(cfg.alg_name, cfg.hyparams['env_name'], args.seed))
    tk = ParallelTask(config=cfg, eval_func=MujocoEnv)
    result = tk.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='', help='config file')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    args.config_file = os.path.join(os.path.abspath("."), args.config_file)
    main()
