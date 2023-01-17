import os
import argparse
from parl.utils import logger
import numpy
from esbox.core import Config, Task
from esbox.problems.functions import FuncProblem


class MyProblem(FuncProblem):
    def __init__(self, func_name='ackley', dim=2, scale=False):
        FuncProblem.__init__(self, func_name=func_name, dim=dim, scale=scale)
        self.dim = dim

    def evaluate(self, weight):
        result = {'value': 0, 'info': {}}
        value = self.problem(weight)
        # print(outputs, value)

        result['value'] = -value
        return result


def main():
    cfg = Config(config_file=args.config_file)
    cfg.hyparams['seed'] = args.seed
    logger.set_dir('./train_log/{}/{}_dim{}_{}'.format(cfg.alg_name, cfg.hyparams['func_name'], cfg.hyparams['dim'],
                                                       args.seed))
    tk = Task(config=cfg, eval_func=MyProblem)
    result = tk.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='', help='config file')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    args.config_file = os.path.join(os.path.abspath("."), args.config_file)
    main()
