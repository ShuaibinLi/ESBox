import os
import argparse
import numpy as np

from esbox.core import Config, Task
from esbox.problems import ProblemBase


# (Undesired) Step 1. Define your model class
# Step 2. Define your problem class
class MyFuncEnv(ProblemBase):
    """Please define your problem by inherting `ProblemBase` class.
    
    And please write the evaluation step in `evaluate()` function.
    """

    def __init__(self):
        super(MyFuncEnv, self).__init__()

        # Please define the dimension of your problem
        self.dim = 2

    def evaluate(self, weight):
        # use the weight to evaluate
        assert len(weight) == self.dim
        value = -(weight[0] - 0.2)**2 - (weight[1] - 0.7)**2 + 1
        step = 1

        # Please return your results in a dictionary,
        # and place the evaluation result in `value` key (necessary)
        # other information in `info` key, e.g. steps (unnecessary)
        return {'value': value, 'info': {"step": step}}


# Step 3. Write your `config.ymal` before running
def main():
    cfg = Config(config_file=args.config_file)
    if args.seed is not None:
        cfg.hyparams['seed'] = args.seed
    tk = Task(config=cfg, eval_func=MyFuncEnv)
    result = tk.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='', help='config file')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    args.config_file = os.path.join(os.path.abspath("."), args.config_file)
    main()
