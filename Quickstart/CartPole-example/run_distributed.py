import os
import argparse
import gym
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import parl
# from esbox.models import TorchModel
from esbox.models import PaddleModel
from esbox.core import Config, ParallelTask
from esbox.problems import ProblemBase


# Step 1. Define your model class
class MyModel(PaddleModel):
    """Please define your model by inherting `TorchModel` or `PaddleModel` class
    according to the deep learning framework that you use.

    And please write out `self.intialization()` at the end of `__init__`.
    """

    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()

        # define your model layers
        self.fc = nn.Linear(input_dim, output_dim)

        # To get weights of your model,
        # please write out this function after defining your model layers
        self.initialization()

    def forward(self, input):
        prob = F.softmax(self.fc(input), axis=-1)
        return prob


# Step 2. Define your problem class
# Please decorate your problem class with `@parl.remote_class(wait=False)` to achieve an acceleration of evaluation.
@parl.remote_class(wait=False)
class CartPoleEnv(ProblemBase):
    """Please define your problem by inherting `ProblemBase` class.
    
    And please write the evaluation step in `evaluate()` function.
    """

    def __init__(self):
        super(CartPoleEnv, self).__init__()
        # definr your problem
        from esbox.utils.rl_wrappers import CompatWrapper
        env = gym.make('CartPole-v1')
        self.env = CompatWrapper(env)

        # Please instantiate the model here if use model
        self.model = MyModel(input_dim=self.env.observation_space.shape[0], output_dim=self.env.action_space.n)

    def evaluate(self, weight):
        # use `model.set_flat_weights(weight)` to set weight of model
        self.model.set_flat_weights(weight)

        # use the model to evaulate your problem
        total_reward = 0.
        steps = 0
        obs = self.env.reset()
        while True:
            obs = paddle.to_tensor(obs, dtype='float32')
            probs = self.model(obs)
            action = int(probs.argmax())
            obs, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += reward
            if done:
                break

        # Please return your results in a dictionary,
        # and place the evaluation result in `value` key (necessary)
        # other information in `info` key, e.g. steps (unnecessary)
        return {'value': total_reward, 'info': {"step": steps}}


# Step 3. Write your `config.ymal` before running
def main():
    cfg = Config(config_file=args.config_file, model_cls=MyModel)
    if args.seed is not None:
        cfg.hyparams['seed'] = args.seed
    tk = ParallelTask(config=cfg, eval_func=CartPoleEnv)
    result = tk.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='', help='config file')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    args.config_file = os.path.join(os.path.abspath("."), args.config_file)
    main()
