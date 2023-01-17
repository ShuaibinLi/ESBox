#
# Summary of all problems [To be supplemented and perfected]
#
# Including:
#     1. single-objective functions
#     2. mujoco problems
#     3. atari problems
#

__all__ = ['SO_FUCNTION', 'MUJOCO_PROBLEM', 'ATARI_PROBLEM']

# single-objective functions: ackley, griewank, zakharov, rastrigin, rosenbrock
"""
Example: TODO
.. code-block:: python
    from esbox.problems import Func_Problem

    func = Func_Problem('ackley', dim=2, )
    x = [0.1, 0.1]
    score = func(x)
"""
SO_FUCNTION = {'ackley', 'griewank', 'zakharov', 'rastrigin', 'rosenbrock'}

MUJOCO_PROBLEM = {
    'Ant-v2',
    'HalfCheetah-v2',
    'Hopper-v2',
    'HumanoidStandup-v2',
    'Humanoid-v2',
    'InvertedDoublePendulum-v2',
    'Invertedpendulum-v2',
    'Reacher-v2',
    'Swimmer-v2',
    'Walker2D-v2',
    'Ant-v4',
    'HalfCheetah-v4',
    'Hopper-v4',
    'HumanoidStandup-v4',
    'Humanoid-v4',
    'InvertedDoublePendulum-v4',
    'Invertedpendulum-v4',
    'Reacher-v4',
    'Swimmer-v4',
    'Walker2D-v4',
}

ATARI_PROBLEM = {
    'Pong-v4',
    'PongNoFrameskip-v4',
    'BreakoutNoFrameskip-v4',
}
