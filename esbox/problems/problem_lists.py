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
    'InvertedPendulum-v5',
    'InvertedDoublePendulum-v5',
    'Reacher-v5',
    'Pusher-v5',
    'HalfCheetah-v5',
    'Hopper-v5',
    'Walker2d-v5',
    'Swimmer-v5',
    'Ant-v5',
    'Humanoid-v5',
    'HumanoidStandup-v5',
    'InvertedPendulum-v4',
    'InvertedDoublePendulum-v4',
    'Reacher-v4',
    'HalfCheetah-v4',
    'Hopper-v4',
    'Walker2d-v4',
    'Swimmer-v4',
    'Ant-v4',
    'Humanoid-v4',
    'HumanoidStandup-v4',
}

ATARI_PROBLEM = {
    'Pong-v0',
    'PongNoFrameskip-v4',
    'BreakoutNoFrameskip-v4',
    'ALE/Pong-v5',              # pip install ale-py, https://ale.farama.org/getting-started/#gymnasium-api
}
