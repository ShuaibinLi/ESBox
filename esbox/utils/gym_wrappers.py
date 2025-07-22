import time
import numpy as np

import ale_py
import gymnasium as gym
from gymnasium.wrappers import (
    TimeLimit,
    RecordEpisodeStatistics,
    NormalizeObservation,
    FrameStackObservation,
    AtariPreprocessing,
    TransformReward,
)

__all__ = ['wrap_mujoco', 'wrap_atari']


def wrap_mujoco(env, norm_obs=True):
    env = RecordEpisodeStatistics(env)
    if norm_obs:
        env = NormalizeObservation(env)
    return env


def wrap_atari(env, reward_func=None, noop_max=30, frame_skip=4, screen_size=84, stack_size=4):
    env = RecordEpisodeStatistics(env)
    if reward_func is not None:
        # """Bin reward to {+1, 0, -1} by its sign."""
        # env = TransformReward(env, np.sign)
        env = TransformReward(env, reward_func)
    if 'NoFrameskip' in env.spec.id and frame_skip > 1:
        frame_skip = frame_skip
    else:
        frame_skip = 1
    env = AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        # terminal_on_life_loss=True,
        screen_size=screen_size,
        # grayscale_obs=False,
        # grayscale_newaxis=False
    )
    if stack_size > 0:
        env = FrameStackObservation(env, stack_size=stack_size)
    return env


if __name__ == "__main__":
    env = gym.make("HalfCheetah-v5")
    env = wrap_mujoco(env)

    print(env)
    obs, info = env.reset()
    print(obs)
    rewards = 0.0
    step = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        rewards += reward
        step += 1
        if terminated or truncated:
            print(info, rewards, step)
            break

    # env = gym.make('ALE/Pong-v5')
    # env = wrap_atari(env, reward_func=np.sign)

    # # env = wrap_atari('BreakoutNoFrameskip-v4', reward_func=np.sign)
    # print(env)
    breakpoint()
    # obs, info = env.reset()
    # print(obs.shape)
    # while True:
    #     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    #     if terminated or truncated:
    #         print(info)
    #         break
