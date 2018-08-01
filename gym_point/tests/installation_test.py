import pytest


def test_installation():
    import gym
    import gym_point
    env = gym.make('PointEnv-v0')
    env = gym.make('PointGoalEnv-v0')
