import pytest
import gym.spaces
import gym
import gym_point

import numpy as np


def test_random_action():

    env = gym.make('PointEnv-v0')
    print(env.seed())
    obs = env.reset()

    for _ in range(100):
        # env.render()
        obs, reward, done, info = env.step(
            env.action_space.sample())  # take a random action

        print('observation: %s, reward: %s' % (obs, reward))
        if done:
            print('found solution')
            env.close()
            break


def test_step():
    env = gym.make('PointEnv-v0')

    goal = np.zeros(2)
    state_ones = np.ones_like(goal)

    action_no_move = np.zeros_like(goal)
    action_move_positive = np.ones_like(goal)
    action_move_negative = np.ones_like(goal) * -1

    env.reset()

    # Access variables directly through the wrapper layer (TimeLimit)
    env.env.state = state_ones
    env.env.goal_position = goal

    # Without action, the state should not change
    state, reward, done, info = env.step(action_no_move)
    assert np.array_equal(state_ones, state)

    # Since the state space is limited to [-1, 1], nothing should change if the
    # agent tries to run out.
    state, reward, done, info = env.step(action_move_positive)
    assert np.array_equal(state_ones, state)

    # Finally the point moves with the maximum speed
    state, reward, done, info = env.step(action_move_negative)
    assert np.array_equal(
        (state_ones + env.env.max_speed * action_move_negative), state)


def test_reward_done():

    env = gym.make('PointEnv-v0')

    goal = np.zeros(2)

    state_not_reached = np.ones_like(goal)
    state_reached = np.zeros_like(goal)

    action_no_move = np.zeros_like(goal)

    # Test return when goal was not reached
    reward, done = env.compute_reward(state_not_reached, goal, {})
    assert reward == 0
    assert not done

    # Start episode
    env.reset()

    # Access variables directly through the wrapper layer (TimeLimit)
    env.env.state = state_not_reached
    env.env.goal_position = goal
    state, reward, done, info = env.step(action_no_move)

    assert np.array_equal(state, state_not_reached)
    assert reward == 0
    assert not done
    assert info == {}

    env.env.state = state_reached
    state, reward, done, info = env.step(action_no_move)
    assert reward == 1
    assert done

    env.close()
