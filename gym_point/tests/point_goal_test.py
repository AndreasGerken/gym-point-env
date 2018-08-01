import pytest
import gym
import gym_point

from gym import error


def test_reset():
    # This test overtakes the tests from gym.GoalEnv.reset
    env = gym.make('PointGoalEnv-v0')

    if not isinstance(env.observation_space, gym.spaces.Dict):
        raise Exception.Error(
            'GoalEnv requires an observation space of type gym.spaces.Dict')

    result = env.reset()

    for key in ['observation', 'achieved_goal', 'desired_goal']:
        if key not in result:
            raise error.Error(
                'GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key))


def test_random_action():

    env = gym.make('PointGoalEnv-v0')
    print(env.seed())
    obs = env.reset()

    for _ in range(100):
        # env.render()
        obs, reward, done, info = env.step(
            env.action_space.sample())  # take a random action

        print ('observation: %s, reward: %s' % (obs, reward))
        if done:
            print ('found solution')
            env.close()
            break
