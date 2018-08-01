import pytest
import gym
import gym_point


def test_random_action():

    env = gym.make('PointEnv-v0')
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
