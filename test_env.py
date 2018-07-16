import gym
import gym_point
env = gym.make('PointEnv-v0')
env = gym.make('PointGoalEnv-v0')

print env.seed()
obs = env.reset()

for _ in range(100000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action

    print obs
    print reward
    #print done
    #print info
    if done:
        print 'found solution'
        break
