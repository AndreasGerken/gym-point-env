from gym.envs.registration import register

# The maximum required steps with a perfect policy is from corner to corner of
# the state space. Each dimension has the maximum speed so tha the goal can
# always be reached in 2 / max_speed steps. With max_speed = 0.05, it is 40.
# We provide double this steps so that a suboptimal agent can still reach the
# goal. The required steps for a perfect agent are dependent on the goal
# position and the initial state.

register(
    id='PointEnv-v0',
    entry_point='gym_point.envs:PointSimpleEnv',
    max_episode_steps=80
)

register(
    id='PointGoalEnv-v0',
    entry_point='gym_point.envs:PointGoalEnv',
    max_episode_steps=80
)
# register(
#     id='foo-extrahard-v0',
#     entry_point='gym_foo.envs:FooExtraHardEnv',
# )
