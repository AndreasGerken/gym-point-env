from gym.envs.registration import register

register(
    id='PointEnv-v0',
    entry_point='gym_point.envs:PointSimpleEnv',
    max_episode_steps=1000
)

register(
    id='PointGoalEnv-v0',
    entry_point='gym_point.envs:PointGoalEnv',
    max_episode_steps=1000
)
# register(
#     id='foo-extrahard-v0',
#     entry_point='gym_foo.envs:FooExtraHardEnv',
# )
