# Point environment for OpenAi gym

This is a simple environment, created for the use with OpenAi [gym](https://github.com/openai/gym) and [baselines](https://github.com/openai/baselines). It features an n dimensional state and action space. The PointGoalEnv also features a goal space which has the same dimensionality. All spaces are continuous.

The state of the environment is a point. The action is a move command for the point and the goal is a point with a margin. The reward function is binary and gives a reward, only if the goal was reached.

The environment was created with the help of these tutorials [[1](https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym
), [2](https://stackoverflow.com/questions/45068568/is-it-possible-to-create-a-new-gym-environment-in-openai
)].

## installation, usage and testing

After cloning, the package can be installed with.

```
pip install -e .
```

There are automatic tests to verify the installation.

```
pip install pytest
pytest
```

If an agent uses the environment it should `import gym_point`. The package is compatible to python3.

The following environments are available:
- PointEnv-v0
- PointGoalEnv-v0
