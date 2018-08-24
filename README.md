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
### PointEnv-v0

The point environment has the standard 2 dimensional state and action space. The action is a move command to the state. The goal is a two dimensional circle with a position and a fixed margin. The goal position is set randomly with the initialization.


The environment is compatible with [DDPG](https://github.com/openai/baselines/tree/master/baselines/ddpg). Convergence is not yet proven.
```
 python3 -m baselines.ddpg.main --env-id PointEnv-v0
 ```

### PointGoalEnv-v0
This environment is an environment with variable goals. It implements gym.GoalEnv with all the required functions.

The environment is compatible with [HER](https://github.com/openai/baselines/tree/master/baselines/her). Convergence is not yet proven.
```
python3 -m baselines.her.experiment.train --env PointGoalEnv-v0 --num_cpu 3
```
