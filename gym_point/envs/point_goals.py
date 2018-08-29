#!/usr/bin/python


import gym
from gym import spaces, logger
from gym.utils import seeding

import numpy


class PointGoalEnv(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    reward_range = {0, 1}
    state_range = {-1, 1}
    action_range = {-1, 1}
    goal_range = {-1, 1}
    spec = None

    dimensions = 2
    state_space = spaces.Box(
        low=min(state_range), high=max(state_range), shape=(dimensions,), dtype=numpy.float32)
    goal_space = spaces.Box(low=min(goal_range), high=max(goal_range), shape=(
        dimensions,), dtype=numpy.float32)
    action_space = spaces.Box(
        low=min(action_range), high=max(action_range), shape=(dimensions,), dtype=numpy.float32)

    observation_space = spaces.Dict(dict(
        desired_goal=goal_space,
        achieved_goal=goal_space,
        observation=state_space
    ))

    def __init__(self):
        super(PointGoalEnv, self).__init__()

        self.dimensions = 2

        if self.dimensions <= 0:
            logger.error('The dimensions have to be at least 1')
        elif self.dimensions > 2:
            logger.warn(
                'The dimensions are bigger than 2, only the first 2 dimensions are visualized')

        self.max_speed = 0.05

        # The goal can either be fixed or set randomly with every reset
        self.set_goal_randomly = False
        self.goal = None
        self.goal_margin = 0.1

        self.viewer = None

        self.seed()
        self.reset()

    def seed(self, given_seed=None):
        self.np_random, seed = seeding.np_random(given_seed)

        # TODO: The seed should be passed from seeding
        gym.spaces.np_random.seed(given_seed)
        return [seed]

    def compute_reward(self, achieved_goal, desired_goal=None, info=None):
        desired_goal = self.goal if desired_goal is None else desired_goal

        distance = numpy.linalg.norm(
            achieved_goal - desired_goal, axis=-1)

        done = distance < self.goal_margin
        reward = done * 1.0

        return reward

    def step(self, action):
        self.state += action * self.max_speed
        self.state = numpy.clip(
            self.state, min(self.state_range), max(self.state_range))

        reward = self.compute_reward(self.state, self.goal, {})
        done = True if reward >= 1.0 else False

        return {'observation': self.state, 'achieved_goal': self.state, 'desired_goal': self.goal}, reward, done, dict(is_success=done)

    def reset(self):
        self.state = self.state_space.sample()

        if self.set_goal_randomly:
            self.goal = self.goal_space.sample()
        else:
            self.goal = numpy.zeros(self.dimensions)

        # TODO: The achieved goal has to be an array of achieved goals instead of one state
        return dict(observation=self.state, achieved_goal=self.state, desired_goal=self.goal)

    def render(self, mode='human', close=False):
        screen_width = 400
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width

        if self.viewer is None:
            # Borrowing rendering from classic control
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.pointtrans = rendering.Transform()
            point = rendering.make_circle(5)
            point.add_attr(self.pointtrans)
            point.set_color(.5, .5, .5)
            self.viewer.add_geom(point)

            self.goaltrans = rendering.Transform()
            goal = rendering.make_circle(self.goal_margin * scale)
            goal.add_attr(self.goaltrans)
            goal.set_color(0, 1, 0)
            self.viewer.add_geom(goal)

        if self.dimensions == 1:
            # If the environment is only one dimensional, add a dimension which
            # is 0 for rendering
            point = [self.state[0], 0]
            goal = [self.goal[0], 0]
        else:
            point = self.state
            goal = self.goal

        # print 'point'
        # print point
        # print 'goal'
        # print goal

        self.pointtrans.set_translation(
            (point[0] - self.min_position) * scale, (point[1] - self.min_position) * scale)
        self.goaltrans.set_translation(
            (goal[0] - self.min_position) * scale, (goal[1] - self.min_position) * scale)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
