#!/usr/bin/python

import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding

import numpy as np

class PointGoalEnv(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.dimensions = 2

        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.0
        self.max_position = 1.0
        self.max_speed = 0.07

        # Fixed goal to 0.5 in all dimensions. For variable goal use PointGoalEnv.
        self.goal = None
        self.goal_margin = 0.1

        self.viewer = None

        self.goal_space = spaces.Dict({'position':spaces.Box(
            low=self.min_position, high=self.max_position, shape=(self.dimensions,), dtype=np.float32)
            })
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(self.dimensions,), dtype=np.float32)
        self.observation_space = spaces.Dict({'position':spaces.Box(
            low=self.min_position, high=self.max_position, shape=(self.dimensions,), dtype=np.float32)
            })
        if self.dimensions <= 0:
            logger.error('The dimensions have to be at least 1')
        elif self.dimensions > 2:
            logger.warn('The dimensions are bigger than 2, only the first 2 dimensions are visualized')

        self.seed()
        self.reset()

    def seed(self, given_seed=None):
        self.np_random, seed = seeding.np_random(given_seed)

        # TODO: The seed should be passed from seeding
        gym.spaces.np_random.seed(given_seed)
        return [seed]

    def compute_reward(self, achieved_goal, desired_goal, info):

        distance = np.linalg.norm(achieved_goal['position'] - desired_goal['position'])

        done = distance < self.goal_margin
        reward = done * 1.0

        return reward, done

    def step(self, action):
        self.state['position'] += action * self.max_speed
        self.state['position'] = np.clip(self.state['position'], self.min_position, self.max_position)

        reward, done = self.compute_reward(self.state, self.goal, {})

        return self.state, reward, done, {}

    def reset(self):
        self.state = self.observation_space.sample()
        self.goal = self.goal_space.sample()
        return self.state

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
            # If the environment is only one dimensional, add a dimension which is 0 for rendering
            point = [self.state['position'][0], 0]
            goal = [self.goal['position'][0], 0]
        else:
            point = self.state['position']
            goal = self.goal['position']

        # print 'point'
        # print point
        # print 'goal'
        # print goal

        self.pointtrans.set_translation(
            (point[0] - self.min_position) * scale, (point[1] - self.min_position) * scale)
        self.goaltrans.set_translation(
            (goal[0] - self.min_position) * scale, (goal[1] - self.min_position) * scale)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
