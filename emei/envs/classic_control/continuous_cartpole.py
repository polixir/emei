"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union
from abc import abstractmethod, ABC

import numpy as np
import pygame
from pygame import gfxdraw

import gym
from gym import spaces, logger
from gym.utils import seeding

from emei.envs.classic_control.cartpole import BaseCartPoleEnv

class ContinuousCartPoleHoldingEnv(BaseCartPoleEnv):
    def __init__(self, freq_rate=1):
        super(ContinuousCartPoleHoldingEnv, self).__init__(freq_rate=freq_rate)
        action_high = np.ones(1, dtype=np.float32)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

    def _get_force(self, action):
        return self.force_mag * action[0]

    def _is_terminal(self):
        return not (-self.x_threshold < self.state[0] < self.x_threshold and
                    -self.theta_threshold_radians < self.state[2] < self.theta_threshold_radians)

    def _get_reward(self):
        return 1.0


class CartPoleSwingUpEnv(BaseCartPoleEnv):
    def __init__(self, freq_rate=1):
        super(CartPoleSwingUpEnv, self).__init__(freq_rate=freq_rate)
        self.x_threshold = 5
        action_high = np.ones(1, dtype=np.float32)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

    def _get_force(self, action):
        return self.force_mag * action[0]

    def _is_terminal(self):
        return not (-self.x_threshold < self.state[0] < self.x_threshold)

    def _get_reward(self):
        return (math.cos(self.state[2]) + 1) / 2


if __name__ == '__main__':
    from emei.envs.util import random_policy_test
    env = CartPoleSwingUpEnv()
    random_policy_test(env, is_render=True)
