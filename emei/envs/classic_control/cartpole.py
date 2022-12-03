"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math

import numpy as np
import pygame
from pygame import gfxdraw

from gym import spaces
from emei.envs.classic_control.base_control import BaseControlEnv


class BaseCartPoleEnv(BaseControlEnv):
    def __init__(self, freq_rate: int = 1, real_time_scale: float = 0.02, integrator: str = "euler", **kwargs):
        super(BaseCartPoleEnv, self).__init__(
            freq_rate=freq_rate, real_time_scale=real_time_scale, integrator=integrator, **kwargs
        )

        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_pole + self.mass_cart
        self.length = 0.5  # actually half the pole's length
        self.force_mag = 10.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

        self._transition_graph = np.array(
            [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 1]]  # x  # theta  # v  # omega  # action
        )
        self._reward_mech_graph = None
        self._termination_graph = None

    def _dsdt(self, s_augmented):
        x, theta, x_dot, theta_dot, force = s_augmented

        pole_mass_length = self.mass_pole * self.length
        cos_theta = math.cos(self.get_absolute_theta(theta))
        sin_theta = math.sin(self.get_absolute_theta(theta))
        temp = (force + pole_mass_length * theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / self.total_mass

        return np.array([x_dot, theta_dot, x_acc, theta_acc, 0], dtype=np.float32)

    def get_absolute_theta(self, theta):
        return theta

    def get_batch_init_state(self, batch_size):
        return self.np_random.uniform(low=-0.05, high=0.05, size=(batch_size, 4)).astype(np.float32)

    @property
    def current_obs(self):
        state = self.state.copy()
        state[1] = (state[1] + np.pi) % (2 * np.pi) - np.pi
        return state

    def draw(self):
        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        x = self.state
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-self.get_absolute_theta(x[1]))
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))


class CartPoleBalancingEnv(BaseCartPoleEnv):
    def __init__(self, freq_rate: int = 1, real_time_scale: float = 0.02, integrator: str = "euler", **kwargs):
        super(CartPoleBalancingEnv, self).__init__(
            freq_rate=freq_rate, real_time_scale=real_time_scale, integrator=integrator, **kwargs
        )

    def _extract_action(self, action):
        return self.force_mag if action == 1 else -self.force_mag

    def get_batch_terminal(self, next_obs, obs=None, action=None, next_state=None, state=None):
        x = next_obs[:, 0]
        theta = self.get_absolute_theta(next_obs[:, 1])
        notdone = (np.abs(theta) < self.theta_threshold_radians) & (np.abs(x) < self.x_threshold)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])

    def get_batch_reward(self, next_obs, obs=None, action=None, next_state=None, state=None):
        return np.ones([next_obs.shape[0], 1])


class CartPoleSwingUpEnv(BaseCartPoleEnv):
    def __init__(self, freq_rate: int = 1, real_time_scale: float = 0.02, integrator: str = "euler", **kwargs):
        super(CartPoleSwingUpEnv, self).__init__(
            freq_rate=freq_rate, real_time_scale=real_time_scale, integrator=integrator, **kwargs
        )
        self.x_threshold = 10

    def _extract_action(self, action):
        return self.force_mag if action == 1 else -self.force_mag

    def get_batch_terminal(self, next_obs, obs=None, action=None, next_state=None, state=None):
        x = next_obs[:, 0]
        notdone = np.abs(x) < self.x_threshold
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])

    def get_batch_reward(self, next_obs, obs=None, action=None, next_state=None, state=None):
        theta = self.get_absolute_theta(next_obs[:, 1])
        rewards = (np.cos(theta) + 1) / 2
        return rewards.reshape([next_obs.shape[0], 1])

    def get_absolute_theta(self, theta):
        return theta + np.pi


class ContinuousCartPoleBalancingEnv(CartPoleBalancingEnv):
    def __init__(self, freq_rate: int = 1, real_time_scale: float = 0.02, integrator: str = "euler", **kwargs):
        super(ContinuousCartPoleBalancingEnv, self).__init__(
            freq_rate=freq_rate, real_time_scale=real_time_scale, integrator=integrator, **kwargs
        )
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def _extract_action(self, action):
        return self.force_mag * action[0]


class ContinuousCartPoleSwingUpEnv(CartPoleSwingUpEnv):
    def __init__(self, freq_rate: int = 1, real_time_scale: float = 0.02, integrator: str = "euler", **kwargs):
        super(ContinuousCartPoleSwingUpEnv, self).__init__(
            freq_rate=freq_rate, real_time_scale=real_time_scale, integrator=integrator, **kwargs
        )
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def _extract_action(self, action):
        return self.force_mag * action[0]
