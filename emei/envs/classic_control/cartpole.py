"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np
import pygame
from pygame import gfxdraw

from gym import spaces
from emei.envs.classic_control.base_control import BaseControlEnv


class BaseCartPoleEnv(BaseControlEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self,
                 freq_rate=1,
                 time_step=0.02):
        super(BaseCartPoleEnv, self).__init__(freq_rate=freq_rate,
                                              time_step=time_step)
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_pole + self.mass_cart
        self.length = 0.5  # actually half the pole's length
        self.force_mag = 10.0
        self.time_step = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        state_high = np.full(4, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-state_high, state_high, dtype=np.float32)

    def _ds_dt(self, s_augmented):
        x, x_dot, theta, theta_dot, force = s_augmented

        pole_mass_length = self.mass_pole * self.length
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        temp = (force + pole_mass_length * theta_dot ** 2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
                self.length * (4.0 / 3.0 - self.mass_pole * cos_theta ** 2 / self.total_mass))
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / self.total_mass

        return np.array([x_dot, x_acc, theta_dot, theta_acc, 0.0], dtype=np.float32)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
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
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
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

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.is_open


class CartPoleHoldingEnv(BaseCartPoleEnv):
    def __init__(self, freq_rate=1, time_step=0.02):
        super(CartPoleHoldingEnv, self).__init__(freq_rate=freq_rate, time_step=time_step)
        self.action_space = spaces.Discrete(2)

    def _extract_action(self, action):
        return self.force_mag if action == 1 else -self.force_mag

    def _is_terminal(self):
        return not (-self.x_threshold < self.state[0] < self.x_threshold and
                    -self.theta_threshold_radians < self.state[2] < self.theta_threshold_radians)

    def _get_initial_state(self):
        return self.np_random.uniform(low=-0.05, high=0.05, size=(4,))

    def _get_reward(self):
        return 1.0


class CartPoleSwingUpEnv(BaseCartPoleEnv):
    def __init__(self, freq_rate=1, time_step=0.02):
        super(CartPoleSwingUpEnv, self).__init__(freq_rate=freq_rate, time_step=time_step)
        self.x_threshold = 5
        self.action_space = spaces.Discrete(2)

    def _extract_action(self, action):
        return self.force_mag if action == 1 else -self.force_mag

    def _is_terminal(self):
        return not (-self.x_threshold < self.state[0] < self.x_threshold)

    def _get_initial_state(self):
        return self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) + [0, 0, np.pi, 0]

    def _get_reward(self):
        return (math.cos(self.state[2]) + 1) / 2


class ContinuousCartPoleHoldingEnv(BaseCartPoleEnv):
    def __init__(self, freq_rate=1, time_step=0.02):
        super(ContinuousCartPoleHoldingEnv, self).__init__(freq_rate=freq_rate, time_step=time_step)
        action_high = np.ones(1, dtype=np.float32)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

    def _extract_action(self, action):
        return self.force_mag * action[0]

    def _is_terminal(self):
        return not (-self.x_threshold < self.state[0] < self.x_threshold and
                    -self.theta_threshold_radians < self.state[2] < self.theta_threshold_radians)

    def _get_initial_state(self):
        return self.np_random.uniform(low=-0.05, high=0.05, size=(4,))

    def _get_reward(self):
        return 1.0


class ContinuousCartPoleSwingUpEnv(BaseCartPoleEnv):
    def __init__(self, freq_rate=1, time_step=0.02):
        super(ContinuousCartPoleSwingUpEnv, self).__init__(freq_rate=freq_rate, time_step=time_step)
        self.x_threshold = 5
        action_high = np.ones(1, dtype=np.float32)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

    def _extract_action(self, action):
        return self.force_mag * action[0]

    def _is_terminal(self):
        return not (-self.x_threshold < self.state[0] < self.x_threshold)

    def _get_initial_state(self):
        return self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) + [0, 0, np.pi, 0]

    def _get_reward(self):
        return (math.cos(self.state[2]) + 1) / 2


if __name__ == '__main__':
    from emei.util import random_policy_test

    env = ContinuousCartPoleSwingUpEnv()
    # env.freeze()
    random_policy_test(env, is_render=True)
