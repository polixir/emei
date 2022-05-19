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


class BaseDoubleCartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]], ABC):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, freq_rate=1):
        self.freq_rate = freq_rate

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 1.0
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.time_step = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        state_high = np.full(4, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-state_high, state_high, dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

    def _dsdt(self, s_augmented):
        x, x_dot, theta, theta_dot, force = s_augmented

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        return (x_dot, xacc, theta_dot, thetaacc, 0.0)

    @abstractmethod
    def _get_force(self, action):
        return 0.0

    @abstractmethod
    def _is_terminal(self):
        return False

    @abstractmethod
    def _get_reward(self):
        return 0.0

    def step(self, action):
        force = self._get_force(action)
        tau = self.time_step / self.freq_rate
        for i in range(self.freq_rate):
            s_augmented = np.append(self.state, force)
            x_dot, xacc, theta_dot, thetaacc, _ = self._dsdt(s_augmented)
            self.state += np.array([x_dot, xacc, theta_dot, thetaacc]) * tau

        return np.array(self.state, dtype=np.float32), self._get_reward(), self._is_terminal(), {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

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
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class CartPoleHoldingEnv(BaseCartPoleEnv):
    def __init__(self, freq_rate=1):
        super(CartPoleHoldingEnv, self).__init__(freq_rate=freq_rate)
        self.action_space = spaces.Discrete(2)

    def _get_force(self, action):
        return self.force_mag if action == 1 else -self.force_mag

    def _is_terminal(self):
        return not (-self.x_threshold < self.state[0] < self.x_threshold and
                    -self.theta_threshold_radians < self.state[2] < self.theta_threshold_radians)

    def _get_reward(self):
        return 1.0


class CartPoleSwingUpEnv(BaseCartPoleEnv):
    def __init__(self, freq_rate=1):
        super(CartPoleSwingUpEnv, self).__init__(freq_rate=freq_rate)
        self.x_threshold = 5
        self.action_space = spaces.Discrete(2)

    def _get_force(self, action):
        return self.force_mag if action == 1 else -self.force_mag

    def _is_terminal(self):
        return not (-self.x_threshold < self.state[0] < self.x_threshold)

    def _get_reward(self):
        return (math.cos(self.state[2]) + 1) / 2


if __name__ == '__main__':
    from emei.envs.util import random_policy_test
    env = CartPoleHoldingEnv()
    random_policy_test(env, is_render=True)
