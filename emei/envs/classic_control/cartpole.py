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

import emei
from emei.envs.classic_control.base_control import BaseControlEnv


class BaseCartPoleEnv(BaseControlEnv):
    def __init__(self, freq_rate: int = 1, real_time_scale: float = 0.02, integrator: str = "euler", **kwargs):
        super(BaseCartPoleEnv, self).__init__(
            freq_rate=freq_rate, real_time_scale=real_time_scale, integrator=integrator, **kwargs
        )

        self.gravity = kwargs.get("gravity", 9.8)
        self.mass_cart = kwargs.get("mass_cart", 1.0)
        self.mass_pole = kwargs.get("mass_pole", 0.1)
        self.total_mass = self.mass_pole + self.mass_cart
        self.length = kwargs.get("length", 0.5)  # actually half the pole's length
        self.force_mag = kwargs.get("force_mag", 10.0)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.action_space = spaces.Discrete(2)
        high = np.array([np.inf, np.pi, np.inf, np.inf])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

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

    def state2obs(self, batch_state):
        batch_state = batch_state.copy()
        batch_state[:, 1] = (batch_state[:, 1] + np.pi) % (2 * np.pi) - np.pi
        return batch_state

    def obs2state(self, batch_obs, batch_extra_obs):
        state = batch_obs.copy()
        state[:, 1:2] = batch_extra_obs[:]
        return state

    def get_batch_extra_obs(self, batch_state):
        batch_state = batch_state.copy()
        return batch_state[:, 1].astype(np.float32)

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

    def get_batch_terminal(self, next_state, state=None, action=None):
        x = next_state[:, 0]
        theta = self.get_absolute_theta(next_state[:, 1])
        notdone = (np.abs(theta) < self.theta_threshold_radians) & (np.abs(x) < self.x_threshold)
        return np.logical_not(notdone).reshape([next_state.shape[0], 1])

    def get_batch_reward(self, next_state, state=None, action=None):
        return np.ones([next_state.shape[0], 1])


class CartPoleSwingUpEnv(BaseCartPoleEnv):
    def __init__(self, freq_rate: int = 1, real_time_scale: float = 0.02, integrator: str = "euler", **kwargs):
        super(CartPoleSwingUpEnv, self).__init__(
            freq_rate=freq_rate, real_time_scale=real_time_scale, integrator=integrator, **kwargs
        )
        self.x_threshold = 10

    def _extract_action(self, action):
        return self.force_mag if action == 1 else -self.force_mag

    def get_batch_terminal(self, next_state, state=None, action=None):
        x = next_state[:, 0]
        notdone = np.abs(x) < self.x_threshold
        return np.logical_not(notdone).reshape([next_state.shape[0], 1])

    def get_batch_reward(self, next_state, state=None, action=None):
        theta = self.get_absolute_theta(next_state[:, 1])
        rewards = (np.cos(theta) + 1) / 2
        return rewards.reshape([next_state.shape[0], 1])

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


class ParallelContinuousCartPoleSwingUpEnv(BaseControlEnv):
    def __init__(self, parallel_num=3, freq_rate: int = 1, real_time_scale: float = 0.02, integrator: str = "euler", **kwargs):
        self.parallel_num = parallel_num

        self.envs = [ContinuousCartPoleSwingUpEnv()] * parallel_num
        self.action_space = spaces.Box(-1, 1, shape=(parallel_num,), dtype=np.float32)
        high = np.array([np.inf, np.pi, np.inf, np.inf] * parallel_num)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._transition_graph = np.zeros((5 * parallel_num, 4 * parallel_num))
        for i in range(parallel_num):
            self._transition_graph[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4] = np.array(
                [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 1]]
            )
            self._transition_graph[4 * parallel_num + i, i * 4 : (i + 1) * 4] = np.array([0, 0, 1, 1])
        self._reward_mech_graph = None
        self._termination_graph = None

        super().__init__(
            self.observation_space,
            parallel_num=parallel_num,
            freq_rate=freq_rate,
            real_time_scale=real_time_scale,
            integrator=integrator,
            **kwargs
        )

    def _extract_action(self, action):
        return self.envs[0].force_mag * action

    def get_batch_terminal(self, next_state, state=None, action=None):
        ts = np.concatenate(
            [env.get_batch_terminal(next_state[:, i * 4 : (i + 1) * 4]) for i, env in enumerate(self.envs)], -1
        )
        return ts.all(-1).reshape(-1, 1)

    def get_batch_reward(self, next_state, state=None, action=None):
        rs = np.concatenate([env.get_batch_reward(next_state[:, i * 4 : (i + 1) * 4]) for i, env in enumerate(self.envs)], -1)
        return rs.sum(-1).reshape(-1, 1)

    def _dsdt(self, s_augmented):
        s_augmented_dot = np.empty(self.parallel_num * 5, dtype=np.float32)
        for i, env in enumerate(self.envs):
            indices = [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3, 4 * self.parallel_num + i]
            s_augmented_dot[indices] = env._dsdt(s_augmented[indices])
        return s_augmented_dot

    def get_batch_init_state(self, batch_size):
        obs = np.concatenate([env.get_batch_init_state(batch_size) for env in self.envs], -1)
        return obs

    def state2obs(self, batch_state):
        batch_state = np.concatenate(
            [env.state2obs(batch_state[:, i * 4 : (i + 1) * 4]) for i, env in enumerate(self.envs)], -1
        )
        return batch_state

    def obs2state(self, batch_obs, batch_extra_obs):
        batch_state = np.concatenate(
            [
                env.obs2state(
                    batch_obs[:, i * 4 : (i + 1) * 4],
                    batch_extra_obs[:, i : i + 1],
                )
                for i, env in enumerate(self.envs)
            ],
            -1,
        )
        return batch_state

    def get_batch_extra_obs(self, batch_state):
        batch_state = batch_state.copy()
        return batch_state[:, [4 * i + 1 for i in range(self.parallel_num)]].astype(np.float32)


if __name__ == "__main__":
    env = ParallelContinuousCartPoleSwingUpEnv(render_mode="human")
    obs, info = env.reset()

    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        # env.render()

    obs, reward, terminal, truncated, info = env.step(env.action_space.sample())
    state = env.obs2state(obs[None], info["next_extra_obs"][None])[0]
    print(obs, info, state)
