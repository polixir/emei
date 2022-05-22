import math
import numpy as np
import pygame
from pygame import gfxdraw

from gym import spaces
from emei.envs.classic_control.base_control import BaseControlEnv


class BaseChargedBallEnv(BaseControlEnv):
    def __init__(self,
                 freq_rate=1,
                 time_step=0.02):
        super(BaseChargedBallEnv, self).__init__(freq_rate=freq_rate,
                                             time_step=time_step)
        self.gravity = 9.8
        self.mass_ball = 1.0
        self.radius = 1.0
        self.charge = 10.0
        self.time_step = 0.02  # seconds between state updates

        state_high = np.full(4, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-state_high, state_high, dtype=np.float32)

    def _ds_dt(self, s_augmented):
        x, y, v_x, v_y, electric_force = s_augmented
        if x ** 2 + y ** 2 < 1:
            return np.array(v_x, v_y, electric_force / self.mass_ball, -self.gravity)
        else:
            omega = v_y / x

        print(s_augmented)
        print(x ** 2 + y ** 2)


    def _get_initial_state(self):
        theta, omega = self.np_random.uniform(low=-0.05, high=0.05, size=(2,)) + [np.pi, 0]
        sin_theta_radius, cos_theta_radius = math.sin(theta) * self.radius, math.cos(theta) * self.radius
        return [sin_theta_radius, cos_theta_radius, - cos_theta_radius * omega, sin_theta_radius * omega]

    def _is_terminal(self):
        return False

    def render(self, mode="human"):
        pass


class ChargedBallEnv(BaseChargedBallEnv):
    def __init__(self, freq_rate=1, time_step=0.02):
        super(ChargedBallEnv, self).__init__(freq_rate=freq_rate, time_step=time_step)
        self.action_space = spaces.Discrete(2)

    def _extract_action(self, action):
        return self.charge if action == 1 else -self.charge

    def _get_reward(self):
        return 1.0


if __name__ == '__main__':
    env = ChargedBallEnv()

    obs = env.reset()
    print(obs)

    env.step(1)
