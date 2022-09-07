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
        self.gravity_acc = 9.8
        self.mass_ball = 1.0
        self.radius = 1.0
        self.charge = 10.0
        self.time_step = 0.02  # seconds between state updates

        state_high = np.full(4, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-state_high, state_high, dtype=np.float32)

        self.screen_width = 600
        self.screen_height = 600

    def circle_to_free(self, circle_state):
        theta, omega = circle_state
        x, y = math.sin(theta) * self.radius, math.cos(theta) * self.radius
        return [x, y, omega * y, - omega * x]

    def _get_angle(self, x, y):
        scale = math.sqrt(x ** 2 + y ** 2)
        if y > 0:
            angle = math.asin(x / (scale * self.radius + 1e-8))
        else:
            angle = np.pi - math.asin(x / (scale * self.radius + 1e-8))
        return angle % (2 * np.pi)

    def _angle_greater(self, a1, a2):
        if abs(a1 - a2) < np.pi:
            return a1 > a2
        else:
            return a1 < a2

    def free_to_circle(self, free_state):
        x, y, v_x, v_y = free_state
        theta = self._get_angle(x, y)
        v_angle = self._get_angle(v_x, v_y)
        if self._angle_greater(v_angle, theta):
            omega = math.sqrt(v_x ** 2 + v_y ** 2) / self.radius
        else:
            omega = - math.sqrt(v_x ** 2 + v_y ** 2) / self.radius
        return [theta, omega]

    def update_state(self, updated):
        on_circle = self.state["on_circle"]
        if on_circle:
            ds_dt, flag = updated
            self.state["circle_state"] += ds_dt * (self.time_step / self.freq_rate)
            self.state["free_state"] = self.circle_to_free(self.state["circle_state"])
            if flag:
                self.state["on_circle"] = False
        else:
            self.state["free_state"] += updated * (self.time_step / self.freq_rate)
            if self.state["free_state"][0] ** 2 + self.state["free_state"][1] ** 2 > self.radius ** 2 + 0.001:
                self.state["on_circle"] = True
                self.state["circle_state"] = self.free_to_circle(self.state["free_state"])

    def _get_update_info(self, electric_force):
        on_circle = self.state["on_circle"]
        theta, omega = self.state["circle_state"]
        x, y, v_x, v_y = self.state["free_state"]
        if on_circle:
            sin_theta, cos_theta = math.sin(theta), math.cos(theta)
            centrifugal_force = self.mass_ball * omega ** 2 * self.radius
            gravity = self.mass_ball * self.gravity_acc
            theta_acc = (sin_theta * gravity + cos_theta * electric_force) / (self.mass_ball * self.radius)
            flag = centrifugal_force + sin_theta * electric_force < cos_theta * gravity
            return np.array([omega, theta_acc]), flag
        else:
            acc_x = electric_force / self.mass_ball
            acc_y = -self.gravity_acc
            return np.array([v_x, v_y, acc_x, acc_y])

    def _get_initial_state(self):
        circle_state = self.np_random.uniform(low=-0.5, high=0.5, size=(2,)) + [np.pi, 0]
        state = dict(on_circle=True,
                     circle_state=circle_state,
                     free_state=self.circle_to_free(circle_state))
        return state

    def _get_obs(self, state):
        return np.array(state["free_state"], dtype=np.float32)

    def set_state_by_obs(self, obs):
        self.state = dict(on_circle=True,
                          circle_state=np.empty(2, dtype=np.float32),
                          free_state=np.empty(4, dtype=np.float32))
        self.state["free_state"] = obs
        if obs[0] ** 2 + obs[1] ** 2 >= self.radius ** 2:
            self.state["on_circle"] = True
            self.state["circle_state"] = self.free_to_circle(self.state["free_state"])

    def get_batch_terminal(self, next_obs):
        return False

    def draw(self):
        world_width = self.radius * 3
        scale = self.screen_width / world_width
        ball_width = 5
        tube_width = self.radius * scale

        x, y, v_x, v_y = self.state["free_state"]

        self.surf.fill((255, 255, 255))

        ball_x = x * scale + self.screen_width / 2.0
        ball_y = y * scale + self.screen_height / 2.0

        gfxdraw.aacircle(
            self.surf,
            int(self.screen_width / 2),
            int(self.screen_height / 2),
            int(tube_width),
            (0, 0, 0)
        )

        gfxdraw.aacircle(
            self.surf,
            int(ball_x),
            int(ball_y),
            ball_width,
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(ball_x),
            int(ball_y),
            ball_width,
            (129, 132, 203),
        )


class ChargedBallCenteringEnv(BaseChargedBallEnv):
    def __init__(self, freq_rate=1, time_step=0.02):
        super(ChargedBallCenteringEnv, self).__init__(freq_rate=freq_rate, time_step=time_step)
        self.action_space = spaces.Discrete(2)

    def _extract_action(self, action):
        return self.charge if action == 1 else -self.charge

    def get_batch_reward(self, obs):
        x, y, v_x, v_y = obs
        return 1 - math.sqrt(x ** 2 + y ** 2) / self.radius


class ContinuousChargedBallCenteringEnv(BaseChargedBallEnv):
    def __init__(self, freq_rate=1, time_step=0.02):
        super(ContinuousChargedBallCenteringEnv, self).__init__(freq_rate=freq_rate, time_step=time_step)
        action_high = np.ones(1, dtype=np.float32)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

    def _extract_action(self, action):
        return self.charge * action[0]

    def get_batch_reward(self, obs):
        x, y, v_x, v_y = obs
        return 1 - math.sqrt(x ** 2 + y ** 2) / self.radius


if __name__ == '__main__':
    from emei.util import random_policy_test

    env = ContinuousChargedBallCenteringEnv()
    random_policy_test(env, is_render=True, default_action=[1])
