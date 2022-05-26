import numpy as np
from gym import utils
from emei.envs.mujoco.base_mujoco import BaseMujocoEnv
import math
import os


class BaseInvertedDoublePendulumEnv(BaseMujocoEnv, utils.EzPickle):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler"):
        BaseMujocoEnv.__init__(self,
                               model_path="inverted_double_pendulum.xml",
                               freq_rate=freq_rate,
                               time_step=time_step,
                               integrator=integrator)
        utils.EzPickle.__init__(self)

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1,
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.1225


class ReboundInvertedDoublePendulumHoldingEnv(BaseInvertedDoublePendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler"):
        BaseInvertedDoublePendulumEnv.__init__(self,
                                               freq_rate=freq_rate,
                                               time_step=time_step,
                                               integrator=integrator)

    def get_single_reward_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs
        y = math.cos(theta1) + math.cos(theta1 + theta2)
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        vel_penalty = 1e-3 * omega1 ** 2 + 5e-3 * omega2 ** 2
        alive_bonus = 10
        return (alive_bonus - dist_penalty - vel_penalty) / 10

    def get_single_terminal_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs
        y = math.cos(theta1) + math.cos(theta1 + theta2)
        notdone = np.isfinite(next_obs).all() and y > 1.5 and (abs(omega1) < 1e5 and abs(omega2) < 1e5)
        return not notdone


class BoundaryInvertedDoublePendulumHoldingEnv(BaseInvertedDoublePendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler"):
        BaseInvertedDoublePendulumEnv.__init__(self,
                                               freq_rate=freq_rate,
                                               time_step=time_step,
                                               integrator=integrator)

    def get_single_reward_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs
        y = math.cos(theta1) + math.cos(theta1 + theta2)
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        vel_penalty = 1e-3 * omega1 ** 2 + 5e-3 * omega2 ** 2
        alive_bonus = 10
        return (alive_bonus - dist_penalty - vel_penalty) / 10

    def get_single_terminal_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs
        x_left, x_right = self.model.jnt_range[0]
        y = math.cos(theta1) + math.cos(theta1 + theta2)
        notdone = np.isfinite(next_obs).all() and (x_left < x < x_right) and y > 1.5 and (
                abs(omega1) < 1e5 and abs(omega2) < 1e5)
        return not notdone


class ReboundInvertedDoublePendulumSwingUpEnv(BaseInvertedDoublePendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler"):
        BaseInvertedDoublePendulumEnv.__init__(self,
                                               freq_rate=freq_rate,
                                               time_step=time_step,
                                               integrator=integrator)

    def _update_model(self):
        self.model.stat.extent = 10
        self.model.geom_size[1][1] = 3
        self.model.jnt_range[0] = [-3, 3]
        self.model.jnt_range[1] = [-np.inf, np.inf]

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        qpos[1] += np.pi
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_single_reward_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs
        y = math.cos(theta1) + math.cos(theta1 + theta2)
        vel_penalty = 0.1 * abs(omega1) + 0.1 * abs(omega2)
        return (y + 2 - vel_penalty) / 4

    def get_single_terminal_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs
        notdone = np.isfinite(next_obs).all() and (abs(omega1) < 1e5 and abs(omega2) < 1e5)
        return not notdone


class BoundaryInvertedDoublePendulumSwingUpEnv(BaseInvertedDoublePendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler"):
        BaseInvertedDoublePendulumEnv.__init__(self,
                                               freq_rate=freq_rate,
                                               time_step=time_step,
                                               integrator=integrator)

    def _update_model(self):
        self.model.stat.extent = 10
        self.model.geom_size[1][1] = 3
        self.model.jnt_range[0] = [-3, 3]
        self.model.jnt_range[1] = [-np.inf, np.inf]

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        qpos[1] += np.pi
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_single_reward_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs
        y = math.cos(theta1) + math.cos(theta1 + theta2)
        vel_penalty = 0.1 * abs(omega1) + 0.1 * abs(omega2)
        return (y + 2 - vel_penalty) / 4

    def get_single_terminal_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs
        x_left, x_right = self.model.jnt_range[0]
        notdone = np.isfinite(next_obs).all() and (x_left < x < x_right) and (abs(omega1) < 1e5 and abs(omega2) < 1e5)
        return not notdone


if __name__ == '__main__':
    from emei.util import random_policy_test

    env = ReboundInvertedDoublePendulumSwingUpEnv()
    random_policy_test(env, is_render=True)
