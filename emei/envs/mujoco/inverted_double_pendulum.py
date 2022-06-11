import numpy as np
from gym import utils
from emei.envs.mujoco.base_mujoco import BaseMujocoEnv
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

    def x_in_range(self, x):
        x_left, x_right = self.model.jnt_range[0]
        return (x_left < x) & (x < x_right)

    @property
    def causal_graph(self):
        return np.array([[0, 0, 0, 1, 0, 0, 0],  # dot x
                         [0, 0, 0, 0, 1, 0, 0],  # dot theta1
                         [0, 0, 0, 0, 0, 1, 0],  # dot theta2
                         [0, 1, 1, 0, 1, 1, 1],  # dot v
                         [0, 1, 1, 0, 1, 1, 1],  # dot v
                         [0, 1, 1, 0, 1, 1, 1],  # dot omega
                         [0, 0, 0, 0, 0, 0, 0]])  # reward


class ReboundInvertedDoublePendulumHoldingEnv(BaseInvertedDoublePendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler"):
        BaseInvertedDoublePendulumEnv.__init__(self,
                                               freq_rate=freq_rate,
                                               time_step=time_step,
                                               integrator=integrator)

    def get_batch_reward_by_next_obs(self, next_obs):
        return np.ones([next_obs.shape[0], 1])

    def get_batch_terminal_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs.T
        y = np.cos(theta1) + np.cos(theta1 + theta2)
        notdone = np.isfinite(next_obs).all(axis=1) & (y > 1.5)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])


class BoundaryInvertedDoublePendulumHoldingEnv(BaseInvertedDoublePendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler"):
        BaseInvertedDoublePendulumEnv.__init__(self,
                                               freq_rate=freq_rate,
                                               time_step=time_step,
                                               integrator=integrator)

    def get_batch_reward_by_next_obs(self, next_obs):
        return np.ones([next_obs.shape[0], 1])

    def get_batch_terminal_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs.T
        y = np.cos(theta1) + np.cos(theta1 + theta2)
        notdone = np.isfinite(next_obs).all(axis=1) & (y > 1.5) & self.x_in_range(x)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])


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

    def get_batch_reward_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs.T
        y = np.cos(theta1) + np.cos(theta1 + theta2)
        rewards = (y + 2) / 4
        return rewards.reshape([next_obs.shape[0], 1])

    def get_batch_terminal_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs.T
        notdone = np.isfinite(next_obs).all(axis=1)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])

    @property
    def causal_graph(self):
        graph = super(ReboundInvertedDoublePendulumSwingUpEnv, self).causal_graph.copy()
        graph[-1] = [0, 1, 1, 0, 1, 1, 0]
        return graph


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

    def get_batch_reward_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs.T
        y = np.cos(theta1) + np.cos(theta1 + theta2)
        rewards = (y + 2) / 4
        return rewards.reshape([next_obs.shape[0], 1])

    def get_batch_terminal_by_next_obs(self, next_obs):
        x, theta1, theta2, v, omega1, omega2 = next_obs.T
        notdone = np.isfinite(next_obs).all(axis=1) & self.x_in_range(x)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])

    @property
    def causal_graph(self):
        graph = super(BoundaryInvertedDoublePendulumSwingUpEnv, self).causal_graph.copy()
        graph[-1] = [0, 1, 1, 0, 1, 1, 0]
        return graph


if __name__ == '__main__':
    from emei.util import random_policy_test

    env = ReboundInvertedDoublePendulumSwingUpEnv()
    random_policy_test(env, is_render=True)
