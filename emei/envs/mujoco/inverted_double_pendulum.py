import numpy as np
from gym import utils
from emei.envs.mujoco.mujoco_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.1225)),
}


class BaseInvertedDoublePendulumEnv(MujocoEnv, utils.EzPickle):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 # noise
                 reset_noise_scale=0.1):
        MujocoEnv.__init__(self,
                           model_path="inverted_double_pendulum.xml",
                           freq_rate=freq_rate,
                           time_step=time_step,
                           integrator=integrator,
                           camera_config=DEFAULT_CAMERA_CONFIG,
                           reset_noise_scale=reset_noise_scale, )
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        theta1, theta2 = self.data.qpos[1:]
        return np.concatenate(
            [
                self.data.qpos[:1],  # cart x pos
                [np.sin(theta1), np.cos(theta1), np.sin(theta2), np.cos(theta2)],
                self.data.qvel
            ]
        ).ravel()

    def _restore_pos_vel_from_obs(self, obs):
        theta1 = np.angle(1j * obs[1] + obs[2])
        theta2 = np.angle(1j * obs[3] + obs[4])
        return np.array([obs[0], theta1, theta2]), obs[-3:]

    @property
    def causal_graph(self):
        return np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0],  # dot x
                         [0, 0, 0, 0, 1, 0, 0],  # dot sin theta1
                         [0, 0, 0, 0, 0, 1, 0],  # dot cos theta1
                         [0, 0, 0, 0, 1, 0, 0],  # dot sin theta2
                         [0, 0, 0, 0, 0, 1, 0],  # dot cos theta2
                         [0, 1, 1, 0, 1, 1, 1],  # dot v
                         [0, 1, 1, 0, 1, 1, 1],  # dot omega1
                         [0, 1, 1, 0, 1, 1, 1],  # dot omega2
                         [0, 0, 0, 0, 0, 0, 0]])  # reward


class ReboundInvertedDoublePendulumBalancingEnv(BaseInvertedDoublePendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 # noise
                 reset_noise_scale=0.1):
        BaseInvertedDoublePendulumEnv.__init__(self,
                                               freq_rate=freq_rate,
                                               time_step=time_step,
                                               integrator=integrator,
                                               reset_noise_scale=reset_noise_scale)

    def get_batch_reward(self, next_obs, pre_obs=None, action=None):
        return np.ones([next_obs.shape[0], 1])

    def get_batch_terminal(self, next_obs, pred_obs=None, action=None):
        x, theta1, theta2, v, omega1, omega2 = next_obs.T
        y = np.cos(theta1) + np.cos(theta1 + theta2)
        notdone = np.isfinite(next_obs).all(axis=1) & (y > 1.5)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])


class BoundaryInvertedDoublePendulumBalancingEnv(BaseInvertedDoublePendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 # noise
                 reset_noise_scale=0.1):
        BaseInvertedDoublePendulumEnv.__init__(self,
                                               freq_rate=freq_rate,
                                               time_step=time_step,
                                               integrator=integrator,
                                               reset_noise_scale=reset_noise_scale)

    def get_batch_reward(self, next_obs, pre_obs=None, action=None):
        return np.ones([next_obs.shape[0], 1])

    def get_batch_terminal(self, next_obs, pred_obs=None, action=None):
        x_left, x_right = self.model.jnt_range[0]
        x, theta1, theta2, v, omega1, omega2 = next_obs.T
        y = np.cos(theta1) + np.cos(theta1 + theta2)
        notdone = np.isfinite(next_obs).all(axis=1) \
                  & (y > 1.5) \
                  & np.logical_and(x_left < x, x < x_right)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])


class ReboundInvertedDoublePendulumSwingUpEnv(BaseInvertedDoublePendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 # noise
                 reset_noise_scale=0.2):
        BaseInvertedDoublePendulumEnv.__init__(self,
                                               freq_rate=freq_rate,
                                               time_step=time_step,
                                               integrator=integrator,
                                               reset_noise_scale=reset_noise_scale)

    def _update_model(self):
        self.model.stat.extent = 10
        self.model.geom_size[1][1] = 3
        self.model.jnt_range[0] = [-3, 3]
        self.model.jnt_range[1] = [-np.inf, np.inf]
        self.model.body_quat[2] = [0, 0, 1, 0]

    def get_batch_reward(self, next_obs, pre_obs=None, action=None):
        x, theta1, theta2, v, omega1, omega2 = next_obs.T
        y = np.cos(theta1) + np.cos(theta1 + theta2)
        rewards = (2 - y) / 4
        return rewards.reshape([next_obs.shape[0], 1])

    def get_batch_terminal(self, next_obs, pred_obs=None, action=None):
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
                 integrator="standard_euler",
                 # noise
                 reset_noise_scale=0.2):
        BaseInvertedDoublePendulumEnv.__init__(self,
                                               freq_rate=freq_rate,
                                               time_step=time_step,
                                               integrator=integrator,
                                               reset_noise_scale=reset_noise_scale)

    def _update_model(self):
        self.model.stat.extent = 10
        self.model.geom_size[1][1] = 3
        self.model.jnt_range[0] = [-3, 3]
        self.model.jnt_range[1] = [-np.inf, np.inf]
        self.model.body_quat[2] = [0, 0, 1, 0]

    def get_batch_reward(self, next_obs, pre_obs=None, action=None):
        x, theta1, theta2, v, omega1, omega2 = next_obs.T
        y = np.cos(theta1) + np.cos(theta1 + theta2)
        rewards = (2 - y) / 4
        return rewards.reshape([next_obs.shape[0], 1])

    def get_batch_terminal(self, next_obs, pre_obs=None, action=None):
        x_left, x_right = self.model.jnt_range[0]
        x, theta1, theta2, v, omega1, omega2 = next_obs.T
        notdone = np.isfinite(next_obs).all(axis=1) \
                  & np.logical_and(x_left < x, x < x_right)
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
