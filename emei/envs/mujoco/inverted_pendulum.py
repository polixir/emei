import numpy as np
from gym import utils
from emei.envs.mujoco.base_mujoco import BaseMujocoEnv

DEFAULT_CAMERA_CONFIG = {}


class BaseInvertedPendulumEnv(BaseMujocoEnv, utils.EzPickle):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 # noise
                 reset_noise_scale=0.2):
        utils.EzPickle.__init__(self)
        BaseMujocoEnv.__init__(self,
                               model_path="inverted_pendulum.xml",
                               freq_rate=freq_rate,
                               time_step=time_step,
                               integrator=integrator,
                               camera_config=DEFAULT_CAMERA_CONFIG,
                               reset_noise_scale=reset_noise_scale, )

    @property
    def causal_graph(self):
        return np.array([[0, 0, 1, 0, 0],  # dot x
                         [0, 0, 0, 1, 0],  # dot theta
                         [0, 1, 1, 1, 1],  # dot v
                         [0, 1, 0, 1, 1],  # dot omega
                         [0, 0, 0, 0, 0]])  # reward


class ReboundInvertedPendulumBalancingEnv(BaseInvertedPendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 # noise
                 reset_noise_scale=0.05):
        BaseInvertedPendulumEnv.__init__(self,
                                         freq_rate=freq_rate,
                                         time_step=time_step,
                                         integrator=integrator,
                                         reset_noise_scale=reset_noise_scale)

    def get_batch_reward_by_next_obs(self, next_obs, pre_obs=None, action=None):
        return np.ones([next_obs.shape[0], 1])

    def get_batch_terminal_by_next_obs(self, next_obs, pre_obs=None, action=None):
        notdone = (np.abs(next_obs[:, 1]) <= 0.2) & np.isfinite(next_obs).all(axis=1)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])


class BoundaryInvertedPendulumBalancingEnv(BaseInvertedPendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 # noise
                 reset_noise_scale=0.05):
        BaseInvertedPendulumEnv.__init__(self,
                                         freq_rate=freq_rate,
                                         time_step=time_step,
                                         integrator=integrator,
                                         reset_noise_scale=reset_noise_scale)

    def get_batch_reward_by_next_obs(self, next_obs, pre_obs=None, action=None):
        return np.ones([next_obs.shape[0], 1])

    def get_batch_terminal_by_next_obs(self, next_obs, pre_obs=None, action=None):
        x_left, x_right = self.model.jnt_range[0]
        notdone = (np.abs(next_obs[:, 1]) <= 0.2) \
                  & np.logical_and(x_left < next_obs[:, 0], next_obs[:, 0] < x_right) \
                  & np.isfinite(next_obs).all(axis=1)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])


class ReboundInvertedPendulumSwingUpEnv(BaseInvertedPendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 # noise
                 reset_noise_scale=0.2):
        BaseInvertedPendulumEnv.__init__(self,
                                         freq_rate=freq_rate,
                                         time_step=time_step,
                                         integrator=integrator,
                                         reset_noise_scale=reset_noise_scale)

    def _update_model(self):
        self.model.stat.extent = 4
        self.model.geom_size[0][1] = 2
        self.model.jnt_range[0] = [-2, 2]
        self.model.jnt_range[1] = [-np.inf, np.inf]
        self.model.body_quat[2] = [0, 0, 1, 0]

    def get_batch_reward_by_next_obs(self, next_obs, pre_obs=None, action=None):
        rewards = (1 - np.cos(next_obs[:, 1])) / 2
        return rewards.reshape([next_obs.shape[0], 1])

    def get_batch_terminal_by_next_obs(self, next_obs, pre_obs=None, action=None):
        notdone = np.isfinite(next_obs).all(axis=1)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])

    @property
    def causal_graph(self):
        graph = super(ReboundInvertedPendulumSwingUpEnv, self).causal_graph.copy()
        graph[-1] = [0, 1, 0, 1, 0]
        return graph


class BoundaryInvertedPendulumSwingUpEnv(BaseInvertedPendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 # noise
                 reset_noise_scale=0.2):
        BaseInvertedPendulumEnv.__init__(self,
                                         freq_rate=freq_rate,
                                         time_step=time_step,
                                         integrator=integrator,
                                         reset_noise_scale=reset_noise_scale)

    def _update_model(self):
        self.model.stat.extent = 4
        self.model.geom_size[0][1] = 2
        self.model.jnt_range[0] = [-2, 2]
        self.model.jnt_range[1] = [-np.inf, np.inf]
        self.model.body_quat[2] = [0, 0, 1, 0]

    def get_batch_reward_by_next_obs(self, next_obs, pre_obs=None, action=None):
        rewards = (1 - np.cos(next_obs[:, 1])) / 2
        return rewards.reshape([next_obs.shape[0], 1])

    def get_batch_terminal_by_next_obs(self, next_obs, pre_obs=None, action=None):
        x_left, x_right = self.model.jnt_range[0]
        notdone = np.logical_and(x_left < next_obs[:, 0], next_obs[:, 0] < x_right) \
                  & np.isfinite(next_obs).all(axis=1)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])

    @property
    def causal_graph(self):
        graph = super(BoundaryInvertedPendulumSwingUpEnv, self).causal_graph.copy()
        graph[-1] = [0, 1, 0, 1, 0]
        return graph


if __name__ == '__main__':
    from emei.util import random_policy_test
    from gym.wrappers import TimeLimit

    env = TimeLimit(ReboundInvertedPendulumSwingUpEnv(), max_episode_steps=1000)
    random_policy_test(env, is_render=True)
