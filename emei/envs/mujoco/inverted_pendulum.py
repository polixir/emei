import numpy as np
from gym import utils
from emei.envs.mujoco.mujoco_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {}


class BaseInvertedPendulumEnv(MujocoEnv, utils.EzPickle):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 # noise
                 reset_noise_scale=0.1):
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self,
                           model_path="inverted_pendulum.xml",
                           freq_rate=freq_rate,
                           time_step=time_step,
                           integrator=integrator,
                           camera_config=DEFAULT_CAMERA_CONFIG,
                           reset_noise_scale=reset_noise_scale, )

    def get_batch_obs(self, batch_state):
        qpos, qvel = batch_state[:, :self.model.nq], batch_state[:, self.model.nq:]
        batch_obs = np.concatenate([qpos[:, :1], np.sin(qpos[:, 1:]), np.cos(qpos[:, 1:]), qvel], axis=-1)
        return batch_obs

    def get_batch_state(self, batch_obs):
        theta = np.angle(1j * batch_obs[:, 1] + batch_obs[:, 2])
        return np.concatenate([batch_obs[:, :1], theta[:, None], batch_obs[:, -2:]], axis=-1)

    @property
    def causal_graph(self):
        return np.array([[0, 0, 0, 1, 0, 0],  # dot x
                         [0, 0, 1, 0, 1, 0],  # dot sin theta
                         [0, 1, 0, 0, 1, 0],  # dot cos theta
                         [0, 1, 1, 0, 1, 1],  # dot v
                         [0, 1, 1, 0, 1, 1],  # dot omega
                         [0, 0, 0, 0, 0, 0]])  # reward


class ReboundInvertedPendulumBalancingEnv(BaseInvertedPendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 # noise
                 reset_noise_scale=0.1):
        BaseInvertedPendulumEnv.__init__(self,
                                         freq_rate=freq_rate,
                                         time_step=time_step,
                                         integrator=integrator,
                                         reset_noise_scale=reset_noise_scale)

    def get_batch_reward(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        return np.ones([obs.shape[0], 1])

    def get_batch_terminal(self, obs: np.ndarray, pre_obs=None, action=None, state=None, pre_state=None):
        y = obs[:, 2]
        notdone = (y >= 0.9) & np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])


class BoundaryInvertedPendulumBalancingEnv(BaseInvertedPendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 # noise
                 reset_noise_scale=0.1):
        BaseInvertedPendulumEnv.__init__(self,
                                         freq_rate=freq_rate,
                                         time_step=time_step,
                                         integrator=integrator,
                                         reset_noise_scale=reset_noise_scale)

    def get_batch_reward(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        return np.ones([obs.shape[0], 1])

    def get_batch_terminal(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        x, y = obs[:, 0], obs[:, 2]
        x_left, x_right = self.model.jnt_range[0]
        notdone = (y >= 0.9) \
                  & np.logical_and(x_left < x, x < x_right) \
                  & np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])


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

    def get_batch_reward(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        y = obs[:, 2]
        rewards = (1 - y) / 2
        return rewards.reshape([obs.shape[0], 1])

    def get_batch_terminal(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        notdone = np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])

    @property
    def causal_graph(self):
        graph = super(ReboundInvertedPendulumSwingUpEnv, self).causal_graph.copy()
        graph[-1] = [0, 1, 1, 0, 1, 0]
        return graph


class BoundaryInvertedPendulumSwingUpEnv(BaseInvertedPendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 # reward weight
                 forward_reward_weight=1,
                 ctrl_cost_weight=0.1,
                 healthy_reward=1,
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

    def get_batch_reward(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        y = obs[:, 2]
        rewards = (1 - y) / 2
        return rewards.reshape([obs.shape[0], 1])

    def get_batch_terminal(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        x = obs[:, 0]
        x_left, x_right = self.model.jnt_range[0]
        notdone = np.logical_and(x_left < x, x < x_right) \
                  & np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])

    @property
    def causal_graph(self):
        graph = super(BoundaryInvertedPendulumSwingUpEnv, self).causal_graph.copy()
        graph[-1] = [0, 1, 1, 0, 1, 0]
        return graph


if __name__ == '__main__':
    from emei.util import random_policy_test
    from gym.wrappers import TimeLimit

    env = TimeLimit(ReboundInvertedPendulumSwingUpEnv(), max_episode_steps=1000, new_step_api=True)
    random_policy_test(env, is_render=True)

    # env = ReboundInvertedPendulumSwingUpEnv()
    # b_s = np.random.rand(5, 4)
    # b_o = env.get_batch_obs(b_s)
    # print(b_s)
    # print(env.get_batch_state(b_o))
    # print((b_s == env.get_batch_state(b_o)).all())
