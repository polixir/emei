from typing import Tuple, Union, Dict

import numpy as np
from gym import utils
from gym.spaces import Box

from emei.envs.mujoco.mujoco_env import EmeiMujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.1225)),
}


class BaseInvertedDoublePendulumEnv(EmeiMujocoEnv, utils.EzPickle):
    def __init__(
        self,
        freq_rate: int = 1,
        real_time_scale: float = 0.02,
        integrator="euler",
        # noise
        init_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 5e-3,
        obs_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 0.0,
        **kwargs
    ):
        utils.EzPickle.__init__(self, freq_rate, real_time_scale, integrator, init_noise_params, obs_noise_params, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        EmeiMujocoEnv.__init__(
            self,
            model_path="inverted_double_pendulum.xml",
            observation_space=observation_space,
            freq_rate=freq_rate,
            real_time_scale=real_time_scale,
            integrator=integrator,
            camera_config=DEFAULT_CAMERA_CONFIG,
            init_noise_params=init_noise_params,
            obs_noise_params=obs_noise_params,
            **kwargs
        )

        self._causal_graph = np.array(
            [
                [0, 0, 0, 0, 0, 0],  # x
                [0, 0, 0, 1, 1, 1],  # theta1
                [0, 0, 0, 1, 1, 1],  # theta2
                [1, 0, 0, 0, 0, 0],  # v
                [0, 1, 0, 1, 1, 1],  # omega1
                [0, 0, 1, 1, 1, 1],  # omega2
                [0, 0, 0, 1, 1, 1],  # action
            ]
        )
        self._reward_mech_graph = None
        self._termination_graph = None

    @property
    def current_obs(self):
        state = self.state_vector().copy()
        state[1:3] = (state[1:3] + np.pi) % 2 * np.pi - np.pi
        return state


class ReboundInvertedDoublePendulumBalancingEnv(BaseInvertedDoublePendulumEnv):
    def __init__(
        self,
        freq_rate: int = 1,
        real_time_scale: float = 0.02,
        integrator="euler",
        # noise
        init_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 5e-3,
        obs_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 0.0,
        **kwargs
    ):
        BaseInvertedDoublePendulumEnv.__init__(
            self,
            freq_rate=freq_rate,
            real_time_scale=real_time_scale,
            integrator=integrator,
            init_noise_params=init_noise_params,
            obs_noise_params=obs_noise_params,
            **kwargs
        )

    def get_batch_reward(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        return np.ones([obs.shape[0], 1])

    def get_batch_terminal(self, obs: np.ndarray, pre_obs=None, action=None, state=None, pre_state=None):
        y = np.cos(obs[:, 1]) + np.cos(obs[:, 1] + obs[:, 2])
        notdone = (y >= 1.5) & np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])


class BoundaryInvertedDoublePendulumBalancingEnv(BaseInvertedDoublePendulumEnv):
    def __init__(
        self,
        freq_rate: int = 1,
        real_time_scale: float = 0.02,
        integrator="euler",
        # noise
        init_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 5e-3,
        obs_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 0.0,
        **kwargs
    ):
        BaseInvertedDoublePendulumEnv.__init__(
            self,
            freq_rate=freq_rate,
            real_time_scale=real_time_scale,
            integrator=integrator,
            init_noise_params=init_noise_params,
            obs_noise_params=obs_noise_params,
            **kwargs
        )

    def get_batch_reward(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        return np.ones([obs.shape[0], 1])

    def get_batch_terminal(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        y = np.cos(obs[:, 1]) + np.cos(obs[:, 1] + obs[:, 2])
        x = obs[:, 0]
        x_left, x_right = self.model.jnt_range[0]
        notdone = (y >= 0) & np.logical_and(x_left < x, x < x_right) & np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])


class ReboundInvertedDoublePendulumSwingUpEnv(BaseInvertedDoublePendulumEnv):
    def __init__(
        self,
        freq_rate: int = 1,
        real_time_scale: float = 0.02,
        integrator="euler",
        # noise
        init_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 5e-3,
        obs_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 0.0,
        **kwargs
    ):
        BaseInvertedDoublePendulumEnv.__init__(
            self,
            freq_rate=freq_rate,
            real_time_scale=real_time_scale,
            integrator=integrator,
            init_noise_params=init_noise_params,
            obs_noise_params=obs_noise_params,
            **kwargs
        )

    def _update_model(self):
        self.model.jnt_range[1] = [-np.inf, np.inf]
        self.model.body_quat[2] = [0, 0, 1, 0]

    def get_batch_reward(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        y = np.cos(obs[:, 1]) + np.cos(obs[:, 1] + obs[:, 2])
        rewards = (2 - y) / 4
        return rewards.reshape([obs.shape[0], 1])

    def get_batch_terminal(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        notdone = np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])


class BoundaryInvertedDoublePendulumSwingUpEnv(BaseInvertedDoublePendulumEnv):
    def __init__(
        self,
        freq_rate: int = 1,
        real_time_scale: float = 0.02,
        integrator="euler",
        # noise
        init_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 5e-3,
        obs_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 0.0,
        **kwargs
    ):
        BaseInvertedDoublePendulumEnv.__init__(
            self,
            freq_rate=freq_rate,
            real_time_scale=real_time_scale,
            integrator=integrator,
            init_noise_params=init_noise_params,
            obs_noise_params=obs_noise_params,
            **kwargs
        )

    def _update_model(self):
        self.model.jnt_range[1] = [-np.inf, np.inf]
        self.model.body_quat[2] = [0, 0, 1, 0]

    def get_batch_reward(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        y = np.cos(obs[:, 1]) + np.cos(obs[:, 1] + obs[:, 2])
        omega1, omega2 = obs[:, -2:].T
        vel_penalty = 5e-3 * omega1**2 + 1e-4 * omega2**2
        rewards = (2 - y) / 4 - vel_penalty
        return rewards.reshape([obs.shape[0], 1])

    def get_batch_terminal(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        x = obs[:, 0]
        x_left, x_right = self.model.jnt_range[0]
        notdone = np.logical_and(x_left < x, x < x_right) & np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])
