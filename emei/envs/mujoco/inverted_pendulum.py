from typing import Tuple, Union, Dict

import numpy as np
from gym import utils
from gym.spaces import Box

from emei.envs.mujoco.mujoco_env import EmeiMujocoEnv

DEFAULT_CAMERA_CONFIG = {}


class BaseInvertedPendulumEnv(EmeiMujocoEnv, utils.EzPickle):
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

        high = np.array([np.inf, np.pi, np.inf, np.inf])
        observation_space = Box(low=-high, high=high, dtype=np.float64)
        EmeiMujocoEnv.__init__(
            self,
            model_path="inverted_pendulum.xml",
            observation_space=observation_space,
            freq_rate=freq_rate,
            real_time_scale=real_time_scale,
            integrator=integrator,
            camera_config=DEFAULT_CAMERA_CONFIG,
            init_noise_params=init_noise_params,
            obs_noise_params=obs_noise_params,
            **kwargs
        )

        self._transition_graph = np.array(
            [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 1]]  # x  # theta  # v  # omega  # action
        )
        self._reward_mech_graph = None
        self._termination_graph = None

    @property
    def current_obs(self):
        state = self.state_vector().copy()
        state[1] = (state[1] + np.pi) % (2 * np.pi) - np.pi
        return state.astype(np.float32)


class ReboundInvertedPendulumBalancingEnv(BaseInvertedPendulumEnv):
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
        BaseInvertedPendulumEnv.__init__(
            self,
            freq_rate=freq_rate,
            real_time_scale=real_time_scale,
            integrator=integrator,
            init_noise_params=init_noise_params,
            obs_noise_params=obs_noise_params,
            **kwargs
        )

    def get_batch_reward(self, next_obs, obs=None, action=None, next_state=None, state=None):
        return np.ones([next_obs.shape[0], 1])

    def get_batch_terminal(self, next_obs: np.ndarray, obs=None, action=None, next_state=None, state=None):
        y = np.cos(next_obs[:, 1])
        notdone = (y >= 0.9) & np.isfinite(next_obs).all(axis=1)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])


class BoundaryInvertedPendulumBalancingEnv(BaseInvertedPendulumEnv):
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
        BaseInvertedPendulumEnv.__init__(
            self,
            freq_rate=freq_rate,
            real_time_scale=real_time_scale,
            integrator=integrator,
            init_noise_params=init_noise_params,
            obs_noise_params=obs_noise_params,
            **kwargs
        )

    def get_batch_reward(self, next_obs, obs=None, action=None, next_state=None, state=None):
        return np.ones([next_obs.shape[0], 1])

    def get_batch_terminal(self, next_obs, obs=None, action=None, next_state=None, state=None):
        y = np.cos(next_obs[:, 1])
        x = next_obs[:, 0]
        x_left, x_right = self.model.jnt_range[0]
        notdone = (y >= 0) & np.logical_and(x_left < x, x < x_right) & np.isfinite(next_obs).all(axis=1)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])


class ReboundInvertedPendulumSwingUpEnv(BaseInvertedPendulumEnv):
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
        BaseInvertedPendulumEnv.__init__(
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

    def get_batch_reward(self, next_obs, obs=None, action=None, next_state=None, state=None):
        y = np.cos(next_obs[:, 1])
        rewards = (1 - y) / 2
        return rewards.reshape([next_obs.shape[0], 1])

    def get_batch_terminal(self, next_obs, obs=None, action=None, next_state=None, state=None):
        notdone = np.isfinite(next_obs).all(axis=1)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])


class BoundaryInvertedPendulumSwingUpEnv(BaseInvertedPendulumEnv):
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
        BaseInvertedPendulumEnv.__init__(
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

    def get_batch_reward(self, next_obs, obs=None, action=None, next_state=None, state=None):
        y = np.cos(next_obs[:, 1])
        rewards = (1 - y) / 2
        return rewards.reshape([next_obs.shape[0], 1])

    def get_batch_terminal(self, next_obs, obs=None, action=None, next_state=None, state=None):
        x = next_obs[:, 0]
        x_left, x_right = self.model.jnt_range[0]
        notdone = np.logical_and(x_left < x, x < x_right) & np.isfinite(next_obs).all(axis=1)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])
