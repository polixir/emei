from typing import Tuple, Union, Dict

import numpy as np

from gym import utils
from gym.spaces import Box
from emei.envs.mujoco.mujoco_env import EmeiMujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class HopperRunningEnv(EmeiMujocoEnv, utils.EzPickle):
    def __init__(
        self,
        freq_rate: int = 4,
        real_time_scale: float = 0.002,
        integrator: str = "rk4",
        # noise
        init_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 5e-3,
        obs_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 0.0,
        # reward mech
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        # termination mech
        terminate_when_unhealthy: bool = True,
        # range
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            freq_rate,
            real_time_scale,
            integrator,
            init_noise_params,
            obs_noise_params,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            **kwargs
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward

        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        EmeiMujocoEnv.__init__(
            self,
            model_path="hopper.xml",
            observation_space=observation_space,
            freq_rate=freq_rate,
            real_time_scale=real_time_scale,
            integrator=integrator,
            camera_config=DEFAULT_CAMERA_CONFIG,
            init_noise_params=init_noise_params,
            obs_noise_params=obs_noise_params,
            **kwargs
        )

    def is_healthy(self, next_obs):
        z, angle = next_obs[:, 1:3].T
        state = next_obs[:, 2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state), axis=1)
        healthy_z = np.logical_and(min_z < z, z < max_z)
        healthy_angle = np.logical_and(min_angle < angle, angle < max_angle)

        is_healthy = np.logical_and(healthy_state, healthy_z, healthy_angle)

        return is_healthy

    def get_batch_reward(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        x_velocity = (obs[:, 0] - pre_obs[:, 0]) / self.dt
        forward_reward = self._forward_reward_weight * x_velocity
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        healthy_reward = np.logical_or(self.is_healthy(obs), self._terminate_when_unhealthy) * self._healthy_reward

        rewards = healthy_reward + forward_reward - control_cost
        return rewards.reshape([obs.shape[0], 1])

    def get_batch_terminal(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        terminals = ~np.logical_or(self.is_healthy(obs), self._terminate_when_unhealthy)
        return terminals.reshape([obs.shape[0], 1])
