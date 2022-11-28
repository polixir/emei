__credits__ = ["Rushiv Arora"]

from typing import Tuple, Union, Dict

import numpy as np

from gym import utils
from gym.spaces import Box
from emei.envs.mujoco.mujoco_env import EmeiMujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetahRunningEnv(EmeiMujocoEnv, utils.EzPickle):
    def __init__(
        self,
        freq_rate: int = 4,
        real_time_scale: float = 0.002,
        integrator="euler",
        # weight
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        # noise
        init_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 0.1,
        obs_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 0.0,
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
            **kwargs
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        observation_space = Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)
        EmeiMujocoEnv.__init__(
            self,
            model_path="half_cheetah.xml",
            observation_space=observation_space,
            freq_rate=freq_rate,
            real_time_scale=real_time_scale,
            integrator=integrator,
            camera_config=DEFAULT_CAMERA_CONFIG,
            init_noise_params=init_noise_params,
            obs_noise_params=obs_noise_params,
            **kwargs
        )

    def get_batch_reward(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        forward_reward = self._forward_reward_weight * (obs[:, 0] - pre_obs[:, 0]) / self.dt
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        rewards = forward_reward - control_cost
        return rewards.reshape([obs.shape[0], 1])

    def get_batch_terminal(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        notdone = np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])
