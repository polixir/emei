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
            freq_rate: int = 1,
            real_time_scale: float = 0.02,
            integrator="euler",
            # noise
            init_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 0.1,
            obs_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 0.0,
            # reward mech
            forward_reward_weight=1.0,
            ctrl_cost_weight=0.1,
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

        observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
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

    def state2obs(self, batch_state):
        batch_obs = batch_state[:, 1:].copy()
        return batch_obs

    def obs2state(self, batch_obs, batch_extra_obs):
        batch_state = np.concatenate([batch_extra_obs, batch_obs], axis=1)
        return batch_state

    def get_batch_extra_obs(self, batch_state):
        return batch_state[:, 0:1].copy()

    def get_batch_reward(self, next_state, state=None, action=None):
        forward_reward = self._forward_reward_weight * (next_state[:, 0] - state[:, 0]) / self.dt
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        rewards = forward_reward - control_cost
        return rewards.reshape([next_state.shape[0], 1])

    def get_batch_terminal(self, next_state, state=None, action=None):
        notdone = np.isfinite(next_state).all(axis=1)
        return np.logical_not(notdone).reshape([next_state.shape[0], 1])

if __name__ == "__main__":
    env = HalfCheetahRunningEnv(render_mode="human")
    obs, info = env.reset()

    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        env.render()

    obs, reward, terminal, truncated, info = env.step(env.action_space.sample())
    state = env.obs2state(obs[None], info["next_extra_obs"][None])[0]
    print(obs, info, state)
