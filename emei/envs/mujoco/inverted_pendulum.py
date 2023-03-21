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
        utils.EzPickle.__init__(self, freq_rate, real_time_scale, integrator, init_noise_params, obs_noise_params,
                                **kwargs)

        high = np.array([np.inf, np.pi, np.inf, np.inf])
        observation_space = Box(low=-high, high=high, dtype=np.float64)
        state_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        EmeiMujocoEnv.__init__(
            self,
            model_path="inverted_pendulum.xml",
            observation_space=observation_space,
            state_space=state_space,
            freq_rate=freq_rate,
            real_time_scale=real_time_scale,
            integrator=integrator,
            camera_config=DEFAULT_CAMERA_CONFIG,
            init_noise_params=init_noise_params,
            obs_noise_params=obs_noise_params,
            **kwargs
        )

    def state2obs(self, batch_state):
        batch_state = batch_state.copy()
        batch_state[:, 1:2] = (batch_state[:, 1:2] + np.pi) % (2 * np.pi) - np.pi
        return batch_state

    def obs2state(self, batch_obs, batch_extra_obs):
        batch_state = batch_obs.copy()
        batch_state[:, 1:2] = batch_extra_obs[:]
        return batch_state

    def get_batch_extra_obs(self, batch_state):
        batch_state = batch_state.copy()
        return batch_state[:, 1:2].astype(np.float32)

    @property
    def _transition_graph(self):
        return np.array(
            [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 1]]  # x  # theta  # v  # omega  # action
        )

    @property
    def _reward_mech_graph(self):
        return np.empty(0)

    @property
    def _termination_mech_graph(self):
        return np.empty(0)


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

    def get_batch_reward(self, next_state=None, state=None, action=None):
        return np.ones([next_state.shape[0], 1])

    def get_batch_terminal(self, next_state: np.ndarray, state=None, action=None):
        y = np.cos(next_state[:, 1])
        notdone = (y >= 0.9) & np.isfinite(next_state).all(axis=1)
        return np.logical_not(notdone).reshape([next_state.shape[0], 1])


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

    def get_batch_reward(self, next_state, state=None, action=None):
        return np.ones([next_state.shape[0], 1])

    def get_batch_terminal(self, next_state, state=None, action=None):
        y = np.cos(next_state[:, 1])
        x = next_state[:, 0]
        x_left, x_right = self.model.jnt_range[0]
        notdone = (y >= 0) & np.logical_and(x_left < x, x < x_right) & np.isfinite(next_state).all(axis=1)
        return np.logical_not(notdone).reshape([next_state.shape[0], 1])


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

    def get_batch_reward(self, next_state, state=None, action=None):
        y = np.cos(next_state[:, 1])
        rewards = (1 - y) / 2
        return rewards.reshape([next_state.shape[0], 1])

    def get_batch_terminal(self, next_state, state=None, action=None):
        notdone = np.isfinite(next_state).all(axis=1)
        return np.logical_not(notdone).reshape([next_state.shape[0], 1])


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

    def get_batch_reward(self, next_state, state=None, action=None):
        y = np.cos(next_state[:, 1])
        rewards = (1 - y) / 2
        return rewards.reshape([next_state.shape[0], 1])

    def get_batch_terminal(self, next_state, state=None, action=None):
        x = next_state[:, 0]
        x_left, x_right = self.model.jnt_range[0]
        notdone = np.logical_and(x_left < x, x < x_right) & np.isfinite(next_state).all(axis=1)
        return np.logical_not(notdone).reshape([next_state.shape[0], 1])


if __name__ == "__main__":
    env = BoundaryInvertedPendulumSwingUpEnv()
    env.reset()

    for i in range(50000):
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)

    obs, reward, terminal, truncated, info = env.step(np.ones(1, dtype=np.float32))
    state = env.obs2state(obs[None], info["next_extra_obs"][None])[0]
    print(obs, info, state)
