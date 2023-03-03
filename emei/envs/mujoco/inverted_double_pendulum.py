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

        high = np.array([np.inf, np.pi, np.pi, np.inf, np.inf, np.inf])
        observation_space = Box(low=-high, high=high, dtype=np.float64)
        state_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        EmeiMujocoEnv.__init__(
            self,
            "inverted_double_pendulum.xml",
            observation_space,
            state_space,
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

    def state2obs(self, batch_state):
        batch_state = batch_state.copy()
        batch_state[:, 1:3] = (batch_state[:, 1:3] + np.pi) % (2 * np.pi) - np.pi
        return batch_state

    def obs2state(self, batch_obs, batch_extra_obs):
        batch_state = batch_obs.copy()
        batch_state[:, 1:3] = batch_extra_obs[:]
        return batch_state

    def get_batch_extra_obs(self, batch_state):
        batch_state = batch_state.copy()
        return batch_state[:, 1:3].astype(np.float32)


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

    def get_batch_reward(self, next_state, state=None, action=None):
        return np.ones([next_state.shape[0], 1])

    def get_batch_terminal(self, next_obs: np.ndarray, obs=None, action=None, next_state=None, state=None):
        y = np.cos(next_obs[:, 1]) + np.cos(next_obs[:, 1] + next_obs[:, 2])
        notdone = (y >= 1.5) & np.isfinite(next_obs).all(axis=1)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])


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

    def get_batch_reward(self, next_state, state=None, action=None):
        return np.ones([next_state.shape[0], 1])

    def get_batch_terminal(self, next_obs, obs=None, action=None, next_state=None, state=None):
        y = np.cos(next_obs[:, 1]) + np.cos(next_obs[:, 1] + next_obs[:, 2])
        x = next_obs[:, 0]
        x_left, x_right = self.model.jnt_range[0]
        notdone = (y >= 0) & np.logical_and(x_left < x, x < x_right) & np.isfinite(next_obs).all(axis=1)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])


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

    def get_batch_reward(self, next_state, state=None, action=None):
        y = np.cos(next_state[:, 1]) + np.cos(next_state[:, 1] + next_state[:, 2])
        omega1, omega2 = next_state[:, -2:].T
        vel_penalty = 5e-3 * omega1 ** 2 + 1e-4 * omega2 ** 2
        rewards = (2 - y) / 4 - vel_penalty
        return rewards.reshape([next_state.shape[0], 1])

    def get_batch_terminal(self, next_state, state=None, action=None):
        notdone = np.isfinite(next_state).all(axis=1)
        return np.logical_not(notdone).reshape([next_state.shape[0], 1])


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

    def get_batch_reward(self, next_state, state=None, action=None):
        y = np.cos(next_state[:, 1]) + np.cos(next_state[:, 1] + next_state[:, 2])
        omega1, omega2 = next_state[:, -2:].T
        vel_penalty = 5e-2 * omega1**2 + 1e-2 * omega2**2
        rewards = (2 - y) / 4 - vel_penalty
        return rewards.reshape([next_state.shape[0], 1])

    def get_batch_terminal(self, next_state, state=None, action=None):
        x = next_state[:, 0]
        x_left, x_right = self.model.jnt_range[0]
        notdone = np.logical_and(x_left < x, x < x_right) & np.isfinite(next_state).all(axis=1)
        return np.logical_not(notdone).reshape([next_state.shape[0], 1])



if __name__ == "__main__":
    env = BoundaryInvertedDoublePendulumSwingUpEnv()
    env.reset()

    for i in range(50000):
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)

    obs, reward, terminal, truncated, info = env.step(np.ones(1, dtype=np.float32))
    state = env.obs2state(obs[None], info["next_extra_obs"][None])[0]
    print(obs, info, state)
