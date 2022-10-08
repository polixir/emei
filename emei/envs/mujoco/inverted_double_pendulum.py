import numpy as np
from gym import utils
from emei.envs.mujoco.mujoco_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.1225)),
}


class BaseInvertedDoublePendulumEnv(MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        freq_rate: int = 1,
        time_step: float = 0.02,
        integrator="standard_euler",
        # obs
        sin_cos: bool = True,
        # noise
        reset_noise_scale=0.1,
    ):
        self.sin_cos = sin_cos

        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(
            self,
            model_path="inverted_double_pendulum.xml",
            freq_rate=freq_rate,
            time_step=time_step,
            integrator=integrator,
            camera_config=DEFAULT_CAMERA_CONFIG,
            reset_noise_scale=reset_noise_scale,
        )

        if self.sin_cos:
            self._causal_graph = np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],  # x
                    [0, 1, 1, 1, 1, 1, 1, 1],  # sin theta1
                    [0, 1, 1, 1, 1, 1, 1, 1],  # cos theta1
                    [0, 1, 1, 1, 1, 1, 1, 1],  # sin theta2
                    [0, 1, 1, 1, 1, 1, 1, 1],  # cos theta2
                    [1, 0, 0, 0, 0, 0, 0, 0],  # v
                    [0, 1, 1, 1, 1, 1, 1, 1],  # omega1
                    [0, 0, 1, 1, 1, 1, 1, 1],  # omega2
                    [0, 0, 0, 1, 1, 1, 1, 1],
                ]
            )  # action
        else:
            self._causal_graph = np.array(
                [
                    [0, 0, 0, 0, 0, 0],  # x
                    [0, 0, 0, 1, 1, 1],  # theta1
                    [0, 0, 0, 1, 1, 1],  # theta2
                    [1, 0, 0, 0, 0, 0],  # v
                    [0, 1, 0, 1, 1, 1],  # omega1
                    [0, 0, 1, 1, 1, 1],  # omega2
                    [0, 0, 0, 1, 1, 1],
                ]
            )  # action

    def get_batch_obs(self, batch_state):
        if self.sin_cos:
            qpos, qvel = (
                batch_state[:, : self.model.nq],
                batch_state[:, self.model.nq :],
            )
            theta1, theta2 = qpos[:, 1:].T
            theta2 += theta1
            batch_obs = np.concatenate(
                [
                    qpos[:, :1],
                    np.sin(theta1)[:, None],
                    np.cos(theta1)[:, None],
                    np.sin(theta2)[:, None],
                    np.cos(theta2)[:, None],
                    qvel,
                ],
                axis=-1,
            )
        else:
            batch_obs = batch_state.copy()
        return batch_obs

    def get_batch_state(self, batch_obs):
        if self.sin_cos:
            theta1 = np.angle(1j * batch_obs[:, 1] + batch_obs[:, 2])
            theta2 = np.angle(1j * batch_obs[:, 3] + batch_obs[:, 4]) - theta1
            batch_state = np.concatenate(
                [batch_obs[:, :1], theta1[:, None], theta2[:, None], batch_obs[:, -3:]],
                axis=-1,
            )
        else:
            batch_state = batch_obs.copy()
        return batch_state

    def get_batch_init_state(self, batch_size):
        pos_random = self.np_random.standard_normal((batch_size, self.model.nq))
        y1 = np.cos(pos_random[:, 1])
        y2 = np.cos(pos_random[:, 1] + pos_random[:, 2])
        while not ((y1 > 0).all() and (y2 > 0.5).all()):
            pos_random = self.np_random.standard_normal((batch_size, self.model.nq))
            y1 = np.cos(pos_random[:, 1])
            y2 = np.cos(pos_random[:, 1] + pos_random[:, 2])
        pos_random[:, 0] = 0
        qpos = self._reset_noise_scale * pos_random
        qvel = self._reset_noise_scale * self.np_random.standard_normal(
            (batch_size, self.model.nv)
        )
        batch_state = np.concatenate([qpos, qvel], axis=-1)
        return batch_state

    def get_pole_height(self, obs):
        if self.sin_cos:
            y = obs[:, 2] + obs[:, 4]
        else:
            theta1, theta2 = obs[:, 1], obs[:, 2]
            y = np.cos(theta1) + np.cos(theta1 + theta2)
        return y


class ReboundInvertedDoublePendulumBalancingEnv(BaseInvertedDoublePendulumEnv):
    def __init__(
        self,
        freq_rate: int = 1,
        time_step: float = 0.02,
        integrator="standard_euler",
        # obs
        sin_cos: bool = False,
        # noise
        reset_noise_scale=0.1,
    ):
        BaseInvertedDoublePendulumEnv.__init__(
            self,
            freq_rate=freq_rate,
            time_step=time_step,
            integrator=integrator,
            sin_cos=sin_cos,
            reset_noise_scale=reset_noise_scale,
        )

    def get_batch_reward(
        self, obs, pre_obs=None, action=None, state=None, pre_state=None
    ):
        return np.ones([obs.shape[0], 1])

    def get_batch_terminal(
        self, obs: np.ndarray, pre_obs=None, action=None, state=None, pre_state=None
    ):
        y = self.get_pole_height(obs)
        notdone = (y >= 1.5) & np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])


class BoundaryInvertedDoublePendulumBalancingEnv(BaseInvertedDoublePendulumEnv):
    def __init__(
        self,
        freq_rate: int = 1,
        time_step: float = 0.02,
        integrator="standard_euler",
        # obs
        sin_cos: bool = False,
        # noise
        reset_noise_scale=0.3,
    ):
        BaseInvertedDoublePendulumEnv.__init__(
            self,
            freq_rate=freq_rate,
            time_step=time_step,
            integrator=integrator,
            sin_cos=sin_cos,
            reset_noise_scale=reset_noise_scale,
        )

    def get_batch_reward(
        self, obs, pre_obs=None, action=None, state=None, pre_state=None
    ):
        return np.ones([obs.shape[0], 1])

    def get_batch_terminal(
        self, obs, pre_obs=None, action=None, state=None, pre_state=None
    ):
        y = self.get_pole_height(obs)
        x = obs[:, 0]
        x_left, x_right = self.model.jnt_range[0]
        notdone = (
            (y >= 0)
            & np.logical_and(x_left < x, x < x_right)
            & np.isfinite(obs).all(axis=1)
        )
        return np.logical_not(notdone).reshape([obs.shape[0], 1])


class ReboundInvertedDoublePendulumSwingUpEnv(BaseInvertedDoublePendulumEnv):
    def __init__(
        self,
        freq_rate: int = 1,
        time_step: float = 0.02,
        integrator="standard_euler",
        # obs
        sin_cos: bool = True,
        # noise
        reset_noise_scale=0.2,
    ):
        BaseInvertedDoublePendulumEnv.__init__(
            self,
            freq_rate=freq_rate,
            time_step=time_step,
            integrator=integrator,
            sin_cos=sin_cos,
            reset_noise_scale=reset_noise_scale,
        )

    def _update_model(self):
        self.model.jnt_range[1] = [-np.inf, np.inf]
        self.model.body_quat[2] = [0, 0, 1, 0]

    def get_batch_reward(
        self, obs, pre_obs=None, action=None, state=None, pre_state=None
    ):
        y = self.get_pole_height(obs)
        rewards = (2 - y) / 4
        return rewards.reshape([obs.shape[0], 1])

    def get_batch_terminal(
        self, obs, pre_obs=None, action=None, state=None, pre_state=None
    ):
        notdone = np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])


class BoundaryInvertedDoublePendulumSwingUpEnv(BaseInvertedDoublePendulumEnv):
    def __init__(
        self,
        freq_rate: int = 1,
        time_step: float = 0.02,
        integrator="standard_euler",
        # obs
        sin_cos: bool = True,
        # noise
        reset_noise_scale=0.2,
    ):
        BaseInvertedDoublePendulumEnv.__init__(
            self,
            freq_rate=freq_rate,
            time_step=time_step,
            integrator=integrator,
            sin_cos=sin_cos,
            reset_noise_scale=reset_noise_scale,
        )

    def _update_model(self):
        self.model.jnt_range[1] = [-np.inf, np.inf]
        self.model.body_quat[2] = [0, 0, 1, 0]

    def get_batch_reward(
        self, obs, pre_obs=None, action=None, state=None, pre_state=None
    ):
        y = self.get_pole_height(obs)
        omega1, omega2 = obs[:, -2:].T
        vel_penalty = 5e-3 * omega1**2 + 1e-4 * omega2**2
        rewards = (2 - y) / 4 - vel_penalty
        return rewards.reshape([obs.shape[0], 1])

    def get_batch_terminal(
        self, obs, pre_obs=None, action=None, state=None, pre_state=None
    ):
        x = obs[:, 0]
        x_left, x_right = self.model.jnt_range[0]
        notdone = np.logical_and(x_left < x, x < x_right) & np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])


if __name__ == "__main__":
    from emei.util import random_policy_test
    from gym.wrappers import TimeLimit

    env = TimeLimit(
        BoundaryInvertedDoublePendulumBalancingEnv(),
        max_episode_steps=1000,
        new_step_api=True,
    )
    random_policy_test(env, is_render=True)
