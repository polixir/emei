import numpy as np
from gym import utils
from emei.envs.mujoco.mujoco_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {}


class BaseInvertedPendulumEnv(MujocoEnv, utils.EzPickle):
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
            model_path="inverted_pendulum.xml",
            freq_rate=freq_rate,
            time_step=time_step,
            integrator=integrator,
            camera_config=DEFAULT_CAMERA_CONFIG,
            reset_noise_scale=reset_noise_scale,
        )

        if self.sin_cos:
            self._causal_graph = np.array(
                [
                    [0, 0, 0, 0, 0],  # x
                    [0, 1, 1, 1, 1],  # sin theta
                    [0, 1, 1, 1, 1],  # cos theta
                    [1, 0, 0, 0, 0],  # v
                    [0, 1, 1, 1, 1],  # omega
                    [0, 0, 0, 1, 1],
                ]
            )  # action
        else:
            self._causal_graph = np.array(
                [
                    [0, 0, 0, 0],  # x
                    [0, 0, 1, 1],  # theta
                    [1, 0, 0, 0],  # v
                    [0, 1, 1, 1],  # omega
                    [0, 0, 1, 1],
                ]
            )  # action

    def get_batch_obs(self, batch_state):
        if self.sin_cos:
            qpos, qvel = (
                batch_state[:, : self.model.nq],
                batch_state[:, self.model.nq :],
            )
            batch_obs = np.concatenate(
                [qpos[:, :1], np.sin(qpos[:, 1:]), np.cos(qpos[:, 1:]), qvel], axis=-1
            )
        else:
            batch_obs = batch_state.copy()
        return batch_obs

    def get_batch_state(self, batch_obs):
        if self.sin_cos:
            theta = np.angle(1j * batch_obs[:, 1] + batch_obs[:, 2])
            batch_state = np.concatenate(
                [batch_obs[:, :1], theta[:, None], batch_obs[:, -2:]], axis=-1
            )
        else:
            batch_state = batch_obs.copy()
        return batch_state

    def get_batch_init_state(self, batch_size):
        pos_random = self.np_random.standard_normal((batch_size, self.model.nq))
        y = np.cos(pos_random[:, 1])
        while not (y > 0).all():
            pos_random = self.np_random.standard_normal((batch_size, self.model.nq))
            y = np.cos(pos_random[:, 1])
        pos_random[:, 0] = 0
        qpos = self._reset_noise_scale * pos_random
        qvel = self._reset_noise_scale * self.np_random.standard_normal(
            (batch_size, self.model.nv)
        )
        batch_state = np.concatenate([qpos, qvel], axis=-1)
        return batch_state

    def get_pole_height(self, obs):
        if self.sin_cos:
            y = obs[:, 2]
        else:
            theta = obs[:, 1]
            y = np.cos(theta)
        return y


class ReboundInvertedPendulumBalancingEnv(BaseInvertedPendulumEnv):
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
        BaseInvertedPendulumEnv.__init__(
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
        notdone = (y >= 0.9) & np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])


class BoundaryInvertedPendulumBalancingEnv(BaseInvertedPendulumEnv):
    def __init__(
        self,
        freq_rate: int = 1,
        time_step: float = 0.02,
        integrator="standard_euler",
        # obs
        sin_cos: bool = False,
        # noise
        reset_noise_scale=0.5,
    ):
        BaseInvertedPendulumEnv.__init__(
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


class ReboundInvertedPendulumSwingUpEnv(BaseInvertedPendulumEnv):
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
        BaseInvertedPendulumEnv.__init__(
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
        rewards = (1 - y) / 2
        return rewards.reshape([obs.shape[0], 1])

    def get_batch_terminal(
        self, obs, pre_obs=None, action=None, state=None, pre_state=None
    ):
        notdone = np.isfinite(obs).all(axis=1)
        return np.logical_not(notdone).reshape([obs.shape[0], 1])


class BoundaryInvertedPendulumSwingUpEnv(BaseInvertedPendulumEnv):
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
        BaseInvertedPendulumEnv.__init__(
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
        rewards = (1 - y) / 2
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
        BoundaryInvertedPendulumBalancingEnv(),
        max_episode_steps=1000,
        new_step_api=True,
    )
    random_policy_test(env, is_render=True)

    # env = ReboundInvertedPendulumSwingUpEnv(sin_cos=True)
    # print(env.get_causal_graph(2))

    # env = ReboundInvertedPendulumSwingUpEnv()
    # b_s = np.random.rand(5, 4)
    # b_o = env.get_batch_obs(b_s)
    # print(b_s)
    # print(env.get_batch_state(b_o))
    # print((b_s == env.get_batch_state(b_o)).all())
