import numpy as np

from gym import utils
from emei.envs.mujoco.mujoco_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class Walker2dRunningEnv(MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        freq_rate: int = 1,
        time_step: float = 0.02,
        integrator="euler",
        # weight
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        # range
        healthy_z_range=(0.8, 2.0),
        healthy_angle_range=(-1.0, 1.0),
        # noise
        reset_noise_scale=5e-3,
    ):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            model_path="walker2d.xml",
            freq_rate=freq_rate,
            time_step=time_step,
            integrator=integrator,
            camera_config=DEFAULT_CAMERA_CONFIG,
            reset_noise_scale=reset_noise_scale,
        )

    def get_batch_reward(self, next_obs, pre_obs=None, action=None):
        forward_reward = (
            self._forward_reward_weight
            * (next_obs[:, 0] - pre_obs[:, 0])
            / self.time_step
        )
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        rewards = self._healthy_reward + forward_reward - control_cost
        return rewards.reshape([next_obs.shape[0], 1])

    def get_batch_terminal(self, next_obs, pre_obs=None, action=None):
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        z = next_obs[:, 1]
        angle = next_obs[:, 2]
        notdone = (
            np.logical_and(z > min_z, z < max_z)
            & np.logical_and(angle > min_angle, angle < max_angle)
            & np.isfinite(next_obs).all(axis=1)
        )
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])

    def get_batch_agent_obs(self, obs):
        return obs[:, 1:]


if __name__ == "__main__":
    from emei.util import random_policy_test

    env = Walker2dRunningEnv()
    random_policy_test(env, is_render=True)
