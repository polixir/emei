__credits__ = ["Rushiv Arora"]

import numpy as np

from gym import utils
from emei.envs.mujoco.mujoco_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {}


class SwimmerRunningEnv(MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        freq_rate: int = 1,
        time_step: float = 0.02,
        integrator="euler",
        # weight
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-4,
        # noise
        reset_noise_scale=0.1,
    ):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            model_path="swimmer.xml",
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
        rewards = forward_reward - control_cost
        return rewards.reshape([next_obs.shape[0], 1])

    def get_batch_terminal(self, next_obs, pre_obs=None, action=None):
        notdone = np.isfinite(next_obs).all(axis=1)
        return np.logical_not(notdone).reshape([next_obs.shape[0], 1])

    def get_agent_obs(self, obs):
        return obs[2:]


if __name__ == "__main__":
    from emei.util import random_policy_test

    env = SwimmerRunningEnv()
    random_policy_test(env, is_render=True)
