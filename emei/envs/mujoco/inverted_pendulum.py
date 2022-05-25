import numpy as np
from gym import utils
from emei.envs.mujoco.base_mujoco import BaseMujocoEnv
import math
import os


class BaseInvertedPendulumEnv(BaseMujocoEnv, utils.EzPickle):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler"):
        utils.EzPickle.__init__(self)
        BaseMujocoEnv.__init__(self,
                               model_path="inverted_pendulum.xml",
                               freq_rate=freq_rate,
                               time_step=time_step,
                               integrator=integrator)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent


class ReboundInvertedPendulumHoldingEnv(BaseInvertedPendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler"):
        BaseInvertedPendulumEnv.__init__(self,
                                         freq_rate=freq_rate,
                                         time_step=time_step,
                                         integrator=integrator)

    def _get_reward(self):
        return 1.0

    def _is_terminal(self) -> bool:
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
        return not notdone


class BoundaryInvertedPendulumHoldingEnv(BaseInvertedPendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler"):
        BaseInvertedPendulumEnv.__init__(self,
                                         freq_rate=freq_rate,
                                         time_step=time_step,
                                         integrator=integrator)

    def _get_reward(self):
        return 1.0

    def _is_terminal(self) -> bool:
        ob = self._get_obs()
        x_left, x_right = self.model.jnt_range[0]
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2) and (x_left < ob[0] < x_right)
        return not notdone


class ReboundInvertedPendulumSwingUpEnv(BaseInvertedPendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler"):
        BaseInvertedPendulumEnv.__init__(self,
                                         freq_rate=freq_rate,
                                         time_step=time_step,
                                         integrator=integrator)

    def _update_model(self):
        self.model.stat.extent = 6
        self.model.geom_size[0][1] = 3
        self.model.jnt_range[0] = [-3, 3]
        self.model.jnt_range[1] = [-np.inf, np.inf]

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        qpos[1] += np.pi
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reward(self):
        ob = self._get_obs()
        omega = self.sim.data.qvel[1]
        vel_penalty = 0.1 * abs(omega)
        return (math.cos(ob[1]) + 1 - vel_penalty) / 2

    def _is_terminal(self) -> bool:
        ob = self._get_obs()
        omega = self.sim.data.qvel[1]
        notdone = np.isfinite(ob).all() and abs(omega) < 1e4
        return not notdone


class BoundaryInvertedPendulumSwingUpEnv(BaseInvertedPendulumEnv):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler"):
        BaseInvertedPendulumEnv.__init__(self,
                                         freq_rate=freq_rate,
                                         time_step=time_step,
                                         integrator=integrator)

    def _update_model(self):
        self.model.stat.extent = 6
        self.model.geom_size[0][1] = 3
        self.model.jnt_range[0] = [-3, 3]
        self.model.jnt_range[1] = [-np.inf, np.inf]

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        qpos[1] += np.pi
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reward(self):
        ob = self._get_obs()
        omega = self.sim.data.qvel[1]
        vel_penalty = 0.1 * abs(omega)
        return (math.cos(ob[1]) + 1 - vel_penalty) / 2

    def _is_terminal(self) -> bool:
        ob = self._get_obs()
        omega = self.sim.data.qvel[1]
        x_left, x_right = self.model.jnt_range[0]
        notdone = np.isfinite(ob).all() and abs(omega) < 1e4 and (x_left < ob[0] < x_right)
        return not notdone


if __name__ == '__main__':
    from emei.util import random_policy_test

    env = ReboundInvertedPendulumSwingUpEnv()
    # random_policy_test(env, is_render=True)
    d = env.get_dataset('freq_rate=1&time_step=0.02-expert')
    print(sum(d["terminals"]) / len(d["terminals"]))
    # random_policy_test(env, is_render=True)