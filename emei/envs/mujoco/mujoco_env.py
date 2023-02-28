from typing import Optional, Dict, Union, Tuple

import mujoco
import numpy as np
from emei import EmeiEnv
from scipy.spatial.transform import Rotation

from gym import spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv

DEFAULT_SIZE = 480


class EmeiMujocoEnv(EmeiEnv, MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        model_path: str,
        observation_space: spaces.Space,
        freq_rate: int = 1,
        real_time_scale: float = 0.02,
        integrator: str = "euler",
        init_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 5e-3,
        obs_noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]] = 0.0,
        # camera
        camera_config: Optional[Dict] = None,
        # base mujoco env
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        self.freq_rate = freq_rate
        self.real_time_scale = real_time_scale
        self.integrator = integrator
        self.init_noise_params = init_noise_params
        self.obs_noise_params = obs_noise_params

        self._camera_config = camera_config if camera_config is None else {}
        self.metadata["render_fps"] = int(np.round(1.0 / (self.real_time_scale * self.freq_rate)))

        EmeiEnv.__init__(self, env_params=dict(freq_rate=freq_rate, real_time_scale=real_time_scale, integrator=integrator))
        MujocoEnv.__init__(
            self,
            model_path,
            freq_rate,
            observation_space,
            render_mode,
            width,
            height,
            camera_id,
            camera_name,
        )

    def _initialize_simulation(self):
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)

        self._update_model()

        self.model.opt.timestep = self.real_time_scale
        self.euler = False
        if self.integrator == "euler":
            self.model.opt.integrator = 0
            self.euler = True
        elif self.integrator == "semi_implicit_euler":
            self.model.opt.integrator = 0
        elif self.integrator == "rk4":
            self.model.opt.integrator = 1
        else:
            raise NotImplementedError

        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

    def _step_mujoco_simulation(self, ctrl, n_frames=None):
        if n_frames is None:
            n_frames = self.freq_rate
        self.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            old_pos, old_vel = self.data.qpos.copy(), self.data.qvel.copy()
            mujoco.mj_step(self.model, self.data)
            if self.euler:
                new_pos = self.get_euler_pos(old_pos, old_vel).copy()
                new_vel = self.data.qvel.copy()
                self.set_state(new_pos, new_vel)
            if self.obs_noise_params != 0:
                noise_pos, noise_vel = self.additive_gaussian_noise(
                    self.data.qpos.copy()[None],
                    self.data.qvel.copy()[None],
                    self.obs_noise_params,
                )
                self.set_state(noise_pos[0], noise_vel[0])

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def _update_model(self):
        pass

    def freeze(self):
        self.frozen = True
        self.frozen_state = (self.data.qpos.copy(), self.data.qvel.copy())

    def unfreeze(self):
        self.frozen = False
        self.set_state(*self.frozen_state)

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in self._camera_config.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def get_batch_init_state(self, batch_size):
        origin_pos = np.tile(self.init_qpos[None, :], [batch_size, 1])
        origin_vel = np.tile(self.init_qvel[None, :], [batch_size, 1])
        return self.additive_gaussian_noise(origin_pos, origin_vel, self.init_noise_params)

    def state2obs(self, batch_state):
        return batch_state.copy()

    def obs2state(self, batch_obs, batch_extra_obs):
        assert len(batch_obs.shape) == 2
        if batch_obs.shape[1] == (self.model.nq + self.model.nv,):
            return batch_obs.copy()
        else:
            raise NotImplementedError

    def get_batch_extra_obs(self, batch_state):
        return batch_state.copy()

    def get_current_state(self):
        return self.state_vector()

    def step(self, action):
        info = {}

        state = self.get_current_state()
        info["extra_obs"] = self.get_batch_extra_obs(state[None])[0]
        self.do_simulation(action, self.freq_rate)

        next_state = self.get_current_state()
        info["next_extra_obs"] = self.get_batch_extra_obs(next_state[None])[0]

        reward = self.get_batch_reward(next_state[None], state[None], action[None])[0, 0]
        terminal = self.get_batch_terminal(next_state[None], state[None], action[None])[0, 0]
        truncated = False

        return self.state2obs(next_state[None])[0], reward, terminal, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        EmeiEnv.reset(self, seed=seed)

        self._reset_simulation()

        pos, vel = self.get_batch_init_state(1)
        self.set_state(pos[0], vel[0])

        if self.render_mode == "human":
            self.render()
        state = self.get_current_state()
        return self.state2obs(state[None])[0], {}

    def get_batch_next_obs(self, obs, action):
        self.freeze()

        next_obs = np.empty(shape=obs.shape)

        for i, o in enumerate(obs):
            self.set_state(o[: self.model.nq], o[self.model.nq :])
            self.do_simulation(action[i], self.freq_rate)
            next_obs[i] = self.current_obs
        self.unfreeze()

        return next_obs

    def get_euler_pos(
        self,
        old_pos: np.ndarray,
        old_vel: np.ndarray,
    ):
        cur_pos_idx = 0
        cur_vel_idx = 0
        new_pos = old_pos.copy()
        for jnt_id, jnt_type in enumerate(self.model.jnt_type):
            if jnt_type == 0:
                pos_len, vel_len = 7, 6

                new_pos[cur_pos_idx : cur_pos_idx + 3] += old_vel[cur_vel_idx : cur_vel_idx + 3] * self.real_time_scale
                angle = Rotation.from_quat(old_pos[cur_pos_idx + 3 : cur_pos_idx + 7]).as_euler("zyx", degrees=True)
                angle += old_vel[cur_vel_idx + 3 : cur_vel_idx + 6] * self.real_time_scale
                new_pos[cur_pos_idx + 3 : cur_pos_idx + 7] = Rotation.from_euler("xyz", angle, degrees=True).as_quat()
            elif jnt_type == 1:
                raise NotImplementedError
            else:
                pos_len, vel_len = 1, 1
                new_pos[cur_pos_idx : cur_pos_idx + pos_len] += (
                    old_vel[cur_vel_idx : cur_vel_idx + vel_len] * self.real_time_scale
                )

            cur_pos_idx += pos_len
            cur_vel_idx += vel_len
        return new_pos

    def additive_gaussian_noise(
        self,
        origin_pos: np.ndarray,
        origin_vel: np.ndarray,
        noise_params: Union[float, Tuple[float, float], Dict[int, Tuple[float, float]]],
    ):
        """
        0:  free    7-pos   6-vel
        1:
        2:  slide   1-pos   1-vel
        3:  hinge   1-pos   1-vel
        """
        batch_size = origin_pos.shape[0]

        cur_pos_idx = 0
        cur_vel_idx = 0
        noisy_pos = origin_pos.copy()
        noisy_vel = origin_vel.copy()
        for jnt_id, jnt_type in enumerate(self.model.jnt_type):
            noise_dim = 6 if jnt_type == 0 else 1

            if isinstance(noise_params, dict):
                if jnt_id in noise_params:
                    pos_n = (noise_params[jnt_id][0] + np.zeros(noise_dim),)
                    vel_n = noise_params[jnt_id][1] + np.zeros(noise_dim)
                else:
                    pos_n, vel_n = np.zeros(noise_dim), np.zeros(noise_dim)
            elif isinstance(noise_params, tuple):
                pos_n, vel_n = np.ones(noise_dim) * noise_params[0], np.ones(noise_dim) * noise_params[1]
            else:
                pos_n, vel_n = np.ones(noise_dim) * noise_params, np.ones(noise_dim) * noise_params

            if jnt_type == 0:
                pos_len, vel_len = 7, 6

                noisy_pos[:, cur_pos_idx : cur_pos_idx + 3] += np.random.randn(batch_size, 3) * pos_n[:3]
                angle = Rotation.from_quat(origin_pos[:, cur_pos_idx + 3 : cur_pos_idx + 7]).as_euler("zyx", degrees=True)
                angle += np.random.randn(batch_size, 3) * pos_n[3:6]
                noisy_pos[:, cur_pos_idx + 3 : cur_pos_idx + 7] = Rotation.from_euler("xyz", angle, degrees=True).as_quat()

                noisy_vel[:, cur_vel_idx : cur_vel_idx + 3] += np.random.randn(batch_size, 3) * vel_n[:3]
                noisy_vel[:, cur_vel_idx + 3 : cur_vel_idx + 6] += np.random.randn(batch_size, 3) * vel_n[3:6]
            elif jnt_type == 1:
                raise NotImplementedError
            else:
                pos_len, vel_len = 1, 1
                noisy_pos[:, cur_pos_idx : cur_pos_idx + 1] += np.random.randn(batch_size, 1) * pos_n
                noisy_vel[:, cur_vel_idx : cur_vel_idx + 1] += np.random.randn(batch_size, 1) * vel_n

            cur_pos_idx += pos_len
            cur_vel_idx += vel_len

        return noisy_pos, noisy_vel
