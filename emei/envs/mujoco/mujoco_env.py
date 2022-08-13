import os
import mujoco
import numpy as np

from functools import partial
from emei import EmeiEnv
from collections import OrderedDict
from typing import Optional
from scipy.spatial.transform import Rotation

from gym import spaces
from gym.utils.renderer import Renderer
from gym.envs.mujoco.mujoco_env import BaseMujocoEnv

DEFAULT_SIZE = 480


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


def free_joint_forward_euler(pos, del_pos):
    assert pos.shape == (7,) and del_pos.shape == (6,)
    new_pos = np.empty(pos.shape)
    new_pos[:3] = pos[:3] + del_pos[:3]

    ang = Rotation.from_quat(pos[3:])
    rot = Rotation.from_euler("zyx", del_pos[3:] * np.array([-1, 1, -1]), degrees=False)
    rotated_ang = ang * rot
    new_pos[3:] = rotated_ang.as_quat()
    return new_pos


class MujocoEnv(EmeiEnv):
    """Superclass for all MuJoCo environments."""
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(self,
                 model_path,
                 freq_rate: int = 1,
                 time_step: float = 0.02,
                 integrator="standard_euler",
                 camera_config: Optional[dict] = None,
                 reset_noise_scale: float = 0,
                 ):
        EmeiEnv.__init__(self)
        # load model from path
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")

        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self._update_model()

        self.freq_rate = freq_rate
        self.time_step = time_step

        self.model.opt.timestep = time_step / freq_rate
        self.standard_euler = False
        if integrator == "standard_euler":
            self.model.opt.integrator = 0
            self.standard_euler = True
        elif integrator == "semi_euler":
            self.model.opt.integrator = 0
        elif integrator == "rk4":
            self.model.opt.integrator = 1
        else:
            raise NotImplementedError

        self._camera_config = camera_config if camera_config is not None else {}
        self._reset_noise_scale = reset_noise_scale

        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._viewers = {}

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, reward, done, truncated, info = self.step(action)
        assert not done

        self._set_observation_space(observation)

    def _update_model(self):
        pass

    def freeze(self):
        self.frozen_state = [self.data.qpos.copy(), self.data.qvel.copy()]

    def unfreeze(self):
        qpos, qvel = self.frozen_state
        self.set_state(qpos, qvel)

    def _restore_pos_vel_from_obs(self, obs):
        if obs.shape == (self.model.nq + self.model.nv,):
            return obs[:self.model.nq], obs[self.model.nq:]
        else:
            raise NotImplementedError

    def _set_state_by_obs(self, obs):
        self.set_state(*self._restore_pos_vel_from_obs(obs))

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def _get_info(self):
        return {}

    def step(self, action):
        pre_obs = self._get_obs()
        self.do_simulation(action, self.freq_rate)
        obs = self._get_obs()
        reward = self.get_reward(obs, pre_obs, action)
        terminal = self.get_terminal(obs, pre_obs, action)
        truncated = False
        info = self._get_info()
        return obs, reward, terminal, truncated, info

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        qpos = self.init_qpos + self._reset_noise_scale * self.np_random.standard_normal(self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        for key, value in self._camera_config.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    # -----------------------------

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        ob = self.reset_model()
        if not return_info:
            return ob
        else:
            return ob, {}

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.freq_rate

    def do_simulation(self, ctrl, n_frames):
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            old_qpos, old_qvel = self.data.qpos.copy(), self.data.qvel.copy()
            mujoco.mj_step(self.model, self.data)
            if self.standard_euler:
                new_qpos = self.standard_euler_pos(old_qpos, old_qvel).copy()
                new_qvel = self.data.qvel.copy()
                self.set_state(new_qpos, new_qvel)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def standard_euler_pos(self, old_qpos, old_qvel):
        cur_pos_idx = 0
        cur_vel_idx = 0
        new_qpos = np.empty(old_qpos.shape)
        for jnt_id, jnt_type in enumerate(self.model.jnt_type):
            if jnt_type == 0:
                pos_len, vel_len = 7, 6
                new_qpos[cur_pos_idx: cur_pos_idx + pos_len] = \
                    free_joint_forward_euler(old_qpos[cur_pos_idx: cur_pos_idx + pos_len],
                                             old_qvel[cur_vel_idx: cur_vel_idx + vel_len] * self.model.opt.timestep)
            elif jnt_type == 1:
                raise NotImplementedError
            else:
                pos_len, vel_len = 1, 1
                new_qpos[cur_pos_idx: cur_pos_idx + pos_len] = \
                    old_qpos[cur_pos_idx: cur_pos_idx + pos_len] + \
                    old_qvel[cur_vel_idx: cur_vel_idx + vel_len] * self.model.opt.timestep

            cur_pos_idx += pos_len
            cur_vel_idx += vel_len
        return new_qpos

    def render(
            self,
            mode: str = "human",
            width: int = DEFAULT_SIZE,
            height: int = DEFAULT_SIZE,
            camera_id: Optional[int] = None,
            camera_name: Optional[str] = None,
    ):
        assert mode in self.metadata["render_modes"]

        if mode in {
            "rgb_array",
            "single_rgb_array",
            "depth_array",
            "single_depth_array",
        }:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )

                self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode in {"rgb_array", "single_rgb_array"}:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode in {"depth_array", "single_depth_array"}:
            self._get_viewer(mode).render(width, height)
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode, width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                from gym.envs.mujoco import Viewer

                self.viewer = Viewer(self.model, self.data)
            elif mode in {
                "rgb_array",
                "depth_array",
                "single_rgb_array",
                "single_depth_array",
            }:
                from gym.envs.mujoco import RenderContextOffscreen

                self.viewer = RenderContextOffscreen(
                    width, height, self.model, self.data
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos

    def state_vector(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
