import h5py
import itertools
import json
from typing import Union, Optional
import pathlib

import omegaconf
from collections import defaultdict
import gym

from tqdm import tqdm
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm


def get_replay_buffer(model):
    replay_buffer = model.replay_buffer
    unscale_action = model.policy.unscale_action
    pos = replay_buffer.pos
    # SAC of sb3 will scale action automatically, so un-scale it manually.
    samples = {
        "observations": replay_buffer.observations[:pos].reshape(pos, -1),
        "next_observations": replay_buffer.next_observations[:pos].reshape(pos, -1),
        "actions": unscale_action(replay_buffer.actions[:pos].reshape(pos, -1)),
        "rewards": replay_buffer.rewards[:pos].reshape(pos),
        "terminals": replay_buffer.dones[:pos].reshape(pos),
        "timeouts": replay_buffer.timeouts[:pos].reshape(pos),
    }

    return samples


def rollout(
    env: gym.Env,
    total_sample_num: int,
    agent: Optional[BaseAlgorithm] = None,
    deterministic: bool = False,
):
    samples = defaultdict(list)
    current_sample_num = 0
    current_episode_num = 0
    avg_reward = 0.0
    avg_length = 0.0

    env.reset()
    with tqdm(total=total_sample_num) as pbar:
        pbar.set_description("Sampling")
        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_length = 0
            done = False
            obs = env.reset()

            while not done:
                if agent is not None:
                    action, _ = agent.predict(obs, deterministic=deterministic)
                else:
                    action = env.action_space.sample()
                next_obs, reward, done, info = env.step(action)
                timeout = "TimeLimit.truncated" in info
                terminal = done and not timeout

                samples["observations"].append(obs)
                samples["next_observations"].append(next_obs)
                samples["actions"].append(action)
                samples["rewards"].append(reward)
                samples["terminals"].append(float(terminal))
                samples["timeouts"].append(float(timeout))

                episode_reward += reward
                episode_length += 1
                current_sample_num += 1
                obs = next_obs
                pbar.update(1)

            avg_reward += episode_reward
            avg_length += episode_length
            current_episode_num += 1

            if current_sample_num >= total_sample_num:
                break
    avg_reward /= current_episode_num
    avg_length /= current_episode_num
    np_samples = {}
    for key in samples.keys():
        np_samples[key] = np.array(samples[key])

    rollout_info = dict(
        avg_reward=avg_reward,
        avg_length=avg_length,
        total_episode_num=current_episode_num,
    )

    return np_samples, rollout_info


def save_rollout_info(rollout_info: dict, save_name: str):
    if pathlib.Path("rollout_info.json").exists():
        with open("rollout_info.json", "r") as f:
            rollout_info_dict = json.load(f)
    else:
        rollout_info_dict = {}
    rollout_info_dict[save_name] = rollout_info
    with open("rollout_info.json", "w") as f:
        json.dump(rollout_info_dict, f, indent=4)
    return rollout_info


def save_as_h5(dataset, h5file_path):
    with h5py.File(h5file_path, "w") as dataset_file:
        for key in dataset.keys():
            dataset_file[key] = dataset[key]


def load_hydra_cfg(
    results_dir: Union[str, pathlib.Path], reset_device=None
) -> omegaconf.DictConfig:
    results_dir = pathlib.Path(results_dir)
    cfg_file = results_dir / ".hydra" / "config.yaml"
    cfg = omegaconf.OmegaConf.load(cfg_file)
    if not isinstance(cfg, omegaconf.DictConfig):
        raise RuntimeError("Configuration format not a omegaconf.DictConf")

    if reset_device:
        cfg.device = reset_device
    return cfg
