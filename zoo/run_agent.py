import argparse
import pathlib
from typing import cast

import gym
import hydra
import emei
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.base_class import BaseAlgorithm

from zoo.util import load_hydra_cfg


def run(exp_dir, type="expert", device="cuda:0"):
    exp_dir = pathlib.Path(exp_dir)
    cfg = load_hydra_cfg(exp_dir, reset_device=device)

    agent_class: BaseAlgorithm = eval(cfg.algorithm.agent._target_)
    if type == "best":
        file_name = "best_model"
    else:
        file_name = "{}-{}-agent".format(cfg.algorithm.name, type)
    agent = agent_class.load(exp_dir / file_name)

    env = cast(emei.EmeiEnv, gym.make(cfg.task.env_id, render_mode="human", **cfg.task.params))
    obs, info = env.reset(seed=cfg.seed)
    env.action_space.seed(seed=cfg.seed)

    episode_reward = 0
    episode_length = 0
    while True:
        action, state = agent.predict(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        episode_reward += reward
        episode_length += 1
        if terminated or truncated:
            obs, info = env.reset()
            print("reward:{}, length:{}".format(episode_reward, episode_length))
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str)
    parser.add_argument("--type", type=str, default="expert")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run(args.exp_dir, args.type)
