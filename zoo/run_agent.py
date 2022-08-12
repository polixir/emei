import argparse
import datetime
import pathlib

import gym
import emei
import numpy as np
import itertools
import torch
import hydra
import json
from tqdm import tqdm
from collections import defaultdict
from zoo.soft_actor_critic.sac import SAC
from zoo.soft_actor_critic.replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
from zoo.util import to_num, save_as_h5, load_hydra_cfg


def run(exp_dir, agent_type="medium"):
    exp_dir = pathlib.Path(exp_dir)
    args = load_hydra_cfg(exp_dir)
    sac_args = args.algorithm

    kwargs = dict([(item.split("=")[0], to_num(item.split("=")[1])) for item in args.task.params.split("&")])
    env = gym.make(args.task.name, new_step_api=True, **kwargs)

    agent = SAC(env.observation_space.shape[0], env.action_space, sac_args)
    agent.load_checkpoint(exp_dir / "{}-agent.pth".format(agent_type), evaluate=True)

    state = env.reset(seed=10086)
    env.render()
    episode_reward = 0
    episode_length = 0
    while True:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, terminal, truncated, _ = env.step(action)
        env.render()

        episode_reward += reward
        episode_length += 1
        if terminal or truncated:
            state = env.reset()
            print("reward:{}, length:{}".format(episode_reward, episode_length))
            episode_reward = 0
            episode_length = 0
        else:
            state = next_state



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str)
    parser.add_argument("--agent_type", type=str, default="medium")
    args = parser.parse_args()

    run(args.exp_dir, args.agent_type)
