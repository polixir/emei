import gym
import argparse
import hydra
import emei
import pathlib
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.base_class import BaseAlgorithm

from zoo.util import load_hydra_cfg


def run(exp_dir, type="expert", device="cuda:0"):
    exp_dir = pathlib.Path(exp_dir)
    args = load_hydra_cfg(exp_dir, reset_device=device)

    agent_class: BaseAlgorithm = eval(args.algorithm.agent._target_)
    if type == "best":
        file_name = "best_model"
    else:
        file_name = "{}-{}-agent".format(args.algorithm.name, type)
    agent = agent_class.load(exp_dir / file_name)

    env: emei.EmeiEnv = hydra.utils.instantiate(args.algorithm.agent.env)
    obs = env.reset(seed=args.seed)
    env.action_space.seed(seed=args.seed)

    env.render()
    episode_reward = 0
    episode_length = 0
    while True:
        action, state = agent.predict(obs, deterministic=False)
        next_obs, reward, done, _ = env.step(action)
        env.render()

        episode_reward += reward
        episode_length += 1
        if done:
            obs = env.reset()
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
