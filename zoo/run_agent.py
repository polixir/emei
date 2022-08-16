import argparse
import pathlib
import gym

import emei
from zoo.soft_actor_critic.sac import SAC
from typing import cast
from zoo.util import to_num, save_as_h5, load_hydra_cfg


def run(exp_dir,
        agent_type="best",
        device="cuda:0"):
    exp_dir = pathlib.Path(exp_dir)
    args = load_hydra_cfg(exp_dir, reset_device=device)
    sac_args = args.algorithm

    kwargs = dict([(item.split("=")[0], to_num(item.split("=")[1])) for item in args.task.params.split("&")])
    env = gym.make(args.task.name, new_step_api=True, **kwargs)
    env = cast(emei.EmeiEnv, env)

    agent = SAC(env.observation_space, env.action_space, sac_args, env.get_agent_obs)
    agent.load_checkpoint(exp_dir / "{}-agent.pth".format(agent_type),
                          evaluate=True,
                          reset_device=device)

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
    parser.add_argument("--agent_type", type=str, default="best")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run(args.exp_dir, args.agent_type)