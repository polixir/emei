import argparse
import datetime
import pathlib
import json
import gym
import emei
import numpy as np
import itertools
import torch
import hydra
from typing import cast
from tqdm import tqdm
from collections import defaultdict
from zoo.soft_actor_critic.sac import SAC
from zoo.soft_actor_critic.replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
from zoo.util import to_num, save_as_h5


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args):
    sac_args = args.algorithm
    reach_medium = False
    reach_expert = False

    # Environment
    kwargs = dict([(item.split("=")[0], to_num(item.split("=")[1])) for item in args.task.params.split("&")])
    env = gym.make(args.task.name, new_step_api=True, **kwargs)
    env = cast(emei.EmeiEnv, env)
    env.reset(seed=sac_args.seed)
    env.action_space.seed(sac_args.seed)

    torch.manual_seed(sac_args.seed)
    np.random.seed(sac_args.seed)

    # Agent
    agent = SAC(env.observation_space, env.action_space, sac_args)

    # rollout random dataset
    agent.save_checkpoint(save_path="random-agent.pth")
    avg_reward, avg_length = rollout_and_save(env, agent,
                                              total_sample_num=args.task.random_sample_num,
                                              seed=sac_args.seed,
                                              save_name="random")
    print("Rollout Random samples: Avg. Reward: {}, Avg. Length: {}".format(round(avg_reward, 2),
                                                                            round(avg_length, 2)))
    # rollout uniform dataset
    avg_reward, avg_length = rollout_and_save(env, None,
                                              total_sample_num=args.task.uniform_sample_num,
                                              seed=sac_args.seed,
                                              save_name="uniform")
    print("Rollout Uniform Distribution samples: Avg. Reward: {}, Avg. Length: {}".format(round(avg_reward, 2),
                                                                                          round(avg_length, 2)))

    # Tesnorboard
    writer = SummaryWriter(log_dir="./tb/")

    # Memory
    memory = ReplayMemory(sac_args.replay_size, sac_args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    best_eval_rewards = -np.inf

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_length = 0
        terminal = False
        truncated = False
        state = env.reset()

        while not (terminal or truncated):
            if sac_args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > sac_args.batch_size:
                # Number of updates per step in environment
                for i in range(sac_args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         sac_args.batch_size,
                                                                                                         updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, terminal, truncated, _ = env.step(action)  # Step
            episode_length += 1
            total_numsteps += 1
            episode_reward += reward

            memory.push(state, action, reward, next_state, terminal, truncated)  # Append transition to memory

            state = next_state

        if total_numsteps > sac_args.num_steps:
            break

        writer.add_scalar('train/reward', episode_reward, total_numsteps)
        writer.add_scalar('train/length', episode_length, total_numsteps)
        print("Episode: {}, total numsteps: {}, length: {}, reward: {}".format(i_episode, total_numsteps,
                                                                               episode_length,
                                                                               round(episode_reward, 2)))
        if i_episode % sac_args.eval_freq == 0 and sac_args.eval is True:
            avg_reward = 0.
            avg_length = 0.
            episode_num = 10
            for _ in range(episode_num):
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                terminal = False
                truncated = False
                while not (terminal or truncated):
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, terminal, truncated, _ = env.step(action)
                    episode_reward += reward
                    episode_length += 1

                    state = next_state
                avg_reward += episode_reward
                avg_length += episode_length
            avg_reward /= episode_num
            avg_length /= episode_num

            writer.add_scalar('test/reward', avg_reward, total_numsteps)
            writer.add_scalar('test/length', avg_length, total_numsteps)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}, Avg. Length: {}".format(episode_num,
                                                                               round(avg_reward, 2),
                                                                               round(avg_length, 2)))
            print("----------------------------------------")
            if avg_reward > best_eval_rewards:
                best_eval_rewards = avg_reward
                agent.save_checkpoint(save_path="best-agent.pth")

            if avg_reward > args.task.medium_reward and not reach_medium:
                memory.save_buffer(save_path="medium-replay.h5")
                agent.save_checkpoint(save_path="medium-agent.pth")
                avg_reward, avg_length = rollout_and_save(env, agent,
                                                          total_sample_num=args.task.medium_sample_num,
                                                          seed=sac_args.seed,
                                                          save_name="medium")
                print("Rollout Medium samples: Avg. Reward: {}, Avg. Length: {}".format(round(avg_reward, 2),
                                                                                        round(avg_length, 2)))
                reach_medium = True
            if avg_reward > args.task.expert_reward and not reach_expert:
                memory.save_buffer(save_path="expert-replay.h5")
                agent.save_checkpoint(save_path="expert-agent.pth")
                avg_reward, avg_length = rollout_and_save(env, agent,
                                                          total_sample_num=args.task.expert_sample_num,
                                                          seed=sac_args.seed,
                                                          save_name="expert")
                print("Rollout Expert samples: Avg. Reward: {}, Avg. Length: {}".format(round(avg_reward, 2),
                                                                                        round(avg_length, 2)))
                reach_expert = True

        if reach_expert:
            break
    env.close()


def rollout_and_save(env,
                     agent,
                     total_sample_num=int(1e6),
                     seed=0,
                     save_name="random"):
    samples = defaultdict(list)
    current_sample_num = 0
    current_episode_num = 0
    avg_reward = 0.
    avg_length = 0.
    env.reset(seed=seed)

    with tqdm(total=total_sample_num) as pbar:
        pbar.set_description('Sampling')
        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_length = 0
            terminal = False
            truncated = False
            state = env.reset()
            while not (terminal or truncated):
                if agent is not None:
                    action = agent.select_action(state, evaluate=False)
                else:
                    action = env.action_space.sample()
                next_state, reward, terminal, truncated, _ = env.step(action)

                samples['observations'].append(state)
                samples['next_observations'].append(next_state)
                samples['actions'].append(action)
                samples['rewards'].append(reward)
                samples['terminals'].append(float(terminal))
                samples['timeouts'].append(float(truncated))

                episode_reward += reward
                episode_length += 1
                current_sample_num += 1
                state = next_state
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

    if pathlib.Path("sampling_info.txt").exists():
        with open("sampling_info.txt", "r") as f:
            sampling_info = json.load(f)
    else:
        sampling_info = {}
    sampling_info[save_name] = dict(avg_reward=avg_reward,
                                    avg_length=avg_length,
                                    total_episode_num=current_episode_num)
    with open("sampling_info.txt", "w") as f:
        json.dump(sampling_info, f, indent=4)
    save_as_h5(np_samples, "{}.h5".format(save_name))

    return avg_reward, avg_length


if __name__ == "__main__":
    main()
