from stable_baselines3 import SAC
import gym
import numpy as np
from collections import defaultdict, Counter
import h5py
from tqdm import tqdm


def collect_replay_buffer(model):
    replay_buffer = model.replay_buffer
    unscale_action = model.policy.unscale_action
    pos = replay_buffer.pos
    # SAC of sb3 will scale action automatically, so un-scale it manually.
    samples = {'observations': replay_buffer.observations[:pos].reshape(pos, -1),
               'next_observations': replay_buffer.next_observations[:pos].reshape(pos, -1),
               'actions': unscale_action(replay_buffer.actions[:pos].reshape(pos, -1)),
               'rewards': replay_buffer.rewards[:pos].reshape(pos),
               'terminals': replay_buffer.dones[:pos].reshape(pos),
               'timeouts': replay_buffer.timeouts[:pos].reshape(pos)}

    return samples


def collect_by_policy(env_name, sample_num, model=None):
    env = gym.make(env_name)
    episode_rewards = []

    episode_reward = 0
    obs = env.reset()
    samples = defaultdict(list)

    for t in tqdm(range(sample_num)):
        if model is None:
            action = env.action_space.sample()
        else:
            action, state = model.predict(obs, deterministic=True)

        next_obs, reward, done, info = env.step(action)

        episode_reward += reward
        timeout = "TimeLimit.truncated" in info
        terminal = done and not timeout

        samples['observations'].append(obs)
        samples['next_observations'].append(next_obs)
        samples['actions'].append(action)
        samples['rewards'].append(reward)
        samples['terminals'].append(float(terminal))
        samples['timeouts'].append(float(timeout))

        if done:
            obs = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0
        else:
            obs = next_obs

    np_samples = {}
    for key in samples.keys():
        np_samples[key] = np.array(samples[key])

    return np_samples, sum(episode_rewards) / len(episode_rewards)


def save_as_h5(dataset, h5file_path):
    with h5py.File(h5file_path, 'w') as dataset_file:
        for key in dataset.keys():
            dataset_file[key] = dataset[key]
