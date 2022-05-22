from stable_baselines3 import SAC
from gym.wrappers import TimeLimit
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
               'actions': unscale_action(replay_buffer.actions[:pos].reshape(pos, -1)),
               'rewards': replay_buffer.rewards[:pos].reshape(pos),
               'terminals': replay_buffer.dones[:pos].reshape(pos),
               'timeouts': replay_buffer.timeouts[:pos].reshape(pos)}

    return samples


def collect_by_policy(num=int(2e5), policy_path=None):
    env = TimeLimit(ParticleEnv(), max_episode_steps)
    episode_rewards = []

    episode_reward = 0
    obs = env.reset()
    samples = defaultdict(list)
    model = None
    if policy_path is not None:
        model = SAC.load(policy_path)

    for t in tqdm(range(num)):
        if model is None:
            action = env.action_space.sample()
        else:
            action, state = model.predict(obs, deterministic=True)

        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward

        samples['observations'].append(obs)
        samples['actions'].append(action)
        samples['rewards'].append(reward)
        samples['terminals'].append(float(done))
        samples['timeouts'].append(float(0))

        if done:
            obs = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0
        else:
            obs = next_obs

    np_samples = {}
    for key in samples.keys():
        np_samples[key] = np.array(samples[key])

    return np_samples, min(episode_rewards), max(episode_rewards)


def save_as_h5(dataset, h5file_path):
    with h5py.File(h5file_path, 'w') as dataset_file:
        for key in dataset.keys():
            dataset_file[key] = dataset[key]
