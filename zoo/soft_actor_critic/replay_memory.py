import random
import numpy as np
import os
import pickle
from zoo.util import save_as_h5


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, terminal, truncated):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, terminal, truncated)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminal, truncated = map(np.stack, zip(*batch))
        return state, action, reward, next_state, terminal, truncated

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name="", suffix="", save_path=None):
        if save_path is None:
            save_path = "sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        state, action, reward, next_state, terminal, truncated = map(np.stack, zip(*self.buffer))
        samples = {'observations': state.astype(np.float32),
                   'next_observations': next_state.astype(np.float32),
                   'actions': action.astype(np.float32),
                   'rewards': reward.astype(np.float32),
                   'terminals': terminal.astype(np.float32),
                   'timeouts': truncated.astype(np.float32)}
        save_as_h5(samples, save_path)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
