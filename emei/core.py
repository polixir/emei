from abc import ABC, abstractmethod
import os
import urllib.request

import gym
import numpy as np
import h5py
from tqdm import tqdm
from emei.offline_info import URL_INFOS
from collections import defaultdict

DATASET_PATH = os.path.expanduser('~/.emei/offline_data')


class Freezable(ABC):
    def __init__(self):
        self.frozen_state = None

    @abstractmethod
    def freeze(self):
        raise NotImplementedError

    @abstractmethod
    def unfreeze(self):
        raise NotImplementedError

    @abstractmethod
    def query(self, obs, action):
        raise NotImplementedError


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def filepath_from_url(dataset_url):
    s, dataset_name = os.path.split(dataset_url)
    s, param = os.path.split(s)
    s, env_name = os.path.split(s)
    os.makedirs(os.path.join(DATASET_PATH, env_name, param), exist_ok=True)
    dataset_filepath = os.path.join(DATASET_PATH, env_name, param, dataset_name)
    return dataset_filepath


def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath


class Downloadable(object):
    def __init__(self):
        env_name = self.__class__.__name__[:-3]
        self.offline_dataset_names = []
        if env_name in URL_INFOS:
            self.data_url = URL_INFOS[env_name]
            self.offline_dataset_names = []
            for param in self.data_url:
                for dataset in self.data_url[param]:
                    self.offline_dataset_names.append("{}-{}".format(param, dataset))

    @property
    def dataset_names(self):
        return self.offline_dataset_names

    def get_dataset(self, dataset_name):
        assert dataset_name in self.offline_dataset_names

        param, dataset_type = dataset_name.split("-")
        url = self.data_url[param][dataset_type]
        h5path = download_dataset_from_url(url)

        data_dict = {}
        with h5py.File(h5path, 'r') as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals', "timeouts"]:
            assert key in data_dict, 'Dataset is missing key %s' % key

        return data_dict

    def get_qlearning_dataset(self, dataset_name):
        dataset = self.get_dataset(dataset_name)
        N = dataset['rewards'].shape[0]
        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        done_ = []

        for i in range(N - 1):
            obs = dataset['observations'][i].astype(np.float32)
            new_obs = dataset['observations'][i + 1].astype(np.float32)
            action = dataset['actions'][i].astype(np.float32)
            reward = dataset['rewards'][i].astype(np.float32)
            done_bool = bool(dataset['terminals'][i])

            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
        return {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'next_observations': np.array(next_obs_),
            'rewards': np.array(reward_),
            'terminals': np.array(done_),
        }

    def get_sequence_dataset(self, dataset_name):
        dataset = self.get_dataset(dataset_name)
        N = dataset['rewards'].shape[0]
        data = defaultdict(lambda: defaultdict(list))

        sequence_num = 0
        for i in range(N):
            done_bool = bool(dataset['terminals'][i])
            final_timestep = dataset['timeouts'][i]

            for k in dataset:
                data[sequence_num][k].append(dataset[k][i])

            if done_bool or final_timestep:
                sequence_num += 1
        return data