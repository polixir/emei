import os
import h5py
import urllib.request

from abc import ABC, abstractmethod
from gym import Env
import numpy as np
from tqdm import tqdm
from emei.offline_info import URL_INFOS
from collections import defaultdict

DATASET_PATH = os.path.expanduser('~/.emei/offline_data')


class FreezableEnv(ABC, Env):
    def __init__(self):
        """
        Abstract class for all Emei environments to better support model-based RL.
        """
        self.frozen_state = None

    @abstractmethod
    def freeze(self):
        """
        Freeze the environment, for rollout-test or query.
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def unfreeze(self):
        """
        Unfreeze the environment, back to normal interaction.
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def _set_state_by_obs(self, obs):
        """
        Set model state by observation, only for MDPs.
        :param obs: single observation
        :return: None
        """
        raise NotImplementedError

    def single_query(self, obs, action):
        self.freeze()
        self._set_state_by_obs(obs)
        next_obs, reward, done, info = self.step(action)
        self.unfreeze()
        return next_obs, reward, done, info

    def query(self, obs, action):
        """
        Give the environment's reflection to single or batch query.
        :param obs: single or batch observations.
        :param action: single or batch action.
        :return: single or batch (next-obs, reward, done, terminal).
        """
        if len(obs.shape) == 1:  # single obs
            assert len(action.shape) == 1
            return self.single_query(obs, action)
        else:
            next_obs_list, reward_list, done_list, info_list = [], [], [], []
            for i in range(obs.shape[0]):
                next_obs, reward, done, info = self.single_query(obs[i], action[i])
                next_obs_list.append(next_obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)
            return next_obs_list, reward_list, done_list, info_list

    @abstractmethod
    def get_single_reward_by_next_obs(self, next_obs):
        pass

    @abstractmethod
    def get_single_terminal_by_next_obs(self, next_obs):
        pass

    def get_reward_by_next_obs(self, next_obs):
        """
        Return the reward of single or batch next-obs.
        :param next_obs: single or batch observations.
        :return: single or batch reward.
        """
        if len(next_obs.shape) == 1:  # single obs
            return self.get_single_reward_by_next_obs(next_obs)
        else:
            reward_list = []
            for i in range(next_obs.shape[0]):
                reward_list.append(self.get_single_reward_by_next_obs(next_obs))
            return reward_list

    def get_terminal_by_next_obs(self, next_obs):
        """
        Return the terminal of single or batch next-obs.
        :param next_obs: single or batch observations.
        :return: single or batch terminal.
        """
        if len(next_obs.shape) == 1:  # single obs
            return self.get_single_terminal_by_next_obs(next_obs)
        else:
            terminal_list = []
            for i in range(next_obs.shape[0]):
                terminal_list.append(self.get_single_terminal_by_next_obs(next_obs))
            return terminal_list

    # @abstractmethod
    # def get_initial_obs(self, batch_size=1):
    #     """
    #     Return the initial observation .
    #     :param batch_size: batch size of generated data, default is single(batch_size=1).
    #     :return: single or batch observation.
    #     """
    #     pass


def get_keys(h5file):
    """
    Get the keys of h5-file.
    :param h5file: binary file of h5 format.
    :return: keys.
    """
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def filepath_from_url(dataset_url: str) -> str:
    """
    Return the data file path corresponding to the url.
    :param dataset_url: web url.
    :return: local file path.
    """
    env_name, param, dataset_name = dataset_url.split(os.path.sep)[-3:]
    os.makedirs(os.path.join(DATASET_PATH, env_name, param), exist_ok=True)
    dataset_filepath = os.path.join(DATASET_PATH, env_name, param, dataset_name)
    return dataset_filepath


def download_dataset_from_url(dataset_url: str) -> str:
    """
    Return the data file path corresponding to the url, downloads if it does not exist.
    :param dataset_url: web url.
    :return: local file path.
    """
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath


class Downloadable(object):
    def __init__(self):
        """
        Abstract class for all offline Emei environments, which supports offline data download.
        """
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
