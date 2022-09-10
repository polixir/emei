import inspect
import pathlib
from collections import defaultdict
from typing import Union, Dict
from abc import abstractmethod

import gym
import h5py
import numpy as np
from tqdm import tqdm
import urllib.request

from emei.offline_info import URL_INFOS

DATASET_PATH = pathlib.Path.home() / ".emei" / "offline_data"


class Freezable:
    def __init__(self):
        self.frozen_state = None

    def freeze(self):
        """
        Freeze the environment, for rollout-test or query.
        :return: None
        """
        self.frozen_state = self.get_current_state(copy=True)

    def unfreeze(self):
        """
        Unfreeze the environment, back to normal interaction.
        :return: None
        """
        self.set_state(self.frozen_state)

    def get_current_state(self, copy: bool = True):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError


class OfflineEnv(gym.Env):
    def __init__(self):
        env_name = self.__class__.__name__[:-3]
        self._offline_dataset_names = []
        if env_name in URL_INFOS:
            self.data_url = URL_INFOS[env_name]
            self._offline_dataset_names = []
            for param in self.data_url:
                for dataset in self.data_url[param]:
                    self._offline_dataset_names.append("{}-{}".format(param, dataset))

    @property
    def dataset_names(self) -> list:
        return self._offline_dataset_names

    @staticmethod
    def load_h5_data(h5path: Union[str, pathlib.Path]) -> Dict[(str, np.ndarray)]:
        """
        Load data of h5-file.
        :param h5path: path of h5 file.
        :return: dataset.
        """
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        data_dict = {}
        with h5py.File(h5path, 'r') as f:
            f.visititems(visitor)
            for k in tqdm(keys, desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = f[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = f[k][()]
        return data_dict

    @staticmethod
    def get_path_from_url(dataset_url: Union[str, pathlib.Path]) -> pathlib.Path:
        """
        Return the data file path corresponding to the url.
        :param dataset_url: web url.
        :return: local file path.
        """
        env_name, param, dataset_name = dataset_url.split('/')[-3:]
        dataset_dir = DATASET_PATH / env_name / param
        dataset_dir.mkdir(exist_ok=True)
        return dataset_dir / dataset_name

    def download_dataset(self, dataset_url: Union[str, pathlib.Path]) -> pathlib.Path:
        """
        Return the data file path corresponding to the url, downloads if it does not exist.
        :param dataset_url: web url.
        :return: local file path.
        """
        dataset_filepath = self.get_path_from_url(dataset_url)
        if not dataset_filepath.exists():
            print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
            urllib.request.urlretrieve(dataset_url, dataset_filepath)
        if not dataset_filepath.exists():
            raise IOError("Failed to download dataset from %s" % dataset_url)
        return dataset_filepath

    def get_dataset(self, dataset_name: str) -> Dict[(str, np.ndarray)]:
        assert dataset_name in self._offline_dataset_names

        joint_pos = dataset_name.find("-")
        param, dataset_type = dataset_name[:joint_pos], dataset_name[joint_pos + 1:]
        url = self.data_url[param][dataset_type]
        h5path = self.download_dataset(url)

        data_dict = self.load_h5_data(h5path)

        # Run a few quick sanity checks
        for key in ['observations', 'observations', 'actions', 'rewards', 'terminals', "timeouts"]:
            assert key in data_dict, 'Dataset is missing key %s' % key

        return data_dict

    def get_sequence_dataset(self, dataset_name: str) -> Dict[(str, np.ndarray)]:
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


class EmeiEnv(Freezable, OfflineEnv):
    def __init__(self):
        """
        Abstract class for all Emei environments to better support model-based RL and offline RL.
        """
        Freezable.__init__(self)
        OfflineEnv.__init__(self)

    def single_query(self, obs, action):
        self.freeze()
        self.set_state_by_obs(obs)
        next_obs, reward, terminal, truncated, info = self.step(action)
        self.unfreeze()
        return next_obs, reward, terminal, truncated, info

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
            next_obs_list, reward_list, terminal_list, truncated_list, info_list = [], [], [], [], []
            for i in range(obs.shape[0]):
                next_obs, reward, terminal, truncated, info = self.single_query(obs[i], action[i])
                next_obs_list.append(next_obs)
                reward_list.append(reward)
                terminal_list.append(terminal)
                truncated_list.append(truncated)
                info_list.append(info)
            return np.array(next_obs_list), \
                   np.array(reward_list), \
                   np.array(terminal_list), \
                   np.array(truncated_list), \
                   np.array(info_list)

    @staticmethod
    def extend_dim(variable):
        if variable is not None:
            return variable.reshape(1, variable.shape[0])
        else:
            return None

    def get_reward(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        """Return the reward of single or batch interaction data.
        :param obs: single or batch observation after taking action.
        :param pre_obs: single or batch observation before taking action.
        :param action: single or batch action to be taken.
        :param state: single or batch state after taking action.
        :param pre_state: single or batch state before taking action.
        :return: single or batch reward.
        """
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        args.remove("self")
        kwargs = dict(filter(lambda x: x[0] in args, values.items()))
        if len(obs.shape) == 1:  # single obs
            kwargs = dict([(arg, self.extend_dim(value)) for arg, value in kwargs.items()])
            return float(self.get_batch_reward(**kwargs)[0, 0])
        else:
            return self.get_batch_reward(**kwargs)

    def get_terminal(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        args.remove("self")
        kwargs = dict(filter(lambda x: x[0] in args, values.items()))
        if len(obs.shape) == 1:  # single obs
            kwargs = dict([(arg, self.extend_dim(value)) for arg, value in kwargs.items()])
            return bool(self.get_batch_terminal(**kwargs)[0, 0])
        else:
            return self.get_batch_terminal(**kwargs)

    def get_init_obs(self, batch_size):
        if batch_size == 1:
            return self.get_batch_obs(self.get_batch_init_state(batch_size=1))[0]
        else:
            return self.get_batch_obs(self.get_batch_init_state(batch_size=batch_size))

    def get_obs(self, state):
        if len(state.shape) == 1:
            return self.get_batch_obs(state.reshape(1, state.shape[0]))[0]
        else:
            return self.get_batch_obs(state)

    ################################################################################
    # methods to override
    ################################################################################

    @abstractmethod
    def set_state_by_obs(self, obs):
        """
        Set model state by observation, only for MDPs.
        :param obs: single observation
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch_agent_obs(self, obs):
        pass

    @abstractmethod
    def get_batch_reward(self, next_obs, pre_obs=None, action=None, state=None, pre_state=None):
        pass

    @abstractmethod
    def get_batch_terminal(self, next_obs, pre_obs=None, action=None, state=None, pre_state=None):
        pass

    @abstractmethod
    def get_batch_init_state(self, batch_size):
        pass

    ########################################
    # methods maybe to override
    ########################################

    def get_batch_obs(self, batch_state):
        return batch_state
