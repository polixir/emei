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
        self.frozen = False

    def freeze(self):
        """
        Freeze the environment, for rollout-test or query.
        :return: None
        """
        assert not self.frozen, "env has frozen"
        self.frozen = True

    def unfreeze(self):
        """
        Unfreeze the environment, back to normal interaction.
        :return: None
        """
        assert self.frozen, "env has unfrozen"
        self.frozen = False


class OfflineEnv(gym.Env):
    def __init__(self, env_params: Dict[str, Union[str, int, float]]):
        self.env_name = self.__class__.__name__[:-3]
        self.env_params = env_params

        self._offline_dataset_urls = {}
        self._offline_dataset_names = []
        if self.env_name in URL_INFOS:
            if self.env_params_name in URL_INFOS[self.env_name]:
                self._offline_dataset_urls = URL_INFOS[self.env_name][self.env_params_name]
                self._offline_dataset_names = list(self._offline_dataset_urls.keys())

    @property
    def dataset_names(self) -> list:
        return self._offline_dataset_names

    @property
    def env_params_name(self):
        return "&".join("{}={}".format(key, self.env_params[key]) for key in sorted(self.env_params.keys()))

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
        with h5py.File(h5path, "r") as f:
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
        env_name, param, dataset_name = dataset_url.split("/")[-3:]
        dataset_dir = DATASET_PATH / env_name / param.replace("%3D", "=").replace("%26", "&")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir / dataset_name

    def download_dataset(self, dataset_url: Union[str, pathlib.Path]) -> pathlib.Path:
        """
        Return the data file path corresponding to the url, downloads if it does not exist.
        :param dataset_url: web url.
        :return: local file path.
        """
        dataset_filepath = self.get_path_from_url(dataset_url)
        if not dataset_filepath.exists():
            print("Downloading dataset:", dataset_url, "to", dataset_filepath)
            urllib.request.urlretrieve(dataset_url, dataset_filepath)
        if not dataset_filepath.exists():
            raise IOError("Failed to download dataset from %s" % dataset_url)
        return dataset_filepath

    def get_dataset(self, dataset_name: str) -> Dict[(str, np.ndarray)]:
        assert dataset_name in self._offline_dataset_urls

        url = self._offline_dataset_urls[dataset_name]
        h5path = self.download_dataset(url)

        data_dict = self.load_h5_data(h5path)

        # Run a few quick sanity checks
        for key in [
            "observations",
            "observations",
            "actions",
            "rewards",
            "dones",
            "timeouts",
        ]:
            assert key in data_dict, "Dataset is missing key %s" % key

        return data_dict


class EmeiEnv(Freezable, OfflineEnv):
    def __init__(self, env_params: Dict[str, Union[str, int, float]]):
        """
        Abstract class for all Emei environments to better support model-based RL and offline RL.
        """
        Freezable.__init__(self)
        OfflineEnv.__init__(self, env_params=env_params)
        self._transition_graph = None
        self._reward_mech_graph = None
        self._termination_mech_graph = None

    def get_transition_graph(self, repeat_times=1):
        g = self._transition_graph.copy()
        num_obs, num_action = (
            self.observation_space.shape[0],
            self.action_space.shape[0],
        )
        assert g.shape == (num_obs + num_action, num_obs)

        if repeat_times == 1:
            return g
        else:
            aug_g = np.zeros([num_obs + num_action, num_obs + num_action])
            aug_g[:, :num_obs] = g.copy()

            prod_g = aug_g.copy()
            sum_g = np.zeros([num_obs + num_action, num_obs + num_action])
            for i in range(repeat_times):
                sum_g += prod_g
                prod_g = np.matmul(prod_g, aug_g)
            return (sum_g > 0).astype(int)[:, :num_obs]

    def get_reward_mech_graph(self):
        return self._reward_mech_graph

    def get_termination_mech_graph(self):
        return self._termination_mech_graph

    def transform_state_to_obs(self, batch_state):
        return batch_state.copy()

    def transform_obs_to_state(self, batch_obs):
        return batch_obs.copy()

    @abstractmethod
    def get_batch_init_state(self, batch_size):
        raise NotImplementedError

    def get_batch_init_obs(self, batch_size):
        return self.transform_state_to_obs(self.get_batch_init_state(batch_size=batch_size))

    @abstractmethod
    def get_batch_reward(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        raise NotImplementedError

    @abstractmethod
    def get_batch_terminal(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        raise NotImplementedError

    @abstractmethod
    def get_batch_next_obs(self, obs, pre_obs=None, action=None, state=None, pre_state=None):
        assert self.frozen
        raise NotImplementedError
