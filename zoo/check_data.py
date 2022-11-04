import h5py
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def load_dataset(data_path):
    data_dict = {}
    with h5py.File(data_path, "r") as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    for key in [
        "observations",
        "observations",
        "actions",
        "rewards",
        "terminals",
        "timeouts",
    ]:
        assert key in data_dict, "Dataset is missing key %s" % key

    return data_dict


def show_obs_rew(data_dict):
    diff_obs = data_dict["next_observations"] - data_dict["observations"]
    rew = data_dict["rewards"]

    for i in range(data_dict["observations"].shape[-1]):
        plt.hist(data_dict["observations"][:, i], bins=50, log=True)
        plt.title(f"obs_dim{i}")
        plt.show()
    for i in range(diff_obs.shape[-1]):
        plt.hist(diff_obs[:, i], bins=50, log=True)
        plt.title(f"diff_obs_dim{i}")
        plt.show()
    plt.hist(rew, bins=50, log=True)
    plt.title(f"reward")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, default=None)
    args = parser.parse_args()

    data_dict = load_dataset(args.data_path)
    show_obs_rew(data_dict)
