import h5py
from typing import Union
import pathlib
import omegaconf


def to_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def save_as_h5(dataset, h5file_path):
    with h5py.File(h5file_path, 'w') as dataset_file:
        for key in dataset.keys():
            dataset_file[key] = dataset[key]


def load_hydra_cfg(results_dir: Union[str, pathlib.Path],
                   reset_device=None) -> omegaconf.DictConfig:
    results_dir = pathlib.Path(results_dir)
    cfg_file = results_dir / ".hydra" / "config.yaml"
    cfg = omegaconf.OmegaConf.load(cfg_file)
    if not isinstance(cfg, omegaconf.DictConfig):
        raise RuntimeError("Configuration format not a omegaconf.DictConf")

    if reset_device:
        cfg.device = reset_device
    return cfg
