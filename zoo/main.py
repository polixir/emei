import os
import hydra
from omegaconf import DictConfig, OmegaConf
import gym
import emei
from zoo.algorithm import sac

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    sac(cfg)


if __name__ == "__main__":
    main()
