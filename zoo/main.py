import os
import hydra
from omegaconf import DictConfig, OmegaConf
import gym
import emei
from zoo.algorithm import sac_train
from zoo.collect_data import collect_by_policy, collect_replay_buffer, save_as_h5


def train_and_collect(algo, cfg):
    # random dataset
    random_samples, random_mean_rewards = collect_by_policy(cfg.env.name,
                                                            cfg.env.params,
                                                            cfg.env.random_samples)
    save_as_h5(random_samples, "random.h5")
    # medium dataset
    medium_model = algo(cfg.env.name,
                        cfg.env.params,
                        cfg.env.eval_freq, cfg.env.batch_size, cfg.env.medium_reward, "medium")
    medium_replay_samples = collect_replay_buffer(medium_model)
    medium_samples, medium_mean_rewards = collect_by_policy(cfg.env.name,
                                                            cfg.env.params,
                                                            cfg.env.medium_samples, medium_model)
    save_as_h5(medium_replay_samples, "medium-replay.h5")
    save_as_h5(medium_samples, "medium.h5")
    # expert dataset
    expert_model = algo(cfg.env.name,
                        cfg.env.params,
                        cfg.env.eval_freq, cfg.env.batch_size, cfg.env.expert_reward, "expert")
    expert_replay_samples = collect_replay_buffer(expert_model)
    expert_samples, medium_mean_rewards = collect_by_policy(cfg.env.name,
                                                            cfg.env.params,
                                                            cfg.env.expert_samples, expert_model)
    save_as_h5(expert_replay_samples, "expert-replay.h5")
    save_as_h5(expert_samples, "expert.h5")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    train_and_collect(sac_train, cfg)


if __name__ == "__main__":
    main()
