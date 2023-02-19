from typing import cast, Optional
from collections import defaultdict

import gym
import hydra
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure
from omegaconf import OmegaConf

import emei
from emei.core import get_params_str
from zoo.util import rollout, save_as_h5, get_replay_buffer, save_rollout_info


def rollout_and_save(
        env,
        sample_num,
        model,
        deterministic,
        save_name
):
    samples, rollout_info = rollout(env, sample_num, model, deterministic)
    save_as_h5(samples, "{}.h5".format(save_name))
    save_rollout_info(rollout_info, save_name)

    print("{}: {}".format(save_name, rollout_info))


class SaveExtraObsCallback(BaseCallback):
    def __init__(
            self,
            verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.saved_info = defaultdict(list)

    def _on_step(self) -> bool:
        self.saved_info["extra_obs"].extend([info['extra_obs'] for info in self.locals["infos"]])
        self.saved_info["next_extra_obs"].extend([info['next_extra_obs'] for info in self.locals["infos"]])
        return True


class SaveMediumAndExpertData(BaseCallback):
    def __init__(
            self,
            rollout_env: gym.Env,
            algorithm_name: str,
            medium_reward_threshold: float,
            expert_reward_threshold: float,
            medium_sample_num: int,
            expert_sample_num: int,
            save_extra_obs_callback: SaveExtraObsCallback,
            verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.env = rollout_env
        self.algorithm_name = algorithm_name
        self.medium_reward_threshold = medium_reward_threshold
        self.expert_reward_threshold = expert_reward_threshold
        self.medium_sample_num = medium_sample_num
        self.expert_sample_num = expert_sample_num

        self.save_extra_obs_callback = save_extra_obs_callback

        self.reached_medium = False
        self.reached_expert = False

    def _on_step(self) -> bool:
        assert isinstance(self.parent, EvalCallback)

        best_mean_reward = float(self.parent.best_mean_reward)
        if best_mean_reward > self.medium_reward_threshold and not self.reached_medium:
            self.model.save("SAC-medium-agent")
            save_as_h5(
                get_replay_buffer(self.model, self.save_extra_obs_callback.saved_info),
                self.algorithm_name + "-medium-replay.h5"
            )
            rollout_and_save(
                self.env,
                self.medium_sample_num,
                self.model,
                False,
                self.algorithm_name + "-medium",
            )
            self.reached_medium = True
            print("medium reached!")

        if best_mean_reward > self.expert_reward_threshold and not self.reached_expert:
            self.model.save("SAC-expert-agent")
            save_as_h5(
                get_replay_buffer(self.model, self.save_extra_obs_callback.saved_info),
                self.algorithm_name + "-expert-replay.h5"
            )
            rollout_and_save(
                self.env,
                self.expert_sample_num,
                self.model,
                False,
                self.algorithm_name + "-expert",
            )
            self.reached_expert = True
            print("expert reached!")

        return not self.reached_expert


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg):
    if cfg.wandb:
        import wandb

        wandb.init(
            project="emei",
            group=cfg.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            sync_tensorboard=True,
        )

    partial_model = hydra.utils.instantiate(cfg.algorithm.agent)

    env = cast(emei.EmeiEnv, gym.make(cfg.task.env_id, **cfg.task.params))
    model: BaseAlgorithm = partial_model(env=env)

    eval_env = cast(emei.EmeiEnv, gym.make(cfg.task.env_id, **cfg.task.params))
    eval_env.reset(seed=cfg.seed)
    eval_env.action_space.seed(seed=cfg.seed)

    save_extra_obs_callback = SaveExtraObsCallback()
    save_offline_callback = SaveMediumAndExpertData(
        eval_env,
        algorithm_name=cfg.algorithm.name,
        medium_reward_threshold=cfg.task.medium_reward,
        expert_reward_threshold=cfg.task.expert_reward,
        medium_sample_num=cfg.task.medium_sample_num,
        expert_sample_num=cfg.task.expert_sample_num,
        save_extra_obs_callback=save_extra_obs_callback,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./",
        callback_on_new_best=save_offline_callback,
        log_path="./",
        n_eval_episodes=cfg.task.n_eval_episodes,
        eval_freq=cfg.task.eval_freq,
        deterministic=False,
        render=False,
    )

    # save random and uniform dataset
    model.save("SAC-random-agent")
    rollout_and_save(
        eval_env,
        cfg.task.random_sample_num,
        model,
        False,
        cfg.algorithm.name + "-random",
    )
    rollout_and_save(eval_env, cfg.task.uniform_sample_num, None, False, "uniform")

    logger = configure("tb", format_strings=["tensorboard"])
    model.set_logger(logger)
    model.learn(
        total_timesteps=cfg.task.num_steps,
        callback=CallbackList([eval_callback, save_extra_obs_callback])
    )


if __name__ == "__main__":
    OmegaConf.register_new_resolver("to_str", get_params_str)
    main()
