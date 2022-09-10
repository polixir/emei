import gym
import hydra
import emei
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.base_class import BaseAlgorithm

from zoo.util import rollout, save_as_h5, get_replay_buffer, save_rollout_info


def rollout_and_save(env,
                     sample_num,
                     model,
                     deterministic,
                     save_name):
    samples, rollout_info = rollout(env, sample_num, model, deterministic)
    save_as_h5(samples, "{}.h5".format(save_name))
    save_rollout_info(rollout_info, save_name)

    print("{}: {}".format(save_name, rollout_info))


class SaveMediumAndExpertData(BaseCallback):
    def __init__(self,
                 rollout_env: gym.Env,
                 algorithm_name: str,
                 medium_reward_threshold: float,
                 expert_reward_threshold: float,
                 medium_sample_num: int,
                 expert_sample_num: int,
                 verbose: int = 0):
        super().__init__(verbose=verbose)
        self.env = rollout_env
        self.algorithm_name = algorithm_name
        self.medium_reward_threshold = medium_reward_threshold
        self.expert_reward_threshold = expert_reward_threshold
        self.medium_sample_num = medium_sample_num
        self.expert_sample_num = expert_sample_num

        self.reached_medium = False
        self.reached_expert = False

    def _on_step(self) -> bool:
        assert isinstance(self.parent, EvalCallback)

        best_mean_reward = float(self.parent.best_mean_reward)
        if best_mean_reward > self.medium_reward_threshold and not self.reached_medium:
            save_as_h5(get_replay_buffer(self.model), "{}.h5".format(self.algorithm_name + "-medium-replay.h5"))
            rollout_and_save(self.env, self.medium_sample_num, self.model, False, self.algorithm_name + "-medium")
            self.model.save("SAC-medium")
            self.reached_medium = True
            print("medium reached!")

        if best_mean_reward > self.expert_reward_threshold and not self.reached_expert:
            save_as_h5(get_replay_buffer(self.model), "{}.h5".format(self.algorithm_name + "-expert-replay.h5"))
            rollout_and_save(self.env, self.expert_sample_num, self.model, False, self.algorithm_name + "-expert")
            self.model.save("SAC-expert")
            self.reached_expert = True
            print("expert reached!")

        return not self.reached_expert


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args):
    model: BaseAlgorithm = hydra.utils.instantiate(args.algorithm.agent)

    eval_env: emei.EmeiEnv = hydra.utils.instantiate(args.task.env)
    eval_env.reset(seed=args.seed)
    eval_env.action_space.seed(seed=args.seed)

    save_offline_callback = SaveMediumAndExpertData(eval_env,
                                                    algorithm_name=args.algorithm.name,
                                                    medium_reward_threshold=args.task.medium_reward,
                                                    expert_reward_threshold=args.task.expert_reward,
                                                    medium_sample_num=args.task.medium_sample_num,
                                                    expert_sample_num=args.task.expert_sample_num)
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path="./",
                                 callback_on_new_best=save_offline_callback,
                                 log_path="./",
                                 eval_freq=1000,
                                 deterministic=True,
                                 render=False)

    # save random and uniform dataset
    rollout_and_save(eval_env, args.task.random_sample_num, model, False, args.algorithm.name + "-random")
    rollout_and_save(eval_env, args.task.uniform_sample_num, None, False, "uniform")

    model.learn(total_timesteps=args.task.num_steps, callback=eval_callback, tb_log_name="tb")


if __name__ == "__main__":
    main()
