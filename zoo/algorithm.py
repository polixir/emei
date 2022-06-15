import gym
import emei
import os
import torch
import stable_baselines3
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3 import SAC


def sac_train(env_name, eval_freq=1000, batch_size=256, reward_threshold=100, level="expert"):
    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_callback = EvalCallback(eval_env, eval_freq=eval_freq,
                                 callback_on_new_best=callback_on_best, verbose=1)
    model = SAC('MlpPolicy', env, tensorboard_log="./{}_log".format(level), batch_size=batch_size)

    model.learn(int(1e7), callback=eval_callback)
    return model


def sac_eval(env_name, model_path):
    env = gym.make(env_name)
    model = SAC.load(model_path)
    episode_reward = 0
    obs = env.reset()
    env.render()

    while True:
        action, state = model.predict(obs, deterministic=True)
        print(action)
        next_obs, reward, done, info = env.step(action)
        env.render()
        episode_reward += reward
        if done:
            obs = env.reset()
            env.render()
            print(episode_reward)
            episode_reward = 0
        else:
            obs = next_obs


if __name__ == "__main__":
    sac_eval("BoundaryInvertedDoublePendulumSwingUp-v0",
             r"C:\Users\frank\Documents\Project\emei\zoo_exp\zoo_exp\SAC\default\BoundaryInvertedDoublePendulumSwingUp-v0\2022.06.06\214549\medium_logs\best_model\best_model.zip")
