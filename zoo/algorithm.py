import gym
import emei
import os
import torch
import stable_baselines3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3 import SAC


def sac_train(env_name, eval_freq=1000, batch_size=256, timesteps=int(1e6), level="expert"):
    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    eval_callback = EvalCallback(eval_env, best_model_save_path='./{}_logs/best_model'.format(level),
                                 log_path='./{}_logs/results'.format(level), eval_freq=eval_freq)
    model = SAC('MlpPolicy', env, tensorboard_log="./{}_log".format(level), batch_size=batch_size)

    model.learn(timesteps, callback=eval_callback)
    return model


def sac_eval(env_name, model_path):
    env = gym.make(env_name)
    model = SAC.load(model_path)
    episode_reward = 0
    obs = env.reset()
    env.render()

    while True:
        action, state = model.predict(obs, deterministic=True)

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
    sac_eval("BoundaryInvertedPendulumSwingUp-v0",
             r"C:\Users\frank\Documents\Project\emei\zoo_exp\SAC\default\BoundaryInvertedPendulumSwingUp-v0\2022.06.04\194730\expert_logs\best_model\best_model.zip")
