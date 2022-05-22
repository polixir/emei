import gym
import emei
import os
import torch
import stable_baselines3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3 import SAC


def sac(cfg):
    env = gym.make(cfg.env.name)
    eval_env = gym.make(cfg.env.name)

    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model',
                                 log_path='./logs/results', eval_freq=cfg.env.eval_freq)
    model = SAC('MlpPolicy', env, tensorboard_log="./log", batch_size=cfg.env.batch_size)

    model.learn(cfg.env.medium_timesteps, callback=eval_callback)
    return model


if __name__ == "__main__":
    pass