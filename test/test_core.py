import emei
from emei import EmeiEnv

import gym
from gym.wrappers import TimeLimit, OrderEnforcing
from gym.wrappers.env_checker import PassiveEnvChecker


def test_env_params_name():
    env = EmeiEnv(env_params={"a": 3,
                              "b": 5,
                              "d": 0.33,
                              "c": "c"})
    assert env.env_params_name == "a=3&b=5&c=c&d=0.33"

    env = EmeiEnv(env_params=dict(freq_rate=1, time_step=0.02))
    assert env.env_params_name == "freq_rate=1&real_time_scale=0.02"


def test_freeze():
    env = EmeiEnv(env_params=dict(freq_rate=1, time_step=0.02))
    env.freeze()
    assert env.frozen

    env.unfreeze()
    assert not env.frozen