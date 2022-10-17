import emei
from emei import EmeiEnv

import gym
from gym.wrappers import TimeLimit, OrderEnforcing
from gym.wrappers.env_checker import PassiveEnvChecker


def test_env():
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0")
    assert isinstance(env, TimeLimit)
    assert isinstance(env.env, OrderEnforcing)
    assert isinstance(env.env.env, PassiveEnvChecker)
    assert isinstance(env.env.env.env, EmeiEnv)
    obs = env.reset()
    env.step(env.action_space.sample())
