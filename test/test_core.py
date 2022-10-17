import emei
import gym
from emei.core import EmeiEnv


def test_env():
    env = gym.make("BoundaryInvertedPendulumSwingUp-v0")
    obs = env.reset()
    env.step(env.action_space.sample())
