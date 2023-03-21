import numpy as np

from emei.envs.classic_control.cartpole import (
    BaseCartPoleEnv,
    CartPoleBalancingEnv,
    CartPoleSwingUpEnv,
    ContinuousCartPoleBalancingEnv,
    ContinuousCartPoleSwingUpEnv,
)


def test_cartpole():
    env = BaseCartPoleEnv()

    try:
        env.reset()
        assert False
    except NotImplementedError:
        pass


def test_cartpole_balancing():
    env = CartPoleBalancingEnv()
    obs, info = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        if terminal:
            assert True
            break


def test_cartpole_swingup():
    env = CartPoleSwingUpEnv()
    obs, info = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        # env.render()
        if terminal:
            assert True
            break


def test_continuous_cartpole_balancing():
    env = ContinuousCartPoleBalancingEnv()
    obs, info = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        if terminal:
            assert True
            break


def test_continuous_cartpole_swingup():
    env = ContinuousCartPoleSwingUpEnv()
    obs, info = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        # env.render()
        if terminal:
            assert True
            break


def test_causal():
    env = ContinuousCartPoleSwingUpEnv()

    assert (
        env.get_transition_graph(5) == np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
    ).all()
