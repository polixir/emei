from emei.envs.mujoco.inverted_pendulum import (
    BaseInvertedPendulumEnv,
    ReboundInvertedPendulumBalancingEnv,
    ReboundInvertedPendulumSwingUpEnv,
    BoundaryInvertedPendulumSwingUpEnv,
    BoundaryInvertedPendulumBalancingEnv,
)


def test_base_IP():
    env = BaseInvertedPendulumEnv()


def test_rebound_IP_balancing():
    env = ReboundInvertedPendulumBalancingEnv()
    env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        if terminal:
            assert True
            break


def test_rebound_IP_swingup():
    env = ReboundInvertedPendulumSwingUpEnv()
    env.reset()

    steps = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        steps += 1
        if terminal:
            assert False
        if steps > 100:
            break


def test_boundary_IP_balancing():
    env = BoundaryInvertedPendulumBalancingEnv()
    env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        if terminal:
            assert True
            break


def test_boundary_IP_swingup():
    env = BoundaryInvertedPendulumSwingUpEnv()
    env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        if terminal:
            assert True
            break
