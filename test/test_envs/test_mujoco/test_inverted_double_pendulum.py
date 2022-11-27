from emei.envs.mujoco.inverted_double_pendulum import (
    BaseInvertedDoublePendulumEnv,
    ReboundInvertedDoublePendulumBalancingEnv,
    ReboundInvertedDoublePendulumSwingUpEnv,
    BoundaryInvertedDoublePendulumSwingUpEnv,
    BoundaryInvertedDoublePendulumBalancingEnv,
)


def test_base_I2P():
    env = BaseInvertedDoublePendulumEnv()


def test_rebound_I2P_balancing():
    env = ReboundInvertedDoublePendulumBalancingEnv()
    env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        if terminal:
            assert True
            break


def test_rebound_I2P_swingup():
    env = ReboundInvertedDoublePendulumSwingUpEnv()
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


def test_boundary_I2P_balancing():
    env = BoundaryInvertedDoublePendulumBalancingEnv()
    env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        if terminal:
            assert True
            break


def test_boundary_I2P_swingup():
    env = BoundaryInvertedDoublePendulumSwingUpEnv()
    env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        if terminal:
            assert True
            break
