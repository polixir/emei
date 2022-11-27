from emei.envs.classic_control.cartpole import BaseCartPoleEnv, CartPoleBalancingEnv, CartPoleSwingUpEnv


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
        if terminal:
            assert True
            break
