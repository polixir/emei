import numpy as np

from emei.envs.mujoco.hopper import HopperRunningEnv


def test_hopper_fn():
    env = HopperRunningEnv()

    is_healthy = env.is_healthy(np.ones([128, 12]))
    assert is_healthy.shape == (128,) and np.all(is_healthy)

    is_healthy = env.is_healthy(np.ones([128, 12]) * 101)
    assert is_healthy.shape == (128,) and not np.any(is_healthy)

    reward = env.get_batch_reward(
        next_obs=np.ones([128, 12]),
        pre_obs=np.ones([128, 12]),
        action=np.ones([128, 3]),
    )
    assert reward.shape == (128, 1)

    terminal = env.get_batch_terminal(
        next_obs=np.ones([128, 12]),
    )
    assert terminal.shape == (128, 1)


def test_hopper_step():
    env = HopperRunningEnv()
    env.reset()

    action = env.action_space.sample()
    obs, reward, terminal, truncated, info = env.step(action)

    assert obs.shape == (12, )
