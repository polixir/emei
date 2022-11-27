import numpy as np

from emei.envs.mujoco.hopper import HopperRunningEnv


def test_hopper_fn():
    env = HopperRunningEnv()

    is_healthy = env.is_healthy(np.ones([128, 12]))
    assert is_healthy.shape == (128,) and np.all(is_healthy)

    is_healthy = env.is_healthy(np.ones([128, 12]) * 101)
    assert is_healthy.shape == (128,) and not np.any(is_healthy)

    reward = env.get_batch_reward(
        obs=np.ones([128, 12]),
        pre_obs=np.ones([128, 12]),
        action=np.ones([128, 3]),
    )
    assert reward.shape == (128, 1)

    terminal = env.get_batch_terminal(
        obs=np.ones([128, 12]),
    )
    assert terminal.shape == (128, 1)


def test_hopper_step():
    env = HopperRunningEnv()
    obs, info = env.reset()
    assert obs.shape == (12,)

    action = env.action_space.sample()
    obs, reward, terminal, truncated, info = env.step(action)
    assert obs.shape == (12,)


def test_hopper_noise():
    env = HopperRunningEnv(obs_noise_params=0.01)
    obs, info = env.reset()

    action = env.action_space.sample()
    obs, reward, terminal, truncated, info = env.step(action)
    assert obs.shape == (12,)

    env = HopperRunningEnv(obs_noise_params={0: (0.1, 0.1)})
    obs, info = env.reset()

    action = env.action_space.sample()
    obs, reward, terminal, truncated, info = env.step(action)
    assert obs.shape == (12,)


def test_hopper_render():
    env = HopperRunningEnv(render_mode="rgb_array")
    obs, info = env.reset()

    action = env.action_space.sample()
    obs, reward, terminal, truncated, info = env.step(action)
    assert env.render().shape == (480, 480, 3)
