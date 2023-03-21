import numpy as np

from emei.envs.mujoco.half_cheetah import HalfCheetahRunningEnv


def test_step():
    env = HalfCheetahRunningEnv()
    obs, info = env.reset()
    assert obs.shape == (18,)

    action = env.action_space.sample()
    obs, reward, terminal, truncated, info = env.step(action)
    assert obs.shape == (18,)


# def test_render():
#     env = HalfCheetahRunningEnv(render_mode="rgb_array")
#     next_obs, info = env.reset()
#
#     action = env.action_space.sample()
#     next_obs, reward, terminal, truncated, info = env.step(action)
#     assert env.render().shape == (480, 480, 3)
