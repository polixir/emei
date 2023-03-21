from emei.envs.toy_examples.local_2d_navigation import Local2DNavigation


def test_2d_navigation():
    env = Local2DNavigation()
    env.reset()

    # while True:
    #     action = env.action_space.sample()
    #     next_obs, reward, terminal, truncated, info = env.step(action)
    #     # env.render()
    #     if terminal:
    #         assert True
    #         break
