def random_policy_test(env, is_render=False):
    def render():
        if is_render:
            env.render()

    episode_len = 0
    episode_rewards = 0
    obs = env.reset()
    render()
    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        episode_len += 1
        episode_rewards += reward
        render()

        if done:
            obs = env.reset()
            render()
            print("episode length: {}\tepisode rewards: {}".format(episode_len, episode_rewards))
            episode_len = 0
            episode_rewards = 0
