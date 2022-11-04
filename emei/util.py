import time
import math


def random_policy_test(env, is_render=False, sleep=None, default_action=None):
    def render():
        if is_render:
            env.render()

    episode_len = 0
    episode_rewards = 0
    obs = env.reset()
    render()
    while True:
        action = env.action_space.sample() if default_action is None else default_action
        next_obs, reward, terminal, truncated, _ = env.step(action)

        episode_len += 1
        episode_rewards += reward
        render()

        if terminal or truncated:
            obs = env.reset()
            render()
            print(
                "episode length: {}\tepisode rewards: {}".format(
                    episode_len, episode_rewards
                )
            )
            episode_len = 0
            episode_rewards = 0
        else:
            obs = next_obs

        if sleep is not None:
            time.sleep(sleep)
