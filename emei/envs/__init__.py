from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

# Hook to load plugins from entry points
_load_env_plugins()

# Classic
# ----------------------------------------
register(
    id="CartPoleHolding-v0",
    entry_point="emei.envs.classic_control:CartPoleHoldingEnv",
    max_episode_steps=500,
)
register(
    id="CartPoleSwingUp-v0",
    entry_point="emei.envs.classic_control:CartPoleSwingUpEnv",
    max_episode_steps=1000,
)
register(
    id="ContinuousCartPoleHolding-v0",
    entry_point="emei.envs.classic_control:ContinuousCartPoleHoldingEnv",
    max_episode_steps=500,
)
register(
    id="ContinuousCartPoleSwingUp-v0",
    entry_point="emei.envs.classic_control:ContinuousCartPoleSwingUpEnv",
    max_episode_steps=1000,
)
