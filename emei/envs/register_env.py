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
register(
    id="ChargedBallCentering-v0",
    entry_point="emei.envs.classic_control:ChargedBallCenteringEnv",
    max_episode_steps=500,
)
register(
    id="ContinuousChargedBallCentering-v0",
    entry_point="emei.envs.classic_control:ContinuousChargedBallCenteringEnv",
    max_episode_steps=1000,
)

# Mujoco
# ----------------------------------------
register(
    id="ReboundInvertedPendulumSwingUp-v0",
    entry_point="emei.envs.mujoco:ReboundInvertedPendulumSwingUpEnv",
    max_episode_steps=1000,
)
register(
    id="ReboundInvertedPendulumHolding-v0",
    entry_point="emei.envs.mujoco:ReboundInvertedPendulumHoldingEnv",
    max_episode_steps=1000,
)
register(
    id="BoundaryInvertedPendulumSwingUp-v0",
    entry_point="emei.envs.mujoco:BoundaryInvertedPendulumSwingUpEnv",
    max_episode_steps=1000,
)
register(
    id="BoundaryInvertedPendulumHolding-v0",
    entry_point="emei.envs.mujoco:BoundaryInvertedPendulumHoldingEnv",
    max_episode_steps=1000,
)
register(
    id="ReboundInvertedDoublePendulumSwingUp-v0",
    entry_point="emei.envs.mujoco:ReboundInvertedDoublePendulumSwingUpEnv",
    max_episode_steps=1000,
)
register(
    id="ReboundInvertedDoublePendulumHolding-v0",
    entry_point="emei.envs.mujoco:ReboundInvertedDoublePendulumHoldingEnv",
    max_episode_steps=1000,
)
register(
    id="BoundaryInvertedDoublePendulumSwingUp-v0",
    entry_point="emei.envs.mujoco:BoundaryInvertedDoublePendulumSwingUpEnv",
    max_episode_steps=1000,
)
register(
    id="BoundaryInvertedDoublePendulumHolding-v0",
    entry_point="emei.envs.mujoco:BoundaryInvertedDoublePendulumHoldingEnv",
    max_episode_steps=1000,
)