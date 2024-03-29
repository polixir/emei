from typing import Union, Optional, Callable
from abc import abstractmethod

from gym import logger, spaces
from gym.error import DependencyNotInstalled
import numpy as np

from emei import EmeiEnv


class BaseControlEnv(EmeiEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self, freq_rate: int = 1, real_time_scale: float = 0.02, integrator: str = "euler", render_mode: Optional[str] = None
    ):
        self.freq_rate = freq_rate
        self.real_time_scale = real_time_scale
        self.integrator = integrator

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = np.empty(0, dtype=np.float32)

        EmeiEnv.__init__(self, env_params=dict(freq_rate=freq_rate, real_time_scale=real_time_scale, integrator=integrator))

    def freeze(self) -> None:
        self.frozen_state = self.state.copy()

    def unfreeze(self) -> None:
        self.state = self.frozen_state.copy()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.state = self.get_batch_init_state(1)[0]
        return self.state.copy(), {}

    @abstractmethod
    def _extract_action(self, action: Union[int, np.ndarray]) -> np.ndarray:
        return action

    @abstractmethod
    def _dsdt(self, s_augmented: np.ndarray) -> np.ndarray:
        return s_augmented

    @abstractmethod
    def draw(self):
        pass

    def step(self, action: Union[int, np.ndarray]):
        if isinstance(action, int):
            action = np.asarray(action)

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        pre_obs = self.state.copy()

        x_action = self._extract_action(action)
        s_augmented = np.append(self.state, x_action)
        s_augmented_out = ODE_approximation(self._dsdt, s_augmented, self.real_time_scale, self.freq_rate)
        self.state = s_augmented_out[: len(self.state)]

        obs = self.state.copy()

        reward = self.get_batch_reward(obs[None], pre_obs[None], action[None])[0, 0]
        terminal = self.get_batch_terminal(obs[None], pre_obs[None], action[None])[0, 0]
        truncated = False
        info = {}

        return obs, reward, terminal, truncated, info

    def render(self):
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled("pygame is not installed, run `pip install gym[classic_control]`")

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_width, self.screen_height))

        self.draw()

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def ODE_approximation(
    derivs: Callable[[np.ndarray], np.ndarray], y0: np.ndarray, dt: float, steps: int, method: str = "euler"
):
    """
    Integrate 1-D or N-D system of ODEs.

    Example for 2D system:

        >>> def derivs(x):
        ...     d1 =  x[0] + 2*x[1]
        ...     d2 =  -3*x[0] + 4*x[1]
        ...     return d1, d2

        >>> y0 = np.array([1,2])
        >>> dt = 0.05
        >>> steps = 20
        >>> y_out = ODE_approximation(derivs, y0, dt, steps)

    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
        y0: initial state vector
        dt: interval btween two sample time
        steps: sample times

    Returns:
        y_out: approximation of the ODE
    """
    y = y0.copy()

    for _ in range(steps):
        if method == "euler":
            y += derivs(y) * dt
        elif method == "rk4":
            k1 = np.asarray(derivs(y))
            k2 = np.asarray(derivs(y + dt * k1 / 2))
            k3 = np.asarray(derivs(y + dt * k2 / 2))
            k4 = np.asarray(derivs(y + dt * k3))
            y += (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        else:
            raise NotImplementedError("approximation method `{}` is not suppoerted yet.".format(method))
    return y
