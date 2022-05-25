from typing import Union, Optional

from abc import abstractmethod

import numpy as np
import pygame

import gym
from emei import Freezable, Downloadable


class BaseControlEnv(gym.Env[np.ndarray, Union[int, np.ndarray]], Freezable, Downloadable):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02) -> None:
        Freezable.__init__(self)
        Downloadable.__init__(self)
        self.freq_rate = freq_rate
        self.time_step = time_step

        self.screen = None
        self.is_open = True
        self.clock = None
        self.state = None
        self.screen_width = 600
        self.screen_height = 400

    def freeze(self) -> None:
        self.frozen_state = self.state

    def unfreeze(self) -> None:
        self.state = self.frozen_state

    @abstractmethod
    def _get_update_info(self, action):
        return []

    @abstractmethod
    def _extract_action(self, action: Union[int, np.ndarray]) -> np.ndarray:
        return action

    @abstractmethod
    def _is_terminal(self) -> bool:
        return False

    @abstractmethod
    def _get_reward(self) -> float:
        return 0.0

    @abstractmethod
    def _get_initial_state(self):
        return 0.0

    def _get_obs(self):
        return self.state

    def _set_state(self, obs):
        self.state = obs

    def query(self, obs, action):
        self.freeze()
        self._set_state(obs)
        obs, reward, done, info = self.step(action)
        self.unfreeze()
        return obs, reward, done, info

    def update_state(self, updated):
        self.state += updated * (self.time_step / self.freq_rate)

    def step(self, action: Union[int, np.ndarray]):
        extracted_action = self._extract_action(action)
        for i in range(self.freq_rate):
            updated = self._get_update_info(extracted_action)
            self.update_state(updated)
        return self._get_obs(), self._get_reward(), self._is_terminal(), {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self._get_initial_state()
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}

    def draw(self):
        pass

    def render(self, mode="human"):
        if self.state is None:
            return None

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_width, self.screen_height))

        self.draw()

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.is_open

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.is_open = False
