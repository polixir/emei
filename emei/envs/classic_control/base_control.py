from typing import Union, Optional

from abc import abstractmethod

import numpy as np
import pygame

import gym
from emei import Freezable


class BaseControlEnv(gym.Env[np.ndarray, Union[int, np.ndarray]], Freezable):
    def __init__(self,
                 freq_rate: int = 1,
                 time_step: float = 0.02) -> None:
        super(BaseControlEnv, self).__init__()
        self.freq_rate = freq_rate
        self.time_step = time_step

        self.screen = None
        self.is_open = True
        self.clock = None
        self.state = None

    def freeze(self) -> None:
        self.frozen_state = self.state

    def unfreeze(self) -> None:
        self.state = self.frozen_state

    @abstractmethod
    def _ds_dt(self, s_augmented: np.ndarray) -> np.ndarray:
        return s_augmented

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

    def step(self, action: Union[int, np.ndarray]):
        extracted_action = self._extract_action(action)
        tau = self.time_step / self.freq_rate
        s_augmented = np.append(self.state, extracted_action)
        for i in range(self.freq_rate):
            ds_dt = self._ds_dt(s_augmented)
            s_augmented += ds_dt * tau
        self.state = s_augmented[:len(self.state)]
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
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}


    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.is_open = False
