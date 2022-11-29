from typing import Optional

from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from emei.core import EmeiEnv


class Local2DNavigation(EmeiEnv):
    def __init__(
        self,
        mode: str = "DIAG",
        render_mode: Optional[str] = None,
    ):
        EmeiEnv.__init__(self, env_params=dict(mode=mode))

        self.mode = mode
        self.render_mode = render_mode

        self.observation_space = spaces.Box(np.zeros((2,)), np.ones((2,)), dtype=np.float32)
        self.action_space = spaces.Box(-np.ones((2,)), np.ones((2,)), dtype=np.float32)

        self.state = np.zeros((2,))

    def get_batch_init_state(self, batch_size):
        return self.np_random.uniform(low=-0.05, high=0.05, size=(batch_size, 4))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # by default starts the agent in bottom left
        self.state = np.random.random((2,)) * 0.15

        # if UP, starts the agent on the bottom, but not center
        if self.mode == "UP":
            if np.random.random() > 0.5:
                self.state[0] = np.random.random() * 0.4 + 0.6
            else:
                self.state[0] = np.random.random() * 0.4
        # if RIGHT, starts the agent on the left, but not center
        elif self.mode == "RIGHT":
            if np.random.random() > 0.5:
                self.state[1] = np.random.random() * 0.4
            else:
                self.state[1] = np.random.random() * 0.4 + 0.6

        return self.state

    def render(self):
        plt.rcParams["figure.figsize"] = (3, 3)

        fig1, ax1 = plt.subplots()

        ax1.set_ylim([0, 1])
        ax1.set_xlim([0, 1])
        ax1.add_patch(plt.Circle(tuple(self.state), 0.04, color="b"))
        ax1.add_patch(plt.Rectangle((0.0, 0.4), 1, 0.2, facecolor="r", alpha=0.1))
        ax1.add_patch(plt.Rectangle((0.4, 0.0), 0.2, 1.0, facecolor="r", alpha=0.1))

        plt.show()
        clear_output(wait=True)

    def step(self, action):
        if np.all(self.state > 0.5):
            self.state -= action[::-1] * 0.02

        self.state += action * 0.05

        reward = -1.0

        terminal = False
        if self.mode == "UP":
            if self.state[1] > 0.9:
                reward = 0.0
                terminal = True
        elif self.mode == "RIGHT":
            if self.state[0] > 0.9:
                reward = 0.0
                terminal = True
        elif self.mode == "DIAG":
            if np.all(self.state > 0.8) and np.all(self.state < 0.9):
                reward = 0.0
                terminal = True
        else:
            raise NotImplementedError

        return self.state.copy(), reward, terminal, False, {}
