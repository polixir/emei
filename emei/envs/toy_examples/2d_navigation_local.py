class toyenv(Env):
    def __init__(self, mode=toymode.DIAG):
        super().__init__()

        self.observation_space = spaces.Box(np.zeros((2,)), np.ones((2,)), dtype=np.float32)
        self.action_space = spaces.Box(-np.ones((2,)), np.ones((2,)), dtype=np.float32)

        self.state = np.zeros((2,))

        self.mode = mode
        self.num_steps = 0

    def reset(self):
        self.num_steps = 0

        # by default starts the agent in bottom left
        self.state = np.random.random((2,)) * 0.15

        # if UP, starts the agent on the bottom, but not center
        if self.mode in [toymode.UP, toymode.UP_L]:
            if self.mode == toymode.UP_L or np.random.random() > 0.5:
                self.state[0] = np.random.random() * 0.4 + 0.6
            else:
                self.state[0] = np.random.random() * 0.4

        # if RIGHT, starts the agent on the left, but not center
        if self.mode in [toymode.RIGHT, toymode.RIGHT_L]:
            if self.mode == toymode.RIGHT_L or np.random.random() > 0.5:
                self.state[1] = np.random.random() * 0.4
            else:
                self.state[1] = np.random.random() * 0.4 + 0.6

        return self.state

    def render(self):

        plt.rcParams["figure.figsize"] = (3, 3)

        fig1, ax1 = plt.subplots()

        ax1.set_ylim([0, 1])
        ax1.set_xlim([0, 1])
        ax1.add_patch(plt.Circle(self.state, 0.04, color="b"))
        ax1.add_patch(plt.Rectangle((0.0, 0.4), 1, 0.2, facecolor="r", alpha=0.1))
        ax1.add_patch(plt.Rectangle((0.4, 0.0), 0.2, 1.0, facecolor="r", alpha=0.1))

        plt.show()
        clear_output(wait=True)

    def step(self, action):

        self.num_steps += 1

        if np.all(self.state > 0.5):
            self.state -= action[::-1] * 0.02

        self.state += action * 0.05

        reward = -1.0
        done = self.num_steps > 70

        if self.mode in [toymode.UP, toymode.UP_L]:
            if self.state[1] > 0.9:
                reward = 0.0
                done = True
        elif self.mode in [toymode.RIGHT, toymode.RIGHT_L]:
            if self.state[0] > 0.9:
                reward = 0.0
                done = True
        elif self.mode == toymode.DIAG:
            if np.all(self.state > 0.8) and np.all(self.state < 0.9):
                reward = 0.0
                done = True

        return self.state.copy(), reward, done, {}
