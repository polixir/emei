from emei.envs.classic_control import ContinuousCartPoleHoldingEnv, ContinuousCartPoleSwingUpEnv, \
    ContinuousChargedBallCenteringEnv
from emei import Downloadable


class OfflineContinuousCartPoleHoldingEnv(ContinuousCartPoleHoldingEnv, Downloadable):
    def __init__(self):
        Downloadable.__init__(self)


class OfflineContinuousCartPoleSwingUpEnv(ContinuousCartPoleSwingUpEnv, Downloadable):
    def __init__(self):
        Downloadable.__init__(self)


class OfflineContinuousChargedBallCenteringEnv(ContinuousChargedBallCenteringEnv, Downloadable):
    def __init__(self):
        Downloadable.__init__(self)
