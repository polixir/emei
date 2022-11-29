ROOT_PATH = r"https://github.com/FrankTianTT/emei/raw/dev/offline_data"

ENV_NAMES = [
    "BoundaryInvertedPendulumBalancing",
    "BoundaryInvertedPendulumSwingUp",
    "BoundaryInvertedDoublePendulumBalancing",
    "BoundaryInvertedDoublePendulumSwingUp",
    "HopperRunning",
    "Walker2dRunning",
    "HalfCheetahRunning",
    "AntRunning",
    "HumanoidRunning",
    "SwimmerRunning",
]
DATASETS = [
    "uniform",
    "SAC-random",
    "SAC-medium",
    "SAC-expert",
    "SAC-medium-replay",
    "SAC-expert-replay",
]
params = [
    "freq_rate=1,integrator=euler,real_time_scale=0.02",
    "freq_rate=2,integrator=euler,real_time_scale=0.01",
    "freq_rate=5,integrator=euler,real_time_scale=0.01",
]

URL_INFOS = {}
for env_name in ENV_NAMES:
    URL_INFOS[env_name] = {}
    for param in params:
        URL_INFOS[env_name][param] = {}
        for dataset in DATASETS:
            URL_INFOS[env_name][param][dataset] = "{}/{}-v0/{}/{}.h5".format(
                ROOT_PATH,
                env_name,
                param.replace("=", "%3D").replace("&", "%26"),
                dataset,
            )
