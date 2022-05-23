ROOT_PATH = ""
ENV_NAMES = ["ContinuousCartPoleHolding", "ContinuousCartPoleSwingUpEnv", "ContinuousChargedBallCenteringEnv"]
DATASETS = ["random", "medium", "expert", "medium-replay", "expert-replay"]

URL_INFOS = {}
for env_name in ENV_NAMES:
    URL_INFOS[env_name] = {}
    for dataset in DATASETS:
        URL_INFOS[env_name][dataset] = "{}/{}-{}-v0.h5".format(ROOT_PATH, dataset, env_name)
