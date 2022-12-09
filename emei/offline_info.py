import requests
from bs4 import BeautifulSoup

ROOT_PATH = r"https://github.com/FrankTianTT/emei/raw/dev/offline_data"

URL_INFOS = {}

DATASETS = [
    "uniform",
    "SAC-random",
    "SAC-medium",
    "SAC-expert",
    "SAC-medium-replay",
    "SAC-expert-replay",
]


root_text = requests.get(ROOT_PATH).text
env_name_elements = BeautifulSoup(root_text, features="html.parser").find_all(
    "a", attrs={"class": "js-navigation-open Link--primary"}
)

for env_name_element in env_name_elements:
    env_name = env_name_element.text
    URL_INFOS[env_name] = {}

    env_text = requests.get("{}/{}".format(ROOT_PATH, env_name)).text
    params_elements = BeautifulSoup(env_text, features="html.parser").find_all(
        "a", attrs={"class": "js-navigation-open Link--primary"}
    )
    for params_element in params_elements:
        params = params_element.text
        URL_INFOS[env_name][params] = {}

        for dataset in DATASETS:
            URL_INFOS[env_name][params][dataset] = "{}/{}/{}/{}.h5".format(
                ROOT_PATH,
                env_name,
                params.replace("=", "%3D"),
                dataset,
            )
