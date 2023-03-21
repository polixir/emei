import pathlib
import json

import requests
from bs4 import BeautifulSoup

EMEI_PATH = pathlib.Path.home() / ".emei"
DATASET_PATH = EMEI_PATH / "offline_data"
ROOT_URL = r"https://github.com/FrankTianTT/emei/tree/offline-data"

DATASETS = [
    "uniform",
    "SAC-random",
    "SAC-medium",
    "SAC-expert",
    "SAC-medium-replay",
    "SAC-expert-replay",
]


def update_url_info():
    url_infos = {}

    root_text = requests.get(ROOT_URL).text
    env_name_elements = BeautifulSoup(root_text, features="html.parser").find_all(
        "a", attrs={"class": "js-navigation-open Link--primary"}
    )

    for env_name_element in env_name_elements:
        env_name = env_name_element.text if "/" not in env_name_element.text else env_name_element.text.split("/")[0]
        url_infos[env_name] = {}

        env_text = requests.get("{}/{}".format(ROOT_URL, env_name)).text
        params_elements = BeautifulSoup(env_text, features="html.parser").find_all(
            "a", attrs={"class": "js-navigation-open Link--primary"}
        )
        for params_element in params_elements:
            params = params_element.text
            url_infos[env_name][params] = {}

            for dataset in DATASETS:
                url_infos[env_name][params][dataset] = "{}/{}/{}/{}.h5".format(
                    ROOT_URL.replace("/tree/", "/raw/"),
                    env_name,
                    params.replace("=", "%3D"),
                    dataset,
                )

    with open(EMEI_PATH / "url_infos.json", "w") as f:
        json.dump(url_infos, f, indent=4)


if not (EMEI_PATH / "url_infos.json").exists():
    EMEI_PATH.mkdir(exist_ok=True)
    update_url_info()

with open(EMEI_PATH / "url_infos.json", "r") as f:
    URL_INFOS = json.load(f)


if __name__ == '__main__':
    update_url_info()