from pathlib import Path

from setuptools import find_packages, setup


def parse_requirements_file(path):
    return [line.rstrip() for line in open(path, "r")]


reqs_main = parse_requirements_file("requirements/main.txt")
reqs_dev = parse_requirements_file("requirements/dev.txt")

init_str = Path("emei/__init__.py").read_text()
version = init_str.split("__version__ = ")[1].rstrip().strip('"')

setup(
    name="emei",
    packages=[package for package in find_packages() if package.startswith("emei")],
    package_data={
        "emei": [
            "envs/mujoco/assets/*.xml",
        ]
    },
    description="Emei is a toolkit for developing causal reinforcement learning algorithms.",
    author="Honglong Tian",
    url="https://github.com/FrankTianTT/emei",
    author_email="franktian424@qq.com",
    license="MIT",
    version=version,
    python_requires=">=3.7",
    install_requires=reqs_main,
    extras_require={
        "dev": reqs_main + reqs_dev,
    },
)
