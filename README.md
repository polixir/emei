<img src="doc/emei_logo.png" align="right" width="40%"/>

<a href="https://github.com/FrankTianTT/emei"><img src="https://github.com/FrankTianTT/emei/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://github.com/FrankTianTT/emei"><img src="https://codecov.io/github/FrankTianTT/emei/branch/main/graph/badge.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/FrankTianTT/causal-mbrl/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
<a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white"></a>
<a href="https://www.python.org/downloads/release/python-380/"><img src="https://img.shields.io/badge/python-3.8-brightgreen"></a>

# Emei

Emei is an open source Python library for developing of **causal model-based reinforcement learning** algorithms by
providing a standard API to communicate between learning algorithms and environments, as well as a standard set of
environments compliant with that API. Emei is a re-encapsulation of [Openai Gym](https://github.com/openai/gym)(will be
replaced by [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) soon).

To better support the **model-based** and the **causal** characteristics, Emei has the following features:

- providing a causal diagram corresponding to the environment
- friendly model-based RL interface
    - the reward and terminal functions that can be obtained directly
    - freeze and unfreeze is supported
- coming with offline dataset
- adjustment of frequency ratio is supported
    - for Mujoco, **forward-euler method** is added

# install

## install by cloning from github

```shell
# clone the repository
git clone https://github.com/FrankTianTT/emei.git
cd emei
# create conda env
conda create -n emei python=3.8
conda activate emei
# install emei and its dependent packages
pip install -e .
```

If there is no `cuda` in your device, it's convenient to install `cuda` and `pytorch` from conda directly (refer
to [pytorch](https://pytorch.org/get-started/locally/)):

````shell
# for example, in the case of cuda=11.3
conda install pytorch cudatoolkit=11.3 -c pytorch
````

## install using pip

coming soon.
