# Emei

Emei is an open source toolkit proposed for causal reinforcement learning. Here are the main features:

- providing a causal diagram corresponding to the environment;
- friendly model-based RL interface;
- coming with offline dataset.

# install

```shell
conda create -n emei python=3.8
conda activate emei
# install pytorch
conda install pytorch cudatoolkit=11.3 -c pytorch
# install stable-baselines3
pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests # -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# install this package
pip install -e .
```