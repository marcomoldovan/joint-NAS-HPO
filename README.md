

<div align="center">

# Simultaneous Neural Architecture Search and Hyperparameter Optimization of a CNN

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

Your task is to automatically improve and analyze the performance of a neural network for a
fashion classification1 dataset. Instead of only considering the architecture and hyperparameters
seperately you should build a system to jointly optimize them.
You are allowed a maximum runtime of 6 hours. We have provided a standard vision model
as a baseline. In the end, you should convince us that you indeed improved the performance of the
network when compared to the default approach. To this end, you could consider one or several
of the following:

- (must) Apply HPO to obtain a well-performing hyperparameter configuration (e.g., BO or EAs);
- (must) Apply NAS (e.g., BOHB or DARTS) to improve the architecture of the network;
- (can) Extend the configuration space to cover preprocessing, data augmentation and regularization;
- (can) Apply one or several of the speedup techniques for HPO/NAS;
- (can) Apply meta-learning, such as algorithm selection or warmstarting, to improve the performance;
- (can) Apply a learning to learn approach to learn how to optimize the network;
- (can) Determine the importance of the algorithm’s hyperparameters; 

From the optional approaches (denoted by can), pick the ones that you think are most appropriate.
To evaluate your approach please choose the way you evaluate well; you could consider the
following:

- Measure and compare against the default performance of the given network;
- Plot a confusion matrix;
- Plot the performance of your AutoML approach over time;
- Apply a statistical test;

### Experimental Constrains

- Your code for making design decisions should run no longer than 6 hours (without additional validation) on a single machine.
- You can use any kind of hardware that is available to you. For example, you could also
consider using Google Colab (which repeatedly offers a VM with a GPU for at most
12h for free) or Amazon SageMaker (which offers quite some resources for free if you
are a first-time customer). Don’t forget to state in your paper what kind of hardware
you used!

### Metrics

- The final performance has to be measured in terms of missclassification error.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/marcomoldovan/cross-modal-speech-segment-retrieval
cd cross-modal-speech-segment-retrieval

# install the correct python version
sudo apt-get install python3.10 # Linux, Python 3.7 or higher
brew install python@3.10 #MacOS, Python 3.7 or higher
choco install python --version=3.9 # Windows, Python 3.7-3.9

# create python virtual environment and activate it
python3 -m venv myenv
source myenv/bin/activate

# if you have several version of python you can create a virtual environment with a specific version:
virtualenv --python=/usr/bin/<python3.x> myenv
myenv\Scripts\activate.bat

# [ALTERNATIVE] create conda environment
conda create -n myenv python=<3.x>
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```
#### Default training

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

#### Hyperparameter search

To run a hyperparameter search with [Optuna](https://optuna.org/) you can use the following command

```bash
python train.py -m hparams_search=fashion_mnist_optuna experiment=example
```

Running a hyperparameter sweep with [Weights and Biases](https://wandb.ai/site) is also supported.

```bash
wandb sweep configs/hparams_search/fashion_mnist_wandb.yaml
wandb agent <sweep_id>
```

**Note #1:** [configs/logger/wandb.yaml](configs/logger/wandb.yaml) contain my personal project and entitiy names. You can change them to your own to ensure proper logging.
**Note #2:** When running a wandb sweep on Windows you need to set the correct python executable in [configs/hparams_search/fashion_mnist_wandb.yaml](configs/hparams_search/fashion_mnist_wandb.yaml)