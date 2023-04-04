# Star Wars Image Generator

[![license](https://img.shields.io/github/license/TomaszKaleczyc/star_wars_image_generator)](LICENSE)

The purpose of this project is to build a [Denoising Diffusion Probabilistic Model](https://arxiv.org/pdf/2006.11239.pdf) and train it to generate images of Star Wars characters. The model will be built from scratch to provide in-depth insight into how diffusion models work. Implementation will be done in [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/).

&nbsp;

## Resources
* Working environment pre-requisites: Ubuntu18.04 LTS / Python 3.6.9 / unzip / virtualenv / CUDA version >=11.6
* Dataset: [Star Wars Images dataset from Kaggle](https://www.kaggle.com/datasets/mathurinache/star-wars-images) - use: `make download-dataset` in the project root to collect

&nbsp;

## Project structure 

```
├── data                             # project source data
├── environment                      # project dependencies 
├── output                           # stored PyTorch Lightning output of training
└── src                              # project source code
    ├── dataset                      # dataset creation tools
    └── model                        # model definitions
```

&nbsp;

## Makefile

The project [Makefile](Makefile) allows performing the following tasks by typing the following commands in terminal from the project root:

* `create-env` - build virtual environment to run the project code
* `activate-env-command` - generate command to activate the project virtual environment
* `download-dataset` - download the Kaggle dataset (requires Kaggle account and token)
* `train` - run training based on the contents of [train.py](src/train.py)
* `purge-output` - deletes all output generated by PyTorch Lightning after the training
* `run-tensorboard` - creates link to tensorboard with training results

&nbsp;

## Methodology

The repository contains definitions of two modle architecture types that can be used for training:
* **Unet** - network described in [this paper](https://arxiv.org/pdf/2006.11239.pdf). Implementation based on [this repository by lucidrains](https://github.com/lucidrains/denoising-diffusion-pytorch)
* **Recurrent Interface Network (RIN)** - network described in [this paper](https://arxiv.org/abs/2212.11972). Implementation based on [this repository by lucidrains](https://github.com/lucidrains/recurrent-interface-network-pytorch/blob/main/rin_pytorch/rin_pytorch.py)
Many thanks to all the authors!