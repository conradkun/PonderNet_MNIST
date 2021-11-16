# PonderNet: complexity in MNIST

Welcome! This repo contains the code presented in this article on PonderNet; make sure to give it a look to learn more about the background!

As the title suggests, this is an implementation of PonderNet that solves the MNIST task. In order to learn more about how pondering works, we suggest two experiments: _interpolation_, where we train PonderNet to classify a vanilla MNIST, and _extrapolation_, where we train on slightly rotated images and evaluate on different degrees of rotation to account for different levels of complexity.

## Setup
As always, create a new virtual environment an install the requirements with:
```
> pip install -r requirements.txt
```
Note that depending on your OS you may want to install PyTorch separately to have it work with CUDA.

If you decide to use Weights & Biases as your logger (you probably should!) and it's your first time using it, you will also need to log in and follow the instructions when running:
```
> wandb login
```
## Running experiments
To run the two experiments, interpolation and extrapolation, just call `python` on the respective file:

```
> python run_interpolation.py
> python run_extrapolation.py
```

If you want to play around a bit, you can change the hyperparameters of the network in the file `config.py`.

## Notebook
This repo also maintains a notebook with nearly-identical code. You can run the notebook locally or use [this copy](https://colab.research.google.com/drive/1ZIfcrpV_Pv6WCfRMHJw9x_0PMolOn6O_?usp=sharing) on Google Colab.