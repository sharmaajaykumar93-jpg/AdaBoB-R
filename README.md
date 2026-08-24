# AdaBoB-R

AdaBoB-R is a robust optimization algorithm that combines belief-based variance estimation with residual-aware dynamic learning-rate bounds. It is designed to provide stable and adaptive training in the presence of noisy or abnormal gradients.

## Features

* Belief-based variance estimation
* Residual-aware learning-rate adjustment
* Dynamic lower and upper learning-rate bounds
* Warm-up-based stabilization
* Robust handling of noisy gradients
* PyTorch implementation
* Experiments on multiple image-classification datasets
* Hyperparameter sensitivity analysis

## Repository Structure
AdaBoB-R/
├── AdaBoBR MNIST.py
├── AdaBobR Cifar10.py
├── AdaBobRcifar100.py
├── AdaBobR Tiny ImageNet.py
├── AdaBobR Sensitivity.py
└── README.md

The experiment files correspond to the following datasets:

* `AdaBoBR MNIST.py`: MNIST
* `AdaBobR Cifar10.py`: CIFAR-10
* `AdaBobRcifar100.py`: CIFAR-100
* `AdaBobR Tiny ImageNet.py`: Tiny ImageNet
* `AdaBobR Sensitivity.py`: Hyperparameter sensitivity analysis

## Requirements

The code requires Python 3 and the following packages:
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn

A CUDA-enabled GPU is recommended for faster training, especially for the Tiny ImageNet experiments.

## AdaBoB-R Parameters

The main AdaBoB-R hyperparameters are:

| Parameter | Description                                      | Default value |
| --------- | ------------------------------------------------ | ------------: |
| `alpha_0` | Initial learning rate                            |       `0.001` |
| `alpha_f` | Final learning-rate bound                        |         `0.1` |
| `beta1`   | Exponential decay rate for the first moment      |         `0.9` |
| `beta2`   | Exponential decay rate for the variance estimate |       `0.999` |
| `gamma`   | Convergence rate of the dynamic bounds           |       `0.001` |
| `eps`     | Numerical-stability constant                     |        `1e-8` |
| `lambda`  | Strength of residual-aware robust adjustment     |         `0.5` |
| `T_w`     | Number of warm-up iterations or epochs           |          `50` |

## Sensitivity Analysis

The sensitivity experiment evaluates the influence of the following AdaBoB-R parameters:

* Robustness coefficient (\lambda)
* Warm-up length (T_w)
* Bound-convergence parameter (\gamma)
* Final learning-rate bound (\alpha_f)

Each parameter is varied independently while the remaining parameters are fixed at their default values.

## Datasets

The repository includes experiments for:

* MNIST
* CIFAR-10
* CIFAR-100
* Tiny ImageNet

MNIST, CIFAR-10, and CIFAR-100 can be downloaded automatically through `torchvision`. Tiny ImageNet may need to be downloaded separately and placed in the directory expected by the corresponding experiment script.


## License

This project is intended for academic and research purposes. Please add an appropriate open-source license before redistributing or modifying the code.

