# AdaBoB-R Optimizer

**AdaBoB-R** is an adaptive optimization algorithm for deep learning, designed to provide stable and effective parameter updates using momentum, belief-based variance estimation, and robust adaptive step-size clipping.

## Overview

AdaBoB-R combines adaptive learning rates, momentum, bias correction, warmup, and robust bounds to optimize neural network training.

## Datasets

AdaBoB-R is evaluated on four image classification datasets:

- **MNIST** — 10 handwritten digit classes
- **CIFAR-10** — 10 image classes
- **CIFAR-100** — 100 image classes
- **Tiny ImageNet** — 200 image classes

## Model Architecture

The experimental framework uses ResNet-18 for image classification.

## Evaluation Metrics

The experimental framework includes:

- Training and test accuracy
- Training and test loss
- Precision, recall, and F1-score
- Convergence analysis
- Hyperparameter sensitivity analysis

## Experimental Results

Training scripts and CSV result files are provided in this repository for experimental analysis and reproducibility.


## Citation

If you use AdaBoB-R in your research, please cite the associated publication when its citation details become available.

## License

Please refer to the repository's LICENSE file, if provided.
