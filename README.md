# ANN_ParallelComputing

## Overview

**ANN_ParallelComputing** is a deep learning project aimed at exploring a simple Artificial Neural Network (ANN) architecture and implementing parallel computing through OpenMP and OpenCL to enhance the performance of training and inference. This project focuses on deep learning applications using the MNIST dataset and CIFAR-10 dataset.

## Features and Goals

- Use a simple ANN architecture for deep learning.
- Implement parallelization on multi-core CPUs using OpenMP to accelerate the training process.
- Implement parallelization on GPUs or other accelerators using OpenCL for further performance improvement.
- Provide training and testing functionality for both MNIST and CIFAR-10 datasets.
- Support custom network configurations and hyperparameter tuning.

## Key Components

### `mnist.hpp`

Contains functionalities and structure definitions related to the MNIST dataset. The MNIST dataset is a classic handwritten digit recognition dataset commonly used for testing and experiments in deep learning and machine learning.

### `readCIFAR10.hpp`

Contains the definition of the `readCIFAR10` function, which is used to read the CIFAR-10 dataset. CIFAR-10 is a widely used image classification benchmark dataset containing images from multiple classes.

### `initGPU.hpp`

Contains functionalities for initializing GPU resources and uses OpenCL as the library for GPU computations. This library aids in configuring and preparing the GPU for parallel computation.

### `MNIST.hpp`

Contains functionalities and structure definitions for processing image data from the MNIST dataset. The MNIST dataset is a classic handwritten digit recognition benchmark dataset commonly used for experiments in deep learning and machine learning.

### `ANN.cpp`

Contains the implementation of a program for training an Artificial Neural Network (ANN). This program is used for building, training, and evaluating ANN models, utilizing the backpropagation algorithm for training.

### `ANNOCL.cpp`

An implementation of ANN.cpp with OpenCL integration.

### `ANNOMP.cpp`

An implementation of ANN.cpp with OpenMP integration.

### `CIFAR10.cpp`

Contains a C++ function for reading the CIFAR-10 dataset.

### `GPU.cpp`

Contains functionalities for executing GPU computations using OpenCL.

### `main.cpp`

Contains an example of implementing a deep learning autoencoder using the code.

### `MNIST.cpp`

Contains a C++ function for reading the MNIST dataset.

## Acknowledgments

This project is inspired by and receives support from Professor You-Ming Hsieh, and we appreciate the files and guidance provided by him. This project is for learning and educational purposes only and is not intended for any commercial use.
