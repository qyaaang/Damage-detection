# A Deep Learning-Based Framework for Damage Detection with Time Series

## Environmental Requirements

- Python 3.7
- Pytorch 1.6.0
- Sklearn 0.23.1
- Numpy 1.8.5
- Pandas 0.25.3
- Matplotlib 3.2.1

## Introduction

We propose a novel framework for damage detection consisted of two modules, namely signal processing and damage recognition. The module of signal processing aims at transferring the denoised signal into frequency domain. We design a network in form of encoder-decoder-encoder for the module of damage recognition. The network is trained with only undamaged samples to acquire the pattern in the undamaged state as the damage detector. The samples from different damage states are then fed into the trained network to output the probability of damage.

## Framework

![framework](https://github.com/qryang/Damage-detection/blob/main/figures/Framework.tiff)

## Architecture of damage recognition network

- Multilayer perceptron

![MLP](/Users/qunyang/Dropbox (Personal)/Work/Journal paper/Paper1/Figures/MLP.tiff)

- Convolution-Deconvolution

![Conv-DeConv](/Users/qunyang/Dropbox (Personal)/Work/Journal paper/Paper1/Figures/Conv2D.tiff)

## Probability of damage

- Multilayer perceptron

![MLP](https://github.com/qryang/Damage-detection/blob/main/figures/PoD_MLP.png)

- Convolution-Deconvolution

![Conv-DeConv](https://github.com/qryang/Damage-detection/blob/main/figures/PoD_Conv2D.png)

