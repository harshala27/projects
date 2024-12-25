Project Overview

This project implements and trains a LeNet-5 Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using PyTorch. It evaluates model performance with metrics such as precision, recall, and F1-Score and supports testing custom images uploaded through Google Colab and Google Drive.

Features

LeNet-5 Architecture:

Two convolutional layers, three fully connected layers.

ReLU activation and average pooling.

Data Augmentation and Normalization:

Resizing, grayscale conversion, and normalization for better generalization.

Hyperparameter Tuning:

Configurable learning rates and batch sizes tested over multiple combinations.

Evaluation Metrics:

Confusion matrix, precision, recall, and F1-Score.

Custom Image Testing:

Supports single-image classification for user-uploaded images via Google Colab.

Dataset

MNIST Handwritten Digits Dataset

Train Set: 60,000 samples

Test Set: 10,000 samples

Requirements

Python 3.x

PyTorch

Torchvision

Matplotlib

Pillow (PIL)

OpenCV

Google Colab (for execution)

Install dependencies via pip:

pip install torch torchvision matplotlib pillow opencv-python-headless

File Structure

|-- lenet5_mnist.py  # Main script for training and evaluation
|-- test_images/     # Folder containing custom test images
|-- README.md        # Documentation

Usage

1. Training and Evaluation

Upload the code to Google Colab.

Download the MNIST dataset automatically via PyTorch.

Train the model and evaluate performance with confusion matrix and metrics.

2. Custom Image Testing

Mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')

Place test images in /content/drive/My Drive/test_images.

The script preprocesses and tests each image.

Outputs predictions and visualizations.

Key Results

Best Hyperparameters: Tested across learning rates and batch sizes.

Final Test Accuracy: ~98% on MNIST test set.

Custom Image Predictions: Accurate digit predictions from uploaded images.

Model Details

Architecture:

Conv1: 6 filters, 5x5 kernel, stride=1, padding=2

Conv2: 16 filters, 5x5 kernel, stride=1

FC Layers: 120, 84, and 10 output classes

Optimizer: Adam with learning rate 0.001
Loss Function: CrossEntropyLoss
Scheduler: StepLR to reduce learning rate during training

Metrics Example

Confusion Matrix:

[[980   0   1 ... 0   0   0]
 [  0 1134  1 ... 0   0   0]
 ...

Precision, Recall, F1-Score:

Class 0: Precision=0.99, Recall=0.98, F1-Score=0.98
Class 1: Precision=0.98, Recall=0.99, F1-Score=0.99
...
Average Precision: 0.99
Average Recall: 0.99
Average F1-Score: 0.99

References

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition.

PyTorch Documentation: https://pytorch.org/docs/stable/index.html

MNIST Dataset: http://yann.lecun.com/exdb/mnist/
