# -Image-classifier-using-Artificial-Neural-Networks-ANN-
# Simple Image Classifier using TensorFlow and Keras on MNIST dataset

This repository contains a simple image classifier built using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. In this README, we'll walk you through the code and explain how the classifier works.

## Prerequisites

Make sure you have the following libraries installed:

- TensorFlow (Version 2.14.0)
- Matplotlib
- Numpy
- Pandas
- Seaborn

You can install these libraries using pip:

```bash
pip install tensorflow matplotlib numpy pandas seaborn
```
Loading and Preprocessing the Data

  We start by loading the MNIST dataset, which consists of hand-written digits. The dataset is split into training and test sets. Here's a summary of the data:
  
    -Training data: 60,000 samples, each with a shape of 28x28 pixels.
    -Test data: 10,000 samples, also 28x28 pixels each.
  We scale the pixel values between 0 and 1 by dividing them by 255. Additionally, we create a validation dataset with 5,000 samples for model evaluation.

Visualizing Data
  We demonstrate how to visualize the data using Matplotlib and Seaborn. You can see an example image from the training data as well as a heatmap representation.

Creating the Neural Network
  We build a simple neural network using TensorFlow and Keras. The architecture consists of:
      Input layer: 28x28 neurons
      Hidden layer 1: 300 neurons with ReLU activation
      Hidden layer 2: 100 neurons with ReLU activation
      Output layer: 10 neurons with softmax activation for classification

Model Summary
  We print a summary of the model, which includes information about the layers and the number of parameters in each layer.

Compiling the Model
  We define the loss function, optimizer, and metrics for our model:
    Loss function: Sparse categorical cross-entropy
    Optimizer: Stochastic Gradient Descent (SGD)
    Metrics: Accuracy

The model is then compiled and ready for training.

Feel free to explore the code in this repository and adapt it for your image classification tasks. You can train the model and evaluate its performance on various datasets, not just MNIST.
