# Feedforward Neural Network with Visualization

![image](https://github.com/user-attachments/assets/e39698f0-c275-434a-9f47-0ce5a0aaee70)


This repository contains a simple implementation of a feedforward neural network with one hidden layer. The network can be trained using backpropagation with either the **Sigmoid** or **ReLU** activation functions. The neural network is designed to solve the XOR problem, where the input data consists of 4 binary vectors, and the output is the XOR of the inputs.

## Features
- **Activation Functions**: Supports both **Sigmoid** and **ReLU** activation functions.
- **Training**: Implements basic backpropagation to train the network on sample data.
- **Visualization**: Provides a visualization of the neural network's architecture, including nodes, layers, and edge weights, using `networkx` and `matplotlib`.
- **Loss Function**: Uses Mean Squared Error (MSE) for loss calculation during training.

## Requirements
This project requires the following Python libraries:
- **Python 3.x**
- **NumPy**
- **matplotlib**
- **networkx**

You can install the required libraries with the following command:

```bash
pip install numpy matplotlib networkx
