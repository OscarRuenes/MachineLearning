'''
Group Members: Lalith Vennapusa, Oscar Ruenes
Date: 10/30/2025
Title: CS 4375 Assignment 4 Neural Networks
Desc: Neural network implementation trained on given learning rate, training/testing data, 
number of iterations, and number of hidden layers and hidden nodes.
'''

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("train_path", type=str, help="File path to training data")
parser.add_argument("test_path", type=str, help="File path to test data")
parser.add_argument("num_hidden_layers", type=int, help="Number of hidden layers")
parser.add_argument("num_hidden_nodes", type=int, help="Number of hidden nodes")
parser.add_argument("learning_rate", type=float, help="Learning rate for training")
parser.add_argument("num_iterations", type=int, help="Number of training iterations")


args = parser.parse_args()

try:
    with open(args.train_path, 'r') as file:
        training_data_lines = file.read().splitlines()
except:
    print(f"Error with train_path occured")
    exit(1)

try:
    with open(args.test_path, 'r') as file:
        test_data_lines = file.read().splitlines()
except:
    print(f"Error with test_path occured")
    exit(1)

class NeuralNetworkNode:
    def __init__(self, num_inputs: int):
        self.weights = np.zeros(num_inputs)
        self.bias = 1.0
        self.inputs = []
        hidden = True

    def activate(self, inputs: np.ndarray) -> float:
        z = np.dot(self.weights, inputs) + self.bias
        return 1 / (1 + np.exp(-z))


def main():
    if len(vars(args)) != 6:
        print("Invalid number of arguments, use train_path, test_path, num_hidden_layers, num_hidden_nodes, learning_rate, num_iterations")
        exit(1)
    neural_network = []
    for i in range(0, args.num_hidden_layers):
        print(f"Creating hidden layer {i+1} with {args.num_hidden_nodes} nodes")
        layer = []
        for j in range(0, args.num_hidden_nodes):
            node = NeuralNetworkNode(0)
            print(f"Created node {j+1} in hidden layer {i+1}")
            layer.append(node)
        neural_network.append(layer)
    print(f"Neural network structure created successfully: {neural_network}")

main()